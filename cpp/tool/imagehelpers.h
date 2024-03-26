/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: BSD-3-Clause
 */

// Visualizing and Communicating Errors in Rendered Images
// Ray Tracing Gems II, 2021,
// by Pontus Andersson, Jim Nilsson, and Tomas Akenine-Moller.
// Pointer to the chapter: https://research.nvidia.com/publication/2021-08_Visualizing-and-Communicating.

// Visualizing Errors in Rendered High Dynamic Range Images
// Eurographics 2021,
// by Pontus Andersson, Jim Nilsson, Peter Shirley, and Tomas Akenine-Moller.
// Pointer to the paper: https://research.nvidia.com/publication/2021-05_HDR-FLIP.

// FLIP: A Difference Evaluator for Alternating Images
// High Performance Graphics 2020,
// by Pontus Andersson, Jim Nilsson, Tomas Akenine-Moller,
// Magnus Oskarsson, Kalle Astrom, and Mark D. Fairchild.
// Pointer to the paper: https://research.nvidia.com/publication/2020-07_FLIP.

// Code by Pontus Ebelin (formerly Andersson), Jim Nilsson, and Tomas Akenine-Moller.

#pragma once
#include <string>
#include <iostream>
#include <algorithm>

#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace ImageHelpers
{
    bool ldrLoad(const std::string& filename, int& imgWidth, int& imgHeight, float*& pixels)
    {
        int bpp;
        unsigned char* ldrPixels = stbi_load(filename.c_str(), &imgWidth, &imgHeight, &bpp, 3);
        if (!ldrPixels)
        {
            return false;
        }

        pixels = new float[3 * imgWidth * imgHeight];
#pragma omp parallel for
        for (int y = 0; y < imgHeight; y++)
        {
            for (int x = 0; x < imgWidth; x++)
            {
                int linearIdx = 3 * (y * imgWidth + x);
                pixels[linearIdx + 0] = ldrPixels[linearIdx + 0] / 255.0f;
                pixels[linearIdx + 1] = ldrPixels[linearIdx + 1] / 255.0f;
                pixels[linearIdx + 2] = ldrPixels[linearIdx + 2] / 255.0f;

            }
        }
        delete[] ldrPixels;
        return true;
    }

    bool hdrLoad(const std::string& fileName, int& imgWidth, int& imgHeight, float*& hdrPixels)
    {
        EXRVersion exrVersion;
        EXRImage exrImage;
        EXRHeader exrHeader;
        InitEXRHeader(&exrHeader);
        InitEXRImage(&exrImage);

        {
            int ret;
            const char* errorString;

            ret = ParseEXRVersionFromFile(&exrVersion, fileName.c_str());
            if (ret != TINYEXR_SUCCESS || exrVersion.multipart || exrVersion.non_image)
            {
                std::cerr << "Unsupported EXR version or type!" << std::endl;
                return false;
            }

            ret = ParseEXRHeaderFromFile(&exrHeader, &exrVersion, fileName.c_str(), &errorString);
            if (ret != TINYEXR_SUCCESS)
            {
                std::cerr << "Error loading EXR header: " << errorString << std::endl;
                return false;
            }

            for (int i = 0; i < exrHeader.num_channels; i++)
            {
                if (exrHeader.pixel_types[i] == TINYEXR_PIXELTYPE_HALF)
                {
                    exrHeader.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
                }
            }

            ret = LoadEXRImageFromFile(&exrImage, &exrHeader, fileName.c_str(), &errorString);
            if (ret != TINYEXR_SUCCESS)
            {
                std::cerr << "Error loading EXR file: " << errorString << std::endl;
                return false;
            }
        }

        imgWidth = exrImage.width;
        imgHeight = exrImage.height;

        int idxR = -1;
        int idxG = -1;
        int idxB = -1;
        int numRecognizedChannels = 0;
        for (int c = 0; c < exrHeader.num_channels; c++)
        {
            std::string channelName = exrHeader.channels[c].name;
            std::transform(channelName.begin(), channelName.end(), channelName.begin(), ::tolower);
            if (channelName == "r")
            {
                idxR = c;
                ++numRecognizedChannels;
            }
            else if (channelName == "g")
            {
                idxG = c;
                ++numRecognizedChannels;
            }
            else if (channelName == "b")
            {
                idxB = c;
                ++numRecognizedChannels;
            }
            else if (channelName == "a")
            {
                ++numRecognizedChannels;
            }
            else
            {
                std::cerr << "Undefined EXR channel name: " << exrHeader.channels[c].name << std::endl;
            }
        }
        if (numRecognizedChannels < exrHeader.num_channels)
        {
            std::cerr << "EXR channels may be loaded in the wrong order." << std::endl;
            idxR = 0;
            idxG = 1;
            idxB = 2;
        }

        auto rawImgChn = reinterpret_cast<float**>(exrImage.images);
        bool loaded = false;

        hdrPixels = new float[imgWidth * imgHeight * 3];

        // 1 channel images can be loaded into either scalar or vector formats.
        if (exrHeader.num_channels == 1)
        {
            for (int y = 0; y < imgHeight; y++)
            {
                for (int x = 0; x < imgWidth; x++)
                {
                    int linearIdx = y * imgWidth + x;
                    float color(rawImgChn[0][linearIdx]);
                    hdrPixels[3 * linearIdx + 0] = color;
                    hdrPixels[3 * linearIdx + 1] = color;
                    hdrPixels[3 * linearIdx + 2] = color;
                }
            }
            loaded = true;
        }

        // 2 channel images can only be loaded into vector2/3/4 formats.
        if (exrHeader.num_channels == 2)
        {
            assert(idxR != -1 && idxG != -1);

#pragma omp parallel for
            for (int y = 0; y < imgHeight; y++)
            {
                for (int x = 0; x < imgWidth; x++)
                {
                    int linearIdx = y * imgWidth + x;
                    hdrPixels[3 * linearIdx + 0] = rawImgChn[idxR][linearIdx];
                    hdrPixels[3 * linearIdx + 1] = rawImgChn[idxG][linearIdx];
                    hdrPixels[3 * linearIdx + 2] = 0.0f;
                }
            }
            loaded = true;
        }

        // 3 channel images can only be loaded into vector3/4 formats.
        if (exrHeader.num_channels == 3)
        {
            assert(idxR != -1 && idxG != -1 && idxB != -1);

#pragma omp parallel for
            for (int y = 0; y < imgHeight; y++)
            {
                for (int x = 0; x < imgWidth; x++)
                {
                    int linearIdx = y * imgWidth + x;
                    hdrPixels[3 * linearIdx + 0] = rawImgChn[idxR][linearIdx];
                    hdrPixels[3 * linearIdx + 1] = rawImgChn[idxG][linearIdx];
                    hdrPixels[3 * linearIdx + 2] = rawImgChn[idxB][linearIdx];
                }
            }
            loaded = true;
        }

        // 4 channel images can only be loaded into vector4 formats.
        if (exrHeader.num_channels == 4)
        {
            assert(idxR != -1 && idxG != -1 && idxB != -1);

#pragma omp parallel for
            for (int y = 0; y < imgHeight; y++)
            {
                for (int x = 0; x < imgWidth; x++)
                {
                    int linearIdx = y * imgWidth + x;
                    hdrPixels[3 * linearIdx + 0] = rawImgChn[idxR][linearIdx];
                    hdrPixels[3 * linearIdx + 1] = rawImgChn[idxG][linearIdx];
                    hdrPixels[3 * linearIdx + 2] = rawImgChn[idxB][linearIdx];
                }
            }
            loaded = true;
        }

        FreeEXRHeader(&exrHeader);
        FreeEXRImage(&exrImage);

        if (!loaded)
        {
            std::cerr << "Insufficient target channels when loading EXR: need " << exrHeader.num_channels << std::endl;
            return false;
        }
        else
        {
            return true;
        }
    }

    // Note that when an image us loaded, the variable pixels is allocated, and it is up to the user to deallocate it later.
    bool loadImage(const std::string& fileName, int& imgWidth, int& imgHeight, float*& pixels)
    {
        bool bOk = false;
        std::string extension = fileName.substr(fileName.find_last_of(".") + 1);
        if (extension == "png" || extension == "bmp" || extension == "tga")
        {
            bOk = ldrLoad(fileName, imgWidth, imgHeight, pixels);
        }
        else if (extension == "exr")
        {
            bOk = hdrLoad(fileName, imgWidth, imgHeight, pixels);
        }

        return bOk;
    }

    bool load(FLIP::image<FLIP::color3>& dstImage, const std::string& fileName)
    {
        int imgWidth;
        int imgHeight;
        float* pixels;
        if (loadImage(fileName, imgWidth, imgHeight, pixels))
        {
            dstImage.setPixels(pixels, imgWidth, imgHeight);
        }
        return false;
    }

    bool pngSave(const std::string& filename, FLIP::image<FLIP::color3>& image)
    {
        unsigned char* pixels = new unsigned char[3 * image.getWidth() * image.getHeight()];

#ifdef FLIP_ENABLE_CUDA
        image.synchronizeHost();
#endif

#pragma omp parallel for
        for (int y = 0; y < image.getHeight(); y++)
        {
            for (int x = 0; x < image.getWidth(); x++)
            {
                int index = image.index(x, y);
                FLIP::color3 color = image.get(x, y);

                pixels[3 * index + 0] = (unsigned char)(255.0f * std::clamp(color.x, 0.0f, 1.0f) + 0.5f);
                pixels[3 * index + 1] = (unsigned char)(255.0f * std::clamp(color.y, 0.0f, 1.0f) + 0.5f);
                pixels[3 * index + 2] = (unsigned char)(255.0f * std::clamp(color.z, 0.0f, 1.0f) + 0.5f);
            }
        }

        int ok = stbi_write_png(filename.c_str(), image.getWidth(), image.getHeight(), 3, pixels, 3 * image.getWidth());
        delete[] pixels;

        return (ok != 0);
    }

    bool exrSave(const std::string& fileName, FLIP::image<FLIP::color3>& image)
    {
#ifdef FLIP_ENABLE_CUDA
        image.synchronizeHost();
#endif
        constexpr int channels = 3;

        float* vpImage[channels] = {};
        std::vector<float> vImages[channels];
        for (int i = 0; i < channels; ++i)
        {
            vImages[i].resize(image.getWidth() * image.getHeight());
        }
        int pixelIndex = 0;
        for (int y = 0; y < image.getHeight(); y++)
        {
            for (int x = 0; x < image.getWidth(); x++)
            {
                FLIP::color3 p = image.get(x, y);
                vImages[0][pixelIndex] = p.r;
                vImages[1][pixelIndex] = p.g;
                vImages[2][pixelIndex] = p.b;
                pixelIndex++;
            }
        }
        vpImage[0] = &(vImages[2].at(0));  // B
        vpImage[1] = &(vImages[1].at(0));  // G
        vpImage[2] = &(vImages[0].at(0));  // R

        EXRHeader exrHeader;
        InitEXRHeader(&exrHeader);
        exrHeader.num_channels = channels;
        exrHeader.channels = (EXRChannelInfo*)malloc(channels * sizeof(EXRChannelInfo));
        exrHeader.pixel_types = (int*)malloc(channels * sizeof(int));
        exrHeader.requested_pixel_types = (int*)malloc(channels * sizeof(int));
        exrHeader.compression_type = TINYEXR_COMPRESSIONTYPE_ZIP;
        for (int i = 0; i < channels; i++)
        {
            exrHeader.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
            exrHeader.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
            exrHeader.channels[i].name[1] = '\0';
        }
        exrHeader.channels[0].name[0] = 'B';
        exrHeader.channels[1].name[0] = 'G';
        exrHeader.channels[2].name[0] = 'R';

        EXRImage exrImage;
        InitEXRImage(&exrImage);
        exrImage.num_channels = channels;
        exrImage.images = (unsigned char**)vpImage;
        exrImage.width = image.getWidth();
        exrImage.height = image.getHeight();

        const char* error;
        int ret = SaveEXRImageToFile(&exrImage, &exrHeader, fileName.c_str(), &error);
        if (ret != TINYEXR_SUCCESS)
        {
            std::cerr << "Failed to save EXR file <" << fileName << ">: " << error << "\n";
            return false;
        }

        free(exrHeader.channels);
        free(exrHeader.pixel_types);
        free(exrHeader.requested_pixel_types);

        return true;
    }
}