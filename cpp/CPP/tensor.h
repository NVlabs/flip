/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES
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

// Code by Pontus Andersson, Jim Nilsson, and Tomas Akenine-Moller.

#if 0
#pragma once

#include <algorithm>
#include <cstdlib>

#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "sharedflip.h"


namespace FLIP
{

    static const float ToneMappingCoefficients[3][6] =
    {
        { 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f },                                                 // Reinhard.
        { 0.6f * 0.6f * 2.51f, 0.6f * 0.03f, 0.0f, 0.6f * 0.6f * 2.43f, 0.6f * 0.59f, 0.14f },  // ACES, 0.6 is pre-exposure cancellation.
        { 0.231683f, 0.013791f, 0.0f, 0.18f, 0.3f, 0.018f },                                    // Hable.
    };

    union int3
    {
        struct { int x, y, z; };
    };

    template<typename T = color3>
    class tensor
    {
    protected:
        int3 mDim;
        int mArea, mVolume;
        T* mvpHostData;

    protected:

        bool allocateHost(void)
        {
            this->mvpHostData = (T*)malloc(this->mVolume * sizeof(T));

            if (this->mvpHostData == nullptr)
            {
                return false;
            }

            return true;
        }

        void init(const int3 dim, bool bClear = false, T clearColor = T(0.0f))
        {
            this->mDim = dim;
            this->mArea = dim.x * dim.y;
            this->mVolume = dim.x * dim.y * dim.z;

            allocateHost();

            if (bClear)
            {
                this->clear(clearColor);
            }
        }

    public:

        tensor()
        {
        }

        tensor(const int width, const int height, const int depth)
        {
            this->init({ width, height, depth });
        }

        tensor(const int width, const int height, const int depth, const T clearColor)
        {
            this->init({ width, height, depth }, true, clearColor);
        }

        tensor(const int3 dim, const T clearColor)
        {
            this->init(dim, true, clearColor);
        }

        tensor(tensor& image)
        {
            this->init(image.mDim);
            this->copy(image);
        }

        tensor(const color3* pColorMap, int size)
        {
            this->init({ size, 1, 1 });
            memcpy(this->mvpHostData, pColorMap, size * sizeof(color3));
        }

        ~tensor(void)
        {
            free(this->mvpHostData);
        }

        T* getHostData(void)
        {
            return this->mvpHostData;
        }

        inline int index(int x, int y = 0, int z = 0) const
        {
            return (z * this->mDim.y + y) * mDim.x + x;
        }

        T get(int x, int y, int z) const
        {
            return this->mvpHostData[this->index(x, y, z)];
        }

        void set(int x, int y, int z, T value)
        {
            this->mvpHostData[this->index(x, y, z)] = value;
        }

        int3 getDimensions(void) const
        {
            return this->mDim;
        }

        int getWidth(void) const
        {
            return this->mDim.x;
        }

        int getHeight(void) const
        {
            return this->mDim.y;
        }

        int getDepth(void) const
        {
            return this->mDim.z;
        }

        void colorMap(tensor<float>& srcImage, tensor<color3>& colorMap)
        {
            for (int z = 0; z < this->getDepth(); z++)
            {
#pragma omp parallel for
                for (int y = 0; y < this->getHeight(); y++)
                {
                    for (int x = 0; x < this->getWidth(); x++)
                    {
                        this->set(x, y, z, colorMap.get(int(srcImage.get(x, y, z) * 255.0f + 0.5f) % colorMap.getWidth(), 0, 0));
                    }
                }
            }
        }

        void sRGB2YCxCz(void)
        {
            for (int z = 0; z < this->getDepth(); z++)
            {
#pragma omp parallel for
                for (int y = 0; y < this->getHeight(); y++)
                {
                    for (int x = 0; x < this->getWidth(); x++)
                    {
                        this->set(x, y, z, color3::XYZ2YCxCz(color3::LinearRGB2XYZ(color3::sRGB2LinearRGB(this->get(x, y, z)))));
                    }
                }
            }
        }

        void LinearRGB2sRGB(void)
        {
            for (int z = 0; z < this->getDepth(); z++)
            {
#pragma omp parallel for
                for (int y = 0; y < this->getHeight(); y++)
                {
                    for (int x = 0; x < this->getWidth(); x++)
                    {
                        this->set(x, y, z, color3::LinearRGB2sRGB(this->get(x, y, z)));
                    }
                }
            }
        }

        void clear(const T color = T(0.0f))
        {
            for (int z = 0; z < this->getDepth(); z++)
            {
#pragma omp parallel for
                for (int y = 0; y < this->getHeight(); y++)
                {
                    for (int x = 0; x < this->getWidth(); x++)
                    {
                        this->set(x, y, z, color);
                    }
                }
            }
        }

        void clamp(float low = 0.0f, float high = 1.0f)
        {
            for (int z = 0; z < this->getDepth(); z++)
            {
#pragma omp parallel for
                for (int y = 0; y < this->getHeight(); y++)
                {
                    for (int x = 0; x < this->getWidth(); x++)
                    {
                        this->set(x, y, z, color3::clamp(this->get(x, y, z), low, high));
                    }
                }
            }
        }

        void toneMap(std::string tm)
        {
            int toneMapper = 1;
            if (tm == "reinhard")
            {
                for (int z = 0; z < this->getDepth(); z++)
                {
                    for (int y = 0; y < this->getHeight(); y++)
                    {
                        for (int x = 0; x < this->getWidth(); x++)
                        {
                            color3 color = this->get(x, y, z);
                            float luminance = color3::linearRGB2Luminance(color);
                            float factor = 1.0f / (1.0f + luminance);
                            this->set(x, y, z, color * factor);
                        }
                    }
                }
                return;
            }

            if (tm == "aces")
                toneMapper = 1;
            if (tm == "hable")
                toneMapper = 2;

            for (int z = 0; z < this->getDepth(); z++)
            {
#pragma omp parallel for
                for (int y = 0; y < this->getHeight(); y++)
                {
                    for (int x = 0; x < this->getWidth(); x++)
                    {
                        const float* tc = ToneMappingCoefficients[toneMapper];
                        color3 color = this->get(x, y, z);
                        this->set(x, y, z, color3(((color * color) * tc[0] + color * tc[1] + tc[2]) / (color * color * tc[3] + color * tc[4] + tc[5])));
                    }
                }
            }
        }

        void copy(tensor<T>& srcImage)
        {
            if (this->mDim.x == srcImage.getWidth() && this->mDim.y == srcImage.getHeight() && this->mDim.z == srcImage.getDepth())
            {
                memcpy(this->mvpHostData, srcImage.getHostData(), this->mVolume * sizeof(T));
            }
        }

        void copyFloat2Color3(tensor<float>& srcImage)
        {
            for (int z = 0; z < this->getDepth(); z++)
            {
#pragma omp parallel for
                for (int y = 0; y < this->getHeight(); y++)
                {
                    for (int x = 0; x < this->getWidth(); x++)
                    {
                        this->set(x, y, z, color3(srcImage.get(x, y, z)));
                    }
                }
            }
        }

        bool load(const std::string& fileName, const int z = 0)
        {
            bool bOk = false;
            std::string extension = fileName.substr(fileName.find_last_of(".") + 1);
            if (extension == "png" || extension == "bmp" || extension == "tga")
            {
                bOk = this->imageLoad(fileName, z);
            }
            else if (extension == "exr")
            {
                bOk = this->exrLoad(fileName, z);
            }

            return bOk;
        }

        bool imageLoad(const std::string& filename, const int z = 0)
        {
            int width, height, bpp;
            unsigned char* pixels = stbi_load(filename.c_str(), &width, &height, &bpp, 3);
            if (!pixels)
            {
                return false;
            }

            this->init({ width, height, z + 1 });

#pragma omp parallel for
            for (int y = 0; y < this->mDim.y; y++)
            {
                for (int x = 0; x < this->mDim.x; x++)
                {
                    this->set(x, y, z, color3(&pixels[3 * this->index(x, y)]));
                }
            }
            delete[] pixels;

            return true;
        }

        inline static float fClamp(float value) { return std::max(0.0f, std::min(1.0f, value)); }

        bool pngSave(const std::string& filename, int z = 0)
        {
            unsigned char* pixels = new unsigned char[3 * this->mDim.x * this->mDim.y];

#pragma omp parallel for
            for (int y = 0; y < this->mDim.y; y++)
            {
                for (int x = 0; x < this->mDim.x; x++)
                {
                    int index = this->index(x, y);
                    color3 color = this->mvpHostData[this->index(x, y, z)];
                    pixels[3 * index + 0] = (unsigned char)(255.0f * fClamp(color.x) + 0.5f);
                    pixels[3 * index + 1] = (unsigned char)(255.0f * fClamp(color.y) + 0.5f);
                    pixels[3 * index + 2] = (unsigned char)(255.0f * fClamp(color.z) + 0.5f);
                }
            }

            int ok = stbi_write_png(filename.c_str(), this->mDim.x, this->mDim.y, 3, pixels, 3 * this->mDim.x);
            delete[] pixels;

            return (ok != 0);
        }

        bool exrLoad(const std::string& fileName, const int z = 0)
        {
            EXRVersion exrVersion;
            EXRImage exrImage;
            EXRHeader exrHeader;
            InitEXRHeader(&exrHeader);
            InitEXRImage(&exrImage);
            int width, height;

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

            width = exrImage.width;
            height = exrImage.height;

            this->init({ width, height, z + 1 });

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

            // 1 channel images can be loaded into either scalar or vector formats.
            if (exrHeader.num_channels == 1)
            {
                for (int y = 0; y < this->mDim.y; y++)
                {
                    for (int x = 0; x < this->mDim.x; x++)
                    {
                        float color(rawImgChn[0][this->index(x, y)]);
                        this->set(x, y, z, color3(color));
                    }
                }
                loaded = true;
            }

            // 2 channel images can only be loaded into vector2/3/4 formats.
            if (exrHeader.num_channels == 2)
            {
                assert(idxR != -1 && idxG != -1);

#pragma omp parallel for
                for (int y = 0; y < this->mDim.y; y++)
                {
                    for (int x = 0; x < this->mDim.x; x++)
                    {
                        size_t linearIdx = this->index(x, y);
                        color3 color;
                        color.x = rawImgChn[idxR][linearIdx];
                        color.y = rawImgChn[idxG][linearIdx];
                        this->set(x, y, z, color);
                    }
                }
                loaded = true;
            }

            // 3 channel images can only be loaded into vector3/4 formats.
            if (exrHeader.num_channels == 3)
            {
                assert(idxR != -1 && idxG != -1 && idxB != -1);

#pragma omp parallel for
                for (int y = 0; y < this->mDim.y; y++)
                {
                    for (int x = 0; x < this->mDim.x; x++)
                    {
                        size_t linearIdx = this->index(x, y);
                        color3 color;
                        color.x = rawImgChn[idxR][linearIdx];
                        color.y = rawImgChn[idxG][linearIdx];
                        color.z = rawImgChn[idxB][linearIdx];
                        this->set(x, y, z, color);
                    }
                }
                loaded = true;
            }

            // 4 channel images can only be loaded into vector4 formats.
            if (exrHeader.num_channels == 4)
            {
                assert(idxR != -1 && idxG != -1 && idxB != -1);

#pragma omp parallel for
                for (int y = 0; y < this->mDim.y; y++)
                {
                    for (int x = 0; x < this->mDim.x; x++)
                    {
                        size_t linearIdx = this->index(x, y);
                        color3 color;
                        color.x = rawImgChn[idxR][linearIdx];
                        color.y = rawImgChn[idxG][linearIdx];
                        color.z = rawImgChn[idxB][linearIdx];
                        this->set(x, y, z, color);
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

    };
}
#endif