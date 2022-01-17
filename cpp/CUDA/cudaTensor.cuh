/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES
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

#pragma once

#include <algorithm>
#include <cstdlib>

#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace FLIP
{
    enum class CombineOperation
    {
        Add,
        Subtract,
        Multiply,
        L1,
        L2
    };

}
#define USING_CUDA
#include "sharedflip.h"
#include "cudaKernels.cuh"

namespace FLIP
{
    const dim3 DEFAULT_KERNEL_BLOCK_DIM = { 32, 32, 1 };  //  1.2s

    enum class CudaTensorState
    {
        UNINITIALIZED,
        ALLOCATED,
        HOST_ONLY,
        DEVICE_ONLY,
        SYNCHRONIZED
    };

    template<typename T = color3>
    class tensor
    {
    private:
        CudaTensorState mState = CudaTensorState::UNINITIALIZED;

    protected:
        int3 mDim;
        int mArea, mVolume;
        dim3 mBlockDim, mGridDim;
        T* mvpHostData;
        T* mvpDeviceData;

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

        bool allocateDevice(void)
        {
            int deviceVolume = this->mGridDim.x * this->mGridDim.y * this->mGridDim.z * this->mBlockDim.x * this->mBlockDim.y * this->mBlockDim.z;
            cudaError cudaError = cudaMalloc((void**)&(this->mvpDeviceData), deviceVolume * sizeof(T));

            if (cudaError != cudaSuccess)
            {
                std::cerr << "cudaMalloc() failed: " << cudaGetErrorString(cudaError) << "\n";
                this->~tensor();
                return false;
            }

            return true;
        }

        void init(const int3 dim, bool bClear = false, T clearColor = T(0.0f))
        {
            this->mDim = dim;
            this->mArea = dim.x * dim.y;
            this->mVolume = dim.x * dim.y * dim.z;

            this->mGridDim.x = (this->mDim.x + this->mBlockDim.x - 1) / this->mBlockDim.x;
            this->mGridDim.y = (this->mDim.y + this->mBlockDim.y - 1) / this->mBlockDim.y;
            this->mGridDim.z = (this->mDim.z + this->mBlockDim.z - 1) / this->mBlockDim.z;

            cudaError_t cudaError = cudaSetDevice(0);
            if (cudaError != cudaSuccess)
            {
                std::cerr << "cudaSetDevice() failed: " << cudaGetErrorString(cudaError) << "\n";
                this->~tensor();
                exit(-1);
            }

            allocateDevice();
            allocateHost();

            this->mState = CudaTensorState::ALLOCATED;

            if (bClear)
            {
                this->clear(clearColor);
            }
        }

    public:

        tensor(const dim3 blockDim = DEFAULT_KERNEL_BLOCK_DIM)
        {
            this->mBlockDim = blockDim;
        }

        tensor(const int width, const int height, const int depth, const dim3 blockDim = DEFAULT_KERNEL_BLOCK_DIM)
        {
            this->mBlockDim = blockDim;
            this->init({ width, height, depth });
        }

        tensor(const int width, const int height, const int depth, const T clearColor, const dim3 blockDim = DEFAULT_KERNEL_BLOCK_DIM)
        {
            this->mBlockDim = blockDim;
            this->init({ width, height, depth }, true, clearColor);
        }

        tensor(const int3 dim, const T clearColor, const dim3 blockDim = DEFAULT_KERNEL_BLOCK_DIM)
        {
            this->mBlockDim = blockDim;
            this->init(dim, true, clearColor);
        }

        tensor(tensor& image, const dim3 blockDim = DEFAULT_KERNEL_BLOCK_DIM)
        {
            this->mBlockDim = blockDim;
            this->init(image.mDim);
            this->copy(image);
        }

        tensor(const color3* pColorMap, int size, const dim3 blockDim = DEFAULT_KERNEL_BLOCK_DIM)
        {
            this->mBlockDim = blockDim;
            this->init({ size, 1, 1 });

            cudaError_t cudaError = cudaMemcpy(this->mvpDeviceData, pColorMap, size * sizeof(color3), cudaMemcpyHostToDevice);
            if (cudaError != cudaSuccess)
            {
                std::cout << "copy() failed: " << cudaGetErrorString(cudaError) << "\n";
                exit(-1);
            }
            this->mState = CudaTensorState::DEVICE_ONLY;
        }

        ~tensor(void)
        {
            free(this->mvpHostData);
            cudaFree(this->mvpDeviceData);
        }

        T* getHostData(void)
        {
            return this->mvpHostData;
        }

        T* getDeviceData(const int z = 0)
        {
            return this->mvpDeviceData + z * this->mArea;
        }

        inline dim3 getBlockDim() const 
        {
            return mBlockDim;
        }

        inline dim3 getGridDim() const
        {
            return mGridDim;
        }

        inline int index(int x, int y = 0, int z = 0)
        {
            return (z * this->mDim.y + y) * mDim.x + x;
        }

        T get(int x, int y, int z)
        {
            this->synchronizeHost();
            return this->mvpHostData[this->index(x, y, z)];
        }

        void set(int x, int y, int z, T value)
        {
            this->synchronizeHost();
            this->mvpHostData[this->index(x, y, z)] = value;
            this->mState = CudaTensorState::HOST_ONLY;
        }

        inline void setState(CudaTensorState state)
        {
            this->mState = state;
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

        void synchronizeHost(void)
        {
            if (this->mState == CudaTensorState::DEVICE_ONLY)
            {
                cudaError_t cudaError = cudaMemcpy(this->mvpHostData, this->mvpDeviceData, this->mVolume * sizeof(T), cudaMemcpyDeviceToHost);
                if (cudaError != cudaSuccess)
                {
                    std::cout << "cudaMemcpy(), DEVICE -> HOST, failed: " << cudaGetErrorString(cudaError) << "\n";
                    exit(-1);
                }
                this->mState = CudaTensorState::SYNCHRONIZED;
            }
        }

        void synchronizeDevice(void)
        {
            if (this->mState == CudaTensorState::HOST_ONLY)
            {
                cudaError_t cudaError = cudaMemcpy(this->mvpDeviceData, this->mvpHostData, this->mVolume * sizeof(T), cudaMemcpyHostToDevice);
                if (cudaError != cudaSuccess)
                {
                    std::cout << "cudaMemcpy(), HOST -> DEVICE, failed: " << cudaGetErrorString(cudaError) << "\n";
                    exit(-1);
                }
                this->mState = CudaTensorState::SYNCHRONIZED;
            }
        }

        void colorMap(tensor<float>& srcImage, tensor<color3>& colorMap)
        {
            srcImage.synchronizeDevice();
            FLIP::kernelColorMap << <this->mGridDim, this->mBlockDim >> > (this->getDeviceData(), srcImage.getDeviceData(), colorMap.getDeviceData(), this->mDim, colorMap.getWidth());
            checkStatus("kernelColorMap");
            this->mState = CudaTensorState::DEVICE_ONLY;
        }

        void sRGB2YCxCz(void)
        {
            this->synchronizeDevice();
            kernelsRGB2YCxCz << <this->mGridDim, this->mBlockDim >> > (this->mvpDeviceData, this->mDim);
            checkStatus("kernelsRGB2YCxCz");
            this->mState = CudaTensorState::DEVICE_ONLY;
        }

        void YCxCz2CIELab(void)
        {
            this->synchronizeDevice();
            FLIP::kernelYCxCz2CIELab << <this->mGridDim, this->mBlockDim >> > (this->mvpDeviceData, this->mDim);
            checkStatus("kernelYCxCz2CIELab");
            this->mState = CudaTensorState::DEVICE_ONLY;
        }

        void LinearRGB2sRGB(void)
        {
            this->synchronizeDevice();
            FLIP::kernelLinearRGB2sRGB << <this->mGridDim, this->mBlockDim >> > (this->mvpDeviceData, this->mDim);
            checkStatus("kernelLinearRGB2sRGB");
            this->mState = CudaTensorState::DEVICE_ONLY;
        }

        static void checkStatus(std::string kernelName)
        {
            cudaError_t cudaError = cudaGetLastError();
            if (cudaError != cudaSuccess)
            {
                std::cerr << kernelName << "() failed: " << cudaGetErrorString(cudaError) << "\n";
                exit(-1);
            }

            // Used if debugging.
            if (true)
            {
                deviceSynchronize(kernelName);
            }
        }

        // Used if debugging.
        static void deviceSynchronize(std::string kernelName)
        {
            cudaError_t cudaError = cudaDeviceSynchronize();
            if (cudaError != cudaSuccess)
            {
                std::cerr << kernelName << "(): cudeDeviceSynchronize: " << cudaGetErrorString(cudaError) << "\n";
                exit(-1);
            }
        }

        void clear(const T color = T(0.0f))
        {
            FLIP::kernelClear << <this->mGridDim, this->mBlockDim >> > (this->mvpDeviceData, this->mDim, color);
            checkStatus("kernelClear");
            this->mState = CudaTensorState::DEVICE_ONLY;
        }

        void clamp(float low = 0.0f, float high = 1.0f)
        {
            this->synchronizeDevice();
            FLIP::kernelClamp << <this->mGridDim, this->mBlockDim >> > (this->mvpDeviceData, this->mDim, low, high);
            checkStatus("kernelClamp");
            this->mState = CudaTensorState::DEVICE_ONLY;
        }

        void toneMap(std::string tm)
        {
            int toneMapper = 1;
            if (tm == "reinhard")
                toneMapper = 0;
            if (tm == "aces")
                toneMapper = 1;
            if (tm == "hable")
                toneMapper = 2;

            FLIP::kernelToneMap << <this->mGridDim, this->mBlockDim >> > (this->mvpDeviceData, this->mDim, toneMapper);
            checkStatus("kernelToneMap");
            this->mState = CudaTensorState::DEVICE_ONLY;
        }

        void copy(tensor<T>& srcImage)
        {
            srcImage.synchronizeDevice();
            if (this->mDim.x == srcImage.getWidth() && this->mDim.y == srcImage.getHeight() && this->mDim.z == srcImage.getDepth())
            {
                cudaError_t cudaError = cudaMemcpy(this->mvpDeviceData, srcImage.getDeviceData(), this->mVolume * sizeof(T), cudaMemcpyDeviceToDevice);
                if (cudaError != cudaSuccess)
                {
                    std::cout << "copy() failed: " << cudaGetErrorString(cudaError) << "\n";
                    exit(-1);
                }
            }
            else
            {
                kernelBilinearCopy << <this->mGridDim, this->mBlockDim >> > (this->mvpDeviceData, srcImage.getDeviceData(), this->mDim, srcImage.getDimensions());
            }
            this->mState = CudaTensorState::DEVICE_ONLY;
        }

        void copyFloat2Color3(tensor<float>& srcImage)
        {
            srcImage.synchronizeDevice();
            FLIP::kernelFloat2Color3<<<this->mGridDim, this->mBlockDim>>>(this->mvpDeviceData, srcImage.getDeviceData(), this->mDim);
            checkStatus("kernelFloat2Color3");
            this->mState = CudaTensorState::DEVICE_ONLY;
        }

        bool load(const std::string& fileName, const int z = 0)
        {
            bool bOk = false;
            std::string extension = fileName.substr(fileName.find_last_of(".") + 1);
            if (extension == "png")
            {
                bOk = this->pngLoad(fileName, z);
            }
            else if (extension == "exr")
            {
                bOk = this->exrLoad(fileName, z);
            }

            return bOk;
        }

        bool pngLoad(const std::string& filename, const int z = 0)
        {
            int width, height, bpp;
            unsigned char* pixels = stbi_load(filename.c_str(), &width, &height, &bpp, 3);
            if (!pixels)
            {
                return false;
            }

            if (this->mState == CudaTensorState::UNINITIALIZED)
            {
                this->init({ width, height, z + 1 });
            }

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

            this->synchronizeHost();

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

            if (this->mState == CudaTensorState::UNINITIALIZED)
            {
                this->init({ width, height, z + 1 });
            }

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
