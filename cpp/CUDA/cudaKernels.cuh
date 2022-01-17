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

#include "stdio.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


namespace FLIP
{
    __constant__ struct
    {
        float gqc = 0.7f;
        float gpc = 0.4f;
        float gpt = 0.95f;
        float gw = 0.082f;
        float gqf = 0.5f;
    } DeviceFLIPConstants;

    enum ReduceOperation
    {
        Add,
        Max,
        Min
    };

    template <ReduceOperation op>
    inline __device__ float reduce(volatile float a, volatile float b)
    {
        switch (op)
        {
        default:
        case ReduceOperation::Add:
            return a + b;
            break;
        case ReduceOperation::Max:
            return Max(a, b);
            break;
        case ReduceOperation::Min:
            return Min(a, b);
            break;
        }
    }

    template <ReduceOperation op>
    inline __device__ color3 reduce(const color3& a, const color3& b)
    {
        switch (op)
        {
        default:
        case ReduceOperation::Add:
            return a + b;
            break;
        case ReduceOperation::Max:
            return color3::max(a, b);
            break;
        case ReduceOperation::Min:
            return color3::min(a, b);
            break;
        }
    }

    template <unsigned int blockSize, ReduceOperation op>
    __device__ void warpReduce(volatile float* pData, int tId)
    {
        if (blockSize >= 64) pData[tId] = reduce<op>(pData[tId], pData[tId + 32]);
        if (blockSize >= 32) pData[tId] = reduce<op>(pData[tId], pData[tId + 16]);
        if (blockSize >= 16) pData[tId] = reduce<op>(pData[tId], pData[tId + 8]);
        if (blockSize >= 8) pData[tId] = reduce<op>(pData[tId], pData[tId + 4]);
        if (blockSize >= 4) pData[tId] = reduce<op>(pData[tId], pData[tId + 2]);
        if (blockSize >= 2) pData[tId] = reduce<op>(pData[tId], pData[tId + 1]);
    }

    template <unsigned int blockSize, ReduceOperation op>
    __device__ void warpReduce(color3* pData, int tId)
    {
        if (blockSize >= 64) pData[tId] = reduce<op>(pData[tId], pData[tId + 32]);
        if (blockSize >= 32) pData[tId] = reduce<op>(pData[tId], pData[tId + 16]);
        if (blockSize >= 16) pData[tId] = reduce<op>(pData[tId], pData[tId + 8]);
        if (blockSize >= 8) pData[tId] = reduce<op>(pData[tId], pData[tId + 4]);
        if (blockSize >= 4) pData[tId] = reduce<op>(pData[tId], pData[tId + 2]);
        if (blockSize >= 2) pData[tId] = reduce<op>(pData[tId], pData[tId + 1]);
    }

    template <unsigned int blockSize, ReduceOperation op>
    __global__ void kernelReduce(float* pDstData, float* pSrcData, const int dim)
    {
        extern __shared__ float vpSharedData[];

        int tId = threadIdx.x;
        int i = blockIdx.x * (blockSize * 2) + threadIdx.x;
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int gridSize = blockSize * 2 * gridDim.x;

        switch (op)
        {
        case ReduceOperation::Add:
            vpSharedData[tId] = 0.0f;
            break;
        case ReduceOperation::Max:
            vpSharedData[tId] = 1e-30f;
            break;
        case ReduceOperation::Min:
            vpSharedData[tId] = 1e30f;
            break;
        }

        if (index >= dim)
            return;

        while (i < dim)
        {
            if (i + blockSize < dim)
            {
                vpSharedData[tId] = reduce<op>(pSrcData[i], pSrcData[i + blockSize]);
            }
            else
            {
                vpSharedData[tId] = pSrcData[i];
            }
            i += gridSize;
        }
        __syncthreads();

        if (blockSize >= 1024) { if (tId < 512) { vpSharedData[tId] = reduce<op>(vpSharedData[tId], vpSharedData[tId + 512]); } __syncthreads(); }
        if (blockSize >= 512) { if (tId < 256) { vpSharedData[tId] = reduce<op>(vpSharedData[tId], vpSharedData[tId + 256]); } __syncthreads(); }
        if (blockSize >= 256) { if (tId < 128) { vpSharedData[tId] = reduce<op>(vpSharedData[tId], vpSharedData[tId + 128]); } __syncthreads(); }
        if (blockSize >= 128) { if (tId < 64) { vpSharedData[tId] = reduce<op>(vpSharedData[tId], vpSharedData[tId + 64]); } __syncthreads(); }

        if (tId < 32)
        {
            warpReduce<blockSize, op>(vpSharedData, tId);
        }

        if (tId == 0)
        {
            pDstData[blockIdx.x] = vpSharedData[0];
        }
    }


    template <unsigned int blockSize, ReduceOperation op>
    __global__ void kernelReduce(color3* pDstData, color3* pSrcData, const int dim)
    {
        extern __shared__ color3 vpSharedDataC[];

        int tId = threadIdx.x;
        int i = blockIdx.x * (blockSize * 2) + threadIdx.x;
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int gridSize = blockSize * 2 * gridDim.x;

        switch (op)
        {
        case ReduceOperation::Add:
            vpSharedDataC[tId] = color3(0.0f);
            break;
        case ReduceOperation::Max:
            vpSharedDataC[tId] = color3(1e-30f);
            break;
        case ReduceOperation::Min:
            vpSharedDataC[tId] = color3(1e30f);
            break;
        }

        if (index >= dim)
            return;

        while (i < dim)
        {
            if (i + blockSize < dim)
            {
                vpSharedDataC[tId] = reduce<op>(pSrcData[i], pSrcData[i + blockSize]);
            }
            else
            {
                vpSharedDataC[tId] = pSrcData[i];
            }
            i += gridSize;
        }
        __syncthreads();

        if (blockSize >= 1024) { if (tId < 512) { vpSharedDataC[tId] = reduce<op>(vpSharedDataC[tId], vpSharedDataC[tId + 512]); } __syncthreads(); }
        if (blockSize >= 512) { if (tId < 256) { vpSharedDataC[tId] = reduce<op>(vpSharedDataC[tId], vpSharedDataC[tId + 256]); } __syncthreads(); }
        if (blockSize >= 256) { if (tId < 128) { vpSharedDataC[tId] = reduce<op>(vpSharedDataC[tId], vpSharedDataC[tId + 128]); } __syncthreads(); }
        if (blockSize >= 128) { if (tId < 64) { vpSharedDataC[tId] = reduce<op>(vpSharedDataC[tId], vpSharedDataC[tId + 64]); } __syncthreads(); }

        if (tId < 32)
        {
            warpReduce<blockSize, op>(vpSharedDataC, tId);
        }

        if (tId == 0)
        {
            pDstData[blockIdx.x] = vpSharedDataC[0];
        }
    }

    __global__ static void kernelColorMap(color3* pDstImage, const float* pSrcImage, const color3* pColorMap, const int3 dim, const int mapSize)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int i = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y || z >= dim.z) return;

        pDstImage[i] = pColorMap[int(pSrcImage[i] * 255.0f + 0.5f) % mapSize];
    }

    __global__ static void kernelFloat2Color3(color3* pDstImage, float* pSrcImage, const int3 dim)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int i = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y || z >= dim.z) return;

        pDstImage[i] = color3(pSrcImage[i]);
    }

    __global__ static void kernelFinalError(float* pDstImage, color3* pColorFeatureDifference, const int3 dim)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int i = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y || z >= dim.z) return;

        const float cdiff = pColorFeatureDifference[i].x;
        const float fdiff = pColorFeatureDifference[i].y;
        const float errorFLIP = std::pow(cdiff, 1.0f - fdiff);

        pDstImage[i] = errorFLIP;
    }

    __global__ static void kernelSetMaxExposure(float* pDstErrorMap, float* pSrcErrorMap, float* pExposureMap, const int3 dim, float exposure)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int i = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y || z >= dim.z) return;

        float srcValue = pSrcErrorMap[i];
        float dstValue = pDstErrorMap[i];

        if (srcValue > dstValue)
        {
            pExposureMap[i] = exposure;
            pDstErrorMap[i] = srcValue;
        }
    }

    __global__ static void kernelCombine(float* pDstImage, float* pSrcImageA, float* pSrcImageB, const int3 dim, CombineOperation operation)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int i = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y || z >= dim.z) return;

        switch (operation)
        {
        case CombineOperation::Add:
            pDstImage[i] = pSrcImageA[i] + pSrcImageB[i];
            break;
        case CombineOperation::Subtract:
            pDstImage[i] = pSrcImageA[i] - pSrcImageB[i];
            break;
        case CombineOperation::Multiply:
            pDstImage[i] = pSrcImageA[i] * pSrcImageB[i];
            break;
        case CombineOperation::L1:
            pDstImage[i] = abs(pSrcImageA[i] - pSrcImageB[i]);
            break;
        case CombineOperation::L2:
            pDstImage[i] = sqrt(pSrcImageA[i] * pSrcImageA[i] + pSrcImageB[i] * pSrcImageB[i]);
            break;
        }
    }

    __global__ static void kernelCombine(color3* pDstImage, color3* pSrcImageA, color3* pSrcImageB, const int3 dim, CombineOperation operation)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int i = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y || z >= dim.z) return;

        switch (operation)
        {
        case CombineOperation::Add:
            pDstImage[i] = pSrcImageA[i] + pSrcImageB[i];
            break;
        case CombineOperation::Subtract:
            pDstImage[i] = pSrcImageA[i] - pSrcImageB[i];
            break;
        case CombineOperation::Multiply:
            pDstImage[i] = pSrcImageA[i] * pSrcImageB[i];
            break;
        case CombineOperation::L1:
            pDstImage[i] = color3::abs(pSrcImageA[i] - pSrcImageB[i]);
            break;
        case CombineOperation::L2:
            pDstImage[i] = color3::sqrt(pSrcImageA[i] * pSrcImageA[i] + pSrcImageB[i] * pSrcImageB[i]);
            break;
        }
    }

    __global__ static void kernelHuntAdjustment(color3* pImage, const int3 dim)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int i = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y || z >= dim.z) return;

        color3 pixel = pImage[i];
        pixel.y = color3::Hunt(pixel.x, pixel.y);
        pixel.z = color3::Hunt(pixel.x, pixel.z);
        pImage[i] = pixel;
    }

    __global__ static void kernelYCxCz2CIELab(color3* pImage, const int3 dim)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int i = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y || z >= dim.z) return;

        pImage[i] = color3::XYZ2CIELab(color3::YCxCz2XYZ(pImage[i]));
    }

    __global__ static void kernelsRGB2YCxCz(color3* pImage, const int3 dim)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int i = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y || z >= dim.z) return;

        pImage[i] = color3::XYZ2YCxCz(color3::LinearRGB2XYZ(color3::sRGB2LinearRGB(pImage[i])));
    }

    __global__ static void kernelLinearRGB2sRGB(color3* pImage, const int3 dim)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int i = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y || z >= dim.z) return;

        pImage[i] = color3::LinearRGB2sRGB(pImage[i]);
    }


    //  General kernels

    __global__ static void kernelClear(color3* pImage, const int3 dim, const color3 color = { 0.0f, 0.0f, 0.0f })
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int i = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y || z >= dim.z) return;

        pImage[i] = color;
    }


    __global__ static void kernelClear(float* pImage, const int3 dim, const float color = 0.0f)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int i = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y || z >= dim.z) return;

        pImage[i] = color;
    }


    __global__ static void kernelMultiplyAndAdd(color3* pImage, const int3 dim, color3 m, color3 a)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int i = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y || z >= dim.z) return;

        pImage[i] = pImage[i] * m + a;
    }


    __global__ static void kernelMultiplyAndAdd(color3* pImage, const int3 dim, float m, float a)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int i = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y || z >= dim.z) return;

        pImage[i] = pImage[i] * m + a;
    }


    __global__ static void kernelMultiplyAndAdd(float* pImage, const int3 dim, float m, float a)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int i = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y || z >= dim.z) return;

        pImage[i] = pImage[i] * m + a;
    }


    __device__ const float ToneMappingCoefficients[3][6] =
    {
        { 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f },                                                 // Reinhard.
        { 0.6f * 0.6f * 2.51f, 0.6f * 0.03f, 0.0f, 0.6f * 0.6f * 2.43f, 0.6f * 0.59f, 0.14f },  // ACES, 0.6 is pre-exposure cancellation.
        { 0.231683f, 0.013791f, 0.0f, 0.18f, 0.3f, 0.018f },                                    // Hable.
    };


    __global__ static void kernelToneMap(color3* pImage, const int3 dim, int toneMapper)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int i = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y || z >= dim.z) return;

        if (toneMapper == 0)
        {
            color3 color = pImage[i];
            float luminance = color3::linearRGB2Luminance(color);
            pImage[i] /= (1.0f + luminance);
            return;
        }

        const float* tc = ToneMappingCoefficients[toneMapper];
        color3 color = pImage[i];
        pImage[i] = color3(((color * color) * tc[0] + color * tc[1] + tc[2]) / (color * color * tc[3] + color * tc[4] + tc[5]));
    }


    __global__ static void kernelClamp(color3* pImage, const int3 dim, float low, float high)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int i = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y || z >= dim.z) return;

        pImage[i] = color3::clamp(pImage[i], low, high);
    }


    __global__ static void kernelConvolve(color3* dstImage, color3* srcImage, color3* pFilter, const int3 dim, int filterWidth, int filterHeight)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int dstIndex = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y || z >= dim.z) return;

        const int halfFilterWidth = filterWidth / 2;
        const int halfFilterHeight = filterHeight / 2;

        color3 colorSum = { 0.0f, 0.0f, 0.0f };

        for (int iy = -halfFilterHeight; iy <= halfFilterHeight; iy++)
        {
            int yy = Min(Max(0, y + iy), dim.y - 1);
            for (int ix = -halfFilterWidth; ix <= halfFilterWidth; ix++)
            {
                int xx = Min(Max(0, x + ix), dim.x - 1);

                int filterIndex = (iy + halfFilterHeight) * filterWidth + (ix + halfFilterWidth);
                int srcIndex = yy * dim.x + xx;
                const color3 weights = pFilter[filterIndex];
                const color3 srcColor = srcImage[srcIndex];

                colorSum += weights * srcColor;
            }
        }

        dstImage[dstIndex] = colorSum;
    }


    __global__ static void kernelConvolve(color3* dstImage, color3* srcImage, color3* pFilter, const int3 dim, int3 filterDim)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int dstIndex = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y || z >= dim.z) return;

        const int3 halfFilterDim = { filterDim.x / 2, filterDim.y / 2, filterDim.z / 2 };

        color3 colorSum = { 0.0f, 0.0f, 0.0f };

        for (int iz = -halfFilterDim.z; iz <= halfFilterDim.z; iz++)
        {
            int zz = Min(Max(0, z + iz), dim.z - 1);
            for (int iy = -halfFilterDim.y; iy <= halfFilterDim.y; iy++)
            {
                int yy = Min(Max(0, y + iy), dim.y - 1);
                for (int ix = -halfFilterDim.x; ix <= halfFilterDim.x; ix++)
                {
                    int xx = Min(Max(0, x + ix), dim.x - 1);

                    int filterIndex = (zz * filterDim.y + (iy + halfFilterDim.y)) * filterDim.x + (ix + halfFilterDim.x);
                    int srcIndex = yy * dim.x + xx;
                    const color3 weights = pFilter[filterIndex];
                    const color3 srcColor = srcImage[srcIndex];

                    colorSum += weights * srcColor;
                }
            }
        }

        dstImage[dstIndex] = colorSum;
    }

    __global__ static void kernelConvolve(float* dstImage, float* srcImage, float* pFilter, const int3 dim, int filterWidth, int filterHeight)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int dstIndex = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y || z >= dim.z) return;

        const int halfFilterWidth = filterWidth / 2;
        const int halfFilterHeight = filterHeight / 2;

        float colorSum = 0.0f;

        for (int iy = -halfFilterHeight; iy <= halfFilterHeight; iy++)
        {
            int yy = Min(Max(0, y + iy), dim.y - 1);
            for (int ix = -halfFilterWidth; ix <= halfFilterWidth; ix++)
            {
                int xx = Min(Max(0, x + ix), dim.x - 1);

                int filterIndex = (iy + halfFilterHeight) * filterWidth + (ix + halfFilterWidth);
                int srcIndex = yy * dim.x + xx;
                const float weight = pFilter[filterIndex];
                const float srcColor = srcImage[srcIndex];

                colorSum += weight * srcColor;

            }
        }

        dstImage[dstIndex] = colorSum;
    }

    // Convolve in x direction (1st and 2nd derivative for filter in x direction, 0th derivative for filter in y direction).
    // For details on the convolution, see the note on separable filters in the FLIP repository.
    __global__ static void kernelFeatureFilterFirstDir(color3* dstImage1, color3* srcImage1, color3* dstImage2, color3* srcImage2, color3* pFilter, const int3 dim, int3 filterDim)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int dstIndex = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y) return;

        const float halfFilterWidth = filterDim.x / 2;

        float  edge1X = 0.0f, edge2X = 0.0f, point1X = 0.0f, point2X = 0.0f;
        float  gaussianFiltered1 = 0.0f, gaussianFiltered2 = 0.0f;

        const float oneOver116 = 1.0f / 116.0f;
        const float sixteenOver116 = 16.0f / 116.0f;
        for (int ix = -halfFilterWidth; ix <= halfFilterWidth; ix++)
        {
            int xx = Min(Max(0, x + ix), dim.x - 1);

            int filterIndex = ix + halfFilterWidth;
            int srcIndex = y * dim.x + xx;
            const color3 featureWeights = pFilter[filterIndex];
            float src1 = srcImage1[srcIndex].x;
            float src2 = srcImage2[srcIndex].x;

            // Normalize the gray values to [0,1].
            src1 = src1 * oneOver116 + sixteenOver116;
            src2 = src2 * oneOver116 + sixteenOver116;

            edge1X += featureWeights.y * src1;
            edge2X += featureWeights.y * src2;
            point1X += featureWeights.z * src1;
            point2X += featureWeights.z * src2;

            gaussianFiltered1 += featureWeights.x * src1;
            gaussianFiltered2 += featureWeights.x * src2;
        }

        dstImage1[dstIndex] = color3(edge1X, point1X, gaussianFiltered1);
        dstImage2[dstIndex] = color3(edge2X, point2X, gaussianFiltered2);
    }

    // Convolve in y direction (1st and 2nd derivative for filter in y direction, 0th derivative for filter in x direction), then compute difference.
    // For details on the convolution, see the note on separable filters in the FLIP repository.
    __global__ static void kernelFeatureFilterSecondDirAndFeatureDifference(color3* dstImage, color3* srcImage1, color3* srcImage2, color3* pFilter, const int3 dim, int3 filterDim)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int dstIndex = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y) return;

        const float normalizationFactor = 1.0f / std::sqrt(2.0f);

        const float halfFilterWidth = filterDim.x / 2;

        float  edge1X = 0.0f, edge2X = 0.0f, point1X = 0.0f, point2X = 0.0f;
        float  edge1Y = 0.0f, edge2Y = 0.0f, point1Y = 0.0f, point2Y = 0.0f;

        for (int iy = -halfFilterWidth; iy <= halfFilterWidth; iy++)
        {
            int yy = Min(Max(0, y + iy), dim.y - 1);

            int filterIndex = iy + halfFilterWidth;
            int srcIndex = yy * dim.x + x;
            const color3 featureWeights = pFilter[filterIndex];
            const color3 src1 = srcImage1[srcIndex];
            const color3 src2 = srcImage2[srcIndex];

            edge1X += featureWeights.x * src1.x;
            edge2X += featureWeights.x * src2.x;
            point1X += featureWeights.x * src1.y;
            point2X += featureWeights.x * src2.y;

            edge1Y += featureWeights.y * src1.z;
            edge2Y += featureWeights.y * src2.z;
            point1Y += featureWeights.z * src1.z;
            point2Y += featureWeights.z * src2.z;
        }

        const float edgeValueRef = std::sqrt(edge1X * edge1X + edge1Y * edge1Y);
        const float edgeValueTest = std::sqrt(edge2X * edge2X + edge2Y * edge2Y);
        const float pointValueRef = std::sqrt(point1X * point1X + point1Y * point1Y);
        const float pointValueTest = std::sqrt(point2X * point2X + point2Y * point2Y);

        const float edgeDifference = std::abs(edgeValueRef - edgeValueTest);
        const float pointDifference = std::abs(pointValueRef - pointValueTest);

        const float featureDifference = std::pow(normalizationFactor * Max(edgeDifference, pointDifference), DeviceFLIPConstants.gqf);

        dstImage[dstIndex].y = featureDifference;
    }

    // Performs spatial filtering in the x direction on both the reference and test image at the same time (for better performance).
    // Filtering has been changed to using separable filtering for better performance.
    // For details on the convolution, see the note on separable filters in the FLIP repository.
    __global__ static void kernelSpatialFilterFirstDir(color3* dstImageARG1, color3* dstImageBY1, color3* srcImage1, color3* dstImageARG2, color3* dstImageBY2, color3* srcImage2, color3* pFilterARG, color3* pFilterBY, const int3 dim, int3 filterDim)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int dstIndex = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y) return;

        const float halfFilterWidth = filterDim.x / 2;

        // filter in x direction
        color3 colorSumARG1 = { 0.0f, 0.0f, 0.0f };
        color3 colorSumBY1 = { 0.0f, 0.0f, 0.0f };
        color3 colorSumARG2 = { 0.0f, 0.0f, 0.0f };
        color3 colorSumBY2 = { 0.0f, 0.0f, 0.0f };

        for (int ix = -halfFilterWidth; ix <= halfFilterWidth; ix++)
        {
            int xx = Min(Max(0, x + ix), dim.x - 1);

            int filterIndex = ix + halfFilterWidth;
            int srcIndex = y * dim.x + xx;
            const color3 weightsARG = pFilterARG[filterIndex];
            const color3 weightsBY = pFilterBY[filterIndex];
            const color3 srcColor1 = srcImage1[srcIndex];
            const color3 srcColor2 = srcImage2[srcIndex];

            colorSumARG1 += color3(weightsARG.x * srcColor1.x, weightsARG.y * srcColor1.y, 0.0f);
            colorSumARG2 += color3(weightsARG.x * srcColor2.x, weightsARG.y * srcColor2.y, 0.0f);
            colorSumBY1 += color3(weightsBY.x * srcColor1.z, weightsBY.y * srcColor1.z, 0.0f);
            colorSumBY2 += color3(weightsBY.x * srcColor2.z, weightsBY.y * srcColor2.z, 0.0f);
        }

        dstImageARG1[dstIndex] = color3(colorSumARG1.x, colorSumARG1.y, 0.0f);
        dstImageBY1[dstIndex] = color3(colorSumBY1.x, colorSumBY1.y, 0.0f);
        dstImageARG2[dstIndex] = color3(colorSumARG2.x, colorSumARG2.y, 0.0f);
        dstImageBY2[dstIndex] = color3(colorSumBY2.x, colorSumBY2.y, 0.0f);
    }

    // Performs spatial filtering in the y direction (and clamps the results) on both the reference and test image at the same time (for better performance).
    // Filtering has been changed to using separable filtering for better performance. For details on the convolution, see the note on separable filters in the FLIP repository.
    // After filtering, compute color differences.
    __global__ static void kernelSpatialFilterSecondDirAndColorDifference(color3* dstImage, color3* srcImageARG1, color3* srcImageBY1, color3* srcImageARG2, color3* srcImageBY2, color3* pFilterARG, color3* pFilterBY, const int3 dim, int3 filterDim)
    {
        // Color difference constants.
        const float cmax = color3::computeMaxDistance(DeviceFLIPConstants.gqc);
        const float pccmax = DeviceFLIPConstants.gpc * cmax;

        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int dstIndex = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y) return;

        const float halfFilterWidth = filterDim.x / 2;

        // Filter in y direction.
        color3 colorSumARG1 = { 0.0f, 0.0f, 0.0f };
        color3 colorSumBY1 = { 0.0f, 0.0f, 0.0f };
        color3 colorSumARG2 = { 0.0f, 0.0f, 0.0f };
        color3 colorSumBY2 = { 0.0f, 0.0f, 0.0f };

        for (int iy = -halfFilterWidth; iy <= halfFilterWidth; iy++)
        {
            int yy = Min(Max(0, y + iy), dim.y - 1);

            int filterIndex = iy + halfFilterWidth;
            int srcIndex = yy * dim.x + x;
            const color3 weightsARG = pFilterARG[filterIndex];
            const color3 weightsBY = pFilterBY[filterIndex];
            const color3 srcColorARG1 = srcImageARG1[srcIndex];
            const color3 srcColorBY1 = srcImageBY1[srcIndex];
            const color3 srcColorARG2 = srcImageARG2[srcIndex];
            const color3 srcColorBY2 = srcImageBY2[srcIndex];

            colorSumARG1 += color3(weightsARG.x * srcColorARG1.x, weightsARG.y * srcColorARG1.y, 0.0f);
            colorSumARG2 += color3(weightsARG.x * srcColorARG2.x, weightsARG.y * srcColorARG2.y, 0.0f);
            colorSumBY1 += color3(weightsBY.x * srcColorBY1.x, weightsBY.y * srcColorBY1.y, 0.0f);
            colorSumBY2 += color3(weightsBY.x * srcColorBY2.x, weightsBY.y * srcColorBY2.y, 0.0f);
        }

        // Clamp to [0,1] in linear RGB.
        color3 color1 = color3(colorSumARG1.x, colorSumARG1.y, colorSumBY1.x + colorSumBY1.y);
        color3 color2 = color3(colorSumARG2.x, colorSumARG2.y, colorSumBY2.x + colorSumBY2.y);
        color1 = FLIP::color3::clamp(FLIP::color3::XYZ2LinearRGB(FLIP::color3::YCxCz2XYZ(color1)));
        color2 = FLIP::color3::clamp(FLIP::color3::XYZ2LinearRGB(FLIP::color3::YCxCz2XYZ(color2)));

        // Move from linear RGB to CIELab.
        color1 = color3::XYZ2CIELab(color3::LinearRGB2XYZ(color1));
        color2 = color3::XYZ2CIELab(color3::LinearRGB2XYZ(color2));

        // Apply Hunt adjustment.
        color1.y = color3::Hunt(color1.x, color1.y);
        color1.z = color3::Hunt(color1.x, color1.z);
        color2.y = color3::Hunt(color2.x, color2.y);
        color2.z = color3::Hunt(color2.x, color2.z);

        float colorDifference = color3::HyAB(color1, color2);

        colorDifference = powf(colorDifference, DeviceFLIPConstants.gqc);

        // Re-map error to the [0, 1] range. Values between 0 and pccmax are mapped to the range [0, gpt],
        // while the rest are mapped to the range (gpt, 1]
        if (colorDifference < pccmax)
        {
            colorDifference *= DeviceFLIPConstants.gpt / pccmax;
        }
        else
        {
            colorDifference = DeviceFLIPConstants.gpt + ((colorDifference - pccmax) / (cmax - pccmax)) * (1.0f - DeviceFLIPConstants.gpt);
        }

        dstImage[dstIndex] = color3(colorDifference, 0.0f, 0.0f);
    }
}