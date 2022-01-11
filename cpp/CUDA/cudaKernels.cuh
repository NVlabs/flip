/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2020-2021 NVIDIA CORPORATION & AFFILIATES
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

    __device__ FLIPConstants DeviceFLIPConstants;


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


    __global__ static void kernelLab2Gray(color3* pDstImage, const color3* pSrcImage, const int3 dim)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int i = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y || z >= dim.z) return;

        float l = (pSrcImage[i].x + 16.0f) / 116.0f;  // make it [0,1]
        pDstImage[i] = color3(l, l, 0.0f);  // luminance [0,1] stored in both x and y since we apply both horizontal and verticals filters at the same time
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

    __global__ static void kernelComputeFeatureDifference(color3* pDstImage, color3* pEdgeReference, color3* pEdgeTest, color3* pPointReference, color3* pPointTest, const int3 dim, const float ppd)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int i = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y || z >= dim.z) return;

        const float normalizationFactor = 1.0f / std::sqrt(2.0f);

        color3 er = pEdgeReference[i];
        color3 et = pEdgeTest[i];
        color3 pr = pPointReference[i];
        color3 pt = pPointTest[i];

        const float edgeValueRef = std::sqrt(er.x * er.x + er.y * er.y);
        const float edgeValueTest = std::sqrt(et.x * et.x + et.y * et.y);
        const float pointValueRef = std::sqrt(pr.x * pr.x + pr.y * pr.y);
        const float pointValueTest = std::sqrt(pt.x * pt.x + pt.y * pt.y);

        const float edgeDifference = std::abs(edgeValueRef - edgeValueTest);
        const float pointDifference = std::abs(pointValueRef - pointValueTest);

        const float featureDifference = std::pow(normalizationFactor * Max(edgeDifference, pointDifference), DeviceFLIPConstants.gqf);

        pDstImage[i] = color3(featureDifference, 0.0f, 0.0f);
    }

    __global__ static void kernelFinalError(float* pDstImage, color3* pColorDifference, color3* pFeatureDifference, const int3 dim)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int i = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y || z >= dim.z) return;

        const float cdiff = pColorDifference[i].x;
        const float fdiff = pFeatureDifference[i].x;
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

    __global__ static void kernelColorDifference(color3* pDstImage, color3* pSrcReferenceImage, color3* pSrcTestImage, const int3 dim)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int i = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y || z >= dim.z) return;

        const float cmax = color3::computeMaxDistance(FLIP::DeviceFLIPConstants.gqc);
        const float pccmax = FLIP::DeviceFLIPConstants.gpc * cmax;

        color3 refPixel = pSrcReferenceImage[i];
        color3 testPixel = pSrcTestImage[i];

        float colorDifference = color3::HyAB(refPixel, testPixel);

        colorDifference = powf(colorDifference, FLIP::DeviceFLIPConstants.gqc);

        // Re-map error to the [0, 1] range. Values between 0 and pccmax are mapped to the range [0, gpt],
        // while the rest are mapped to the range (gpt, 1]
        if (colorDifference < pccmax)
        {
            colorDifference *= FLIP::DeviceFLIPConstants.gpt / pccmax;
        }
        else
        {
            colorDifference = FLIP::DeviceFLIPConstants.gpt + ((colorDifference - pccmax) / (cmax - pccmax)) * (1.0f - FLIP::DeviceFLIPConstants.gpt);
        }

        pDstImage[i] = color3(colorDifference, 0.0f, 0.0f);
    }

    __global__ static void kernelFeatureDifference(color3* pDstImage, color3* pPointReference, color3* pPointTest, color3* pEdgeReference, color3* pEdgeTest, const int3 dim)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int i = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y || z >= dim.z) return;

        const float normalizationFactor = 1.0f / std::sqrt(2.0f);

        color3 er = pEdgeReference[i];
        color3 et = pEdgeTest[i];
        color3 pr = pPointReference[i];
        color3 pt = pPointTest[i];

        const float edgeValueRef = std::sqrt(er.x * er.x + er.y * er.y);
        const float edgeValueTest = std::sqrt(et.x * et.x + et.y * et.y);
        const float pointValueRef = std::sqrt(pr.x * pr.x + pr.y * pr.y);
        const float pointValueTest = std::sqrt(pt.x * pt.x + pt.y * pt.y);

        const float edgeDifference = std::abs(edgeValueRef - edgeValueTest);
        const float pointDifference = std::abs(pointValueRef - pointValueTest);

        const float featureDifference = std::pow(normalizationFactor * Max(edgeDifference, pointDifference), DeviceFLIPConstants.gqf);

        pDstImage[i] = color3(featureDifference, 0.0f, 0.0f);
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

        pImage[i] = color3::XYZ2CIELab(color3::LinearRGB2XYZ(color3::clamp(color3::XYZ2LinearRGB(color3::YCxCz2XYZ(pImage[i])))));
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

    __global__ static void kernelYCxCz2Gray(float* dstImage, color3* srcImage, const int3 dim)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int i = y * dim.x + x;

        if (x >= dim.x || y >= dim.y) return;

        dstImage[i] = color3::YCxCz2Gray(srcImage[i]);
    }

    __global__ static void kernelYCxCz2Gray(color3* dstImage, color3* srcImage, const int3 dim)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int i = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y || z >= dim.z) return;

        float gray = color3::YCxCz2Gray(srcImage[i]);
        dstImage[i] = color3(gray);
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
        { 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f },  //  Reinhard
        { 0.6f * 0.6f * 2.51f, 0.6f * 0.03f, 0.0f, 0.6f * 0.6f * 2.43f, 0.6f * 0.59f, 0.14f },  //  Aces, 0.6 is pre-exposure cancellation
        { 0.231683f, 0.013791f, 0.0f, 0.18f, 0.3f, 0.018f },  //  Hable
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

    __global__ static void kernelConvolve2images(color3* dstImage1, color3* srcImage1, color3* dstImage2, color3* srcImage2, color3* pFilter, const int3 dim, int3 filterDim)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int dstIndex = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y || z >= dim.z) return;

        const int3 halfFilterDim = { filterDim.x / 2, filterDim.y / 2, filterDim.z / 2 };

        color3 colorSum1 = { 0.0f, 0.0f, 0.0f };
        color3 colorSum2 = { 0.0f, 0.0f, 0.0f };

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
                    const color3 srcColor1 = srcImage1[srcIndex];
                    const color3 srcColor2 = srcImage2[srcIndex];

                    colorSum1 += weights * srcColor1;
                    colorSum2 += weights * srcColor2;
                }
            }
        }
        dstImage1[dstIndex] = colorSum1;
        dstImage2[dstIndex] = colorSum2;
    }


}
