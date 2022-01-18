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
 * THIS SOFTWARE IS PROVIDED Cz THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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

    // Convolve in x direction (1st and 2nd derivative for filter in x direction, 0th derivative for filter in y direction).
    // For details on the convolution, see separated_convolutions.pdf in the FLIP repository.
    // We filter both reference and test image simultaneously (for better performance).
    __global__ static void kernelFeatureFilterFirstDir(color3* intermediateFeaturesImageReference, color3* referenceImage, color3* intermediateFeaturesImageTest, color3* testImage, color3* pFilter, const int3 dim, int3 filterDim)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int dstIndex = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y) return;

        const float halfFilterWidth = filterDim.x / 2;

        float dxReference = 0.0f, dxTest = 0.0f, ddxReference = 0.0f, ddxTest = 0.0f;
        float gaussianFilteredReference = 0.0f, gaussianFilteredTest = 0.0f;

        const float oneOver116 = 1.0f / 116.0f;
        const float sixteenOver116 = 16.0f / 116.0f;
        for (int ix = -halfFilterWidth; ix <= halfFilterWidth; ix++)
        {
            int xx = Min(Max(0, x + ix), dim.x - 1);

            int filterIndex = ix + halfFilterWidth;
            int srcIndex = y * dim.x + xx;
            const color3 featureWeights = pFilter[filterIndex];
            float yReference = referenceImage[srcIndex].x;
            float yTest = testImage[srcIndex].x;

            // Normalize the Y values to [0,1].
            float yReferenceNormalized = yReference * oneOver116 + sixteenOver116;
            float yTestNormalized = yTest * oneOver116 + sixteenOver116;

            // Image multiplied by 1st and 2nd x-derivatives.
            dxReference += featureWeights.y * yReferenceNormalized;
            dxTest += featureWeights.y * yTestNormalized;
            ddxReference += featureWeights.z * yReferenceNormalized;
            ddxTest += featureWeights.z * yTestNormalized;

            // Image multiplied by 0th derivative.
            gaussianFilteredReference += featureWeights.x * yReferenceNormalized;
            gaussianFilteredTest += featureWeights.x * yTestNormalized;
        }

        intermediateFeaturesImageReference[dstIndex] = color3(dxReference, ddxReference, gaussianFilteredReference);
        intermediateFeaturesImageTest[dstIndex] = color3(dxTest, ddxTest, gaussianFilteredTest);
    }

    // Convolve in y direction (1st and 2nd derivative for filter in y direction, 0th derivative for filter in x direction), then compute difference.
    // For details on the convolution, see separated_convolutions.pdf in the FLIP repository.
    // We filter both reference and test image simultaneously (for better performance).
    __global__ static void kernelFeatureFilterSecondDirAndFeatureDifference(color3* featureDifferenceImage, color3* intermediateFeaturesImageReference, color3* intermediateFeaturesImageTest, color3* pFilter, const int3 dim, int3 filterDim)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int dstIndex = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y) return;

        const float normalizationFactor = 1.0f / std::sqrt(2.0f);

        const float halfFilterWidth = filterDim.x / 2;

        float dxReference = 0.0f, dxTest = 0.0f, ddxReference = 0.0f, ddxTest = 0.0f;
        float dyReference = 0.0f, dyTest = 0.0f, ddyReference = 0.0f, ddyTest = 0.0f;

        for (int iy = -halfFilterWidth; iy <= halfFilterWidth; iy++)
        {
            int yy = Min(Max(0, y + iy), dim.y - 1);

            int filterIndex = iy + halfFilterWidth;
            int srcIndex = yy * dim.x + x;
            const color3 featureWeights = pFilter[filterIndex];
            const color3 intermediateFeaturesReference = intermediateFeaturesImageReference[srcIndex];
            const color3 intermediateFeatureTest = intermediateFeaturesImageTest[srcIndex];

            // Intermediate images (1st and 2nd derivative in x) multiplied by 0th derivative.
            dxReference += featureWeights.x * intermediateFeaturesReference.x;
            dxTest += featureWeights.x * intermediateFeatureTest.x;
            ddxReference += featureWeights.x * intermediateFeaturesReference.y;
            ddxTest += featureWeights.x * intermediateFeatureTest.y;

            // Intermediate image (0th derivative) multiplied by 1st and 2nd y-derivatives.
            dyReference += featureWeights.y * intermediateFeaturesReference.z;
            dyTest += featureWeights.y * intermediateFeatureTest.z;
            ddyReference += featureWeights.z * intermediateFeaturesReference.z;
            ddyTest += featureWeights.z * intermediateFeatureTest.z;
        }

        const float edgeValueRef = std::sqrt(dxReference * dxReference + dyReference * dyReference);
        const float edgeValueTest = std::sqrt(dxTest * dxTest + dyTest * dyTest);
        const float pointValueRef = std::sqrt(ddxReference * ddxReference + ddyReference * ddyReference);
        const float pointValueTest = std::sqrt(ddxTest * ddxTest + ddyTest * ddyTest);

        const float edgeDifference = std::abs(edgeValueRef - edgeValueTest);
        const float pointDifference = std::abs(pointValueRef - pointValueTest);

        const float featureDifference = std::pow(normalizationFactor * Max(edgeDifference, pointDifference), DeviceFLIPConstants.gqf);

        featureDifferenceImage[dstIndex].y = featureDifference;
    }

    // Performs spatial filtering in the x direction on both the reference and test image at the same time (for better performance).
    // Filtering has been changed to using separable filtering for better performance.
    // For details on the convolution, see separated_convolutions.pdf in the FLIP repository.
    __global__ static void kernelSpatialFilterFirstDir(color3* intermediateYCxImageReference, color3* intermediateCzImageReference, color3* referenceImage, color3* intermediateYCxImageTest, color3* intermediateCzImageTest, color3* testImage, color3* pFilterYCx, color3* pFilterCz, const int3 dim, int3 filterDim)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int dstIndex = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y) return;

        const float halfFilterWidth = filterDim.x / 2;

        // Filter in x direction.
        color3 intermediateYCxReference = { 0.0f, 0.0f, 0.0f };
        color3 intermediateYCxTest = { 0.0f, 0.0f, 0.0f };
        color3 intermediateCzReference = { 0.0f, 0.0f, 0.0f };
        color3 intermediateCzTest = { 0.0f, 0.0f, 0.0f };

        for (int ix = -halfFilterWidth; ix <= halfFilterWidth; ix++)
        {
            int xx = Min(Max(0, x + ix), dim.x - 1);

            int filterIndex = ix + halfFilterWidth;
            int srcIndex = y * dim.x + xx;
            const color3 weightsYCx = pFilterYCx[filterIndex];
            const color3 weightsCz = pFilterCz[filterIndex];
            const color3 referenceColor = referenceImage[srcIndex];
            const color3 testColor = testImage[srcIndex];

            intermediateYCxReference += color3(weightsYCx.x * referenceColor.x, weightsYCx.y * referenceColor.y, 0.0f);
            intermediateYCxTest += color3(weightsYCx.x * testColor.x, weightsYCx.y * testColor.y, 0.0f);
            intermediateCzReference += color3(weightsCz.x * referenceColor.z, weightsCz.y * referenceColor.z, 0.0f);
            intermediateCzTest += color3(weightsCz.x * testColor.z, weightsCz.y * testColor.z, 0.0f);
        }

        intermediateYCxImageReference[dstIndex] = color3(intermediateYCxReference.x, intermediateYCxReference.y, 0.0f);
        intermediateYCxImageTest[dstIndex] = color3(intermediateYCxTest.x, intermediateYCxTest.y, 0.0f);
        intermediateCzImageReference[dstIndex] = color3(intermediateCzReference.x, intermediateCzReference.y, 0.0f);
        intermediateCzImageTest[dstIndex] = color3(intermediateCzTest.x, intermediateCzTest.y, 0.0f);
    }

    // Performs spatial filtering in the y direction (and clamps the results) on both the reference and test image at the same time (for better performance).
    // Filtering has been changed to using separable filtering for better performance. For details on the convolution, see separated_convolutions.pdf in the FLIP repository.
    // After filtering, compute color differences.
    __global__ static void kernelSpatialFilterSecondDirAndColorDifference(color3* colorDifferenceImage, color3* intermediateYCxImageReference, color3* intermediateCzImageReference, color3* intermediateYCxImageTest, color3* intermediateCzImageTest, color3* pFilterYCx, color3* pFilterCz, const int3 dim, const int3 filterDim, const float cmax, const float pccmax)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int dstIndex = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y) return;

        const float halfFilterWidth = filterDim.x / 2;

        // Filter in y direction.
        color3 filteredYCxReference = { 0.0f, 0.0f, 0.0f };
        color3 filteredYCxTest = { 0.0f, 0.0f, 0.0f };
        color3 filteredCzReference = { 0.0f, 0.0f, 0.0f };
        color3 filteredCzTest = { 0.0f, 0.0f, 0.0f };

        for (int iy = -halfFilterWidth; iy <= halfFilterWidth; iy++)
        {
            int yy = Min(Max(0, y + iy), dim.y - 1);

            int filterIndex = iy + halfFilterWidth;
            int srcIndex = yy * dim.x + x;
            const color3 weightsYCx = pFilterYCx[filterIndex];
            const color3 weightsCz = pFilterCz[filterIndex];
            const color3 intermediateYCxReference = intermediateYCxImageReference[srcIndex];
            const color3 intermediateYCxTest = intermediateYCxImageTest[srcIndex];
            const color3 intermediateCzReference = intermediateCzImageReference[srcIndex];
            const color3 intermediateCzTest = intermediateCzImageTest[srcIndex];

            filteredYCxReference += color3(weightsYCx.x * intermediateYCxReference.x, weightsYCx.y * intermediateYCxReference.y, 0.0f);
            filteredYCxTest += color3(weightsYCx.x * intermediateYCxTest.x, weightsYCx.y * intermediateYCxTest.y, 0.0f);
            filteredCzReference += color3(weightsCz.x * intermediateCzReference.x, weightsCz.y * intermediateCzReference.y, 0.0f);
            filteredCzTest += color3(weightsCz.x * intermediateCzTest.x, weightsCz.y * intermediateCzTest.y, 0.0f);
        }

        // Clamp to [0,1] in linear RGB.
        color3 filteredYCxCzReference = color3(filteredYCxReference.x, filteredYCxReference.y, filteredCzReference.x + filteredCzReference.y);
        color3 filteredYCxCzTest = color3(filteredYCxTest.x, filteredYCxTest.y, filteredCzTest.x + filteredCzTest.y);
        filteredYCxCzReference = FLIP::color3::clamp(FLIP::color3::XYZ2LinearRGB(FLIP::color3::YCxCz2XYZ(filteredYCxCzReference)));
        filteredYCxCzTest = FLIP::color3::clamp(FLIP::color3::XYZ2LinearRGB(FLIP::color3::YCxCz2XYZ(filteredYCxCzTest)));

        // Move from linear RGB to CIELab.
        filteredYCxCzReference = color3::XYZ2CIELab(color3::LinearRGB2XYZ(filteredYCxCzReference));
        filteredYCxCzTest = color3::XYZ2CIELab(color3::LinearRGB2XYZ(filteredYCxCzTest));

        // Apply Hunt adjustment.
        filteredYCxCzReference.y = color3::Hunt(filteredYCxCzReference.x, filteredYCxCzReference.y);
        filteredYCxCzReference.z = color3::Hunt(filteredYCxCzReference.x, filteredYCxCzReference.z);
        filteredYCxCzTest.y = color3::Hunt(filteredYCxCzTest.x, filteredYCxCzTest.y);
        filteredYCxCzTest.z = color3::Hunt(filteredYCxCzTest.x, filteredYCxCzTest.z);

        float colorDifference = color3::HyAB(filteredYCxCzReference, filteredYCxCzTest);

        colorDifference = powf(colorDifference, DeviceFLIPConstants.gqc);

        // Re-map error to the [0, 1] range. Values between 0 and pccmax are mapped to the range [0, gpt],
        // while the rest are mapped to the range (gpt, 1].
        if (colorDifference < pccmax)
        {
            colorDifference *= DeviceFLIPConstants.gpt / pccmax;
        }
        else
        {
            colorDifference = DeviceFLIPConstants.gpt + ((colorDifference - pccmax) / (cmax - pccmax)) * (1.0f - DeviceFLIPConstants.gpt);
        }

        colorDifferenceImage[dstIndex] = color3(colorDifference, 0.0f, 0.0f);
    }
}