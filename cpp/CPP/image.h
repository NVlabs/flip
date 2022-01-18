﻿/*
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

#include "sharedflip.h"
#include "tensor.h"

namespace FLIP
{

    template<typename T>
    class image :
        public tensor<T>
    {
    public:

        image(std::string fileName)
        {
            bool bOk = this->load(fileName);
            if (!bOk)
            {
                std::cout << "Failed to load image <" << fileName << ">" << std::endl;
                exit(-1);
            }
        }

        image(const int width, const int height)
            : tensor<T>(width, height, 1)
        {
        }

        image(const int width, const int height, const T clearColor)
            : tensor<T>(width, height, 1, clearColor)
        {
        }

        image(const int3 dim, const T clearColor)
            : tensor<T>(dim.x, dim.y, 1)
        {
        }

        image(const int3 dim)
            : tensor<T>(dim.x, dim.y, 1)
        {
        }

        image(image& image)
        {
            this->init(image.mDim);
            this->copy(image);
        }

        image(tensor<T>& tensor, const int offset)
        {
            this->init(tensor.getDimensions());
            this->copy(tensor, offset);
        }

        image(const color3* pColorMap, int size)
            : tensor<T>(pColorMap, size)
        {
        }

        ~image(void)
        {
        }

        T get(int x, int y) const
        {
            return this->mvpHostData[this->index(x, y)];
        }

        void set(int x, int y, T value)
        {
            this->mvpHostData[this->index(x, y)] = value;
        }


        //////////////////////////////////////////////////////////////////////////////////

        static void setSpatialFilters(image<color3>& filterYCx, image<color3>& filterCz, float ppd, int filterRadius) // For details, see separatedConvolutions.pdf in the FLIP repository.
        {
            float deltaX = 1.0f / ppd;
            color3 filterSumYCx = { 0.0f, 0.0f, 0.0f };
            color3 filterSumCz = { 0.0f, 0.0f, 0.0f };
            int filterWidth = 2 * filterRadius + 1;

            for (int x = 0; x < filterWidth; x++)
            {
                float ix = (x - filterRadius) * deltaX;

                float ix2 = ix * ix;
                float gY = Gaussian(ix2, GaussianConstants.a1.x, GaussianConstants.b1.x);
                float gCx = Gaussian(ix2, GaussianConstants.a1.y, GaussianConstants.b1.y);
                float gCz1 = GaussianSqrt(ix2, GaussianConstants.a1.z, GaussianConstants.b1.z);
                float gCz2 = GaussianSqrt(ix2, GaussianConstants.a2.z, GaussianConstants.b2.z);
                color3 valueYCx = color3(gY, gCx, 0.0f);
                color3 valueCz = color3(gCz1, gCz2, 0.0f);
                filterYCx.set(x, 0, valueYCx);
                filterCz.set(x, 0, valueCz);
                filterSumYCx += valueYCx;
                filterSumCz += valueCz;
            }

            // Normalize weights.
            color3 normFactorYCx = { 1.0f / filterSumYCx.x, 1.0f / filterSumYCx.y, 1.0f };
            float normFactorCz = 1.0f / std::sqrt(filterSumCz.x * filterSumCz.x + filterSumCz.y * filterSumCz.y);
            for (int x = 0; x < filterWidth; x++)
            {
                color3 pYCx = filterYCx.get(x, 0);
                color3 pCz = filterCz.get(x, 0);

                filterYCx.set(x, 0, color3(pYCx.x * normFactorYCx.x, pYCx.y * normFactorYCx.y, 0.0f));
                filterCz.set(x, 0, color3(pCz.x * normFactorCz, pCz.y * normFactorCz, 0.0f));
            }
        }

        static void setFeatureFilter(image<color3>& filter, const float ppd) // For details, see separatedConvolutions.pdf in the FLIP repository.
        {
            const float stdDev = 0.5f * FLIPConstants.gw * ppd;
            const int radius = int(std::ceil(3.0f * stdDev));
            const int width = 2 * radius + 1;

            float gSum = 0.0f;
            float dgSumNegative = 0.0f;
            float dgSumPositive = 0.0f;
            float ddgSumNegative = 0.0f;
            float ddgSumPositive = 0.0f;

            for (int x = 0; x < width; x++)
            {
                int xx = x - radius;

                float g = Gaussian(float(xx), stdDev);
                gSum += g;

                // 1st derivative.
                float dg = -float(xx) * g;
                if (dg > 0.0f)
                    dgSumPositive += dg;
                else
                    dgSumNegative -= dg;

                // 2nd derivative.
                float ddg = (float(xx) * float(xx) / (stdDev * stdDev) - 1.0f) * g;
                if (ddg > 0.0f)
                    ddgSumPositive += ddg;
                else
                    ddgSumNegative -= ddg;

                filter.set(x, 0, color3(g, dg, ddg));
            }

            // Normalize weights (Gaussian weights should sum to 1; postive and negative weights of 1st and 2nd derivative should sum to 1 and -1, respectively).
            for (int x = 0; x < width; x++)
            {
                color3 p = filter.get(x, 0);

                filter.set(x, 0, color3(p.x / gSum, p.y / (p.y > 0.0f ? dgSumPositive : dgSumNegative), p.z / (p.z > 0.0f ? ddgSumPositive : ddgSumNegative)));
            }
        }

        void FLIP(image<color3>& reference, image<color3>& test, float ppd)
        {
            int width = reference.getWidth();
            int height = reference.getHeight();

            // Transform from sRGB to YCxCz.
            reference.sRGB2YCxCz();
            test.sRGB2YCxCz();

            // Prepare separated spatial filters. Because the filter for the Blue-Yellow channel is a sum of two Gaussians, we need to separate the spatial filter into two
            // (YCx for the Achromatic and Red-Green channels and Cz for the Blue-Yellow channel).
            int spatialFilterRadius = calculateSpatialFilterRadius(ppd);
            int spatialFilterWidth = 2 * spatialFilterRadius + 1;
            image<color3> spatialFilterYCx(spatialFilterWidth, 1);
            image<color3> spatialFilterCz(spatialFilterWidth, 1);
            setSpatialFilters(spatialFilterYCx, spatialFilterCz, ppd, spatialFilterRadius);

            // The next call performs spatial filtering on both the reference and test image at the same time (for better performance).
            // It then computes the color difference between the images. "this" is an image<float> here, so we store the color difference in that image.
            this->computeColorDifference(reference, test, spatialFilterYCx, spatialFilterCz);

            // Prepare separated feature (edge/point) detection filters.
            const float stdDev = 0.5f * FLIPConstants.gw * ppd;
            const int featureFilterRadius = int(std::ceil(3.0f * stdDev));
            int featureFilterWidth = 2 * featureFilterRadius + 1;
            image<color3> featureFilter(featureFilterWidth, 1);
            setFeatureFilter(featureFilter, ppd);

            // The following call convolves referenceImage and testImage with the edge and point detection filters and performs additional
            // computations for the final feature differences, and then computes the final FLIP error and stores in "this".
            this->computeFeatureDifferenceAndFinalError(reference, test, featureFilter);
        }

        // Performs spatial filtering (and clamps the results) on both the reference and test image at the same time (for better performance).
        // Filtering has been changed to separable filtering for better performance. For details on the convolution, see separatedConvolutions.pdf in the FLIP repository.
        // After filtering, compute color differences. referenceImage and testImage are expected to be in YCxCz space.
        void computeColorDifference(const FLIP::image<color3>& referenceImage, const FLIP::image<color3>& testImage, const FLIP::image<color3>& filterYCx, const FLIP::image<color3>& filterCz)
        {
            // Color difference constants
            const float cmax = color3::computeMaxDistance(FLIPConstants.gqc);
            const float pccmax = FLIPConstants.gpc * cmax;

            const int halfFilterWidth = filterYCx.getWidth() / 2; // YCx and Cz filters are the same size.

            const int w = referenceImage.getWidth();
            const int h = referenceImage.getHeight();

            image<color3> intermediateYCxImageReference(w, h);
            image<color3> intermediateYCxImageTest(w, h);
            image<color3> intermediateCzImageReference(w, h);
            image<color3> intermediateCzImageTest(w, h);

            // Filter in x direction.
#pragma omp parallel for
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    color3 intermediateYCxReference = { 0.0f, 0.0f, 0.0f };
                    color3 intermediateYCxTest = { 0.0f, 0.0f, 0.0f };
                    color3 intermediateCzReference = { 0.0f, 0.0f, 0.0f };
                    color3 intermediateCzTest = { 0.0f, 0.0f, 0.0f };

                    for (int ix = -halfFilterWidth; ix <= halfFilterWidth; ix++)
                    {
                        int xx = Min(Max(0, x + ix), w - 1);

                        const color3 weightsYCx = filterYCx.get(ix + halfFilterWidth, 0);
                        const color3 weightsCz = filterCz.get(ix + halfFilterWidth, 0);
                        const color3 referenceColor = referenceImage.get(xx, y);
                        const color3 testColor = testImage.get(xx, y);

                        intermediateYCxReference += color3(weightsYCx.x * referenceColor.x, weightsYCx.y * referenceColor.y, 0.0f);
                        intermediateYCxTest += color3(weightsYCx.x * testColor.x, weightsYCx.y * testColor.y, 0.0f);
                        intermediateCzReference += color3(weightsCz.x * referenceColor.z, weightsCz.y * referenceColor.z, 0.0f);
                        intermediateCzTest += color3(weightsCz.x * testColor.z, weightsCz.y * testColor.z, 0.0f);
                    }

                    intermediateYCxImageReference.set(x, y, intermediateYCxReference);
                    intermediateYCxImageTest.set(x, y, intermediateYCxTest);
                    intermediateCzImageReference.set(x, y, intermediateCzReference);
                    intermediateCzImageTest.set(x, y, intermediateCzTest);
                }
            }

            // Filter in y direction.
#pragma omp parallel for
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    color3 filteredYCxReference = { 0.0f, 0.0f, 0.0f };
                    color3 filteredYCxTest = { 0.0f, 0.0f, 0.0f };
                    color3 filteredCzReference = { 0.0f, 0.0f, 0.0f };
                    color3 filteredCzTest = { 0.0f, 0.0f, 0.0f };

                    for (int iy = -halfFilterWidth; iy <= halfFilterWidth; iy++)
                    {
                        int yy = Min(Max(0, y + iy), h - 1);

                        const color3 weightsYCx = filterYCx.get(iy + halfFilterWidth, 0);
                        const color3 weightsCz = filterCz.get(iy + halfFilterWidth, 0);
                        const color3 intermediateYCxReference = intermediateYCxImageReference.get(x, yy);
                        const color3 intermediateYCxTest = intermediateYCxImageTest.get(x, yy);
                        const color3 intermediateCzReference = intermediateCzImageReference.get(x, yy);
                        const color3 intermediateCzTest = intermediateCzImageTest.get(x, yy);

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

                    colorDifference = powf(colorDifference, FLIPConstants.gqc);

                    // Re-map error to the [0, 1] range. Values between 0 and pccmax are mapped to the range [0, gpt],
                    // while the rest are mapped to the range (gpt, 1].
                    if (colorDifference < pccmax)
                    {
                        colorDifference *= FLIPConstants.gpt / pccmax;
                    }
                    else
                    {
                        colorDifference = FLIPConstants.gpt + ((colorDifference - pccmax) / (cmax - pccmax)) * (1.0f - FLIPConstants.gpt);
                    }

                    this->set(x, y, colorDifference);
                }
            }
        }

        // This includes convolution (using separable filtering) of grayRefImage and grayTestImage for both edge and point filtering.
        // In addition, it computes the final FLIP error and stores in "this". referenceImage and testImage are expected to be in YCxCz space.
        void computeFeatureDifferenceAndFinalError(const image<color3>& referenceImage, const image<color3>& testImage, const image<color3>& featureFilter)
        {
            const float normalizationFactor = 1.0f / std::sqrt(2.0f);
            const int halfFilterWidth = featureFilter.getWidth() / 2;      // The edge and point filters are of the same size.
            const int w = referenceImage.getWidth();
            const int h = referenceImage.getHeight();

            image<color3> intermediateFeaturesImageReference(w, h);
            image<color3> intermediateFeaturesImageTest(w, h);

            // Convolve in x direction (1st and 2nd derivative for filter in x direction, Gaussian in y direction).
            // For details, see separatedConvolutions.pdf in the FLIP repository.
            // We filter both reference and test image simultaneously (for better performance).
            const float oneOver116 = 1.0f / 116.0f;
            const float sixteenOver116 = 16.0f / 116.0f;
#pragma omp parallel for
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    float dxReference = 0.0f, dxTest = 0.0f, ddxReference = 0.0f, ddxTest = 0.0f;
                    float gaussianFilteredReference = 0.0f, gaussianFilteredTest = 0.0f;

                    for (int ix = -halfFilterWidth; ix <= halfFilterWidth; ix++)
                    {
                        int xx = Min(Max(0, x + ix), w - 1);

                        const color3 featureWeights = featureFilter.get(ix + halfFilterWidth, 0);
                        float yReference = referenceImage.get(xx, y).x;
                        float yTest = testImage.get(xx, y).x;

                        // Normalize the Y values to [0,1].
                        float yReferenceNormalized = yReference * oneOver116 + sixteenOver116;
                        float yTestNormalized = yTest * oneOver116 + sixteenOver116;

                        // Image multiplied by 1st and 2nd x-derivatives of Gaussian.
                        dxReference += featureWeights.y * yReferenceNormalized;
                        dxTest += featureWeights.y * yTestNormalized;
                        ddxReference += featureWeights.z * yReferenceNormalized;
                        ddxTest += featureWeights.z * yTestNormalized;

                        // Image multiplied by Gaussian.
                        gaussianFilteredReference += featureWeights.x * yReferenceNormalized;
                        gaussianFilteredTest += featureWeights.x * yTestNormalized;
                    }

                    intermediateFeaturesImageReference.set(x, y, color3(dxReference, ddxReference, gaussianFilteredReference));
                    intermediateFeaturesImageTest.set(x, y, color3(dxTest, ddxTest, gaussianFilteredTest));
                }
            }

            // Convolve in y direction (1st and 2nd derivative for filter in y direction, Gaussian in x direction), then compute difference.
            // For details on the convolution, see separatedConvolutions.pdf in the FLIP repository.
            // We filter both reference and test image simultaneously (for better performance).
#pragma omp parallel for
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    float dxReference = 0.0f, dxTest = 0.0f, ddxReference = 0.0f, ddxTest = 0.0f;
                    float dyReference = 0.0f, dyTest = 0.0f, ddyReference = 0.0f, ddyTest = 0.0f;

                    for (int iy = -halfFilterWidth; iy <= halfFilterWidth; iy++)
                    {
                        int yy = Min(Max(0, y + iy), h - 1);

                        const color3 featureWeights = featureFilter.get(iy + halfFilterWidth, 0);
                        const color3 intermediateFeaturesReference = intermediateFeaturesImageReference.get(x, yy);
                        const color3 intermediateFeatureTest = intermediateFeaturesImageTest.get(x, yy);

                        // Intermediate images (1st and 2nd derivative in x) multiplied by Gaussian.
                        dxReference += featureWeights.x * intermediateFeaturesReference.x;
                        dxTest += featureWeights.x * intermediateFeatureTest.x;
                        ddxReference += featureWeights.x * intermediateFeaturesReference.y;
                        ddxTest += featureWeights.x * intermediateFeatureTest.y;

                        // Intermediate image (Gaussian) multiplied by 1st and 2nd y-derivatives of Gaussian.
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

                    const float featureDifference = std::pow(normalizationFactor * Max(edgeDifference, pointDifference), FLIPConstants.gqf);
                    const float colorDifference = this->get(x, y);

                    const float errorFLIP = std::pow(colorDifference, 1.0f - featureDifference);

                    this->set(x, y, errorFLIP);
                }
            }
        }

        void setMaxExposure(image<float>& errorMap, image<float>& exposureMap, float exposure)
        {
#pragma omp parallel for
            for (int y = 0; y < this->getHeight(); y++)
            {
                for (int x = 0; x < this->getWidth(); x++)
                {
                    float srcValue = errorMap.get(x, y);
                    float dstValue = this->get(x, y);

                    if (srcValue > dstValue)
                    {
                        exposureMap.set(x, y, exposure);
                        this->set(x, y, srcValue);
                    }
                }
            }
        }

        void expose(float level)
        {
            const float m = std::pow(2.0f, level);
#pragma omp parallel for
            for (int y = 0; y < this->getHeight(); y++)
            {
                for (int x = 0; x < this->getWidth(); x++)
                {
                    this->set(x, y, this->get(x, y) * m);
                }
            }
        }

        void computeExposures(std::string tm, float& startExposure, float& stopExposure)
        {
            int toneMapper = 1;
            if (tm == "reinhard")
                toneMapper = 0;
            if (tm == "aces")
                toneMapper = 1;
            if (tm == "hable")
                toneMapper = 2;

            const float* tc = FLIP::ToneMappingCoefficients[toneMapper];

            float t = 0.85f;
            float a = tc[0] - t * tc[3];
            float b = tc[1] - t * tc[4];
            float c = tc[2] - t * tc[5];

            float xMin = 0.0f;
            float xMax = 0.0f;
            solveSecondDegree(xMin, xMax, a, b, c);

            float Ymin = 1e30f;
            float Ymax = -1e30f;
            std::vector<float> luminances;
            for (int y = 0; y < this->mDim.y; y++)
            {
                for (int x = 0; x < this->mDim.x; x++)
                {
                    float luminance = color3::linearRGB2Luminance(this->mvpHostData[this->index(x, y)]);
                    luminances.push_back(luminance);
                    if (luminance != 0.0f)
                    {
                        Ymin = std::min(luminance, Ymin);
                    }
                    Ymax = std::max(luminance, Ymax);
                }
            }
            std::sort(luminances.begin(), luminances.end());
            size_t medianLocation = luminances.size() / 2 - 1;
            float Ymedian = (luminances[medianLocation] + luminances[medianLocation + 1]) * 0.5f;

            startExposure = log2(xMax / Ymax);
            stopExposure = log2(xMax / Ymedian);
        }

        bool exrSave(const std::string& fileName)
        {
            constexpr int channels = 3;

            float* vpImage[channels];
            std::vector<float> vImages[channels];
            for (int i = 0; i < channels; ++i)
            {
                vImages[i].resize(this->mDim.x * this->mDim.y);
            }
            int pixelIndex = 0;
            for (int y = 0; y < this->mDim.y; y++)
            {
                for (int x = 0; x < this->mDim.x; x++)
                {
                    color3 p = this->get(x, y);
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
            exrImage.width = this->mWidth;
            exrImage.height = this->mHeight;

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

    };

    template <>
    bool image<float>::exrSave(const std::string& fileName)
    {
        EXRHeader exrHeader;
        InitEXRHeader(&exrHeader);
        exrHeader.num_channels = 1;
        exrHeader.channels = (EXRChannelInfo*)malloc(sizeof(EXRChannelInfo));
        exrHeader.pixel_types = (int*)malloc(sizeof(int));
        exrHeader.requested_pixel_types = (int*)malloc(sizeof(int));
        exrHeader.compression_type = TINYEXR_COMPRESSIONTYPE_ZIP;
        exrHeader.pixel_types[0] = TINYEXR_PIXELTYPE_FLOAT;
        exrHeader.requested_pixel_types[0] = TINYEXR_PIXELTYPE_FLOAT;
        exrHeader.channels[0].name[0] = 'R';
        exrHeader.channels[0].name[1] = '\0';

        EXRImage exrImage;
        InitEXRImage(&exrImage);
        exrImage.num_channels = 1;
        exrImage.images = (unsigned char**)&(this->mvpHostData);
        exrImage.width = this->mDim.x;
        exrImage.height = this->mDim.y;

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
