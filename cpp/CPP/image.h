﻿/*
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

#include "tensor.h"

namespace FLIP
{
    FLIPConstants HostFLIPConstants;

    static const struct
    {
        color3 a1 = { 1.0f, 1.0f, 34.1f };
        color3 b1 = { 0.0047f, 0.0053f, 0.04f };
        color3 a2 = { 0.0f, 0.0f, 13.5f };
        color3 b2 = { 1.0e-5f, 1.0e-5f, 0.025f };
    } GaussianConstants;  // constants for Gaussians -- see paper for details.

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
        static float Gaussian2D(const float x, const float y, const float sigma) // standard 2D Gaussian
        {
            return std::exp(-(x * x + y * y) / (2.0f * sigma * sigma));
        }

        static float Gaussian(const float x2, const float a, const float b) // 1D Gaussian in alternative form (see FLIP paper)
        {
            const float pi = float(PI);
            const float pi_sq = float(PI * PI);
            return a * std::sqrt(pi / b) * std::exp(-pi_sq * x2 / b);
        }

        static float GaussianSqrt(const float x2, const float a, const float b) // Needed to separate sum of Gaussians filters
        {
            const float pi = float(PI);
            const float pi_sq = float(PI * PI);
            return std::sqrt(a * std::sqrt(pi / b)) * std::exp(-pi_sq * x2 / b);
        }

        static int calculateSpatialFilterRadius(const float ppd)
        {
            const float deltaX = 1.0f / ppd;
            const float pi_sq = float(PI * PI);

            float maxScaleParameter = std::max(std::max(std::max(GaussianConstants.b1.x, GaussianConstants.b1.y), std::max(GaussianConstants.b1.z, GaussianConstants.b2.x)), std::max(GaussianConstants.b2.y, GaussianConstants.b2.z));
            int radius = int(std::ceil(3.0f * std::sqrt(maxScaleParameter / (2.0f * pi_sq)) * ppd)); // Set radius based on largest scale parameter

            return radius;
        }

        static void setSpatialFilters(image<color3>& filterARG, image<color3>& filterBY, float ppd, int filterRadius)
        {
            float deltaX = 1.0f / ppd;
            color3 filterSumARG = { 0.0f, 0.0f, 0.0f };
            color3 filterSumBY = { 0.0f, 0.0f, 0.0f };
            int filterWidth = 2 * filterRadius + 1;

            for (int x = 0; x < filterWidth; x++)
            {
                float ix = (x - filterRadius) * deltaX;

                float ix2 = ix * ix;
                float gA = Gaussian(ix2, GaussianConstants.a1.x, GaussianConstants.b1.x);
                float gRG = Gaussian(ix2, GaussianConstants.a1.y, GaussianConstants.b1.y);
                float gBY1 = GaussianSqrt(ix2, GaussianConstants.a1.z, GaussianConstants.b1.z);
                float gBY2 = GaussianSqrt(ix2, GaussianConstants.a2.z, GaussianConstants.b2.z);
                color3 valueARG = color3(gA, gRG, 0.0f);
                color3 valueBY = color3(gBY1, gBY2, 0.0f);
                filterARG.set(x, 0, valueARG);
                filterBY.set(x, 0, valueBY);
                filterSumARG += valueARG;
                filterSumBY += valueBY;
            }

            // normalize weights
            color3 normFactorARG = { 1.0f / filterSumARG.x, 1.0f / filterSumARG.y, 1.0f };
            float normFactorBY = 1.0f / std::sqrt(filterSumBY.x * filterSumBY.x + filterSumBY.y * filterSumBY.y);
            for (int x = 0; x < filterWidth; x++)
            {
                color3 pARG = filterARG.get(x, 0);
                color3 pBY = filterBY.get(x, 0);

                filterARG.set(x, 0, color3(pARG.x * normFactorARG.x, pARG.y * normFactorARG.y, 0.0f));
                filterBY.set(x, 0, color3(pBY.x * normFactorBY, pBY.y * normFactorBY, 0.0f));
            }
        }

        static void setFeatureFilter(image<color3>& filter, const float ppd)
        {
            const float stdDev = 0.5f * HostFLIPConstants.gw * ppd;
            const int radius = int(std::ceil(3.0f * stdDev));
            const int width = 2 * radius + 1;

            float weight1, weight2;
            float weightSum = 0.0f;
            float negativeWeightsSum1 = 0.0f;
            float positiveWeightsSum1 = 0.0f;
            float negativeWeightsSum2 = 0.0f;
            float positiveWeightsSum2 = 0.0f;

            for (int x = 0; x < width; x++)
            {
                int xx = x - radius;

                // 0th derivative
                float G = Gaussian2D(float(xx), 0, stdDev);
                weightSum += G;

                // 1st derivative
                weight1 = -float(xx) * G;
                if (weight1 > 0.0f)
                    positiveWeightsSum1 += weight1;
                else
                    negativeWeightsSum1 += -weight1;

                // 2nd derivative
                weight2 = (float(xx) * float(xx) / (stdDev * stdDev) - 1.0f) * G;
                if (weight2 > 0.0f)
                    positiveWeightsSum2 += weight2;
                else
                    negativeWeightsSum2 += -weight2;

                filter.set(x, 0, color3(G, weight1, weight2));
            }

            for (int x = 0; x < width; x++)
            {
                color3 p = filter.get(x, 0);

                filter.set(x, 0, color3(p.x / weightSum, p.y / (p.y > 0.0f ? positiveWeightsSum1 : negativeWeightsSum1), p.z / (p.z > 0.0f ? positiveWeightsSum2 : negativeWeightsSum2)));
            }
        }

        void FLIP(image<color3>& reference, image<color3>& test, float ppd)
        {
            int width = reference.getWidth();
            int height = reference.getHeight();

            //  temporary images (on device)
            image<color3> referenceImage(reference), testImage(test);
            image<color3> preprocessedReference(width, height), preprocessedTest(width, height);
            image<color3> colorDifference(width, height);
            image<color3> featureDifference(width, height);

            //  move from sRGB to YCxCz
            referenceImage.sRGB2YCxCz();
            testImage.sRGB2YCxCz();

            //  spatial filtering
            int spatialFilterRadius = calculateSpatialFilterRadius(ppd);
            int spatialFilterWidth = 2 * spatialFilterRadius + 1;
            image<color3> spatialFilterARG(spatialFilterWidth, 1);
            image<color3> spatialFilterBY(spatialFilterWidth, 1);
            setSpatialFilters(spatialFilterARG, spatialFilterBY, ppd, spatialFilterRadius);
            convolve2images(referenceImage, preprocessedReference, testImage, preprocessedTest, spatialFilterARG, spatialFilterBY);

            //  move from YCxCz to CIELab
            preprocessedReference.YCxCz2CIELab();
            preprocessedTest.YCxCz2CIELab();

            //  Hunt adjustment
            preprocessedReference.huntAdjustment();
            preprocessedTest.huntAdjustment();

            //  color difference
            colorDifference.computeColorDifference(preprocessedReference, preprocessedTest);

            //  feature (point/edge) filtering
            const float stdDev = 0.5f * HostFLIPConstants.gw * ppd;
            const int featureFilterRadius = int(std::ceil(3.0f * stdDev));
            int featureFilterWidth = 2 * featureFilterRadius + 1;
            image<color3> featureFilter(featureFilterWidth, 1);
            setFeatureFilter(featureFilter, ppd);

            //  grayscale images needed for feature detection
            image<color3> grayReference(width, height), grayTest(width, height);
            grayReference.YCxCz2Gray(referenceImage);
            grayTest.YCxCz2Gray(testImage);

            featureDifference.computeFeatureDifference(grayReference, grayTest, featureFilter);

            this->finalError(colorDifference, featureDifference);
        }

        void computeFeatureDifference(const image<color3>& grayRefImage, const image<color3>& grayTestImage, const image<color3>& featureFilter)
        {
            const float normalizationFactor = 1.0f / std::sqrt(2.0f);
            const int halfFilterWidth = featureFilter.getWidth() / 2;      // The edge and point filters are of the same size.
            const int w = grayRefImage.getWidth();
            const int h = grayRefImage.getHeight();

            image<color3> iRefFeatures(w, h);
            image<color3> iTestFeatures(w, h);

            // convolve in x direction (1st and 2nd derivative for filter in x direction, 0th derivative for filter in y direction)
#pragma omp parallel for
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    float  iEdgeRefX = 0.0f, iEdgeTestX = 0.0f, iPointRefX = 0.0f, iPointTestX = 0.0f;
                    float  refGaussianFiltered = 0.0f, testGaussianFiltered = 0.0f;
                    
                    for (int ix = -halfFilterWidth; ix <= halfFilterWidth; ix++)
                    {
                        int xx = Min(Max(0, x + ix), w - 1);

                        const color3 featureWeights = featureFilter.get(ix + halfFilterWidth, 0);
                        const float grayRef = grayRefImage.get(xx, y).x;
                        const float grayTest = grayTestImage.get(xx, y).x;

                        iEdgeRefX += featureWeights.y * grayRef;
                        iEdgeTestX += featureWeights.y * grayTest;
                        iPointRefX += featureWeights.z * grayRef;
                        iPointTestX += featureWeights.z * grayTest;
                        
                        refGaussianFiltered += featureWeights.x * grayRef;
                        testGaussianFiltered += featureWeights.x * grayTest;
                    }

                    iRefFeatures.set(x, y, color3(iEdgeRefX, iPointRefX, refGaussianFiltered));
                    iTestFeatures.set(x, y, color3(iEdgeTestX, iPointTestX, testGaussianFiltered));
                }
            }

            // convolve in y direction (1st and 2nd derivative for filter in y direction, 0th derivative for filter in x direction), then compute difference
#pragma omp parallel for
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    float  edgeRefX = 0.0f, edgeTestX = 0.0f, pointRefX = 0.0f, pointTestX = 0.0f;
                    float  edgeRefY = 0.0f, edgeTestY = 0.0f, pointRefY = 0.0f, pointTestY = 0.0f;

                    for (int iy = -halfFilterWidth; iy <= halfFilterWidth; iy++)
                    {
                        int yy = Min(Max(0, y + iy), h - 1);

                        const color3 featureWeights = featureFilter.get(iy + halfFilterWidth, 0);
                        const color3 grayRef = iRefFeatures.get(x, yy);
                        const color3 grayTest = iTestFeatures.get(x, yy);

                        edgeRefX += featureWeights.x * grayRef.x;
                        edgeTestX += featureWeights.x * grayTest.x;
                        pointRefX += featureWeights.x * grayRef.y;
                        pointTestX += featureWeights.x * grayTest.y;

                        edgeRefY += featureWeights.y * grayRef.z;
                        edgeTestY += featureWeights.y * grayTest.z;
                        pointRefY += featureWeights.z * grayRef.z;
                        pointTestY += featureWeights.z * grayTest.z;
                    }

                    const float edgeValueRef = std::sqrt(edgeRefX * edgeRefX + edgeRefY * edgeRefY);
                    const float edgeValueTest = std::sqrt(edgeTestX * edgeTestX + edgeTestY * edgeTestY);
                    const float pointValueRef = std::sqrt(pointRefX * pointRefX + pointRefY * pointRefY);
                    const float pointValueTest = std::sqrt(pointTestX * pointTestX + pointTestY * pointTestY);

                    const float edgeDifference = std::abs(edgeValueRef - edgeValueTest);
                    const float pointDifference = std::abs(pointValueRef - pointValueTest);

                    const float featureDifference = std::pow(normalizationFactor * Max(edgeDifference, pointDifference), HostFLIPConstants.gqf);

                    this->set(x, y, color3(featureDifference, 0.0f, 0.0f));
                }
            }
        }

        void finalError(image<color3>& colorDifference, image<color3>& featureDifference)
        {
#pragma omp parallel for
            for (int y = 0; y < this->getHeight(); y++)
            {
                for (int x = 0; x < this->getWidth(); x++)
                {
                    const float cdiff = colorDifference.get(x, y).x;
                    const float fdiff = featureDifference.get(x, y).x;
                    const float errorFLIP = std::pow(cdiff, 1.0f - fdiff);

                    this->set(x, y, errorFLIP);
                }
            }
        }

        void computeColorDifference(image& reference, image& test)
        {
            const float cmax = color3::computeMaxDistance(HostFLIPConstants.gqc);
            const float pccmax = HostFLIPConstants.gpc * cmax;
#pragma omp parallel for
            for (int y = 0; y < this->getHeight(); y++)
            {
                for (int x = 0; x < this->getWidth(); x++)
                {
                    color3 refPixel = reference.get(x, y);
                    color3 testPixel = test.get(x, y);

                    float colorDifference = color3::HyAB(refPixel, testPixel);

                    colorDifference = powf(colorDifference, HostFLIPConstants.gqc);

                    // Re-map error to the [0, 1] range. Values between 0 and pccmax are mapped to the range [0, gpt],
                    // while the rest are mapped to the range (gpt, 1]
                    if (colorDifference < pccmax)
                    {
                        colorDifference *= HostFLIPConstants.gpt / pccmax;
                    }
                    else
                    {
                        colorDifference = HostFLIPConstants.gpt + ((colorDifference - pccmax) / (cmax - pccmax)) * (1.0f - HostFLIPConstants.gpt);
                    }

                    this->set(x, y, color3(colorDifference, 0.0f, 0.0f));
                }
            }
        }

        void huntAdjustment(void)
        {
#pragma omp parallel for
            for (int y = 0; y < this->getHeight(); y++)
            {
                for (int x = 0; x < this->getWidth(); x++)
                {
                    color3 pixel = this->get(x, y);
                    pixel.y = color3::Hunt(pixel.x, pixel.y);
                    pixel.z = color3::Hunt(pixel.x, pixel.z);
                    this->set(x , y, pixel);
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

        void solveSecondDegree(float& xMin, float& xMax, float a, float b, float c)
        {
            //  a * x^2 + b * x + c = 0
            if (a == 0.0f)
            {
                xMin = xMax = -c / b;
                return;
            }

            float d1 = -0.5f * (b / a);
            float d2 = sqrtf((d1 * d1) - (c / a));
            xMin = d1 - d2;
            xMax = d1 + d2;
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

    static void convolve2images(const FLIP::image<color3>& input1, FLIP::image<color3>& output1, const FLIP::image<color3>& input2, FLIP::image<color3>& output2, const FLIP::image<color3>& filterARG, const FLIP::image<color3>& filterBY)
    {
        const int halfFilterWidth = filterARG.getWidth() / 2; // ARG and BY filters are same size

        const int w = input1.getWidth();
        const int h = input1.getHeight();

        image<color3> iImageARG1(w, h);
        image<color3> iImageBY1(w, h);
        image<color3> iImageARG2(w, h);
        image<color3> iImageBY2(w, h);

        // filter in x direction
#pragma omp parallel for
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                color3 colorSumARG1 = { 0.0f, 0.0f, 0.0f };
                color3 colorSumBY1 = { 0.0f, 0.0f, 0.0f };
                color3 colorSumARG2 = { 0.0f, 0.0f, 0.0f };
                color3 colorSumBY2 = { 0.0f, 0.0f, 0.0f };

                for (int ix = -halfFilterWidth; ix <= halfFilterWidth; ix++)
                {
                    int xx = Min(Max(0, x + ix), w - 1);

                    const color3 weightsARG = filterARG.get(ix + halfFilterWidth, 0);
                    const color3 weightsBY = filterBY.get(ix + halfFilterWidth, 0);
                    const color3 srcColor1 = input1.get(xx, y);
                    const color3 srcColor2 = input2.get(xx, y);

                    colorSumARG1 += color3(weightsARG.x * srcColor1.x, weightsARG.y * srcColor1.y, 0.0f);
                    colorSumBY1 += color3(weightsBY.x * srcColor1.z, weightsBY.y * srcColor1.z, 0.0f);
                    colorSumARG2 += color3(weightsARG.x * srcColor2.x, weightsARG.y * srcColor2.y, 0.0f);
                    colorSumBY2 += color3(weightsBY.x * srcColor2.z, weightsBY.y * srcColor2.z, 0.0f);
                }

                iImageARG1.set(x, y, colorSumARG1);
                iImageBY1.set(x, y, colorSumBY1);
                iImageARG2.set(x, y, colorSumARG2);
                iImageBY2.set(x, y, colorSumBY2);
            }
        }

        // filter in y direction
#pragma omp parallel for
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                color3 colorSumARG1 = { 0.0f, 0.0f, 0.0f };
                color3 colorSumBY1 = { 0.0f, 0.0f, 0.0f };
                color3 colorSumARG2 = { 0.0f, 0.0f, 0.0f };
                color3 colorSumBY2 = { 0.0f, 0.0f, 0.0f };

                for (int iy = -halfFilterWidth; iy <= halfFilterWidth; iy++)
                {
                    int yy = Min(Max(0, y + iy), h - 1);

                    const color3 weightsARG = filterARG.get(iy + halfFilterWidth, 0);
                    const color3 weightsBY = filterBY.get(iy + halfFilterWidth, 0);
                    const color3 iColorARG1 = iImageARG1.get(x, yy);
                    const color3 iColorBY1 = iImageBY1.get(x, yy);
                    const color3 iColorARG2 = iImageARG2.get(x, yy);
                    const color3 iColorBY2 = iImageBY2.get(x, yy);

                    colorSumARG1 += color3(weightsARG.x * iColorARG1.x, weightsARG.y * iColorARG1.y, 0.0f);
                    colorSumBY1 += color3(weightsBY.x * iColorBY1.x, weightsBY.y * iColorBY1.y, 0.0f);
                    colorSumARG2 += color3(weightsARG.x * iColorARG2.x, weightsARG.y * iColorARG2.y, 0.0f);
                    colorSumBY2 += color3(weightsBY.x * iColorBY2.x, weightsBY.y * iColorBY2.y, 0.0f);
                }
                output1.set(x, y, color3(colorSumARG1.x, colorSumARG1.y, colorSumBY1.x + colorSumBY1.y));
                output2.set(x, y, color3(colorSumARG2.x, colorSumARG2.y, colorSumBY2.x + colorSumBY2.y));

                if (x == 0 && y == 0)
                {
                    std::cout << colorSumARG2.x << "\n";
                    std::cout << colorSumARG2.y << "\n";
                    std::cout << colorSumBY2.x + colorSumBY2.y << "\n";
                }
            }
        }
    }

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
