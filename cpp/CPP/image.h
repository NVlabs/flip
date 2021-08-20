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

        T get(int x, int y)
        {
            return this->mvpHostData[this->index(x, y)];
        }

        void set(int x, int y, T value)
        {
            this->mvpHostData[this->index(x, y)] = value;
        }


        //////////////////////////////////////////////////////////////////////////////////

        static float GaussSum(const float x2, const float a1, const float b1, const float a2, const float b2)
        {
            const float pi = float(PI);
            const float pi_sq = float(PI * PI);
            return a1 * std::sqrt(pi / b1) * std::exp(-pi_sq * x2 / b1) + a2 * std::sqrt(pi / b2) * std::exp(-pi_sq * x2 / b2);
        }

        static float Gaussian(const float x, const float y, const float sigma)
        {
            return std::exp(-(x * x + y * y) / (2.0f * sigma * sigma));
        }

        static int calculateSpatialFilterRadius(const float ppd)
        {
            const float deltaX = 1.0f / ppd;
            const float pi_sq = float(PI * PI);

            float maxScaleParameter = std::max(std::max(std::max(GaussianConstants.b1.x, GaussianConstants.b1.y), std::max(GaussianConstants.b1.z, GaussianConstants.b2.x)), std::max(GaussianConstants.b2.y, GaussianConstants.b2.z));
            int radius = int(std::ceil(3.0f * std::sqrt(maxScaleParameter / (2.0f * pi_sq)) * ppd)); // Set radius based on largest scale parameter

            return radius;
        }

        static void setSpatialFilter(image<color3>& filter, float ppd, int filterRadius)
        {
            float deltaX = 1.0f / ppd;
            color3 filterSum = { 0.0f, 0.0f, 0.0f };
            int filterWidth = 2 * filterRadius + 1;

            for (int y = 0; y < filterWidth; y++)
            {
                float iy = (y - filterRadius) * deltaX;
                for (int x = 0; x < filterWidth; x++)
                {
                    float ix = (x - filterRadius) * deltaX;

                    float dist2 = ix * ix + iy * iy;
                    float gx = GaussSum(dist2, GaussianConstants.a1.x, GaussianConstants.b1.x, GaussianConstants.a2.x, GaussianConstants.b2.x);
                    float gy = GaussSum(dist2, GaussianConstants.a1.y, GaussianConstants.b1.y, GaussianConstants.a2.y, GaussianConstants.b2.y);
                    float gz = GaussSum(dist2, GaussianConstants.a1.z, GaussianConstants.b1.z, GaussianConstants.a2.z, GaussianConstants.b2.z);
                    color3 value = color3(gx, gy, gz);
                    filter.set(x, y, value);
                    filterSum += value;
                }
            }

            // normalize weights
            filter.multiplyAndAdd(color3(1.0f) / filterSum);
        }

        static void setFeatureFilter(image<color3>& filter, const float ppd, const bool pointDetector)
        {
            const float stdDev = 0.5f * HostFLIPConstants.gw * ppd;
            const int radius = int(std::ceil(3.0f * stdDev));
            const int width = 2 * radius + 1;

            float weightX, weightY;
            float negativeWeightsSumX = 0.0f;
            float positiveWeightsSumX = 0.0f;
            float negativeWeightsSumY = 0.0f;
            float positiveWeightsSumY = 0.0f;

            for (int y = 0; y < width; y++)
            {
                int yy = y - radius;
                for (int x = 0; x < width; x++)
                {
                    int xx = x - radius;
                    float G = Gaussian(float(xx), float(yy), stdDev);
                    if (pointDetector)
                    {
                        weightX = (float(xx) * float(xx) / (stdDev * stdDev) - 1.0f) * G;
                        weightY = (float(yy) * float(yy) / (stdDev * stdDev) - 1.0f) * G;
                    }
                    else
                    {
                        weightX = -float(xx) * G;
                        weightY = -float(yy) * G;
                    }
                    filter.set(x, y, color3(weightX, weightY, 0.0f));

                    if (weightX > 0.0f)
                        positiveWeightsSumX += weightX;
                    else
                        negativeWeightsSumX += -weightX;

                    if (weightY > 0.0f)
                        positiveWeightsSumY += weightY;
                    else
                        negativeWeightsSumY += -weightY;
                }
            }

            // Normalize positive weights to sum to 1 and negative weights to sum to -1
            for (int y = 0; y < width; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    color3 p = filter.get(x, y);

                    filter.set(x, y, color3(p.x / (p.x > 0.0f ? positiveWeightsSumX : negativeWeightsSumX), p.y / (p.y > 0.0f ? positiveWeightsSumY : negativeWeightsSumY), 0.0f));
                }
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
            image<color3> pointReference(width, height), pointTest(width, height);
            image<color3> edgeReference(width, height), edgeTest(width, height);
            image<color3> featureDifference(width, height);

            //  move from sRGB to YCxCz
            referenceImage.sRGB2YCxCz();
            testImage.sRGB2YCxCz();

            //  spatial filtering
            int spatialFilterRadius = calculateSpatialFilterRadius(ppd);
            int spatialFilterWidth = 2 * spatialFilterRadius + 1;
            image<color3> spatialFilter(spatialFilterWidth, spatialFilterWidth);
            setSpatialFilter(spatialFilter, ppd, spatialFilterRadius);
            preprocessedReference.convolve(referenceImage, spatialFilter);
            preprocessedTest.convolve(testImage, spatialFilter);

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
            image<color3> pointFilter(featureFilterWidth, featureFilterWidth);
            image<color3> edgeFilter(featureFilterWidth, featureFilterWidth);
            setFeatureFilter(pointFilter, ppd, true);
            setFeatureFilter(edgeFilter, ppd, false);

            //  grayscale images needed for feature detection
            image<color3> grayReference(width, height), grayTest(width, height);
            grayReference.CIELab2Gray(referenceImage);
            grayTest.CIELab2Gray(testImage);

            //  feature filtering
            pointReference.convolve(grayReference, pointFilter);
            pointTest.convolve(grayTest, pointFilter);
            edgeReference.convolve(grayReference, edgeFilter);
            edgeTest.convolve(grayTest, edgeFilter);

            //  feature difference
            featureDifference.computeFeatureDifference(pointReference, pointTest, edgeReference, edgeTest);

            this->finalError(colorDifference, featureDifference);
        }

        void finalError(image<color3>& colorDifference, image<color3>& featureDifference)
        {
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
            for (int y = 0; y < this->getHeight(); y++)
            {
                for (int x = 0; x < this->getWidth(); x++)
                {
                    const float cmax = color3::computeMaxDistance(HostFLIPConstants.gqc);
                    const float pccmax = HostFLIPConstants.gpc * cmax;

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

        void computeFeatureDifference(image& pointReference, image& pointTest, image& edgeReference, image& edgeTest)
        {
            for (int y = 0; y < this->getHeight(); y++)
            {
                for (int x = 0; x < this->getWidth(); x++)
                {
                    const float normalizationFactor = 1.0f / std::sqrt(2.0f);

                    color3 er = edgeReference.get(x, y);
                    color3 et = edgeTest.get(x, y);
                    color3 pr = pointReference.get(x, y);
                    color3 pt = pointTest.get(x, y);

                    const float edgeValueRef = std::sqrt(er.x * er.x + er.y * er.y);
                    const float edgeValueTest = std::sqrt(et.x * et.x + et.y * et.y);
                    const float pointValueRef = std::sqrt(pr.x * pr.x + pr.y * pr.y);
                    const float pointValueTest = std::sqrt(pt.x * pt.x + pt.y * pt.y);

                    const float edgeDifference = std::abs(edgeValueRef - edgeValueTest);
                    const float pointDifference = std::abs(pointValueRef - pointValueTest);

                    const float featureDifference = std::pow(normalizationFactor * Max(edgeDifference, pointDifference), HostFLIPConstants.gqf);

                    this->set(x, y, color3(featureDifference, 0.0f, 0.0f));
                }
            }
        }

        void setMaxExposure(image<float>& errorMap, image<float>& exposureMap, float exposure)
        {
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
            float m = std::pow(2.0f, level);
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
