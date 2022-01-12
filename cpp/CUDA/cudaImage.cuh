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

#include "cudaTensor.cuh"

namespace FLIP
{
    FLIPConstants HostFLIPConstants;

    static const struct
    {
        float3 a1 = { 1.0f, 1.0f, 34.1f };
        float3 b1 = { 0.0047f, 0.0053f, 0.04f };
        float3 a2 = { 0.0f, 0.0f, 13.5f };
        float3 b2 = { 1.0e-5f, 1.0e-5f, 0.025f };
    } GaussianConstants;  // constants for Gaussians -- see paper for details.

    template<typename T>
    class image :
        public tensor<T>
    {
    public:

        image(std::string fileName, const dim3 blockDim = DEFAULT_KERNEL_BLOCK_DIM)
        {
            this->mBlockDim = blockDim;
            bool bOk = this->load(fileName);
            if (!bOk)
            {
                std::cout << "Failed to load image <" << fileName << ">" << std::endl;
                exit(-1);
            }
            this->setState(CudaTensorState::HOST_ONLY);
        }

        image(const int width, const int height, const dim3 blockDim = DEFAULT_KERNEL_BLOCK_DIM)
            : tensor<T>(width, height, 1, blockDim)
        {
        }

        image(const int width, const int height, const T clearColor, const dim3 blockDim = DEFAULT_KERNEL_BLOCK_DIM)
            : tensor<T>(width, height, 1, clearColor, blockDim)
        {
        }

        image(const int3 dim, const T clearColor, const dim3 blockDim = DEFAULT_KERNEL_BLOCK_DIM)
            : tensor<T>(dim.x, dim.y, 1, blockDim)
        {
        }

        image(const int3 dim, const dim3 blockDim = DEFAULT_KERNEL_BLOCK_DIM)
            : tensor<T>(dim.x, dim.y, 1, blockDim)
        {
        }

        image(image& image, const dim3 blockDim = DEFAULT_KERNEL_BLOCK_DIM)
        {
            this->mBlockDim = blockDim;
            this->init(image.mDim);
            this->copy(image);
        }

        image(tensor<T>& tensor, const int offset, const dim3 blockDim = DEFAULT_KERNEL_BLOCK_DIM)
        {
            this->mBlockDim = blockDim;
            this->init(tensor.getDimensions());
            this->copy(tensor, offset);
        }

        image(const color3* pColorMap, int size, const dim3 blockDim = DEFAULT_KERNEL_BLOCK_DIM)
            : tensor<T>(pColorMap, size, blockDim)
        {
        }

        ~image(void)
        {
        }

        T get(int x, int y)
        {
            this->synchronizeHost();
            return this->mvpHostData[this->index(x, y)];
        }

        void set(int x, int y, T value)
        {
            this->synchronizeHost();
            this->mvpHostData[this->index(x, y)] = value;
            this->setState(CudaTensorState::HOST_ONLY);
        }


        //////////////////////////////////////////////////////////////////////////////////

        void FLIP(image<color3>& reference, image<color3>& test, float ppd);

        void finalError(image<color3>& colorFeatureDifference)
        {
            FLIP::kernelFinalError << <this->mGridDim, this->mBlockDim >> > (this->getDeviceData(), colorFeatureDifference.getDeviceData(), this->mDim);
            image<T>::checkStatus("kernelFinalError");
            this->setState(CudaTensorState::DEVICE_ONLY);
        }

        void computeColorDifference(image& reference, image& test)
        {
            FLIP::kernelColorDifference << <this->mGridDim, this->mBlockDim >> > (this->mvpDeviceData, reference.mvpDeviceData, test.mvpDeviceData, this->mDim);
            image<T>::checkStatus("kernelColorDifference");
            this->setState(CudaTensorState::DEVICE_ONLY);
        }

        void huntAdjustment(void)
        {
            FLIP::kernelHuntAdjustment << <this->mGridDim, this->mBlockDim >> > (this->mvpDeviceData, this->mDim);
            image<T>::checkStatus("kernelHuntAdjustment");
            this->setState(CudaTensorState::DEVICE_ONLY);
        }

        static void featureFilterFirstDir(image& input1, image& output1, image& input2, image& output2, image& featureFilter)
        {
            input1.synchronizeDevice();
            input2.synchronizeDevice();
            featureFilter.synchronizeDevice();
            FLIP::kernelFeatureFilterFirstDir << <output1.mGridDim, output1.mBlockDim >> > (output1.mvpDeviceData, input1.mvpDeviceData, output2.mvpDeviceData, input2.mvpDeviceData, featureFilter.mvpDeviceData, output1.mDim, featureFilter.mDim);
            image<T>::checkStatus("kernelFeatureFilterFirstDir");
            output1.setState(CudaTensorState::DEVICE_ONLY);
            output2.setState(CudaTensorState::DEVICE_ONLY);
        }

        static void featureFilterSecondDirAndFeatureDifference(image& input1, image& input2, image& output1, image& featureFilter)
        {
            input1.synchronizeDevice();
            input2.synchronizeDevice();
            featureFilter.synchronizeDevice();
            FLIP::kernelFeatureFilterSecondDirAndFeatureDifference << <output1.mGridDim, output1.mBlockDim >> > (output1.mvpDeviceData, input1.mvpDeviceData, input2.mvpDeviceData, featureFilter.mvpDeviceData, output1.mDim, featureFilter.mDim);
            image<T>::checkStatus("kernelFeatureFilterSecondDirAndFeatureDifference");
            output1.setState(CudaTensorState::DEVICE_ONLY);
        }

        static void spatialFilterFirstDir(image& input1, image& output1ARG, image& output1BY, image& input2, image& output2ARG, image& output2BY, image& filterARG, image& filterBY)
        {
            input1.synchronizeDevice();
            input2.synchronizeDevice();
            filterARG.synchronizeDevice();
            filterBY.synchronizeDevice();
            FLIP::kernelSpatialFilterFirstDir << <output1ARG.getGridDim(), output1ARG.getBlockDim() >> > (output1ARG.mvpDeviceData, output1BY.mvpDeviceData, input1.mvpDeviceData, output2ARG.mvpDeviceData, output2BY.mvpDeviceData, input2.mvpDeviceData, filterARG.mvpDeviceData, filterBY.mvpDeviceData, output1ARG.mDim, filterARG.mDim); // filter sizes are the same
            checkStatus("kernelSpatialFilterFirstDir");
            output1ARG.setState(CudaTensorState::DEVICE_ONLY);
            output1BY.setState(CudaTensorState::DEVICE_ONLY);
            output2ARG.setState(CudaTensorState::DEVICE_ONLY);
            output2BY.setState(CudaTensorState::DEVICE_ONLY);
        }

        static void spatialFilterSecondDir(image& input1ARG, image& input1BY, image& output1, image& input2ARG, image& input2BY, image& output2, image& filterARG, image& filterBY)
        {
            input1ARG.synchronizeDevice();
            input1BY.synchronizeDevice();
            input2ARG.synchronizeDevice();
            input2BY.synchronizeDevice();
            filterARG.synchronizeDevice();
            filterBY.synchronizeDevice();
            FLIP::kernelSpatialFilterSecondDir << <output1.getGridDim(), output1.getBlockDim() >> > (output1.mvpDeviceData, input1ARG.mvpDeviceData, input1BY.mvpDeviceData, output2.mvpDeviceData, input2ARG.mvpDeviceData, input2BY.mvpDeviceData, filterARG.mvpDeviceData, filterBY.mvpDeviceData, output1.mDim, filterARG.mDim); // filter sizes are the same
            checkStatus("kernelSpatialFilterSecondDir");
            output1.setState(CudaTensorState::DEVICE_ONLY);
            output2.setState(CudaTensorState::DEVICE_ONLY);
        }

        void setMaxExposure(tensor<float>& errorMap, tensor<float>& exposureMap, float exposure)
        {
            errorMap.synchronizeDevice();
            exposureMap.synchronizeDevice();
            FLIP::kernelSetMaxExposure << <this->mGridDim, this->mBlockDim >> > (this->getDeviceData(), errorMap.getDeviceData(), exposureMap.getDeviceData(), this->mDim, exposure);
            image<T>::checkStatus("kernelSetMaxExposure");
            exposureMap.setState(CudaTensorState::DEVICE_ONLY);
            this->setState(CudaTensorState::DEVICE_ONLY);
        }

        void copy(image& srcImage)
        {
            srcImage.synchronizeDevice();
            cudaError_t cudaStatus = cudaMemcpy(this->getDeviceData(), srcImage.getDeviceData(), this->mVolume * sizeof(T), cudaMemcpyDeviceToDevice);
            if (cudaStatus != cudaSuccess)
            {
                std::cout << "copy() failed: " << cudaGetErrorString(cudaStatus) << "\n";
                exit(-1);
            }
            this->setState(CudaTensorState::DEVICE_ONLY);
        }

        void copy(tensor<T>& srcTensor, const int z)
        {
            srcTensor.synchronizeDevice();
            cudaError_t cudaStatus = cudaMemcpy(this->getDeviceData(), srcTensor.getDeviceData(z), this->mArea * sizeof(T), cudaMemcpyDeviceToDevice);
            if (cudaStatus != cudaSuccess)
            {
                std::cout << "copy() failed: " << cudaGetErrorString(cudaStatus) << "\n";
                exit(-1);
            }
            this->setState(CudaTensorState::DEVICE_ONLY);
        }

        void expose(float level)
        {
            float m = std::pow(2.0f, level);
            FLIP::kernelMultiplyAndAdd << <this->mGridDim, this->mBlockDim >> > (this->mvpDeviceData, this->mDim, m, 0.0f);
            image<T>::checkStatus("kernelExpose");
            this->setState(CudaTensorState::DEVICE_ONLY);
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

            const float* tc = ToneMappingCoefficients[toneMapper];

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
            this->synchronizeHost();

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
        this->synchronizeHost();

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


    static int calculateSpatialFilterRadius(const float ppd)
    {
        const float deltaX = 1.0f / ppd;
        const float pi_sq = float(PI * PI);

        float maxScaleParameter = std::max(std::max(std::max(GaussianConstants.b1.x, GaussianConstants.b1.y), std::max(GaussianConstants.b1.z, GaussianConstants.b2.x)), std::max(GaussianConstants.b2.y, GaussianConstants.b2.z));
        int radius = int(std::ceil(3.0f * std::sqrt(maxScaleParameter / (2.0f * pi_sq)) * ppd)); // Set radius based on largest scale parameter

        return radius;
    }

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

    ///////////////////////////////////////////////////////////////////////////

    template<typename T>
    inline void image<T>::FLIP(image<color3>& reference, image<color3>& test, float ppd)
    {
        int width = reference.getWidth();
        int height = reference.getHeight();

        //  temporary images (on device)
        image<color3> referenceImage(reference), testImage(test);
        image<color3> preprocessedReferenceARG(width, height), preprocessedReferenceBY(width, height), preprocessedReference(width, height), preprocessedTestARG(width, height), preprocessedTestBY(width, height), preprocessedTest(width, height);
        image<color3> colorFeatureDifference(width, height);

        //  move from sRGB to YCxCz
        referenceImage.sRGB2YCxCz();
        testImage.sRGB2YCxCz();

        //  spatial filtering
        int spatialFilterRadius = calculateSpatialFilterRadius(ppd);
        int spatialFilterWidth = 2 * spatialFilterRadius + 1;
        image<color3> spatialFilterARG(spatialFilterWidth, 1);
        image<color3> spatialFilterBY(spatialFilterWidth, 1);
        setSpatialFilters(spatialFilterARG, spatialFilterBY, ppd, spatialFilterRadius);
        FLIP::image<color3>::spatialFilterFirstDir(referenceImage, preprocessedReferenceARG, preprocessedReferenceBY, testImage, preprocessedTestARG, preprocessedTestBY, spatialFilterARG, spatialFilterBY);
        FLIP::image<color3>::spatialFilterSecondDir(preprocessedReferenceARG, preprocessedReferenceBY, preprocessedReference, preprocessedTestARG, preprocessedTestBY, preprocessedTest, spatialFilterARG, spatialFilterBY);

        //  move from YCxCz to CIELab
        preprocessedReference.YCxCz2CIELab();
        preprocessedTest.YCxCz2CIELab();

        //  Hunt adjustment
        preprocessedReference.huntAdjustment();
        preprocessedTest.huntAdjustment();

        //  color difference
        colorFeatureDifference.computeColorDifference(preprocessedReference, preprocessedTest);

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

        // feature detection and difference computation
        image<color3> iFeaturesReference(width, height), iFeaturesTest(width, height);
        FLIP::image<color3>::featureFilterFirstDir(grayReference, iFeaturesReference, grayTest, iFeaturesTest, featureFilter);
        FLIP::image<color3>::featureFilterSecondDirAndFeatureDifference(iFeaturesReference, iFeaturesTest, colorFeatureDifference, featureFilter);

        this->finalError(colorFeatureDifference);
    }

}
