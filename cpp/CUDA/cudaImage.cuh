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

#pragma once

#include "sharedflip.h"
#include "cudaTensor.cuh"

namespace FLIP
{

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


        void FLIP(image<color3>& reference, image<color3>& test, float ppd);

        // Perform the x-component of separable spatial filtering of both the reference and the test.
        // referenceImage and testImage are expected to be in YCxCz space.
        static void spatialFilterFirstDir(image& referenceImage, image& intermediateYCxImageReference, image& intermediateCzImageReference, image& testImage, image& intermediateYCxImageTest, image& intermediateCzImageTest, image& filterYCx, image& filterCz)
        {
            referenceImage.synchronizeDevice();
            testImage.synchronizeDevice();
            filterYCx.synchronizeDevice();
            filterCz.synchronizeDevice();
            FLIP::kernelSpatialFilterFirstDir << <intermediateYCxImageReference.getGridDim(), intermediateYCxImageReference.getBlockDim() >> > (intermediateYCxImageReference.mvpDeviceData, intermediateCzImageReference.mvpDeviceData, referenceImage.mvpDeviceData, intermediateYCxImageTest.mvpDeviceData, intermediateCzImageTest.mvpDeviceData, testImage.mvpDeviceData, filterYCx.mvpDeviceData, filterCz.mvpDeviceData, intermediateYCxImageReference.mDim, filterYCx.mDim); // Filter sizes are the same.
            image<T>::checkStatus("kernelSpatialFilterFirstDir");
            intermediateYCxImageReference.setState(CudaTensorState::DEVICE_ONLY);
            intermediateCzImageReference.setState(CudaTensorState::DEVICE_ONLY);
            intermediateYCxImageTest.setState(CudaTensorState::DEVICE_ONLY);
            intermediateCzImageTest.setState(CudaTensorState::DEVICE_ONLY);
        }

        // Perform the y-component of separable spatial filtering of both the reference and the test and compute color difference.
        static void spatialFilterSecondDirAndColorDifference(image& intermediateYCxImageReference, image& intermediateCzImageReference, image& intermediateYCxImageTest, image& intermediateCzImageTest, image& colorDifferenceImage, image& filterYCx, image& filterCz)
        {
            intermediateYCxImageReference.synchronizeDevice();
            intermediateCzImageReference.synchronizeDevice();
            intermediateYCxImageTest.synchronizeDevice();
            intermediateCzImageTest.synchronizeDevice();
            filterYCx.synchronizeDevice();
            filterCz.synchronizeDevice();

            // Set color difference constants.
            const float cmax = color3::computeMaxDistance(FLIPConstants.gqc);
            const float pccmax = FLIPConstants.gpc * cmax;

            FLIP::kernelSpatialFilterSecondDirAndColorDifference << <colorDifferenceImage.getGridDim(), colorDifferenceImage.getBlockDim() >> > (colorDifferenceImage.mvpDeviceData, intermediateYCxImageReference.mvpDeviceData, intermediateCzImageReference.mvpDeviceData, intermediateYCxImageTest.mvpDeviceData, intermediateCzImageTest.mvpDeviceData, filterYCx.mvpDeviceData, filterCz.mvpDeviceData, colorDifferenceImage.mDim, filterYCx.mDim, cmax, pccmax); // Filter sizes are the same.
            image<T>::checkStatus("kernelSpatialFilterSecondDirAndColorDifference");
            colorDifferenceImage.setState(CudaTensorState::DEVICE_ONLY);
        }

        // Perform the x-component of separable feature detection filtering of both the reference and the test.
        // referenceImage and testImage are expected to be in YCxCz space.
        static void featureFilterFirstDir(image& referenceImage, image& intermediateFeaturesImageReference, image& testImage, image& intermediateFeaturesImageTest, image& featureFilter)
        {
            referenceImage.synchronizeDevice();
            testImage.synchronizeDevice();
            featureFilter.synchronizeDevice();
            FLIP::kernelFeatureFilterFirstDir << <intermediateFeaturesImageReference.mGridDim, intermediateFeaturesImageReference.mBlockDim >> > (intermediateFeaturesImageReference.mvpDeviceData, referenceImage.mvpDeviceData, intermediateFeaturesImageTest.mvpDeviceData, testImage.mvpDeviceData, featureFilter.mvpDeviceData, intermediateFeaturesImageReference.mDim, featureFilter.mDim);
            image<T>::checkStatus("kernelFeatureFilterFirstDir");
            intermediateFeaturesImageReference.setState(CudaTensorState::DEVICE_ONLY);
            intermediateFeaturesImageTest.setState(CudaTensorState::DEVICE_ONLY);
        }

        // Perform the y-component of separable feature detection filtering of both the reference and the test and compute feature difference.
        static void featureFilterSecondDirAndFeatureDifference(image& intermediateFeaturesImageReference, image& intermediateFeaturesImageTest, image& featureDifferenceImage, image& featureFilter)
        {
            intermediateFeaturesImageReference.synchronizeDevice();
            intermediateFeaturesImageTest.synchronizeDevice();
            featureFilter.synchronizeDevice();
            FLIP::kernelFeatureFilterSecondDirAndFeatureDifference << <featureDifferenceImage.mGridDim, featureDifferenceImage.mBlockDim >> > (featureDifferenceImage.mvpDeviceData, intermediateFeaturesImageReference.mvpDeviceData, intermediateFeaturesImageTest.mvpDeviceData, featureFilter.mvpDeviceData, featureDifferenceImage.mDim, featureFilter.mDim);
            image<T>::checkStatus("kernelFeatureFilterSecondDirAndFeatureDifference");
            featureDifferenceImage.setState(CudaTensorState::DEVICE_ONLY);
        }

        void finalError(image<color3>& colorFeatureDifference)
        {
            FLIP::kernelFinalError << <this->mGridDim, this->mBlockDim >> > (this->getDeviceData(), colorFeatureDifference.getDeviceData(), this->mDim);
            image<T>::checkStatus("kernelFinalError");
            this->setState(CudaTensorState::DEVICE_ONLY);
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


    ///////////////////////////////////////////////////////////////////////////

    template<typename T>
    inline void image<T>::FLIP(image<color3>& reference, image<color3>& test, float ppd)
    {
        int width = reference.getWidth();
        int height = reference.getHeight();

        // Temporary images (on device).
        image<color3> intermediateReferenceYCx(width, height), intermediateReferenceCz(width, height), intermediateTestYCx(width, height), intermediateTestCz(width, height);
        image<color3> intermediateFeaturesReference(width, height), intermediateFeaturesTest(width, height);
        image<color3> colorFeatureDifference(width, height);

        // Transform from sRGB to YCxCz.
        reference.sRGB2YCxCz();
        test.sRGB2YCxCz();

        // Prepare separated spatial filters. Because the filter for the Blue-Yellow channel is a sum of two Gaussians, we need to separate the spatial filter into two
        // (YCx for the Achromatic and Red-Green channels and Cz for the Blue-Yellow channel). For details, see separatedConvolutions.pdf in the FLIP repository.
        int spatialFilterRadius = calculateSpatialFilterRadius(ppd);
        int spatialFilterWidth = 2 * spatialFilterRadius + 1;
        image<color3> spatialFilterYCx(spatialFilterWidth, 1);
        image<color3> spatialFilterCz(spatialFilterWidth, 1);
        setSpatialFilters(spatialFilterYCx, spatialFilterCz, ppd, spatialFilterRadius);
        
        // The next two calls perform separable spatial filtering on both the reference and test image at the same time (for better performance).
        // The second call also computes the color difference between the images.
        FLIP::image<color3>::spatialFilterFirstDir(reference, intermediateReferenceYCx, intermediateReferenceCz, test, intermediateTestYCx, intermediateTestCz, spatialFilterYCx, spatialFilterCz);
        FLIP::image<color3>::spatialFilterSecondDirAndColorDifference(intermediateReferenceYCx, intermediateReferenceCz, intermediateTestYCx, intermediateTestCz, colorFeatureDifference, spatialFilterYCx, spatialFilterCz);

        // Prepare separated feature (edge/point) detection filters.
        const float stdDev = 0.5f * FLIPConstants.gw * ppd;
        const int featureFilterRadius = int(std::ceil(3.0f * stdDev));
        int featureFilterWidth = 2 * featureFilterRadius + 1;
        image<color3> featureFilter(featureFilterWidth, 1);
        setFeatureFilter(featureFilter, ppd);

        // The following two calls convolve (separably) referenceImage and testImage with the edge and point detection filters and performs additional computations for the feature differences.
        FLIP::image<color3>::featureFilterFirstDir(reference, intermediateFeaturesReference, test, intermediateFeaturesTest, featureFilter);
        FLIP::image<color3>::featureFilterSecondDirAndFeatureDifference(intermediateFeaturesReference, intermediateFeaturesTest, colorFeatureDifference, featureFilter);

        this->finalError(colorFeatureDifference);
    }
}
