/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES
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

#include <cmath>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <algorithm>

namespace FLIP
{
    template <typename T>
    class histogram
    {
    private:
        T mMinValue, mMaxValue;
        size_t mValueCount, mErrorValueCount;
        T mBucketSize;
        std::vector<size_t> mvBuckets;
        size_t mBucketIdRange[2];
        size_t mBucketIdMax;

    public:
        histogram(size_t buckets, T minValue = 0.0, T maxValue = 1.0)
        {
            this->init(buckets, minValue, maxValue);
        }

        void init(size_t buckets, T minValue, T maxValue, size_t value = 0)
        {
            this->mMinValue = minValue;
            this->mMaxValue = maxValue;
            this->mValueCount = 0;
            this->mErrorValueCount = 0;
            this->mBucketIdRange[0] = std::string::npos;
            this->mBucketIdRange[1] = 0;
            this->resize(buckets);
            this->mvBuckets.resize(buckets, value);
        }

        T getBucketSize() const { return mBucketSize; }
        size_t getBucketIdMin() const { return mBucketIdRange[0]; }
        size_t getBucketIdMax() const { return mBucketIdRange[1]; }
        size_t getBucketValue(size_t bucketId) const { return mvBuckets[bucketId]; }
        size_t size() const { return mvBuckets.size(); }
        T getMinValue() const { return this->mMinValue; }
        T getMaxValue() const { return this->mMaxValue; }
        T getBucketStep() const { return (this->mMaxValue - this->mMinValue) / this->mvBuckets.size(); }

        void clear(size_t value = 0)
        {
            this->mvBuckets.resize(mvBuckets.size(), value);
        }

        void resize(size_t buckets)
        {
            this->mBucketSize = (this->mMaxValue - this->mMinValue) / buckets;
            this->mvBuckets.resize(buckets);
        }

        size_t valueBucketId(T value) const
        {
            if (value < this->mMinValue || value > this->mMaxValue)
                return std::string::npos;

            size_t bucketId = size_t(double(value) / this->mBucketSize);

            if (bucketId == this->mvBuckets.size())
            {
                bucketId--;
            }
            return bucketId;
        }

        void inc(T value, size_t amount = 1)
        {
            size_t bucketId = valueBucketId(value);
            if (bucketId != std::string::npos)
            {
                this->mvBuckets[bucketId] += amount;
                this->mValueCount += amount;
                this->mBucketIdRange[0] = std::min(this->mBucketIdRange[0], bucketId);
                this->mBucketIdRange[1] = std::max(this->mBucketIdRange[1], bucketId);
            }
            else
            {
                mErrorValueCount += amount;
            }
        }

        std::string toPython(std::string fileName, size_t numPixels, T meanValue, T maxValue, T minValue, T weightedMedian, T firstWeightedQuartile, T thirdWeightedQuartile, const bool optionLog, const bool includeValues, const float yMax) const
        {
            std::stringstream ss;

            T bucketStep = getBucketStep();

            //  imports
            ss << "import matplotlib.pyplot as plt\n";
            ss << "import os\n";
            ss << "import sys\n";
            ss << "import numpy as np\n";
            ss << "from matplotlib.ticker import (MultipleLocator)\n\n";

            ss << "dimensions = (25, 15)  #  centimeters\n\n";

            ss << "lineColor = 'blue'\n";
            ss << "fillColor = 'lightblue'\n";
            ss << "meanLineColor = 'red'\n";
            ss << "weightedMedianLineColor = 'gray'\n";
            ss << "quartileLineColor = 'purple'\n";
            ss << "fontSize = 14\n";
            ss << "numPixels = " << numPixels << "\n\n";

            ss << "meanValue = " << meanValue << "\n";
            ss << "maxValue = " << maxValue << "\n";
            ss << "minValue = " << minValue << "\n\n";
            ss << "weightedMedianValue = " << weightedMedian << "\n\n";
            ss << "firstWeightedQuartileValue = " << firstWeightedQuartile << "\n\n";
            ss << "thirdWeightedQuartileValue = " << thirdWeightedQuartile << "\n\n";

            //  X-axis
            ss << "dataX = [";
            for (size_t bucketId = 0; bucketId < this->mvBuckets.size(); bucketId++)
            {
                ss << (bucketId > 0 ? ", " : "");
                ss << bucketStep * bucketId + 0.5 * bucketStep;
            }
            ss << "]\n\n";

            // FLIP histogram.
            ss << "dataFLIP = [";
            for (size_t bucketId = 0; bucketId < this->mvBuckets.size(); bucketId++)
            {
                ss << (bucketId > 0 ? ", " : "");
                ss << this->mvBuckets[bucketId];
            }
            ss << "]\n\n";

            // Weighted FLIP histogram.
            ss << "bucketStep = " << bucketStep << "\n";
            ss << "weightedDataFLIP = np.empty(" << this->mvBuckets.size() << ")\n";
            ss << "moments = np.empty(" << this->mvBuckets.size() << ")\n";
            ss << "for i in range(" << this->mvBuckets.size() << ") :\n";
            ss << "\tweight = (i + 0.5) * bucketStep\n";
            ss << "\tweightedDataFLIP[i] = dataFLIP[i] * weight\n";
            ss << "weightedDataFLIP /= (numPixels /(1024 * 1024))  # normalized with the number of megapixels in the image\n\n";
            if (optionLog)
            {
                ss << "for i in range(" << this->mvBuckets.size() << ") :\n";
                ss << "\tif weightedDataFLIP[i] > 0 :\n";
                ss << "\t\tweightedDataFLIP[i] = np.log10(weightedDataFLIP[i])  # avoid log of zero\n\n";
            }

            if (yMax != 0.0f)
            {
                ss << "maxY = " << yMax << "\n\n";
            }
            else
            {
                ss << "maxY = max(weightedDataFLIP)\n\n";
            }

            ss << "sumWeightedDataFLIP = sum(weightedDataFLIP)\n\n";

            ss << "font = { 'family' : 'serif', 'style' : 'normal', 'weight' : 'normal', 'size' : fontSize }\n";
            ss << "lineHeight = fontSize / (dimensions[1] * 15)\n";
            ss << "plt.rc('font', **font)\n";
            ss << "fig = plt.figure()\n";
            ss << "axes = plt.axes()\n";
            ss << "axes.xaxis.set_minor_locator(MultipleLocator(0.1))\n";
            ss << "axes.xaxis.set_major_locator(MultipleLocator(0.2))\n\n";

            ss << "fig.set_size_inches(dimensions[0] / 2.54, dimensions[1] / 2.54)\n";

            if (optionLog)
                ss << "axes.set(title = 'Weighted \\uA7FBLIP Histogram', xlabel = '\\uA7FBLIP error', ylabel = 'log(weighted \\uA7FBLIP sum per megapixel)')\n\n";
            else
                ss << "axes.set(title = 'Weighted \\uA7FBLIP Histogram', xlabel = '\\uA7FBLIP error', ylabel = 'Weighted \\uA7FBLIP sum per megapixel')\n\n";

            ss << "plt.bar(dataX, weightedDataFLIP, width = " << bucketStep << ", color = fillColor, edgecolor = lineColor)\n\n";

            if (includeValues)
            {
                ss << "plt.text(0.99, 1.0 - 1 * lineHeight, 'Mean: ' + str(f'{meanValue:.4f}'), ha = 'right', fontsize = fontSize, transform = axes.transAxes, color=meanLineColor)\n\n";
                ss << "plt.text(0.99, 1.0 - 2 * lineHeight, 'Weighted median: ' + str(f'{weightedMedianValue:.4f}'), ha = 'right', fontsize = fontSize, transform = axes.transAxes, color=weightedMedianLineColor)\n\n";
                ss << "plt.text(0.99, 1.0 - 3 * lineHeight, '1st weighted quartile: ' + str(f'{firstWeightedQuartileValue:.4f}'), ha = 'right', fontsize = fontSize, transform = axes.transAxes, color=quartileLineColor)\n\n";
                ss << "plt.text(0.99, 1.0 - 4 * lineHeight, '3rd weighted quartile: ' + str(f'{thirdWeightedQuartileValue:.4f}'), ha = 'right', fontsize = fontSize, transform = axes.transAxes, color=quartileLineColor)\n\n";
                ss << "plt.text(0.99, 1.0 - 5 * lineHeight, 'Min: ' + str(f'{minValue:.4f}'), ha = 'right', fontsize = fontSize, transform = axes.transAxes)\n";
                ss << "plt.text(0.99, 1.0 - 6 * lineHeight, 'Max: ' + str(f'{maxValue:.4f}'), ha = 'right', fontsize = fontSize, transform = axes.transAxes)\n";
            }

            ss << "axes.set_xlim(0.0, 1.0)\n";
            ss << "axes.set_ylim(0.0, maxY * 1.05)\n";

            if (includeValues)
            {
                ss << "axes.axvline(x = meanValue, color = meanLineColor, linewidth = 1.5)\n\n";
                ss << "axes.axvline(x = weightedMedianValue, color = weightedMedianLineColor, linewidth = 1.5)\n\n";
                ss << "axes.axvline(x = firstWeightedQuartileValue, color = quartileLineColor, linewidth = 1.5)\n\n";
                ss << "axes.axvline(x = thirdWeightedQuartileValue, color = quartileLineColor, linewidth = 1.5)\n\n";
                ss << "axes.axvline(x = minValue, color='black', linestyle = ':', linewidth = 1.5)\n\n";
                ss << "axes.axvline(x = maxValue, color='black', linestyle = ':', linewidth = 1.5)\n\n";
            }

            ss << "plt.savefig(\"" << fileName.substr(0, fileName.size() - 3) << ".pdf\")";

            ss << std::endl;

            return ss.str();
        }
    };

    template <typename T>
    class pooling
    {
    private:
        size_t mValueCount;
        T mValueSum;
        T mSquareValueSum;
        T mMinValue;
        T mMaxValue;
        uint32_t mMinCoord[2];
        uint32_t mMaxCoord[2];
        histogram<T> mHistogram = histogram<T>(100);
        bool mDataSorted = false;
        std::vector<T> mvData;

    public:
        pooling() { clear(); }
        pooling(size_t buckets) { mHistogram.resize(buckets); clear(); }

        histogram<T>& getHistogram() { return mHistogram; }
        T getMinValue(void) const { return mMinValue; }
        T getMaxValue(void) const { return mMaxValue; }
        T getMean(void) const { return mValueSum / mValueCount; }

        double getWeightedPercentile(const double percent) const
        {
            double weight;
            double weightedValue;
            double bucketStep = mHistogram.getBucketStep();
            double sumWeightedDataValue = 0.0;
            for (size_t bucketId = 0; bucketId < mHistogram.size(); bucketId++)
            {
                weight = (bucketId + 0.5) * bucketStep;
                weightedValue = mHistogram.getBucketValue(bucketId) * weight;
                sumWeightedDataValue += weightedValue;
            }

            double sum = 0;
            size_t weightedMedianIndex = 0;
            for (size_t bucketId = 0; bucketId < mHistogram.size(); bucketId++)
            {
                weight = (bucketId + 0.5) * bucketStep;
                weightedValue = mHistogram.getBucketValue(bucketId) * weight;
                weightedMedianIndex = bucketId;
                if (sum + weightedValue > percent * sumWeightedDataValue)
                    break;
                sum += weightedValue;
            }

            weight = (weightedMedianIndex + 0.5) * bucketStep;
            weightedValue = mHistogram.getBucketValue(weightedMedianIndex) * weight;
            double discrepancy = percent * sumWeightedDataValue - sum;
            double linearWeight = discrepancy / weightedValue; // In [0,1].
            double percentile = (weightedMedianIndex + linearWeight) * bucketStep;
            return percentile;
        }

        T getPercentile(const float percent, const bool bWeighted = false)
        {
            if (!mDataSorted)
            {
                std::sort(mvData.begin(), mvData.end());
                mDataSorted = true;
            }

            T percentile = T(0);
            if (bWeighted)
            {
                T runningSum = T(0);
                for (size_t i = 0; i < mvData.size(); i++)
                {
                    runningSum += mvData[i];
                    if (runningSum > percent * mValueSum)
                    {
                        percentile = mvData[i];
                        break;
                    }
                }
            }
            else
            {
                // Using the nearest-rank method.
                percentile = mvData[size_t(std::ceil(mvData.size() * percent))];
            }

            return percentile;
        }

        void clear()
        {
            mValueCount = 0;
            mValueSum = T(0);
            mSquareValueSum = T(0);
            mMinValue = std::numeric_limits<T>::max();
            mMaxValue = std::numeric_limits<T>::min();
            mvData.clear();
            mHistogram.clear();
        }

        void update(uint32_t xcoord, uint32_t ycoord, T value)
        {
            mValueCount++;
            mValueSum += value;
            mSquareValueSum += (value * value);
            mHistogram.inc(value);

            mvData.push_back(value);

            mDataSorted = false;

            if (value < mMinValue)
            {
                mMinValue = value;
                mMinCoord[0] = xcoord;
                mMinCoord[1] = ycoord;
            }

            if (value > mMaxValue)
            {
                mMaxValue = value;
                mMaxCoord[0] = xcoord;
                mMaxCoord[1] = ycoord;
            }
        }

        void save(const std::string& fileName, size_t imgWidth, size_t imgHeight, const bool optionLog, const std::string referenceFileName, const std::string testFileName, const bool includeValues, const float yMax)
        {
            std::ofstream file;
            std::string pyFileName = fileName;

            size_t area = size_t(imgWidth) * size_t(imgHeight);

            // Python output.
            std::ofstream pythonHistogramFile;
            pythonHistogramFile.open(pyFileName);
            pythonHistogramFile << mHistogram.toPython(pyFileName, area, getMean(), getMaxValue(), getMinValue(), getPercentile(0.5f, true), getPercentile(0.25f, true), getPercentile(0.75f, true), optionLog, includeValues, yMax);
            pythonHistogramFile.close();
        }
    };
}
