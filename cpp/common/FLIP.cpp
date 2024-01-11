﻿/*
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

#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <filesystem>

#define NOMINMAX
#define FIXED_DECIMAL_DIGITS(x, d) std::fixed << std::setprecision(d) << (x)


//#define FLIP_USE_CUDA
#include "../CPP/flip_single_header.h"

#include "commandline.h"
#include "filename.h"
#include "pooling.h"

struct
{
    float PPD = 0;                          // If PPD==0.0, then it will be computed from the parameters below.
    float monitorDistance = 0.7f;           // Unit: meters.
    float monitorWidth = 0.7f;              // Unit: meters.
    float monitorResolutionX = 3840.0f;     // Unit: pixels.
} gFLIPOptions;

//  Pixels per degree (PPD).
inline float calculatePPD(const float dist, const float resolutionX, const float monitorWidth)
{
    return dist * (resolutionX / monitorWidth) * (float(FLIP::PI) / 180.0f);
}

inline std::string f2s(float value, size_t decimals = 4)
{
    std::stringstream ss;
    ss << std::string(value < 0.0f ? "m" : "p") << FIXED_DECIMAL_DIGITS(std::abs(value), decimals);
    return ss.str();
}

int main(int argc, char** argv)
{
    std::string FLIPString = "FLIP";
    int MajorVersion = 1;
    int MinorVersion = 2;

    const commandline_options allowedCommandLineOptions =
    {
        "Compute FLIP between reference.<png|exr> and test.<png|exr>.\n"
        "Reference and test(s) must have same resolution and format.\n"
        "If pngs are entered, LDR-FLIP will be evaluated. If exrs are entered, HDR-FLIP will be evaluated.\n"
        "For HDR-FLIP, the reference is used to automatically calculate start exposure and/or stop exposure and/or number of exposures, if they are not entered.",
        {
        { "help", "h", 0, false, "", "show this help message and exit" },
        { "reference", "r", 1, true, "REFERENCE", "Relative or absolute path (including file name and extension) for reference image" },
        { "test", "t", -1, true, "TEST", "Relative or absolute path (including file name and extension) for test image" },
        { "pixels-per-degree", "ppd", 1, false, "PIXELS-PER-DEGREE", "Observer's number of pixels per degree of visual angle. Default corresponds to\nviewing the images at 0.7 meters from a 0.7 meter wide 4K display" },
        { "viewing-conditions", "vc", 3, false, "MONITOR-DISTANCE MONITOR-WIDTH MONITOR-WIDTH-PIXELS", "Distance to monitor (in meters), width of monitor (in meters), width of monitor (in pixels).\nDefault corresponds to viewing the monitor at 0.7 meters from a 0.7 meters wide 4K display" },
        { "tone-mapper", "tm", 1, false, "ACES | HABLE | REINHARD", "Tone mapper used for HDR-FLIP. Supported tone mappers are ACES, Hable, and Reinhard (default: ACES)" },
        { "num-exposures", "n", 1, false, "NUM-EXPOSURES", "Number of exposures between (and including) start and stop exposure used to compute HDR-FLIP" },
        { "start-exposure", "cstart", 1, false, "C-START", "Start exposure used to compute HDR-FLIP" },
        { "stop-exposure", "cstop", 1, false, "C-STOP", "Stop exposure used to comput HDR-FLIP" },
        { "basename", "b", 1, false, "BASENAME", "Basename for outfiles, e.g., error and exposure maps" },
        { "csv", "c", 1, false, "CSV_FILENAME", "Write results to a csv file. Input is the desired file name (including .csv extension).\nResults are appended if the file already exists" },
        { "histogram", "hist", 0, false, "", "Save weighted histogram of the FLIP error map(s)" },
        { "y-max", "", 1, true, "", "Set upper limit of weighted histogram's y-axis" },
        { "log", "lg", 0, false, "", "Use logarithmic scale on y-axis in histogram" },
        { "exclude-pooled-values", "epv", 0, false, "", "Do not include pooled FLIP values in the weighted histogram" },
        { "save-ldr-images", "sli", 0, false, "", "Save all exposure compensated and tone mapped LDR images (png) used for HDR-FLIP" },
        { "save-ldrflip", "slf", 0, false, "", "Save all LDR-FLIP maps used for HDR-FLIP" },
        { "no-magma", "nm", 0, false, "", "Save FLIP error maps in grayscale instead of magma" },
        { "no-exposure-map", "nexm", 0, false, "", "Do not save the HDR-FLIP exposure map" },
        { "no-error-map", "nerm", 0, false, "", "Do not save the FLIP error map" },
        { "exit-on-test", "et", 0, false, "", "Do exit(EXIT_FAILURE) if the selected FLIP QUANTITY is greater than THRESHOLD"},
        { "exit-test-parameters", "etp", 2, false, "QUANTITY = {MEAN (default) | WEIGHTED-MEDIAN | MAX} THRESHOLD (default = 0.05) ", "Test parameters for selected quantity and threshold value (in [0,1]) for exit on test"},
        { "directory", "d", 1, false, "Relative or absolute path to save directory"},
    } };
    commandline commandLine(argc, argv, allowedCommandLineOptions);

    if (commandLine.optionSet("help") || !commandLine.optionSet("reference") || !commandLine.optionSet("test") || commandLine.getError())
    {
        if (commandLine.getError())
        {
            std::cout << commandLine.getErrorString() << "\n";
        }
        std::cout << "FLIP v" << MajorVersion << "." << MinorVersion << ".\n";
        commandLine.print();
        exit(EXIT_FAILURE);
    }

    std::string exitOnTestQuantity = "mean";
    float exitOnTestThresholdValue = 0.05f;
    if (commandLine.optionSet("exit-on-test") && commandLine.optionSet("exit-test-parameters"))
    {
        exitOnTestQuantity = commandLine.getOptionValue("exit-test-parameters", 0);
        std::transform(exitOnTestQuantity.begin(), exitOnTestQuantity.end(), exitOnTestQuantity.begin(), [](unsigned char c) { return std::tolower(c); });
        if (exitOnTestQuantity != "mean" && exitOnTestQuantity != "weighted-median" && exitOnTestQuantity != "max")
        {
            std::cout << "For --exit-test-parameters / -etp, the first paramter needs to be {MEAN | WEIGHTED-MEDIAN | MAX}\n";
            exit(EXIT_FAILURE);
        }
        exitOnTestThresholdValue = std::stof(commandLine.getOptionValue("exit-test-parameters", 1));
        if (exitOnTestThresholdValue < 0.0f || exitOnTestThresholdValue > 1.0f)
        {
            std::cout << "For --exit-test-parameters / -etp, the second paramter needs to be in [0,1]\n";
            exit(EXIT_FAILURE);
        }
    }

    FLIP::filename referenceFileName(commandLine.getOptionValue("reference"));
    std::vector<std::string>& testFileNames = commandLine.getOptionValues("test");
    bool bUseHDR = (referenceFileName.getExtension() == "exr");

    std::string destinationDirectory = ".";
    if (commandLine.optionSet("directory"))
    {
        destinationDirectory = commandLine.getOptionValue("directory");
        std::replace(destinationDirectory.begin(), destinationDirectory.end(), '\\', '/');      // Replace backslash with forwardslash.
        const bool bNoExposureMap = bUseHDR ? commandLine.optionSet("no-exposure-map") : true;
        const bool bSaveLDRImages = bUseHDR ? commandLine.optionSet("save-ldr-images") : false;
        const bool bSaveLDRFLIP   = bUseHDR ? commandLine.optionSet("save-ldrflip") : false;
        const bool willCreateOutput = (!commandLine.optionSet("no-error-map")) || (!bNoExposureMap) || bSaveLDRImages || bSaveLDRFLIP || commandLine.optionSet("histogram");

        if (!std::filesystem::exists(destinationDirectory) && willCreateOutput)     // Create directories if the parameters indicate that some files will be saved.
        {
            std::cout << "Creating new directory(s): <" << destinationDirectory << ">.\n";
            std::filesystem::create_directories(destinationDirectory);
        }
    }

    FLIP::image<FLIP::color3> magmaMap(FLIP::MapMagma, 256);

    if (commandLine.optionSet("pixels-per-degree"))
    {
        gFLIPOptions.PPD = std::stof(commandLine.getOptionValue("pixels-per-degree"));
    }
    else
    {
        if (commandLine.optionSet("viewing-conditions"))
        {
            gFLIPOptions.monitorDistance = std::stof(commandLine.getOptionValue("viewing-conditions", 0));
            gFLIPOptions.monitorWidth = std::stof(commandLine.getOptionValue("viewing-conditions", 1));
            gFLIPOptions.monitorResolutionX = std::stof(commandLine.getOptionValue("viewing-conditions", 2));
        }
        gFLIPOptions.PPD = calculatePPD(gFLIPOptions.monitorDistance, gFLIPOptions.monitorResolutionX, gFLIPOptions.monitorWidth);
    }

    FLIP::image<FLIP::color3> referenceImage(referenceFileName.toString());


    std::cout << "Invoking " << (bUseHDR ? "HDR" : "LDR") << "-FLIP\n";
    std::cout << "     Pixels per degree: " << int(std::round(gFLIPOptions.PPD)) << "\n" << (!bUseHDR ? "\n" : "");

    float startExposure = 0.0f, stopExposure = 0.0f;
    size_t numExposures = 0;
    float exposureStepSize = 0.0f;
    bool bStartexp = commandLine.optionSet("start-exposure");
    bool bStopexp = commandLine.optionSet("stop-exposure");

    //  Set HDR-FLIP parameters.
    std::string optionTonemapper = "aces";
    if (bUseHDR)
    {
        if (commandLine.optionSet("tone-mapper"))
        {
            optionTonemapper = commandLine.getOptionValue("tone-mapper");
            std::transform(optionTonemapper.begin(), optionTonemapper.end(), optionTonemapper.begin(), [](unsigned char c) { return std::tolower(c); });
            if (optionTonemapper != "aces" && optionTonemapper != "reinhard" && optionTonemapper != "hable")
            {
                std::cout << "\nError: unknown tonemapper, should be one of \"aces\", \"reinhard\", or \"hable\"\n";
                exit(-1);
            }
        }

        if (!bStartexp || !bStopexp)
        {
            referenceImage.computeExposures(optionTonemapper, startExposure, stopExposure);
        }
        if (bStartexp)
        {
            startExposure = std::stof(commandLine.getOptionValue("start-exposure"));
        }
        if (bStopexp)
        {
            stopExposure = std::stof(commandLine.getOptionValue("stop-exposure"));
        }

        if (startExposure > stopExposure)
        {
            std::cout << "Start exposure must be smaller than stop exposure!\n";
            exit(-1);
        }

        numExposures = (commandLine.optionSet("num-exposures") ? atoi(commandLine.getOptionValue("num-exposures").c_str()) : size_t(std::max(2.0f, ceil(stopExposure - startExposure))));
        exposureStepSize = (stopExposure - startExposure) / (numExposures - 1);

        std::cout << "     Assumed tone mapper: " << ((optionTonemapper == "aces") ? "ACES" : (optionTonemapper == "hable" ? "Hable" : "Reinhard")) << "\n";
        std::cout << "     Start exposure: " << FIXED_DECIMAL_DIGITS(startExposure, 4) << "\n";
        std::cout << "     Stop exposure: " << FIXED_DECIMAL_DIGITS(stopExposure, 4) << "\n";
        std::cout << "     Number of exposures: " << numExposures << "\n\n";
    }

    std::string basename = (commandLine.optionSet("basename") ? commandLine.getOptionValue("basename") : "");
    FLIP::filename flipFileName("tmp.png");
    FLIP::filename histogramFileName("tmp.py");
    FLIP::filename exposureFileName("tmp.png");

    uint32_t testFileCount = 0;
    uint32_t maxTestFileCount = uint32_t(commandLine.getOptionValues("test").size());
    FLIP::image<FLIP::color3> originalReferenceImage(referenceImage.getWidth(), referenceImage.getHeight());

    for (auto& testFileNameString : commandLine.getOptionValues("test"))
    {
        FLIP::filename testFileName = testFileNameString;

        if (basename != "" && commandLine.getOptionValues("test").size() == 1)
        {
            flipFileName.setName(basename);
            histogramFileName.setName(basename);
            exposureFileName.setName(basename + ".exposure_map");
        }
        else
        {
            flipFileName.setName(referenceFileName.getName() + "." + testFileName.getName() + "." + std::to_string(int(std::round(gFLIPOptions.PPD))) + "ppd");
            if (bUseHDR)
            {
                flipFileName.setName(flipFileName.getName() + ".hdr." + optionTonemapper + "." + f2s(startExposure) + "_to_" + f2s(stopExposure) + "." + std::to_string(numExposures));
            }
            else
            {
                flipFileName.setName(flipFileName.getName() + ".ldr");
            }
            histogramFileName.setName("weighted_histogram." + flipFileName.getName());
            exposureFileName.setName("exposure_map." + flipFileName.getName());
            flipFileName.setName("flip." + flipFileName.getName());
        }

        FLIP::image<FLIP::color3> testImage(testFileName.toString());

        FLIP::image<FLIP::color3> viridisMap(FLIP::MapViridis, 256);
        FLIP::image<float> errorMapFLIP(referenceImage.getWidth(), referenceImage.getHeight(), 0.0f);

        auto t0 = std::chrono::high_resolution_clock::now();
        auto t = t0;
        if (bUseHDR)
        {
            FLIP::image<FLIP::color3> rImage(referenceImage.getWidth(), referenceImage.getHeight());
            FLIP::image<FLIP::color3> tImage(referenceImage.getWidth(), referenceImage.getHeight());
            FLIP::image<float> tmpErrorMap(referenceImage.getWidth(), referenceImage.getHeight(), 0.0f);
            FLIP::image<float> prevTmpErrorMap(referenceImage.getWidth(), referenceImage.getHeight());

            FLIP::image<float> maxErrorExposureMap(referenceImage.getWidth(), referenceImage.getHeight());

            FLIP::filename rFileName("tmp.png");
            FLIP::filename tFileName("tmp.png");

            for (size_t i = 0; i < numExposures; i++)
            {
                float exposure = startExposure + i * exposureStepSize;

                std::stringstream ss;
                ss << std::setfill('0') << std::setw(3) << i;
                std::string expCount = ss.str();
                ss.str(std::string());
                ss << std::string(exposure < 0.0f ? "m" : "p") << std::to_string(std::abs(exposure));
                std::string expString = ss.str();

                rImage.copy(referenceImage);
                tImage.copy(testImage);

                rImage.expose(exposure);
                tImage.expose(exposure);

                rImage.toneMap(optionTonemapper);
                tImage.toneMap(optionTonemapper);

                rImage.clamp();
                tImage.clamp();

                rImage.LinearRGB2sRGB();
                tImage.LinearRGB2sRGB();

                if (commandLine.optionSet("save-ldr-images"))
                {
                    if (basename == "")
                    {
                        rFileName.setName(referenceFileName.getName() + "." + optionTonemapper + "." + expCount + "." + expString);
                        tFileName.setName(testFileName.getName() + "." + optionTonemapper + "." + expCount + "." + expString);
                    }
                    else
                    {
                        rFileName.setName(basename + ".reference." + "." + expCount);
                        tFileName.setName(basename + ".test." + "." + expCount);
                    }
                    rImage.pngSave(destinationDirectory + "/" + rFileName.toString());
                    tImage.pngSave(destinationDirectory + "/" + tFileName.toString());
                }

                tmpErrorMap.FLIP(rImage, tImage, gFLIPOptions.PPD);

                errorMapFLIP.setMaxExposure(tmpErrorMap, maxErrorExposureMap, float(i) / (numExposures - 1));

                if (commandLine.optionSet("save-ldrflip"))
                {
                    FLIP::image<FLIP::color3> pngResult(referenceImage.getWidth(), referenceImage.getHeight());
                    if (!commandLine.optionSet("no-magma"))
                    {
                        pngResult.colorMap(tmpErrorMap, magmaMap);
                    }
                    else
                    {
                        pngResult.copyFloat2Color3(tmpErrorMap);
                    }

                    if (basename == "")
                    {
                        pngResult.pngSave(destinationDirectory + "/" + "flip." + referenceFileName.getName() + "." + testFileName.getName() + "." + std::to_string(int(std::round(gFLIPOptions.PPD))) + "ppd.ldr." + optionTonemapper + "." + expCount + "." + expString + ".png");
                    }
                    else
                    {
                        pngResult.pngSave(destinationDirectory + "/" + basename + "." + expCount + ".png");

                    }

                    pngResult.pngSave(destinationDirectory + "/" + flipFileName.toString());
                }

            }
            t = std::chrono::high_resolution_clock::now();

            if (!commandLine.optionSet("no-exposure-map"))
            {
                FLIP::image<FLIP::color3> pngMaxErrorExposureMap(referenceImage.getWidth(), referenceImage.getHeight());
                pngMaxErrorExposureMap.colorMap(maxErrorExposureMap, viridisMap);
                pngMaxErrorExposureMap.pngSave(destinationDirectory + "/" + exposureFileName.toString());
            }
        }
        else
        {
            if (maxTestFileCount > 1 && testFileCount == 0)     // Are we testing more than one image, and are we testing the first image?
            {
                originalReferenceImage.copy(referenceImage);    // If so, then store the referenceImage in originalReferenceImage, since FLIP() destroys referenceImage, i.e., in the errorMapFLIP.FLIP() just below.
            }
            errorMapFLIP.FLIP(referenceImage, testImage, gFLIPOptions.PPD);
            t = std::chrono::high_resolution_clock::now();
            if (maxTestFileCount > 1)
            {
                referenceImage.copy(originalReferenceImage);    // Restore the original referenceImage for the second test, and all images after.
            }
        }

        if (!commandLine.optionSet("no-error-map"))
        {
            FLIP::image<FLIP::color3> pngResult(referenceImage.getWidth(), referenceImage.getHeight());
            pngResult.copyFloat2Color3(errorMapFLIP);
            if (!commandLine.optionSet("no-magma"))
            {
                pngResult.colorMap(errorMapFLIP, magmaMap);
            }
            pngResult.pngSave(destinationDirectory + "/" + flipFileName.toString());
        }

        pooling<float> pooledValues;
        for (int y = 0; y < errorMapFLIP.getHeight(); y++)
        {
            for (int x = 0; x < errorMapFLIP.getWidth(); x++)
            {
                pooledValues.update(x, y, errorMapFLIP.get(x, y));
            }
        }

        if (commandLine.optionSet("histogram"))
        {
            bool optionLog = commandLine.optionSet("log");
            bool optionExcludeValues = commandLine.optionSet("exclude-pooled-values");
            float yMax = (commandLine.optionSet("y-max") ? std::stof(commandLine.getOptionValue("y-max")) : 0.0f);
            pooledValues.save(destinationDirectory + "/" + histogramFileName.toString(), errorMapFLIP.getWidth(), errorMapFLIP.getHeight(), optionLog, referenceFileName.toString(), testFileName.toString(), !optionExcludeValues, yMax);
        }

        if (commandLine.optionSet("csv"))
        {
            FLIP::filename csvFileName(commandLine.getOptionValue("csv"));

            std::fstream csv;
            csv.open(csvFileName.toString(), std::ios::app);
            if (csv.is_open())
            {
                csv.seekp(0, std::ios_base::end);

                if (csv.tellp() <= 0)
                    csv << "\"Reference\",\"Test\",\"Mean\",\"Weighted median\",\"1st weighted quartile\",\"3rd weighted quartile\",\"Min\",\"Max\",\"Evaluation time\"\n";

                csv << "\"" << referenceFileName.toString() << "\",";
                csv << "\"" << testFileName.toString() << "\",";
                csv << "\"" << FIXED_DECIMAL_DIGITS(pooledValues.getMean(), 6) << "\",";
                csv << "\"" << FIXED_DECIMAL_DIGITS(pooledValues.getPercentile(0.5f, true), 6) << "\",";
                csv << "\"" << FIXED_DECIMAL_DIGITS(pooledValues.getPercentile(0.25f, true), 6) << "\",";
                csv << "\"" << FIXED_DECIMAL_DIGITS(pooledValues.getPercentile(0.75f, true), 6) << "\",";
                csv << "\"" << FIXED_DECIMAL_DIGITS(pooledValues.getMinValue(), 6) << "\",";
                csv << "\"" << FIXED_DECIMAL_DIGITS(pooledValues.getMaxValue(), 6) << "\",";
                csv << "\"" << FIXED_DECIMAL_DIGITS(std::chrono::duration_cast<std::chrono::microseconds>(t - t0).count() / 1000000.0f, 4) << "\"\n";

                csv.close();
            }
            else
            {
                std::cout << "\nError: Could not write csv file " << csvFileName.toString() << "\n";
            }
        }

        std::cout << FLIPString << " between reference image <" << referenceFileName.toString() << "> and test image <" << testFileName.toString() << ">\n";
        std::cout << "     Mean: " << FIXED_DECIMAL_DIGITS(pooledValues.getMean(), 6) << "\n";
        std::cout << "     Weighted median: " << FIXED_DECIMAL_DIGITS(pooledValues.getPercentile(0.5f, true), 6) << "\n";
        std::cout << "     1st weighted quartile: " << FIXED_DECIMAL_DIGITS(pooledValues.getPercentile(0.25f, true), 6) << "\n";
        std::cout << "     3rd weighted quartile: " << FIXED_DECIMAL_DIGITS(pooledValues.getPercentile(0.75f, true), 6) << "\n";
        std::cout << "     Min: " << FIXED_DECIMAL_DIGITS(pooledValues.getMinValue(), 6) << "\n";
        std::cout << "     Max: " << FIXED_DECIMAL_DIGITS(pooledValues.getMaxValue(), 6) << "\n";
        std::cout << "     Evaluation time: " << FIXED_DECIMAL_DIGITS(std::chrono::duration_cast<std::chrono::microseconds>(t - t0).count() / 1000000.0f, 4) << " seconds\n";
        std::cout << (((++testFileCount) < commandLine.getOptionValues("test").size()) ? "\n" : "");

        if (commandLine.optionSet("exit-on-test"))
        {
            float testQuantity;
            if (exitOnTestQuantity == "mean")
            {
                testQuantity = pooledValues.getMean();
            }
            else if (exitOnTestQuantity == "weighted-median")
            {
                testQuantity = pooledValues.getPercentile(0.5f, true);
            }
            else if (exitOnTestQuantity == "max")
            {
                testQuantity = pooledValues.getMaxValue();
            }

            if (testQuantity > exitOnTestThresholdValue)
            {
                std::cout << "Exiting with failure code because the " << exitOnTestQuantity << " of the FLIP error map is " << FIXED_DECIMAL_DIGITS(testQuantity, 6) << ", while the threshold for success is " << FIXED_DECIMAL_DIGITS(exitOnTestThresholdValue, 6) << ".\n";
                exit(EXIT_FAILURE);     // From stdlib.h: equal to 1.
            }
        }
    }
    exit(EXIT_SUCCESS);                 // From stdlib.h: equal to 0.
}
