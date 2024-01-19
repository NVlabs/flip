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

#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <filesystem>

#if defined(_WIN32) && !defined(NOMINMAX)
#define NOMINMAX
#endif

#define FIXED_DECIMAL_DIGITS(x, d) std::fixed << std::setprecision(d) << (x)

#include "FLIP.h"

#include "imagehelpers.h"
#include "commandline.h"
#include "filename.h"
#include "pooling.h"

inline std::string f2s(float value, size_t decimals = 4)
{
    std::stringstream ss;
    ss << std::string(value < 0.0f ? "m" : "p") << FIXED_DECIMAL_DIGITS(std::abs(value), decimals);
    return ss.str();
}

// Here follows a set of helps functions for differet setups in order to avoid clutter in main().

static void setupDestinationDirectory(const bool useHDR, const commandline& commandLine, std::string& destinationDirectory)
{
    if (commandLine.optionSet("directory"))
    {
        destinationDirectory = commandLine.getOptionValue("directory");
        std::replace(destinationDirectory.begin(), destinationDirectory.end(), '\\', '/');      // Replace backslash with forwardslash.
        const bool bNoExposureMap = useHDR ? commandLine.optionSet("no-exposure-map") : true;
        const bool bSaveLDRImages = useHDR ? commandLine.optionSet("save-ldr-images") : false;
        const bool bSaveLDRFLIP = useHDR ? commandLine.optionSet("save-ldrflip") : false;
        const bool willCreateOutput = (!commandLine.optionSet("no-error-map")) || (!bNoExposureMap) || bSaveLDRImages || bSaveLDRFLIP || commandLine.optionSet("histogram");

        if (!std::filesystem::exists(destinationDirectory) && willCreateOutput)     // Create directories if the parameters indicate that some files will be saved.
        {
            std::cout << "Creating new directory(s): <" << destinationDirectory << ">.\n";
            std::filesystem::create_directories(destinationDirectory);
        }
    }
}

static void setupPixelsPerDegree(const commandline& commandLine, FLIP::Parameters& parameters)
{
    // The default value in parameters.PPD is computed as FLIP::calculatePPD(0.7f, 3840.0f, 0.7f); in FLIP.h.
    if (commandLine.optionSet("pixels-per-degree"))
    {
        parameters.PPD = std::stof(commandLine.getOptionValue("pixels-per-degree"));
    }
    else if (commandLine.optionSet("viewing-conditions"))
    {
        const float monitorDistance = std::stof(commandLine.getOptionValue("viewing-conditions", 0));
        const float monitorWidth = std::stof(commandLine.getOptionValue("viewing-conditions", 1));
        const float monitorResolutionX = std::stof(commandLine.getOptionValue("viewing-conditions", 2));
        parameters.PPD = FLIP::calculatePPD(monitorDistance, monitorResolutionX, monitorWidth);
    }
}

static void getExposureParameters(const bool useHDR, const commandline& commandLine, FLIP::Parameters& parameters, bool& returnLDRFLIPImages, bool& returnLDRImages)
{
    if (useHDR)
    {
        if (commandLine.optionSet("tone-mapper"))   // The default in FLIP::Parameters.tonemapper is "aces".
        {
            std::string tonemapper = commandLine.getOptionValue("tone-mapper");
            std::transform(tonemapper.begin(), tonemapper.end(), tonemapper.begin(), [](unsigned char c) { return std::tolower(c); });
            if (tonemapper != "aces" && tonemapper != "reinhard" && tonemapper != "hable")
            {
                std::cout << "\nError: unknown tonemapper, should be one of \"aces\", \"reinhard\", or \"hable\"\n";
                exit(-1);
            }
            parameters.tonemapper = tonemapper;
        }
        if (commandLine.optionSet("start-exposure"))
        {
            parameters.startExposure = std::stof(commandLine.getOptionValue("start-exposure"));
        }
        if (commandLine.optionSet("stop-exposure"))
        {
            parameters.stopExposure = std::stof(commandLine.getOptionValue("stop-exposure"));
        }
        if (commandLine.optionSet("num-exposures"))
        {
            parameters.numExposures = atoi(commandLine.getOptionValue("num-exposures").c_str());
        }
        returnLDRFLIPImages = commandLine.optionSet("save-ldrflip");
        returnLDRImages = commandLine.optionSet("save-ldr-images");
    }
}

static void saveErrorAndExposureMaps(const bool useHDR, commandline& commandLine, const FLIP::Parameters& parameters, const std::string basename,
    FLIP::image<float>& errorMapFLIP, FLIP::image<float>& maxErrorExposureMap, const std::string& destinationDirectory,
    FLIP::filename& referenceFileName, FLIP::filename& testFileName, FLIP::filename& histogramFileName,
    FLIP::filename& flipFileName, FLIP::filename& exposureFileName)
{
    if (basename != "" && commandLine.getOptionValues("test").size() == 1)
    {
        flipFileName.setName(basename);
        histogramFileName.setName(basename);
        exposureFileName.setName(basename + ".exposure_map");
    }
    else
    {
        flipFileName.setName(referenceFileName.getName() + "." + testFileName.getName() + "." + std::to_string(int(std::round(parameters.PPD))) + "ppd");
        if (!useHDR)
        {
            flipFileName.setName(flipFileName.getName() + ".ldr");  // Note that the HDR filename is not complete until after FLIP has been computed, since FLIP may update the exposure parameters.
        }
        histogramFileName.setName("weighted_histogram." + flipFileName.getName());
    }


    if(useHDR) // Updating the flipFileName here, since the computation of FLIP may have updated the exposure parameters.
    {
        std::cout << "     Assumed tone mapper: " << ((parameters.tonemapper == "aces") ? "ACES" : (parameters.tonemapper == "hable" ? "Hable" : "Reinhard")) << "\n";
        std::cout << "     Start exposure: " << FIXED_DECIMAL_DIGITS(parameters.startExposure, 4) << "\n";
        std::cout << "     Stop exposure: " << FIXED_DECIMAL_DIGITS(parameters.stopExposure, 4) << "\n";
        std::cout << "     Number of exposures: " << parameters.numExposures << "\n\n";

        flipFileName.setName(flipFileName.getName() + ".hdr." + parameters.tonemapper + "." + f2s(parameters.startExposure) + "_to_" + f2s(parameters.stopExposure) + "." + std::to_string(parameters.numExposures));
        exposureFileName.setName("exposure_map." + flipFileName.getName());
    }

    if (basename != "" && commandLine.getOptionValues("test").size() == 1)
    {
        flipFileName.setName(basename);
        exposureFileName.setName(basename + ".exposure_map");
    }
    else
    {
        flipFileName.setName("flip." + flipFileName.getName());
    }

    if (!commandLine.optionSet("no-error-map"))
    {
        FLIP::image<FLIP::color3> pngResult(errorMapFLIP.getWidth(), errorMapFLIP.getHeight());
        if (!commandLine.optionSet("no-magma"))
        {
            pngResult.colorMap(errorMapFLIP, FLIP::magmaMap);
        }
        else
        {
            pngResult.copyFloat2Color3(errorMapFLIP);
        }
        ImageHelpers::pngSave(destinationDirectory + "/" + flipFileName.toString(), pngResult);
    }

    if (useHDR)
    {
        if (!commandLine.optionSet("no-exposure-map"))
        {
            FLIP::image<FLIP::color3> pngMaxErrorExposureMap(maxErrorExposureMap.getWidth(), maxErrorExposureMap.getHeight());
            pngMaxErrorExposureMap.colorMap(maxErrorExposureMap, FLIP::viridisMap);
            ImageHelpers::pngSave(destinationDirectory + "/" + exposureFileName.toString(), pngMaxErrorExposureMap);
        }
    }
}

static void setExposureStrings(const int exposureCount, const float exposure, std::string& expCount, std::string& expString)
{
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(3) << exposureCount;
    expCount = ss.str();
    ss.str(std::string());
    ss << std::string(exposure < 0.0f ? "m" : "p") << std::to_string(std::abs(exposure));
    expString = ss.str();
}

// Optionally store the intermediate LDR images and LDR-FLIP error maps produced during the evaluation of HDR-FLIP.
static void saveHDROutputLDRImages(const commandline& commandLine, const FLIP::Parameters& parameters, const std::string& basename, const FLIP::filename& flipFileName,
    const FLIP::filename& referenceFileName, const FLIP::filename& testFileName, const std::string& destinationDirectory,
    std::vector<FLIP::image<float>*> hdrOutputFlipLDRImages, std::vector<FLIP::image<FLIP::color3>*> hdrOutputLDRImages)
{
    if (hdrOutputLDRImages.size() > 0)
    {
        FLIP::filename rFileName("tmp.png");
        FLIP::filename tFileName("tmp.png");
        if (hdrOutputLDRImages.size() != parameters.numExposures * 2)
        {
            std::cout << "FLIP tool error: the number of LDR images from HDR-FLIP is not the expected number.\nExiting.\n";
            exit(EXIT_FAILURE);
        }

        const float exposureStepSize = (parameters.stopExposure - parameters.startExposure) / (parameters.numExposures - 1);
        for (int i = 0; i < parameters.numExposures; i++)
        {
            std::string expCount, expString;
            setExposureStrings(i, parameters.startExposure + i * exposureStepSize, expCount, expString);

            if (basename == "")
            {
                rFileName.setName(referenceFileName.getName() + "." + parameters.tonemapper + "." + expCount + "." + expString);
                tFileName.setName(testFileName.getName() + "." + parameters.tonemapper + "." + expCount + "." + expString);
            }
            else
            {
                rFileName.setName(basename + ".reference." + "." + expCount);
                tFileName.setName(basename + ".test." + "." + expCount);
            }
            FLIP::image<FLIP::color3>* rImage = hdrOutputLDRImages[0];
            FLIP::image<FLIP::color3>* tImage = hdrOutputLDRImages[1];
            hdrOutputLDRImages.erase(hdrOutputLDRImages.begin());
            hdrOutputLDRImages.erase(hdrOutputLDRImages.begin());
            rImage->LinearRGB2sRGB();
            tImage->LinearRGB2sRGB();
            ImageHelpers::pngSave(destinationDirectory + "/" + rFileName.toString(), *rImage);
            ImageHelpers::pngSave(destinationDirectory + "/" + tFileName.toString(), *tImage);
            delete rImage;
            delete tImage;
        }
    }
    if (hdrOutputFlipLDRImages.size() > 0)
    {
        if (hdrOutputFlipLDRImages.size() != parameters.numExposures)
        {
            std::cout << "FLIP tool error: the number of FLIP LDR images from HDR-FLIP is not the expected number.\nExiting.\n";
            exit(EXIT_FAILURE);
        }

        const float exposureStepSize = (parameters.stopExposure - parameters.startExposure) / (parameters.numExposures - 1);
        for (int i = 0; i < parameters.numExposures; i++)
        {
            std::string expCount, expString;
            setExposureStrings(i, parameters.startExposure + i * exposureStepSize, expCount, expString);

            FLIP::image<float>* flipImage = hdrOutputFlipLDRImages[0];
            hdrOutputFlipLDRImages.erase(hdrOutputFlipLDRImages.begin());

            FLIP::image<FLIP::color3> pngResult(flipImage->getWidth(), flipImage->getHeight());

            if (!commandLine.optionSet("no-magma"))
            {
                pngResult.colorMap(*flipImage, FLIP::magmaMap);
            }
            else
            {
                pngResult.copyFloat2Color3(*flipImage);
            }
            if (basename == "")
            {
                ImageHelpers::pngSave(destinationDirectory + "/" + "flip." + referenceFileName.getName() + "." + testFileName.getName() + "." + std::to_string(int(std::round(parameters.PPD))) + "ppd.ldr." + parameters.tonemapper + "." + expCount + "." + expString + ".png", pngResult);
            }
            else
            {
                ImageHelpers::pngSave(destinationDirectory + "/" + basename + "." + expCount + ".png", pngResult);
            }
            ImageHelpers::pngSave(destinationDirectory + "/" + flipFileName.toString(), pngResult);
            delete flipImage;
        }
    }
}

static void gatherStatisticsAndSaveOutput(commandline& commandLine, FLIP::image<float>& errorMapFLIP, const std::string& destinationDirectory,
    const FLIP::filename& referenceFileName, const FLIP::filename& testFileName, const FLIP::filename& histogramFileName,
    const std::string& FLIPString, const float time, const uint32_t testFileCount)
{
    FLIP::pooling<float> pooledValues;
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
            csv << "\"" << FIXED_DECIMAL_DIGITS(time, 4) << "\"\n";

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
    std::cout << "     Evaluation time: " << FIXED_DECIMAL_DIGITS(time, 4) << " seconds\n";
    std::cout << ((testFileCount < commandLine.getOptionValues("test").size()) ? "\n" : "");

    if (commandLine.optionSet("exit-on-test"))
    {
        std::string exitOnTestQuantity = "mean";
        float exitOnTestThresholdValue = 0.05f;
        if (commandLine.optionSet("exit-test-parameters"))
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
        else
        {
            std::cout << "Exiting with failure code because exit-on-text-quantity was " << exitOnTestQuantity << ", and expects to be mean, weighted-median, or max.\n";
            exit(EXIT_FAILURE);     // From stdlib.h: equal to 1.
        }

        if (testQuantity > exitOnTestThresholdValue)
        {
            std::cout << "Exiting with failure code because the " << exitOnTestQuantity << " of the FLIP error map is " << FIXED_DECIMAL_DIGITS(testQuantity, 6) << ", while the threshold for success is " << FIXED_DECIMAL_DIGITS(exitOnTestThresholdValue, 6) << ".\n";
            exit(EXIT_FAILURE);     // From stdlib.h: equal to 1.
        }
    }
}

int main(int argc, char** argv)
{
    std::string FLIPString = "FLIP";
    int MajorVersion = 1;
    int MinorVersion = 3;

    const commandline_options allowedCommandLineOptions =
    {
        "Compute FLIP between reference.<png|exr> and test.<png|exr>.\n"
        "Reference and test(s) must have same resolution and format.\n"
        "If pngs are entered, LDR-FLIP will be evaluated. If exrs are entered, HDR-FLIP will be evaluated.\n"
        "For HDR-FLIP, the reference is used to automatically calculate start exposure and/or stop exposure and/or number of exposures, if they are not entered.",
        {
        { "help", "h", 0, false, "", "show this help message and exit" },
        { "reference", "r", 1, true, "REFERENCE", "Relative or absolute path (including file name and extension) for reference image" },
        { "test", "t", -1, true, "TEST", "Relative or absolute path(s) (including file name and extension) for test image(s)" },
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

    FLIP::Parameters parameters;
    FLIP::filename referenceFileName(commandLine.getOptionValue("reference"));
    bool bUseHDR = (referenceFileName.getExtension() == "exr");
    std::string destinationDirectory = ".";
    std::string basename = (commandLine.optionSet("basename") ? commandLine.getOptionValue("basename") : "");
    FLIP::filename flipFileName("tmp.png"); // Dummy file name, but it keeps the file extension (.png).
    FLIP::filename histogramFileName("tmp.py");
    FLIP::filename exposureFileName("tmp.png");
    FLIP::filename testFileName;
    bool returnLDRImages = false;                                   // Can only happen for HDR.
    bool returnLDRFLIPImages = false;                               // Can only happen for HDR.

    setupDestinationDirectory(bUseHDR, commandLine, destinationDirectory);
    setupPixelsPerDegree(commandLine, parameters);
    getExposureParameters(bUseHDR, commandLine, parameters, returnLDRFLIPImages, returnLDRImages);

    std::cout << "Invoking " << (bUseHDR ? "HDR" : "LDR") << "-FLIP\n";
    std::cout << "     Pixels per degree: " << int(std::round(parameters.PPD)) << "\n" << (!bUseHDR ? "\n" : "");

    FLIP::image<FLIP::color3> referenceImage;
    ImageHelpers::load(referenceImage, referenceFileName.toString());   // Load reference image.
    if (!bUseHDR)
    {
        referenceImage.sRGB2LinearRGB();
    }

    uint32_t testFileCount = 0;
    // Loop over the test images files to be FLIP:ed against the reference image.
    for (auto& testFileNameString : commandLine.getOptionValues("test"))
    {
        std::vector<FLIP::image<float>*> hdrOutputFlipLDRImages;
        std::vector<FLIP::image<FLIP::color3>*> hdrOutputLDRImages;
        testFileName = testFileNameString;

        FLIP::image<FLIP::color3> testImage;
        ImageHelpers::load(testImage, testFileName.toString());     // Load test image.
        if (!bUseHDR)
        {
            testImage.sRGB2LinearRGB();
        }

        FLIP::image<float> errorMapFLIP(referenceImage.getWidth(), referenceImage.getHeight(), 0.0f);
        FLIP::image<float> maxErrorExposureMap(referenceImage.getWidth(), referenceImage.getHeight());

        auto t0 = std::chrono::high_resolution_clock::now();
        FLIP::evaluate(bUseHDR, parameters, referenceImage, testImage, errorMapFLIP, maxErrorExposureMap, returnLDRFLIPImages, hdrOutputFlipLDRImages, returnLDRImages, hdrOutputLDRImages);
        float time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t0).count() / 1000000.0f;

        saveErrorAndExposureMaps(bUseHDR, commandLine, parameters, basename, errorMapFLIP, maxErrorExposureMap, destinationDirectory, referenceFileName, testFileName, histogramFileName, flipFileName, exposureFileName);
        saveHDROutputLDRImages(commandLine, parameters, basename, flipFileName, referenceFileName, testFileName, destinationDirectory, hdrOutputFlipLDRImages, hdrOutputLDRImages);
        gatherStatisticsAndSaveOutput(commandLine, errorMapFLIP, destinationDirectory, referenceFileName, testFileName, histogramFileName, FLIPString, time, ++testFileCount);
    }
    exit(EXIT_SUCCESS);                 // From stdlib.h: equal to 0.
}
