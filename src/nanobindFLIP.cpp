/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES
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

 // Code by Pontus Ebelin (formerly Andersson) and Tomas Akenine-Moller.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include "cpp/FLIP.h"
#include "cpp/tool/FLIPToolHelpers.h"

namespace nb = nanobind;

using Array3D = nb::ndarray<float, nb::numpy, nb::c_contig, nb::shape<-1, -1, -1>, nb::device::cpu>;

Array3D createThreeChannelImage(float*& data, const size_t imageHeight, const size_t imageWidth, const size_t numChannels)
{
    // Create an ndarray from the raw data and an owner to make sure the data is deallocated when no longer used.
    nb::capsule owner(data, [](void* p) noexcept {
        delete[](float*) p;
    });

    // Allocate 3D array
    Array3D image(data, { imageHeight, imageWidth, numChannels }, owner);

    return image;
}

// Load the .exr, .png, .bmp, or .tga file with path fileName.
Array3D load(std::string fileName)
{
    FLIP::image<FLIP::color3> image;
    bool imageOk = ImageHelpers::load(image, fileName);
    if (!imageOk)
    {
        std::cout << "Error: could not read image file <" << fileName << ">. Note that FLIP only loads png, bmp, tga, and exr images. Exiting.\n";
        exit(EXIT_FAILURE);
    }

    const size_t imageWidth = image.getWidth();
    const size_t imageHeight = image.getHeight();
    const size_t channels = 3;

    // Allocate memory for the ndarray.
    float* data = new float[imageWidth * imageHeight * channels * sizeof(float)];

    // Copy the image data to the allocated memory.
    memcpy(data, image.getHostData(), imageWidth * imageHeight * sizeof(float) * channels);

    // Create an ndarray from the raw data and an owner to make sure the data is deallocated when no longer used.
    Array3D numpyImage = createThreeChannelImage(data, imageHeight, imageWidth, channels);

    return numpyImage;
}

// Convert linear RGB image to sRGB.
void sRGBToLinearRGB(float* image, const size_t imageWidth, const size_t imageHeight)
{
#pragma omp parallel for
    for (int y = 0; y < imageHeight; y++)
    {
        for (int x = 0; x < imageWidth; x++)
        {
            size_t idx = (y * imageWidth + x) * 3;
            image[idx] = FLIP::color3::sRGBToLinearRGB(image[idx]);
            image[idx + 1] = FLIP::color3::sRGBToLinearRGB(image[idx + 1]);
            image[idx + 2] = FLIP::color3::sRGBToLinearRGB(image[idx + 2]);
        }
    }
}

// Set parameters for evaluate function based on input settings.
FLIP::Parameters setParameters(nb::dict inputParameters)
{
    FLIP::Parameters parameters;

    for (auto item : inputParameters)
    {
        std::string key = nb::cast<std::string>(item.first);
        std::string errorMessage = "Unrecognized parameter dictionary key or invalid value type. Available ones are \"ppd\" (float), \"startExposure\" (float), \"stopExposure\" (float), \"numExposures\" (int), and \"tonemapper\" (string).";
        if (key == "ppd")
        {
            parameters.PPD = nb::cast<float>(item.second);
        }
        else if (key == "vc")
        {
            auto vc = nb::cast<nb::list>(item.second);
            float distanceToDisplay = nb::cast<float>(vc[0]);
            float displayWidthPixels = nb::cast<float>(vc[1]);
            float displayWidthMeters = nb::cast<float>(vc[2]);
            parameters.PPD = FLIP::calculatePPD(distanceToDisplay, displayWidthPixels, displayWidthMeters);
        }
        else if (key == "startExposure")
        {
            parameters.startExposure = nb::cast<float>(item.second);
        }
        else if (key == "stopExposure")
        {
            parameters.stopExposure = nb::cast<float>(item.second);
        }
        else if (key == "numExposures")
        {
            parameters.numExposures = nb::cast<int>(item.second);
        }
        else if (key == "tonemapper")
        {
            parameters.tonemapper = nb::cast<std::string>(item.second);
        }
        else
        {
            std::cout << errorMessage << std::endl;
        }
    }

    return parameters;
}

// Update parameter dictionary that is returned to the Python side.
void updateInputParameters(const FLIP::Parameters& parameters, nb::dict& inputParameters)
{
    inputParameters["ppd"] = parameters.PPD;
    inputParameters["startExposure"] = parameters.startExposure;
    inputParameters["stopExposure"] = parameters.stopExposure;
    inputParameters["numExposures"] = parameters.numExposures;
    inputParameters["tonemapper"] = parameters.tonemapper;
}

/** A simplified function for evaluating (the image metric called) FLIP between a reference image and a test image. Corresponds to the fourth evaluate() option in FLIP.h.
*
* @param[in] referenceInput Reference input image. For LDR, the content should be in [0,1]. The image is expected to have 3 floats per pixel.
* @param[in] testInput Test input image. For LDR, the content should be in [0,1]. The image is expected to have 3 floats per pixel.
* @param[in] useHDR Set to true if the input images are to be considered containing HDR content, i.e., not necessarily in [0,1].
* @param[in] inputsRGB Set to true if the input images are given in the sRGB color space.
* @param[in] applyMagma A boolean indicating whether the output should have the Magma map applied to it before the image is returned.
* @param[in] computeMeanFLIPError A boolean indicating whether the mean FLIP error should be computed. If false, the returned mean error is -1.
* @param[in,out] inputParameters Contains parameters (e.g., PPD, exposure settings, etc). If the exposures have not been set by the user, then those will be computed (and returned).
* @return tuple containing FLIP error map (in Magma if applyMagma is true), the mean FLIP error (computed if computeMeanFLIPError is true, else -1), and dictionary of parameters.
*/
nb::tuple evaluate(const Array3D referenceInput, const Array3D testInput, const bool useHDR, const bool inputsRGB = true, const bool applyMagma = true, const bool computeMeanFLIPError = true, nb::dict inputParameters = {})
{   
    size_t r_ndim = referenceInput.ndim();
    size_t t_ndim = testInput.ndim();

    // Check number of dimensions and resolution.
    if (r_ndim != 3 || t_ndim != 3)
    {
        std::stringstream message;
        message << "Number of dimensions must be three. The reference image has " << r_ndim << " dimensions, while the test image has "<< t_ndim << " dimensions.";
        throw std::runtime_error(message.str());  
    }
    
    if (referenceInput.shape(0) != testInput.shape(0) || referenceInput.shape(1) != testInput.shape(1))
    {
        std::stringstream message;
        message << "Reference and test image resolutions differ.\nReference image resolution: " << referenceInput.shape(0) << "x" << referenceInput.shape(1) << "\nTest image resolution: "<< testInput.shape(0) << "x" << testInput.shape(0);
        throw std::runtime_error(message.str()); 
    }

    // Image size.
    const size_t imageHeight = referenceInput.shape(0), imageWidth = referenceInput.shape(1);
    const size_t nChannelsOut = applyMagma ? 3 : 1;

    // FLIP
    float* flip = nullptr; // Allocated in FLIP::evaluate().

    // Arrays for reference and test.
    float* referenceImage = new float[imageHeight * imageWidth * 3 * sizeof(float)];
    float* testImage = new float[imageHeight * imageWidth * 3 * sizeof(float)];
    memcpy(referenceImage, referenceInput.data(), imageHeight * imageWidth * sizeof(float) * 3);
    memcpy(testImage, testInput.data(), imageHeight * imageWidth * sizeof(float) * 3);

    // Transform to linear RGB if desired.
    if (inputsRGB)
    {
        sRGBToLinearRGB(referenceImage, imageWidth, imageHeight);
        sRGBToLinearRGB(testImage, imageWidth, imageHeight);
    }

    // Run FLIP.
    FLIP::Parameters parameters = setParameters(inputParameters);
    float meanError = -1;
    FLIP::evaluate(referenceImage, testImage, int(imageWidth), int(imageHeight), useHDR, parameters, applyMagma, computeMeanFLIPError, meanError, &flip);
    
    nb::dict returnParams = {};
    updateInputParameters(parameters, returnParams);

    Array3D flipNumpy = createThreeChannelImage(flip, imageHeight, imageWidth, nChannelsOut);
    
    delete [] referenceImage;
    delete [] testImage;
    
    return nb::make_tuple(flipNumpy, meanError, returnParams);
}

// Create command line, based on the Python command line string, for the FLIP tool to parse.
commandline generateCommandLine(const nb::list argvPy)
{
    size_t argc = argvPy.size();
    char** argv = new char* [argc];

    int counter = 0;
    for (auto item : argvPy)
    {
        const std::string it = nb::steal<nb::str>(item).c_str();
        argv[counter] = strdup(it.c_str());
        counter++;
    }

    commandline cmd = commandline(int(argc), argv, getAllowedCommandLineOptions(false));

    for (int i = 0; i < counter; i++)
    {
        delete [] argv[i];
    }
    delete [] argv;

    return cmd;
}

// Run the FLIP tool based on Python command line string.
int execute(const nb::list argvPy)
{
    commandline commandLine = generateCommandLine(argvPy);
    FLIPTool::execute(commandLine);

    return EXIT_SUCCESS;
}

// Setup the pybind11 module.
NB_MODULE(nbflip, handle)
{
    handle.doc() = "Load images (load), evaluate FLIP (evaluate), or run the full FLIP tool (execute).";
    handle.def("load", &load);
    handle.def("evaluate", &evaluate);
    handle.def("execute", &execute);
}