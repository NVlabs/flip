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

 // Code by Pontus Ebelin (formerly Andersson) and Tomas Akenine-Moller.

#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "../../cpp/FLIP.h"
#include "../../cpp/tool/FLIPToolHelpers.h"

namespace py = pybind11;

// Load the .exr, .png, .bmp, or .tga file with path fileName.
py::array_t<float> load(std::string fileName)
{
    FLIP::image<FLIP::color3> image;
    ImageHelpers::load(image, fileName);

    const int imageWidth = image.getWidth();
    const int imageHeight = image.getHeight();

    py::array_t<float> numpyImage = py::array_t<float>({ imageHeight, imageWidth, 3 });;

    py::buffer_info numpyImage_buf = numpyImage.request();
    float* ptr_numpyImage = static_cast<float*>(numpyImage_buf.ptr);

    memcpy(ptr_numpyImage, image.getHostData(), size_t(image.getWidth()) * image.getHeight() * sizeof(float) * 3);

    return numpyImage;
}

// Convert linear RGB channel value to sRGB.
float sRGBToLinearRGB(float sC)
{
    if (sC <= 0.04045f)
    {
        return sC / 12.92f;
    }
    return powf((sC + 0.055f) / 1.055f, 2.4f);
}

// Convert linear RGB image to sRGB.
void sRGBToLinearRGB(float* image, const int imageWidth, const int imageHeight)
{
#pragma omp parallel for
    for (int y = 0; y < imageHeight; y++)
    {
        for (int x = 0; x < imageWidth; x++)
        {
            int idx = (y * imageWidth + x) * 3;
            image[idx] = sRGBToLinearRGB(image[idx]);
            image[idx + 1] = sRGBToLinearRGB(image[idx + 1]);
            image[idx + 2] = sRGBToLinearRGB(image[idx + 2]);
        }
    }
}

// Set parameters for evaluate function based on input settings.
FLIP::Parameters setParameters(py::dict inputParameters)
{
    FLIP::Parameters parameters;

    for (auto item : inputParameters)
    {
        std::string key = py::cast<std::string>(item.first);
        std::string errorMessage = "Unrecognized parameter dictionary key or invalid value type. Available ones are \"ppd\" (float), \"startExposure\" (float), \"stopExposure\" (float), \"numExposures\" (int), and \"tonemapper\" (string).";
        if (key == "ppd")
        {
            parameters.PPD = py::cast<float>(item.second);
        }
        else if (key == "vc")
        {
            auto vc = py::cast<py::list>(item.second);
            float distanceToDisplay = py::cast<float>(vc[0]);
            float displayWidthPixels = py::cast<float>(vc[1]);
            float displayWidthMeters = py::cast<float>(vc[2]);
            parameters.PPD = FLIP::calculatePPD(distanceToDisplay, displayWidthPixels, displayWidthPixels);
        }
        else if (key == "startExposure")
        {
            parameters.startExposure = py::cast<float>(item.second);
        }
        else if (key == "stopExposure")
        {
            parameters.stopExposure = py::cast<float>(item.second);
        }
        else if (key == "numExposures")
        {
            parameters.numExposures = py::cast<int>(item.second);
        }
        else if (key == "tonemapper")
        {
            parameters.tonemapper = py::cast<std::string>(item.second);
        }
        else
        {
            std::cout << errorMessage << std::endl;
        }
    }

    return parameters;
}

// Update parameter dictionary that is returned to the Python side.
void updateInputParameters(const FLIP::Parameters& parameters, py::dict& inputParameters)
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
* @param[in] testInput Test input image. For LDR, the content should be in [0,1].
* @param[in] useHDR Set to true if the input images are to be considered containing HDR content, i.e., not necessarily in [0,1].
* @param[in] inputsRGB Set to true if the input images are given in the sRGB color space.
* @param[in] applyMagma A boolean indicating whether the output should have the Magma map applied to it before the image is returned.
* @param[in] computeMeanFLIPError A boolean indicating whether the mean FLIP error should be computed. If false, the returned mean error is -1.
* @param[in,out] inputParameters Contains parameters (e.g., PPD, exposure settings, etc). If the exposures have not been set by the user, then those will be computed (and returned).
* @return tuple containing FLIP error map (in Magma if applyMagma is true), the mean FLIP error (computed if computeMeanFLIPError is true, else -1), and dictionary of parameters.
*/
std::tuple<py::array_t<float>, float, py::dict> evaluate(const py::array_t<float> referenceInput, const py::array_t<float> testInput, const bool useHDR, const bool inputsRGB = true, const bool applyMagma = true, const bool computeMeanFLIPError = true, py::dict inputParameters = {})
{
    py::buffer_info r_buf = referenceInput.request(), t_buf = testInput.request();
    
    // Check number of dimensions and resolution.
    if (r_buf.ndim != 3 || t_buf.ndim != 3)
        throw std::runtime_error("Number of dimensions must be three.");
    if (r_buf.shape[0] != t_buf.shape[0] || r_buf.shape[1] != t_buf.shape[1] || r_buf.shape[2] != t_buf.shape[2])
        throw std::runtime_error("Reference and Test image resolutions differ.");
    
    // Arrays for reference and test.
    float* ptr_r = static_cast<float*>(r_buf.ptr);
    float* ptr_t = static_cast<float*>(t_buf.ptr);

    // Image size.
    const int nRows = int(r_buf.shape[0]), nCols = int(r_buf.shape[1]), nChannels = int(r_buf.shape[2]);

    // FLIP
    float* flip;

    // Create NumPy output array.
    py::array_t<float> flipNumpy;
    int nChannelsOut;
    if (applyMagma)
    {
        flipNumpy = py::array_t<float>({ r_buf.shape[0], r_buf.shape[1], r_buf.shape[2] });
        nChannelsOut = 3;
    }
    else
    {
        flipNumpy = py::array_t<float>({ r_buf.shape[0], r_buf.shape[1] });
        nChannelsOut = 1;
    }
    py::buffer_info flipNumpy_buf = flipNumpy.request();
    float* ptr_flipNumpy = static_cast<float*>(flipNumpy_buf.ptr);

    // Transform to linear RGB if desired.
    if (inputsRGB)
    {
        sRGBToLinearRGB(ptr_r, nCols, nRows);
        sRGBToLinearRGB(ptr_t, nCols, nRows);
    }

    // Run FLIP.
    FLIP::Parameters parameters = setParameters(inputParameters);
    float meanError = -1;
    FLIP::evaluate(ptr_r, ptr_t, nCols, nRows, useHDR, parameters, applyMagma, computeMeanFLIPError, meanError, &flip);

    // Move output array info to correct buffer.
#pragma omp parallel for
    for (int i = 0; i < nRows; i++)
    {
        for (int j = 0; j < nCols; j++)
        {
            for (int c = 0; c < nChannelsOut; c++)
            {
                int idx = (i * nCols + j) * nChannelsOut + c;
                ptr_flipNumpy[idx] = flip[idx];
            }
        }
    }
    delete flip;

    updateInputParameters(parameters, inputParameters);

    return std::make_tuple(flipNumpy, meanError, inputParameters);
}

// Create command line, based on the Python command line string, for the FLIP tool to parse.
commandline generateCommandLine(const py::list argvPy)
{
    size_t argc = argvPy.size();
    char** argv = new char* [argc];

    int counter = 0;
    for (auto item : argvPy)
    {
        const std::string it = py::reinterpret_steal<py::str>(item);
        argv[counter] = new char[it.length()];
        std::strcpy(argv[counter], it.c_str());
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
int execute(const py::list argvPy)
{
    commandline commandLine = generateCommandLine(argvPy);
    FLIPTool::execute(commandLine);

    return 0;
}

// Setup the pybind11 module.
PYBIND11_MODULE(pbflip, handle)
{
    handle.doc() = "Load images (load), evaluate FLIP (evaluate), or run the full FLIP tool (run_tool).";
    handle.def("evaluate", &evaluate);
    handle.def("load", &load);
    handle.def("execute", &execute);
}