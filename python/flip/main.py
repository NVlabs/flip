#################################################################################
# Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################

import pbflip
import sys, os

def evaluate(reference, test, dynamicRangeString, inputsRGB=True, applyMagma=True, computeMeanError=True, parameters={}):
    """
    Compute the FLIP error map between two images.

    :param reference: string containing relative or absolute path to reference image OR
                      numpy array containing reference image (with HxWxC layout and nonnegative values in the case of HDR and values in [0,1] in the case of LDR)
    :param test: string containing relative or absolute path to test image OR
                 numpy array containing test image (with HxWxC layout and nonnegative values in the case of HDR and values in [0,1] in the case of LDR)
    :param dynamicRangeString:  string describing the assumed dynamic range of the reference and test image, valid arguments are \"LDR\" (low dynamic range) and \"HDR\" (high dynamic range)
    :param inputsRGB: (optional) bool describing if input images are in sRGB. LDR images are assumed to be sRGB while HDR images are assumed to be in linear sRGB
    :param applyMagma: (optional) bool saying if the magma color map should be applied to the error map, default is True
    :param computeMeanError: (optional) bool saying if mean FLIP error should be computed, default is True. If False, -1 is returned.
    :param parameters: (optional)   dictionary containing non-default settings for the FLIP evaluation:
                                    \"ppd\": float describing assumed pixels per degree of visual angle (default is 67 PPD). Should not be included in the dictionary if \"vc\" is included, and vice versa
                                    \"vc\": list of three floats. First float is the distance to display (meters), second is display width (pixels), and third is display width (meters). Should not be included if \"ppd\" is included, and vice versa
                                    \"startExposure\": float setting the start exposre (c_start). Its value is computed by FLIP if not entered. startExposure must be smaller than or equal to stopExposure
                                    \"startExposure\": float setting the stop exposre (c_stop). Its value is computed by FLIP if not entered. stopExposure must be greater than or equal to startExposure
                                    \"numExposures\": int setting the number of exposure steps. Its value is computed by FLIP if not entered
                                    \"tonemapper\": string setting the assumed tone mapper. Allowed options are \"ACES\", \"Hable\", and \"Reinhard\"
    :return: tuple containing 1: the FLIP error map, 2: the mean FLIP error (computed if computeMeanError is True, else -1), 3: the parameter dictionary used to compute the results
    """
    # Set correct bools based on if HDR- or LDR-FLIP should be used.
    if dynamicRangeString.lower() == "hdr":
        useHDR = True
        inputsRGB = False # Assume HDR input is in linear RGB.
    elif dynamicRangeString.lower() == "ldr":
        useHDR = False
    else:
        print("Third argument to evaluate() must be \"LDR\" or \"HDR\".\nExiting.")
        sys.exit(1)

    # Check that parameters only include settings for either viewing conditions or pixels per degree.
    if "vc" in parameters and "ppd" in parameters:
        print("\"vc\" and \"ppd\" are mutually exclusive. Use only one of the two.\nExiting.")
        sys.exit(1)

    # Check that PPD values are valid:
    if "vc" in parameters:
        if parameters["vc"][0] <= 0 or parameters["vc"][1] <= 0 or parameters["vc"][2] <= 0:
            print("Viewing condition options must be positive.\nExiting.")
            sys.exit(1)
    elif "ppd" in parameters:
        if parameters["ppd"] <= 0:
            print("The number of pixels per degree must be positive.\nExiting.")
            sys.exit(1)

    # Check that tonemapper is valid.
    if "tonemapper" in parameters and parameters["tonemapper"].lower() not in ["aces", "hable", "reinhard"]:
        print("Invalid tonemapper. Valid options are \"ACES\", \"Hable\", and \"Reinhard\".\nExiting.")
        sys.exit(1)

    # If strings are input, load images before evaluation.
    if isinstance(reference, str):
        if not os.path.exists(reference):
            print("Path to reference image is invalid, or the reference image does not exist.\nExiting.")
            sys.exit(1)
        reference = pbflip.load(reference)
    if isinstance(test, str):
        if not os.path.exists(test):
            print("Path to test image is invalid, or the reference image does not exist.\nExiting.")
            sys.exit(1)
        test = pbflip.load(test)

    # Evaluate FLIP. Return error map.
    return pbflip.evaluate(reference, test, useHDR, inputsRGB, computeMeanError, applyMagma, parameters)

def execute(cmdline):
    """
    Run the FLIP tool, based on the C++ tool code.

    :param cmdline: string containing the command line for the FLIP tool (run python flip.py to see all available input)
    """
    pbflip.execute(cmdline)

def load(imgpath):
    """
    Load an image.

    :param imgpath: string containing the relative or absolute path to an image, allowed file types are png, exr, bmp, and tga
    :return: numpy array containing the image (with HxWxC layout)
    """
    return pbflip.load(imgpath)