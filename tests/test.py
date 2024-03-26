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

# Visualizing and Communicating Errors in Rendered Images
# Ray Tracing Gems II, 2021,
# by Pontus Andersson, Jim Nilsson, and Tomas Akenine-Moller.
# Pointer to the chapter: https://research.nvidia.com/publication/2021-08_Visualizing-and-Communicating.

# Visualizing Errors in Rendered High Dynamic Range Images
# Eurographics 2021,
# by Pontus Andersson, Jim Nilsson, Peter Shirley, and Tomas Akenine-Moller.
# Pointer to the paper: https://research.nvidia.com/publication/2021-05_HDR-FLIP.

# FLIP: A Difference Evaluator for Alternating Images
# High Performance Graphics 2020,
# by Pontus Andersson, Jim Nilsson, Tomas Akenine-Moller,
# Magnus Oskarsson, Kalle Astrom, and Mark D. Fairchild.
# Pointer to the paper: https://research.nvidia.com/publication/2020-07_FLIP.

# Code by Pontus Ebelin (formerly Andersson), Jim Nilsson, and Tomas Akenine-Moller.

import subprocess
import os
import sys
import flip

if __name__ == '__main__':
    """
    Test script. Runs FLIP for both LDR and HDR using one of CUDA/CPP/PYTHON based on the commandline argument.
    Both the mean FLIP is tested and the pixel values from the resulting FLIP images.
    """

    if(len(sys.argv) != 2):
        print("Usage: python test.py --{cuda|cpp|python}")
        print("Tip: do not forget to install the FLIP Python API through `pip install .` in `flip/python`.")
        sys.exit()

    # Parse command line argument.
    if(sys.argv[1] == "--cuda" or sys.argv[1] == "cuda" or sys.argv[1] == "-cuda"):
        test_str = "CUDA"
        correct_ldr_image_filename = "correct_ldrflip_cuda.png"
        correct_hdr_image_filename = "correct_hdrflip_cuda.png"
        ldr_cmd = "../cpp/x64/release/flip-cuda.exe --reference ../images/reference.png --test ../images/test.png"
        hdr_cmd = "../cpp/x64/release/flip-cuda.exe --reference ../images/reference.exr --test ../images/test.exr --no-exposure-map"
        expected_ldr_mean = 0.159691
        expected_hdr_mean = 0.283478
    elif(sys.argv[1] == "--cpp" or sys.argv[1] == "cpp" or sys.argv[1] == "-cpp"):
        test_str = "CPP"
        correct_ldr_image_filename = "correct_ldrflip_cpp.png"
        correct_hdr_image_filename = "correct_hdrflip_cpp.png"
        ldr_cmd = "../cpp/x64/release/flip.exe --reference ../images/reference.png --test ../images/test.png"
        hdr_cmd = "../cpp/x64/release/flip.exe --reference ../images/reference.exr --test ../images/test.exr --no-exposure-map"
        expected_ldr_mean = 0.159691
        expected_hdr_mean = 0.283478
    elif(sys.argv[1] == "--python" or sys.argv[1] == "python" or sys.argv[1] == "-python"):
        test_str = "PYTHON"
        correct_ldr_image_filename = "correct_ldrflip_cpp.png" # Python and C++ should give the same results,
        correct_hdr_image_filename = "correct_hdrflip_cpp.png" # as the Python code runs the C++ code via pybind11.
        ldr_cmd = "python ../python/flip.py --reference ../images/reference.png --test ../images/test.png"
        hdr_cmd = "python ../python/flip.py --reference ../images/reference.exr --test ../images/test.exr --no-exposure-map"
        expected_ldr_mean = 0.159691
        expected_hdr_mean = 0.283478
    else:
        print("Error: the argument should be one of --cuda, --cpp, and --python.")
        sys.exit()

    print("Running " + test_str + " tests...")
    print("========================")

    # Load correct LDR/HDR FLIP images in the tests directory.
    ldr_correct_result = flip.load(correct_ldr_image_filename) # LDR, sRGB
    hdr_correct_result = flip.load(correct_hdr_image_filename) # HDR

    # Run FLIP on the reference/test image pairs in the ../images directory.
    ldr_process = subprocess.run(ldr_cmd, stdout=subprocess.PIPE, text=True)
    hdr_process = subprocess.run(hdr_cmd, stdout=subprocess.PIPE, text=True)

    ldr_result_strings = ldr_process.stdout.split('\n')
    subpos = ldr_result_strings[4].find(':')
    ldr_mean = float(ldr_result_strings[4][subpos + 2 : len(ldr_result_strings[4])])

    hdr_result_strings = hdr_process.stdout.split('\n')
    subpos = hdr_result_strings[8].find(':')
    hdr_mean = float(hdr_result_strings[8][subpos + 2 : len(hdr_result_strings[4])])

    # Load the images that were just created.
    result_ldr_file = "flip.reference.test.67ppd.ldr.png"
    result_hdr_file = "flip.reference.test.67ppd.hdr.aces.m12.5423_to_p0.9427.14.png"
    ldr_new_result = flip.load(result_ldr_file) # LDR, sRGB
    hdr_new_result = flip.load(result_hdr_file) # HDR

    if((abs(ldr_correct_result - ldr_new_result) > 0).any()):
        print("LDR: FAILED per-pixel test on FLIP images.")
    else:
        print("LDR: PASSED per-pixel test on FLIP images.")

    if(ldr_mean != expected_ldr_mean):
        print("LDR: FAILED mean test.")
    else:
        print("LDR: PASSED mean test.")

    if((abs(hdr_correct_result - hdr_new_result) > 0).any()):
        print("HDR: FAILED per-pixel test on FLIP images.")
    else:
        print("HDR: PASSED per-pixel test on FLIP images.")

    if(hdr_mean != expected_hdr_mean):
        print("HDR: FAILED mean test.")
    else:
        print("HDR: PASSED mean test.")

    # Remove output created during tests.
    os.remove(result_ldr_file)
    os.remove(result_hdr_file)

    print("========================")
    print("Tests complete!")

