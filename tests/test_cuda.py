#################################################################################
# Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# SPDX-FileCopyrightText: Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES
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

# Code by Pontus Andersson, Jim Nilsson, and Tomas Akenine-Moller.

import subprocess
import os
import numpy as np
import sys
sys.path.insert(1, '../python')
from data import *

if __name__ == '__main__':
	"""
	Test script. Runs flip-cuda.exe from the cpp/x64/release/-directory for both LDR and HDR.
    Both the mean FLIP is tested and the pixel values from the resulting FLIP images.
	"""
	# Set up test results
	test_results = []

	print("Running CUDA tests...")
	print("=====================")

	# Load correct LDR/HDR FLIP image in the images directory
	ldr_correct_result = load_image_array('../images/correct_ldrflip_cuda.png') # LDR, sRGB
	hdr_correct_result = load_image_array('../images/correct_hdrflip_cuda.png') # HDR

	# Run the image pairs in the images directory
	ldr_process = subprocess.run("../cpp/x64/release/flip-cuda.exe --reference ../images/reference.png --test ../images/test.png -v 0 --directory ../images", stdout=subprocess.PIPE, text=True)
	hdr_process = subprocess.run("../cpp/x64/release/flip-cuda.exe --reference ../images/reference.exr --test ../images/test.exr -v 0 --no-exposure-map --directory ../images", stdout=subprocess.PIPE, text=True)

	ldr_result_strings = ldr_process.stdout.split('\n')
	subpos = ldr_result_strings[4].find(':')
	ldr_mean = float(ldr_result_strings[4][subpos + 2 : len(ldr_result_strings[4])])

	hdr_result_strings = hdr_process.stdout.split('\n')
	subpos = hdr_result_strings[8].find(':')
	hdr_mean = float(hdr_result_strings[8][subpos + 2 : len(hdr_result_strings[4])])

   # Load the images that were just created.
	result_ldr_file = "../images/flip.reference.test.67ppd.ldr.png";
	result_hdr_file = "../images/flip.reference.test.67ppd.hdr.aces.m12.5423_to_p0.9427.14.png";
	ldr_new_result = load_image_array(result_ldr_file) # LDR, sRGB
	hdr_new_result = load_image_array(result_hdr_file) # HDR

	if((abs(ldr_correct_result - ldr_new_result) > 0).any()):
		print("LDR: FAILED per-pixel test on FLIP images.")
	else:
		print("LDR: PASSED per-pixel test on FLIP images.")

	if(ldr_mean != 0.159691):
		print("LDR: FAILED mean test.")
	else:
		print("LDR: PASSED mean test.")

	if((abs(hdr_correct_result - hdr_new_result) > 0).any()):
		print("HDR: FAILED per-pixel test on FLIP images.")
	else:
		print("HDR: PASSED per-pixel test on FLIP images.")

	if(hdr_mean != 0.283478):
		print("HDR: FAILED mean test.")
	else:
		print("HDr: PASSED mean test.")

	# Remove output created during tests
	os.remove(result_ldr_file)
	os.remove(result_hdr_file)

	print("=====================")
	print("Tests complete!")

