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

import sys
sys.path.insert(1, '../pytorch')
from flip_loss import LDRFLIPLoss, HDRFLIPLoss
from data import *

if __name__ == '__main__':
	"""
	Test script. Add tests by appending descriptions to the test_description list,
	and set up tests (results should be booleans - True if test passed, False otherwise).
	Add results to the test_results list. Printed output will show which tests succeed and
	which fail.
	"""
	# Test descriptions
	test_descriptions = []
	test_descriptions.append("HDR-FLIP output matches reference HDR-FLIP output")
	test_descriptions.append("LDR-FLIP output matches reference LDR-FLIP output")

	# Set up test results
	test_results = []

	print("Running tests...")
	print("================")

	# HDR test
	# Run the image pairs in the images directory
	hdr_reference = read_exr('../images/reference.exr') # EXR
	hdr_test = read_exr('../images/test.exr') # EXR
	hdrflip_loss_fn = HDRFLIPLoss()
	hdrflip_loss = hdrflip_loss_fn(hdr_test, hdr_reference)
	test_results.append(round(hdrflip_loss.item(), 4) == 0.2835)

	# LDR test
	# Run the image pairs in the images directory
	ldr_reference = load_image_tensor('../images/reference.png') # sRGB
	ldr_test = load_image_tensor('../images/test.png') # sRGB
	ldrflip_loss_fn = LDRFLIPLoss()
	ldrflip_loss = ldrflip_loss_fn(ldr_test, ldr_reference)
	test_results.append(round(ldrflip_loss.item(), 4) == 0.1597)

	for idx, passed in enumerate(test_results):
		print(("PASSED " if passed else "FAILED ") + "test " + str(idx) + " - " + test_descriptions[idx])

	print("================")
	print("Tests complete!")
