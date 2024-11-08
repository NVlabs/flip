""" Functions for image loading and saving """
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

import numpy as np
import torch
from matplotlib.pyplot import imsave
from PIL import Image
import OpenEXR as exr
import Imath

from flip_loss import color_space_transform, tone_map

def HWCtoCHW(x):
	"""
	Transforms an image from HxWxC layout to CxHxW

	:param x: image with HxWxC layout
	:return: image with CxHxW layout
	"""
	return np.rollaxis(x, 2)

def CHWtoHWC(x):
	"""
	Transforms an image from CxHxW layout to HxWxC

	:param x: image with CxHxW layout
	:return: image with HxWxC layout
	"""
	return np.swapaxes(np.swapaxes(x, 0, 1), 1, 2)

def save_image(img_file, img):
	"""
	Saves image as png

	:param img_file: image's filename
	:param img: float image (in the [0,1] range) to save
	"""
	img = CHWtoHWC(img.cpu().numpy()[0])
	imsave(img_file, img)

def load_image_tensor(img_file):
	"""
	Loads an image and transforms it into a numpy array and into the [0, 1] range

	:param img_file: image's filename
	:return: float image (in the [0,1] range) with 1xCxHxW layout
	"""
	img = Image.open(img_file, 'r').convert('RGB')
	img = np.asarray(img).astype(np.float32)
	img = np.rollaxis(img, 2)
	img = img / 255.0
	img = torch.from_numpy(img).unsqueeze(0).cuda()
	return img

def read_exr(filename):
	"""
	Read color data from EXR image file. Set negative values and nans to 0.

	:param filename: string describing file path
	:return: RGB image in float32 format (with 1xCxHxW layout)
	"""
	exrfile = exr.InputFile(filename)
	header = exrfile.header()

	dw = header['dataWindow']
	isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

	channelData = dict()

	# Convert all channels in the image to numpy arrays
	for c in header['channels']:
		C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
		C = np.frombuffer(C, dtype=np.float32)
		C = np.reshape(C, isize)

		channelData[c] = C

	if len(channelData) == 1:
		channelData['R'] = channelData['G'] = channelData['B'] = channelData[next(iter(channelData))]

	colorChannels = ['R', 'G', 'B']
	img_np = np.concatenate([channelData[c][...,np.newaxis] for c in colorChannels], axis=2)

	img_torch = torch.max(torch.nan_to_num(torch.from_numpy(np.array(img_np))), torch.tensor([0.0])).cuda() # added maximum to avoid negative values in images

	return img_torch.permute(2, 0, 1).unsqueeze(0)

def save_ldr_image_range(img_file, hdr_img, epoch=0, tone_mapper="aces", start_exposure=0.0, stop_exposure=0.0):
	"""
	Saves exposure compensated and tone mapped versions of the input HDR image corresponding
	to the exposure range and tone mapper assumed by HDR-FLIP during its calculations

	:param img_file: string describing image's base file path
	:param epoch: (optional) int describing training epoch
	:param tone_mapper: (optional) string describing tone mapper to apply before storing the LDR image
	:param start_exposure: (optional) float describing the shortest exposure in the exposure range
	:param stop_exposure: (optional) float describing the longest exposure in the exposure range
	"""
	start_exposure_sign = "m" if start_exposure < 0 else "p"
	stop_exposure_sign = "m" if stop_exposure < 0 else "p"

	num_exposures = int(max(2.0, np.ceil(stop_exposure.item() - start_exposure.item())))
	step_size = (stop_exposure - start_exposure) / max(num_exposures - 1, 1)

	for i in range(0, num_exposures):
		exposure = start_exposure + i * step_size
		exposure_sign = "m" if exposure < 0 else "p"
		img_tone_mapped = color_space_transform(tone_map(hdr_img, tone_mapper, exposure), "linrgb2srgb")
		save_image((img_file + '_epoch_%04d.%04d.' + start_exposure_sign + '%0.4f_to_' + stop_exposure_sign + '%0.4f.' + exposure_sign + '%0.4f.png')
					% (epoch, i, abs(start_exposure), abs(stop_exposure), abs(exposure)),
					img_tone_mapped)
