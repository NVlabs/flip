""" Simple example of using FLIP as a loss function """
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

# Code by Pontus Andersson, Jim Nilsson, and Tomas Akenine-Moller.
import numpy as np
import torch
import torch.nn as nn
import argparse
import os

from flip_loss import LDRFLIPLoss, HDRFLIPLoss, compute_start_stop_exposures, color_space_transform
from data import *

###############################################################################
# Autoencoder network definition
###############################################################################
class Autoencoder(nn.Module):
	""" Autoencoder class """
	def __init__(self):
		""" Init """
		super(Autoencoder, self).__init__()
		# Encoder
		self.encoder = nn.Sequential(
			nn.Conv2d(3, 48, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(48, 48, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2)
		)
		# Decoder
		self.decoder = nn.Sequential(
			nn.Conv2d(48, 64, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 32, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 3, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Upsample(scale_factor=2)
		)

	def forward(self, x):
		"""
		Autoencoder's forward function

		:param x: tensor with NxCxHxW layout
		:return: tensor with NxCxHxW layout
		"""
		x = self.encoder(x)
		return self.decoder(x)

###############################################################################
# Simple Autoencoder training
###############################################################################
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--reference', type=str, default='../images/reference.png')
parser.add_argument('--test', type=str, default='../images/test.png')
parser.add_argument('--directory', type=str, default='../images/out')
parser.add_argument('--output_ldr_images', action='store_true')

args = parser.parse_args()

# Create output directory if it doesn't exist
if not os.path.isdir(args.directory) and args.output_ldr_images:
	os.makedirs(args.directory)

# Load images and setup loss. If images are HDR, use HDR-FLIP loss, otherwise use LDR-FLIP loss
image_format = args.reference.split('.')[-1]
if image_format == "exr" or image_format == "EXR":
	hdr = True
	reference = read_exr(args.reference)
	test = read_exr(args.test)

	# Compute start and stop exposures automatically for the given reference
	start_exposure, stop_exposure = compute_start_stop_exposures(reference, "aces", 0.85, 0.85)

	# Save the LDR versions of the reference that will be used for HDR-FLIP
	if args.output_ldr_images:
		save_ldr_image_range(args.directory + '/reference', reference, tone_mapper="aces", start_exposure=start_exposure, stop_exposure=stop_exposure)

	loss_fn = HDRFLIPLoss()
elif image_format == "png" or image_format == "PNG":
	hdr = False
	reference = load_image_tensor(args.reference)
	test = load_image_tensor(args.test)

	loss_fn = LDRFLIPLoss()
else:
	sys.exit("Error: Invalid image format. Please use png or exr.")

# Initialize autoencoder and optimizer
model = Autoencoder().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1001):
	pred = model.forward(test)
	if not hdr:
		# LDR-FLIP expects sRGB input per default
		pred.data = color_space_transform(torch.clamp(pred.data, 0, 1), "linrgb2srgb")
	dist = loss_fn.forward(pred, reference)
	optimizer.zero_grad()
	dist.backward()
	optimizer.step()

	# Print loss every 10th epoch
	if epoch % 10 == 0:
		print('iter %d, dist %.3g' % (epoch, dist.item()))
		pred_img = pred.data
		if args.output_ldr_images:
			if hdr:
				if epoch % 100 == 0:
					# Loop over exposures used for HDR-FLIP - apply exposure compensation and tone mapping and save resulting test images
					# (Only do this once per 100 epochs to avoid large amount of images saved)
					save_ldr_image_range(args.directory + '/pred', pred_img, epoch=epoch, tone_mapper="aces", start_exposure=start_exposure, stop_exposure=stop_exposure)
				else:
					# Save output for arbitrary exposure between start and stop exposure
					exposure = 0.7 * (stop_exposure - start_exposure) + start_exposure
					pred_img_tone_mapped = color_space_transform(tone_map(pred_img, "aces", exposure), 'linrgb2srgb')
					save_image(args.directory + '/pred_epoch_%04d.png' % epoch, pred_img_tone_mapped)
			else:
				save_image(args.directory + '/pred_epoch_%04d.png' % epoch, pred_img)
