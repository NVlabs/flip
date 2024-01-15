""" FLIP metric functions """
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
import cv2 as cv
import os
import sys

from data import *

##################################################################################################################################################################################################################################################
##################################################################################################################################################################################################################################################
# Utility functions
##################################################################################################################################################################################################################################################
##################################################################################################################################################################################################################################################

def tone_map(img, exposure, tone_mapper="aces"):
	"""
	Applies exposure compensation and tone mapping.
	Refer to the Visualizing Errors in Rendered High Dynamic Range Images
	paper for details about the formulas

	:param img: float tensor (with CxHxW layout) containing nonnegative values
	:param exposure: float describing the exposure compensation factor
	:param tone_mapper: (optional) string describing the tone mapper to apply
	:return: float tensor (with CxHxW layout) containing exposure compensated and tone mapped image
	"""
	# Clip to 0. Negative values shouldn't be used
	img = np.maximum(img, 0.0)

	# Exposure compensation
	x = (2 ** exposure) * img

	# Set tone mapping coefficients depending on tone_mapper
	if tone_mapper == "reinhard":
		lum_coeff_r = 0.2126
		lum_coeff_g = 0.7152
		lum_coeff_b = 0.0722
		Y = x[0:1, :, :] * lum_coeff_r + x[1:2, :, :] * lum_coeff_g + x[2:3, :, :] * lum_coeff_b

		return np.clip(np.divide(x, 1 + Y), 0.0, 1.0)

	elif tone_mapper == "hable":
		# Source: https://64.github.io/tonemapping/
		A = 0.15
		B = 0.50
		C = 0.10
		D = 0.20
		E = 0.02
		F = 0.30
		k0 = A * F - A * E
		k1 = C * B * F - B * E
		k2 = 0
		k3 = A * F
		k4 = B * F
		k5 = D * F * F

		W = 11.2
		nom = k0 * np.power(W, 2) + k1 * W + k2
		denom = k3 * np.power(W, 2) + k4 * W + k5
		white_scale = denom / nom # = 1 / (nom / denom)

		# Include white scale and exposure bias in rational polynomial coefficients
		k0 = 4 * k0 * white_scale
		k1 = 2 * k1 * white_scale
		k2 = k2 * white_scale
		k3 = 4 * k3
		k4 = 2 * k4
		#k5 = k5
	else:# tone_mapper == "aces":
		# Include pre-exposure cancelation in constants
		k0 = 0.6 * 0.6 * 2.51
		k1 = 0.6 * 0.03
		k2 = 0
		k3 = 0.6 * 0.6 * 2.43
		k4 = 0.6 * 0.59
		k5 = 0.14

	x2 = np.power(x, 2)
	nom = k0 * x2 + k1 * x + k2
	denom = k3 * x2 + k4 * x + k5
	denom[np.isinf(denom)] = 1.0 # if denom is inf, then so is nom => nan. Pixel is very bright. It becomes inf here, but 1 after clamp below
	y = np.divide(nom, denom)
	return np.clip(y, 0.0, 1.0)

def color_space_transform(input_color, fromSpace2toSpace):
	"""
	Transforms inputs between different color spaces

	:param input_color: tensor of colors to transform (with CxHxW layout)
	:param fromSpace2toSpace: string describing transform
	:return: transformed tensor (with CxHxW layout)
	"""
	dim = input_color.shape

	# Assume D65 standard illuminant
	reference_illuminant = np.array([[[0.950428545]], [[1.000000000]], [[1.088900371]]]).astype(np.float32)
	inv_reference_illuminant = np.array([[[1.052156925]], [[1.000000000]], [[0.918357670]]]).astype(np.float32)

	if fromSpace2toSpace == "srgb2linrgb":
		limit = 0.04045
		transformed_color = np.where(input_color > limit, np.power((input_color + 0.055) / 1.055, 2.4), input_color / 12.92)

	elif fromSpace2toSpace == "linrgb2srgb":
		limit = 0.0031308
		transformed_color = np.where(input_color > limit, 1.055 * (input_color ** (1.0 / 2.4)) - 0.055, 12.92 * input_color)

	elif fromSpace2toSpace == "linrgb2xyz" or fromSpace2toSpace == "xyz2linrgb":
		# Source: https://www.image-engineering.de/library/technotes/958-how-to-convert-between-srgb-and-ciexyz
		# Assumes D65 standard illuminant
		if fromSpace2toSpace == "linrgb2xyz":
			a11 = 10135552 / 24577794
			a12 = 8788810  / 24577794
			a13 = 4435075  / 24577794
			a21 = 2613072  / 12288897
			a22 = 8788810  / 12288897
			a23 = 887015   / 12288897
			a31 = 1425312  / 73733382
			a32 = 8788810  / 73733382
			a33 = 70074185 / 73733382
		else:
			# Constants found by taking the inverse of the matrix
			# defined by the constants for linrgb2xyz
			a11 = 3.241003275
			a12 = -1.537398934
			a13 = -0.498615861
			a21 = -0.969224334
			a22 = 1.875930071
			a23 = 0.041554224
			a31 = 0.055639423
			a32 = -0.204011202
			a33 = 1.057148933
		A = np.array([[a11, a12, a13],
					  [a21, a22, a23],
					  [a31, a32, a33]]).astype(np.float32)

		input_color = np.transpose(input_color, (2, 0, 1)) # C(H*W)
		transformed_color = np.matmul(A, input_color)
		transformed_color = np.transpose(transformed_color, (1, 2, 0))

	elif fromSpace2toSpace == "xyz2ycxcz":
		input_color = np.multiply(input_color, inv_reference_illuminant)
		y = 116 * input_color[1:2, :, :] - 16
		cx = 500 * (input_color[0:1, :, :] - input_color[1:2, :, :])
		cz = 200 * (input_color[1:2, :, :] - input_color[2:3, :, :])
		transformed_color = np.concatenate((y, cx, cz), 0)

	elif fromSpace2toSpace == "ycxcz2xyz":
		y = (input_color[0:1, :, :] + 16) / 116
		cx = input_color[1:2, :, :] / 500
		cz = input_color[2:3, :, :] / 200

		x = y + cx
		z = y - cz
		transformed_color = np.concatenate((x, y, z), 0)

		transformed_color = np.multiply(transformed_color, reference_illuminant)

	elif fromSpace2toSpace == "xyz2lab":
		input_color = np.multiply(input_color, inv_reference_illuminant)
		delta = 6 / 29
		delta_square = delta * delta
		delta_cube = delta * delta_square
		factor = 1 / (3 * delta_square)

		input_color = np.where(input_color > delta_cube, np.power(input_color, 1 / 3), (factor * input_color + 4 / 29))

		l = 116 * input_color[1:2, :, :] - 16
		a = 500 * (input_color[0:1,:, :] - input_color[1:2, :, :])
		b = 200 * (input_color[1:2, :, :] - input_color[2:3, :, :])

		transformed_color = np.concatenate((l, a, b), 0)

	elif fromSpace2toSpace == "lab2xyz":
		y = (input_color[0:1, :, :] + 16) / 116
		a =  input_color[1:2, :, :] / 500
		b =  input_color[2:3, :, :] / 200

		x = y + a
		z = y - b

		xyz = np.concatenate((x, y, z), 0)
		delta = 6 / 29
		factor = 3 * delta * delta
		xyz = np.where(xyz > delta,  xyz ** 3, factor * (xyz - 4 / 29))

		transformed_color = np.multiply(xyz, reference_illuminant)

	elif fromSpace2toSpace == "srgb2xyz":
		transformed_color = color_space_transform(input_color, 'srgb2linrgb')
		transformed_color = color_space_transform(transformed_color,'linrgb2xyz')
	elif fromSpace2toSpace == "srgb2ycxcz":
		transformed_color = color_space_transform(input_color, 'srgb2linrgb')
		transformed_color = color_space_transform(transformed_color, 'linrgb2xyz')
		transformed_color = color_space_transform(transformed_color, 'xyz2ycxcz')
	elif fromSpace2toSpace == "linrgb2ycxcz":
		transformed_color = color_space_transform(input_color, 'linrgb2xyz')
		transformed_color = color_space_transform(transformed_color, 'xyz2ycxcz')
	elif fromSpace2toSpace == "srgb2lab":
		transformed_color = color_space_transform(input_color, 'srgb2linrgb')
		transformed_color = color_space_transform(transformed_color, 'linrgb2xyz')
		transformed_color = color_space_transform(transformed_color, 'xyz2lab')
	elif fromSpace2toSpace == "linrgb2lab":
		transformed_color = color_space_transform(input_color, 'linrgb2xyz')
		transformed_color = color_space_transform(transformed_color, 'xyz2lab')
	elif fromSpace2toSpace == "ycxcz2linrgb":
		transformed_color = color_space_transform(input_color, 'ycxcz2xyz')
		transformed_color = color_space_transform(transformed_color, 'xyz2linrgb')
	elif fromSpace2toSpace == "lab2srgb":
		transformed_color = color_space_transform(input_color, 'lab2xyz')
		transformed_color = color_space_transform(transformed_color, 'xyz2linrgb')
		transformed_color = color_space_transform(transformed_color, 'linrgb2srgb')
	elif fromSpace2toSpace == "ycxcz2lab":
		transformed_color = color_space_transform(input_color, 'ycxcz2xyz')
		transformed_color = color_space_transform(transformed_color, 'xyz2lab')
	else:
		sys.exit('Error: The color transform %s is not defined!' % fromSpace2toSpace)

	return transformed_color

##################################################################################################################################################################################################################################################
##################################################################################################################################################################################################################################################
# LDR-FLIP functions
##################################################################################################################################################################################################################################################
##################################################################################################################################################################################################################################################

def generate_spatial_filter(pixels_per_degree, channel):
	"""
	Generates spatial contrast sensitivity filters with width depending on
	the number of pixels per degree of visual angle of the observer

	:param pixels_per_degree: float indicating number of pixels per degree of visual angle
	:param channel: string describing what filter should be generated
	:yield: Filter kernel corresponding to the spatial contrast sensitivity function of the given channel
	"""
	a1_A = 1
	b1_A = 0.0047
	a2_A = 0
	b2_A = 1e-5 # avoid division by 0
	a1_rg = 1
	b1_rg = 0.0053
	a2_rg = 0
	b2_rg = 1e-5 # avoid division by 0
	a1_by = 34.1
	b1_by = 0.04
	a2_by = 13.5
	b2_by = 0.025
	if channel == "A": # Achromatic CSF
		a1 = a1_A
		b1 = b1_A
		a2 = a2_A
		b2 = b2_A
	elif channel == "RG": # Red-Green CSF
		a1 = a1_rg
		b1 = b1_rg
		a2 = a2_rg
		b2 = b2_rg
	elif channel == "BY": # Blue-Yellow CSF
		a1 = a1_by
		b1 = b1_by
		a2 = a2_by
		b2 = b2_by

	# Determine evaluation domain
	max_scale_parameter = max([b1_A, b2_A, b1_rg, b2_rg, b1_by, b2_by])
	r = np.ceil(3 * np.sqrt(max_scale_parameter / (2 * np.pi**2)) * pixels_per_degree)
	r = int(r)
	deltaX = 1.0 / pixels_per_degree
	x, y = np.meshgrid(range(-r, r + 1), range(-r, r + 1))
	z = ((x * deltaX)**2 + (y * deltaX)**2).astype(np.float32)

	# Generate weights
	s = a1 * np.sqrt(np.pi / b1) * np.exp(-np.pi**2 * z / b1) + a2 * np.sqrt(np.pi / b2) * np.exp(-np.pi**2 * z / b2)
	s = s / np.sum(s)

	return s

def spatial_filter(img, s_a, s_rg, s_by):
	"""
	Filters an image with channel specific spatial contrast sensitivity functions
	and clips result to the unit cube in linear RGB

	:param img: image to filter (with CxHxW layout in the YCxCz color space)
	:param s_a: spatial filter matrix for the achromatic channel
	:param s_rg: spatial filter matrix for the red-green channel
	:param s_by: spatial filter matrix for the blue-yellow channel
	:return: input image (with CxHxW layout) transformed to linear RGB after filtering with spatial contrast sensitivity functions
	"""
	# Apply Gaussian filters
	dim = img.shape
	img_tilde_opponent = np.zeros((dim[0], dim[1], dim[2])).astype(np.float32)
	img_tilde_opponent[0:1, :, :] = cv.filter2D(img[0:1, :, :].squeeze(0), ddepth=-1, kernel=s_a, borderType=cv.BORDER_REPLICATE)
	img_tilde_opponent[1:2, :, :] = cv.filter2D(img[1:2, :, :].squeeze(0), ddepth=-1, kernel=s_rg, borderType=cv.BORDER_REPLICATE)
	img_tilde_opponent[2:3, :, :] = cv.filter2D(img[2:3, :, :].squeeze(0), ddepth=-1, kernel=s_by, borderType=cv.BORDER_REPLICATE)

	# Transform to linear RGB for clamp
	img_tilde_linear_rgb = color_space_transform(img_tilde_opponent, 'ycxcz2linrgb')

	# Clamp to RGB box
	return np.clip(img_tilde_linear_rgb, 0.0, 1.0)

def hunt_adjustment(img):
	"""
	Applies Hunt-adjustment to an image

	:param img: image to adjust (with CxHxW layout in the L*a*b* color space)
	:return: Hunt-adjusted image (with CxHxW layout in the Hunt-adjusted L*A*B* color space)
	"""
	# Extract luminance component
	L = img[0:1, :, :]

	# Apply Hunt adjustment
	img_h = np.zeros(img.shape).astype(np.float32)
	img_h[0:1, :, :] = L
	img_h[1:2, :, :] = np.multiply((0.01 * L), img[1:2, :, :])
	img_h[2:3, :, :] = np.multiply((0.01 * L), img[2:3, :, :])

	return img_h

def hyab(reference, test):
	"""
	Computes the HyAB distance between reference and test images

	:param reference: reference image (with CxHxW layout in the standard or Hunt-adjusted L*A*B* color space)
	:param test: test image (with CxHxW layout in the standard or Hunt-adjusted L*A*B* color space)
	:return: matrix (with 1xHxW layout) containing the per-pixel HyAB distance between reference and test
	"""
	delta = reference - test
	return abs(delta[0:1, :, :]) + np.linalg.norm(delta[1:3, :, :], axis=0)

def redistribute_errors(power_deltaE_hyab, cmax):
	"""
	Redistributes exponentiated HyAB errors to the [0,1] range

	:param power_deltaE_hyab: float containing the exponentiated HyAb distance
	:param cmax: float containing the exponentiated, maximum HyAB difference between two colors in Hunt-adjusted L*A*B* space
	:return: matrix (on 1xHxW layout) containing redistributed per-pixel HyAB distances (in range [0,1])
	"""
	# Set redistribution parameters
	pc = 0.4
	pt = 0.95

	# Re-map error to 0-1 range. Values between 0 and
	# pccmax are mapped to the range [0, pt],
	# while the rest are mapped to the range (pt, 1]
	deltaE_c = np.zeros(power_deltaE_hyab.shape)
	pccmax = pc * cmax
	deltaE_c = np.where(power_deltaE_hyab < pccmax, (pt / pccmax) * power_deltaE_hyab, pt + ((power_deltaE_hyab - pccmax) / (cmax - pccmax)) * (1.0 - pt))

	return deltaE_c

def feature_detection(imgy, pixels_per_degree, feature_type):
	"""
	Detects edges and points (features) in the achromatic image

	:param imgy: achromatic image (on 1xHxW layout, containing normalized Y-values from YCxCz)
	:param pixels_per_degree: float describing the number of pixels per degree of visual angle of the observer
	:param feature_type: string indicating the type of feature to detect
	:return: tensor (with layout 2xHxW with values in range [0,1]) containing large values where features were detected
	"""
	# Set peak to trough value (2x standard deviations) of human edge
	# detection filter
	w = 0.082

	# Compute filter radius
	sd = 0.5 * w * pixels_per_degree
	radius = int(np.ceil(3 * sd))

	# Compute 2D Gaussian
	[x, y] = np.meshgrid(range(-radius, radius+1), range(-radius, radius+1))
	g = np.exp(-(x ** 2 + y ** 2) / (2 * sd * sd))

	if feature_type == 'edge': # Edge detector
		# Compute partial derivative in x-direction
		Gx = np.multiply(-x, g)
	else: # Point detector
		# Compute second partial derivative in x-direction
		Gx = np.multiply(x ** 2 / (sd * sd) - 1, g)

	# Normalize positive weights to sum to 1 and negative weights to sum to -1
	negative_weights_sum = -np.sum(Gx[Gx < 0])
	positive_weights_sum = np.sum(Gx[Gx > 0])
	Gx = np.where(Gx < 0, Gx / negative_weights_sum, Gx / positive_weights_sum)

	# Detect features
	featuresX = cv.filter2D(imgy.squeeze(0), ddepth=-1, kernel=Gx, borderType=cv.BORDER_REPLICATE)
	featuresY = cv.filter2D(imgy.squeeze(0), ddepth=-1, kernel=np.transpose(Gx), borderType=cv.BORDER_REPLICATE)

	return np.stack((featuresX, featuresY))

def compute_ldrflip(reference, test, pixels_per_degree=(0.7 * 3840 / 0.7) * np.pi / 180):
	"""
	Computes the FLIP error map between two LDR images,
	assuming the images are observed at a certain number of
	pixels per degree of visual angle

	:param reference: reference image (with CxHxW layout on float32 format with values in the range [0, 1] in the sRGB color space)
	:param test: test image (with CxHxW layout on float32 format with values in the range [0, 1] in the sRGB color space)
	:param pixels_per_degree: (optional) float describing the number of pixels per degree of visual angle of the observer,
							  default corresponds to viewing the images on a 0.7 meters wide 4K monitor at 0.7 meters from the display
	:return: matrix (with 1xHxW layout on float32 format) containing the per-pixel FLIP errors (in the range [0, 1]) between LDR reference and test image
	"""
	# Set color and feature exponents
	qc = 0.7
	qf = 0.5

	# Transform reference and test to opponent color space
	reference = color_space_transform(reference, 'srgb2ycxcz')
	test = color_space_transform(test, 'srgb2ycxcz')

	# --- Color pipeline ---
	# Spatial filtering
	s_a = generate_spatial_filter(pixels_per_degree, 'A')
	s_rg = generate_spatial_filter(pixels_per_degree, 'RG')
	s_by = generate_spatial_filter(pixels_per_degree, 'BY')
	filtered_reference = spatial_filter(reference, s_a, s_rg, s_by)
	filtered_test = spatial_filter(test, s_a, s_rg, s_by)

	# Perceptually Uniform Color Space
	preprocessed_reference = hunt_adjustment(color_space_transform(filtered_reference, 'linrgb2lab'))
	preprocessed_test = hunt_adjustment(color_space_transform(filtered_test, 'linrgb2lab'))

	# Color metric
	deltaE_hyab = hyab(preprocessed_reference, preprocessed_test)
	hunt_adjusted_green = hunt_adjustment(color_space_transform(np.array([[[0.0]], [[1.0]], [[0.0]]]).astype(np.float32), 'linrgb2lab'))
	hunt_adjusted_blue = hunt_adjustment(color_space_transform(np.array([[[0.0]], [[0.0]], [[1.0]]]).astype(np.float32), 'linrgb2lab'))
	cmax = np.power(hyab(hunt_adjusted_green, hunt_adjusted_blue), qc)
	deltaE_c = redistribute_errors(np.power(deltaE_hyab, qc), cmax)

	# --- Feature pipeline ---
	# Extract and normalize achromatic component
	reference_y = (reference[0:1, :, :] + 16) / 116
	test_y = (test[0:1, :, :] + 16) / 116

	# Edge and point detection
	edges_reference = feature_detection(reference_y, pixels_per_degree, 'edge')
	points_reference = feature_detection(reference_y, pixels_per_degree, 'point')
	edges_test = feature_detection(test_y, pixels_per_degree, 'edge')
	points_test = feature_detection(test_y, pixels_per_degree, 'point')

	# Feature metric
	deltaE_f = np.maximum(abs(np.linalg.norm(edges_reference, axis=0) - np.linalg.norm(edges_test, axis=0)),
						  abs(np.linalg.norm(points_test, axis=0) - np.linalg.norm(points_reference, axis=0)))
	deltaE_f = np.power(((1 / np.sqrt(2)) * deltaE_f), qf)

	# --- Final error ---
	return np.power(deltaE_c, 1 - deltaE_f)

##################################################################################################################################################################################################################################################
##################################################################################################################################################################################################################################################
# HDR-FLIP functions
##################################################################################################################################################################################################################################################
##################################################################################################################################################################################################################################################

def compute_exposure_params(reference, tone_mapper="aces", t_max=0.85, t_min=0.85):
	"""
	Computes start and stop exposure for HDR-FLIP based on given tone mapper and reference image.
	Refer to the Visualizing Errors in Rendered High Dynamic Range Images
	paper for details about the formulas

	:param reference: float tensor (with CxHxW layout) containing reference image (nonnegative values)
	:param tone_mapper: (optional) string describing the tone mapper assumed by HDR-FLIP
	:param t_max: (optional) float describing the t value used to find the start exposure
	:param t_max: (optional) float describing the t value used to find the stop exposure
	:return: two floats describing start and stop exposure, respectively, to use for HDR-FLIP
	"""
	if tone_mapper == "reinhard":
		k0 = 0
		k1 = 1
		k2 = 0
		k3 = 0
		k4 = 1
		k5 = 1

		x_max = t_max * k5 / (k1 - t_max * k4)
		x_min = t_min * k5 / (k1 - t_min * k4)
	elif tone_mapper == "hable":
		# Source: https://64.github.io/tonemapping/
		A = 0.15
		B = 0.50
		C = 0.10
		D = 0.20
		E = 0.02
		F = 0.30
		k0 = A * F - A * E
		k1 = C * B * F - B * E
		k2 = 0
		k3 = A * F
		k4 = B * F
		k5 = D * F * F

		W = 11.2
		nom = k0 * np.power(W, 2) + k1 * W + k2
		denom = k3 * np.power(W, 2) + k4 * W + k5
		white_scale = denom / nom # = 1 / (nom / denom)

		# Include white scale and exposure bias in rational polynomial coefficients
		k0 = 4 * k0 * white_scale
		k1 = 2 * k1 * white_scale
		k2 = k2 * white_scale
		k3 = 4 * k3
		k4 = 2 * k4
		#k5 = k5 # k5 is not changed

		c0 = (k1 - k4 * t_max) / (k0 - k3 * t_max)
		c1 = (k2 - k5 * t_max) / (k0 - k3 * t_max)
		x_max = - 0.5 * c0 + np.sqrt(((0.5 * c0) ** 2) - c1)

		c0 = (k1 - k4 * t_min) / (k0 - k3 * t_min)
		c1 = (k2 - k5 * t_min) / (k0 - k3 * t_min)
		x_min = - 0.5 * c0 + np.sqrt(((0.5 * c0) ** 2) - c1)
	else: #tone_mapper == "aces":
		# Source:  ACES approximation: https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
		# Include pre-exposure cancelation in constants
		k0 = 0.6 * 0.6 * 2.51
		k1 = 0.6 * 0.03
		k2 = 0
		k3 = 0.6 * 0.6 * 2.43
		k4 = 0.6 * 0.59
		k5 = 0.14

		c0 = (k1 - k4 * t_max) / (k0 - k3 * t_max)
		c1 = (k2 - k5 * t_max) / (k0 - k3 * t_max)
		x_max = - 0.5 * c0 + np.sqrt(((0.5 * c0) ** 2) - c1)

		c0 = (k1 - k4 * t_min) / (k0 - k3 * t_min)
		c1 = (k2 - k5 * t_min) / (k0 - k3 * t_min)
		x_min = - 0.5 * c0 + np.sqrt(((0.5 * c0) ** 2) - c1)

	# Convert reference to luminance
	lum_coeff_r = 0.2126
	lum_coeff_g = 0.7152
	lum_coeff_b = 0.0722
	Y_reference = reference[0:1, :, :] * lum_coeff_r + reference[1:2, :, :] * lum_coeff_g + reference[2:3, :, :] * lum_coeff_b

	# Compute start exposure
	Y_hi = np.amax(Y_reference)
	if Y_hi == 0:
		return 0, 0
	start_exposure = np.log2(x_max / Y_hi)

	# Compute stop exposure
	Y_lo = np.percentile(Y_reference, 50)
	stop_exposure = np.log2(x_min / Y_lo)

	return start_exposure, stop_exposure

def compute_exposure_map(hdrflip, all_errors, num_exposures):
	"""
	Computes the exposure map corresponding to the HDR-FLIP error map

	:param hdrflip: matrix (with HxW layout) containing per-pixel HDR-FLIP errors
	:param all_errors: tensor of size HxWxN containing LDR-FLIP error maps for different exposures
	:param num_exposures: integer describing the number of exposures used to compute the HDR-FLIP map
	:return: tensor of size HxWx3 in which each pixel describes which exposure yielded the HDR-FLIP error through the viridis color map
	"""
	dim = hdrflip.shape
	exposure_map = np.zeros((dim[0], dim[1], 3))
	viridis_map = get_viridis_map()

	# Decide exposure map color
	for x in range(0, dim[1]):
		for y in range(0, dim[0]):
			pixel_errors = all_errors[y, x, :]
			for i in range(0, num_exposures):
				if pixel_errors[i] == hdrflip[y, x]:
					t = i
					t /= max(num_exposures - 1, 1)
					break

			idx = int(np.round(t * 255.0))
			exposure_map[y, x, :] = viridis_map[idx, :]

	return exposure_map

def compute_hdrflip(reference, test, directory, reference_filename, test_filename, basename, default_basename, pixels_per_degree=(0.7 * 3840 / 0.7) * np.pi / 180, tone_mapper="aces", start_exposure=0.0, stop_exposure=0.0, num_exposures=1, save_ldr_images=False, save_ldrflip=False, no_magma=False):
	"""
	Computes the FLIP error map between two HDR images,
	assuming the images are observed at a certain number of
	pixels per degree of visual angle

	:param reference: reference image (with CxHxW layout on float32 format with nonnegative values)
	:param test: test image (with CxHxW layout on float32 format with nonnegative values)
	:param directory: relative path to directory where results should be saved
	:param reference_filename: string describing basename of reference image
	:param test_filename: string describing basename of test image
	:param basename: string describing basename of output png files
	:param default_basename: bool indicating that the default basename is used
	:param pixels_per_degree: (optional) float describing the number of pixels per degree of visual angle of the observer,
							  default corresponds to viewing the images on a 0.7 meters wide 4K monitor at 0.7 meters from the display
	:param tone_mapper: (optional) string describing what tone mapper HDR-FLIP should assume
	:param start_exposure: (optional) float indicating the shortest exposure HDR-FLIP should use
	:param stop_exposure: (optional) float indicating the longest exposure HDR-FLIP should use
	:param save_ldr_images: (optional) bool indicating if intermediate LDR images used in HDR-FLIP should be saved or not
	:param save_ldrflip: (optional) bool indicating if intermediate LDR-FLIP maps used in HDR-FLIP should be saved or not
	:param no_magma: (optional) bool indicating if FLIP error maps should be saved in grayscale or not
	:return: matrix (with HxW layout on float32 format) containing the per-pixel FLIP errors (in the range [0, 1]) between HDR reference and test image
			 and exposure map in viridis colors (with HxWxC layout)
	"""
	# Find step size
	step_size = (stop_exposure - start_exposure) / max(num_exposures - 1, 1)

	# Perform exposure compensation and tone mapping, and compute LDR-FLIP for each pair of tone mapped image
	dim = reference.shape
	all_errors = np.zeros((dim[1], dim[2], num_exposures)).astype(np.float32)

	# Store sign of start and stop exposures
	stop_exposure_sign = "m" if stop_exposure < 0 else "p"
	start_exposure_sign = "m" if start_exposure < 0 else "p"

	# Loop over exposures in the given exposure range
	for i in range(0, num_exposures):
		exposure = start_exposure + i * step_size
		exposure_sign = "m" if exposure < 0 else "p"

		# Perform exposure compensation and tone mapping, and map to sRGB
		reference_tone_mapped = tone_map(reference, exposure, tone_mapper=tone_mapper)
		test_tone_mapped = tone_map(test, exposure, tone_mapper=tone_mapper)
		reference_srgb = color_space_transform(reference_tone_mapped, "linrgb2srgb")
		test_srgb = color_space_transform(test_tone_mapped, "linrgb2srgb")

		# Compute LDR-FLIP
		deltaE = compute_ldrflip(reference_srgb, test_srgb, pixels_per_degree).squeeze(0)

		# Store result in tensor
		all_errors[:, :, i] = deltaE

		# Save intermediate images and LDR-FLIP maps
		if save_ldr_images:
			if default_basename:
				base = "%s.%s.%s%.4f.png" % (tone_mapper, str(i).zfill(3), exposure_sign, abs(exposure))
				ldr_reference_path = "%s/%s.%s" % (directory, reference_filename, base)
				ldr_test_path = "%s/%s.%s" % (directory, test_filename, base)
			else:
				ldr_reference_path = "%s/%s.reference.%s.png" % (directory, basename, str(i).zfill(3))
				ldr_test_path = "%s/%s.test.%s.png" % (directory, basename, str(i).zfill(3))
			save_image(ldr_reference_path, CHWtoHWC(reference_srgb))
			save_image(ldr_test_path, CHWtoHWC(test_srgb))
		if save_ldrflip:
			if default_basename:
				ldrflip_path = "%s/flip.%s.%s.%dppd.ldr.%s.%s.%s%.4f.png" % (directory, reference_filename, test_filename, pixels_per_degree, tone_mapper, str(i).zfill(3), exposure_sign, abs(exposure))
			else:
				ldrflip_path = "%s/%s.%s.png" % (directory, basename, str(i).zfill(3))
			if no_magma == True:
				error_map = deltaE
			else:
				error_map = CHWtoHWC(index2color(np.round(deltaE * 255.0), get_magma_map()))
			save_image(ldrflip_path, error_map)

	# Final error map and exposure map
	hdrflip = np.max(all_errors, axis=2)
	exposure_map = compute_exposure_map(hdrflip, all_errors, num_exposures)

	return hdrflip, exposure_map