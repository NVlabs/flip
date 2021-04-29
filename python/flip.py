""" FLIP metric tool """
#########################################################################
# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#########################################################################

# Visualizing and Communicating Errors in Rendered Images
# Ray Tracing Gems II, 2021,
# by Pontus Andersson, Jim Nilsson, and Tomas Akenine-Moller.
# Pointer to the article: N/A.

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
import argparse
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator)

from flip_api import *
from data import *

#####################################################################################################################################################################################################################
#####################################################################################################################################################################################################################
# Utility functions
#####################################################################################################################################################################################################################
#####################################################################################################################################################################################################################

def print_pooled_values(error_map, output_textfile, save_dir, test_file_name, verbosity):
	"""
	Prints pooled values of the FLIP error map

	:param error_map: matrix (with HxW layout) containing per-pixel FLIP values in the [0,1] range
	:param output_text: bool describing if output should be written to file or to console
	:param save_dir: string describing relative or absolute path to directory where results should be stored
	:param test_file_name: string describing name of test image
	:param verbosity: (optional) integer describing level of verbosity.
					  0: no printed output,
					  1: print mean FLIP error,
					  2: print pooled FLIP errors and (for HDR-FLIP) start and stop exposure,
					  3: print pooled FLIP errors, warnings, and runtime and (for HDR-FLIP) start and stop exposure and intermediate exposures
	"""
	mean = "%.6f" % np.mean(error_map)
	median = "%.6f" % np.percentile(error_map, 50)
	quartile1 = "%.6f" % np.percentile(error_map, 25)
	quartile3 = "%.6f" % np.percentile(error_map, 75)
	minimum = "%.6f" % np.amin(error_map)
	maximum = "%.6f" % np.amax(error_map)

	if output_textfile == True:
		txt_file_path = save_dir + '/' + test_file_name + '.txt'
		with open(txt_file_path, 'w') as f:
			f.write("Mean: " + mean + "\n")
			f.write("Median: " + median + "\n")
			f.write("1st quartile: " + quartile1 + "\n")
			f.write("3rd quartile: " + quartile3 + "\n")
			f.write("Min: " + minimum + "\n")
			f.write("Max: " + maximum + "\n")
	else:
		if verbosity > 0:
			print("Mean: " + mean)
			if verbosity > 1:
				print("Median: " + median)
				print("1st quartile: " + quartile1)
				print("3rd quartile: " + quartile3)
				print("Min: " + minimum)
				print("Max: " + maximum)

def weighted_flip_histogram(flip_error_map, save_dir, log_scale, pixels_per_degree, y_max):
	"""
	Compute weighted FLIP histogram

	:param flip_error_map: matrix (with HxW layout) containing per-pixel FLIP values in the [0,1] range
	:param save_dir: string describing relative or absolute path to directory where results should be stored
	:param log_scale: bool describing if histogram's y-axis should be in log-scale
	:param pixels_per_degree: float indicating the observer's number of pixels per degree of visual angle
	:param y_max: float indicating largest value on the histogram's y-axis
	"""
	dimensions = (25, 15)  #  histogram image size, centimeters

	lineColor = 'blue'
	fillColor = 'lightblue'
	medianLineColor = 'gray'
	meanLineColor = 'red'
	quartileLineColor = 'purple'
	fontSize = 14

	font = { 'family' : 'serif', 'style' : 'normal', 'weight' : 'normal', 'size' : fontSize }
	lineHeight = fontSize / (dimensions[1] * 15)
	plt.rc('font', **font)
	fig = plt.figure()
	axes = plt.axes()
	axes.xaxis.set_minor_locator(MultipleLocator(0.1))
	axes.xaxis.set_major_locator(MultipleLocator(0.2))

	fig.set_size_inches(dimensions[0] / 2.54, dimensions[1] / 2.54)
	if log_scale == True:
		axes.set(title = 'Weighted \uA7FBLIP Histogram', xlabel = '\uA7FBLIP error', ylabel = 'log(weighted \uA7FBLIP sum per megapixel)')
	else:
		axes.set(title = 'Weighted \uA7FBLIP Histogram', xlabel = '\uA7FBLIP error', ylabel = 'Weighted \uA7FBLIP sum per megapixel')

	counts, bins = np.histogram(flip_error_map, bins=100, range=[0,1])
	bins_weights = bins[:-1] + 0.005
	weighted_hist = (1024 ** 2) * counts * bins_weights / flip_error_map.size
	weighted_hist = weighted_hist if log_scale == False else np.log10(weighted_hist, where = (weighted_hist > 0))

	if y_max is not None:
		y_axis_max = y_max
	else:
		y_axis_max = 1.05 * max(weighted_hist)

	meanValue = np.mean(flip_error_map)
	medianValue = np.percentile(flip_error_map, 50)
	firstQuartileValue = np.percentile(flip_error_map, 25)
	thirdQuartileValue = np.percentile(flip_error_map, 75)
	maxValue = np.amax(flip_error_map)
	minValue = np.amin(flip_error_map)

	plt.hist(bins[:-1], bins=bins, weights = weighted_hist, ec=lineColor, color=fillColor)

	plt.text(0.99, 1.0 - 1 * lineHeight, 'PPD: ' + str(f'{pixels_per_degree:.1f}'), ha = 'right', fontsize = fontSize, transform = axes.transAxes, color='black')

	plt.text(0.99, 1.0 - 2 * lineHeight, 'Mean: ' + str(f'{meanValue:.4f}'), ha = 'right', fontsize = fontSize, transform = axes.transAxes, color=meanLineColor)

	plt.text(0.99, 1.0 - 3 * lineHeight, 'Median: ' + str(f'{medianValue:.4f}'), ha = 'right', fontsize = fontSize, transform = axes.transAxes, color=medianLineColor)

	plt.text(0.99, 1.0 - 4 * lineHeight, '1st quartile: ' + str(f'{firstQuartileValue:.4f}'), ha = 'right', fontsize = fontSize, transform = axes.transAxes, color=quartileLineColor)

	plt.text(0.99, 1.0 - 5 * lineHeight, '3rd quartile: ' + str(f'{thirdQuartileValue:.4f}'), ha = 'right', fontsize = fontSize, transform = axes.transAxes, color=quartileLineColor)

	plt.text(0.99, 1.0 - 6 * lineHeight, 'Min: ' + str(f'{minValue:.4f}'), ha = 'right', fontsize = fontSize, transform = axes.transAxes)

	plt.text(0.99, 1.0 - 7 * lineHeight, 'Max: ' + str(f'{maxValue:.4f}'), ha = 'right', fontsize = fontSize, transform = axes.transAxes)

	axes.set_xlim(0.0, 1.0)
	axes.set_ylim(0.0, y_axis_max)

	axes.axvline(x = meanValue, color = meanLineColor, linewidth = 1.5)

	axes.axvline(x = medianValue, color = medianLineColor, linewidth = 1.5)

	axes.axvline(x = firstQuartileValue, color = quartileLineColor, linewidth = 1.5)

	axes.axvline(x = thirdQuartileValue, color = quartileLineColor, linewidth = 1.5)

	axes.axvline(x = minValue, color='black', linestyle = ':', linewidth = 1.5)

	axes.axvline(x = maxValue, color='black', linestyle = ':', linewidth = 1.5)

	if log_scale == True:
		plt.savefig(save_dir + '/log_weighted_flip_histogram.pdf')
	else:
		plt.savefig(save_dir + '/weighted_flip_histogram.pdf')

def overlapping_weighted_flip_histogram(flip_error_map_array, save_dir, log_scale, pixels_per_degree, y_max, test_images):
	"""
	Compute overlapping weighted FLIP histogram of two error maps

	:param flip_error_map_array: matrix array (with HxWx2 layout) containing per-pixel FLIP values in the [0,1] range for two test images
	:param save_dir: string describing relative path to directory where results should be stored
	:param log_scale: bool describing if histogram's y-axis should be in log-scale
	:param pixels_per_degree: float indicating observer's number of pixels per degree of visual angle
	:param y_max: float indicating largest value on the histogram's y-axis
	:param test_images: string array describing names of the two test images
	"""
	dimensions = (25, 15)  #  histogram image size, centimeters

	lineColor = 'green'
	lineColor2 = 'blue'
	fillColorBelow = [118 / 255, 185 / 255, 0]
	fillColorAbove = 'lightblue' #lightblue
	meanLineColor = [107 / 255, 168 / 255, 0]
	meanLineColor2 = [113 / 255, 171 / 255, 189 / 255]
	fontSize = 14

	# Figure styling
	font = { 'family' : 'serif', 'style' : 'normal', 'weight' : 'normal', 'size' : fontSize }
	lineHeight = fontSize / (dimensions[1] * 15)
	plt.rc('font', **font)

	fig = plt.figure()
	fig.set_size_inches(dimensions[0] / 2.54, dimensions[1] / 2.54)

	meanValue = np.mean(flip_error_map_array[:,:,0])
	meanValue2 = np.mean(flip_error_map_array[:,:,1])

	counts, bins = np.histogram(flip_error_map_array[:,:,0], bins=100, range=[0,1])
	counts2, _ = np.histogram(flip_error_map_array[:,:,1], bins=100, range=[0,1])
	bins_weights = bins[:-1] + 0.005
	weighted_hist = (1024 ** 2) * counts * bins_weights / flip_error_map_array[:,:,0].size
	weighted_hist2 = (1024 ** 2) * counts2 * bins_weights / flip_error_map_array[:,:,0].size
	weighted_hist = weighted_hist if log_scale == False else np.log10(weighted_hist, where = (weighted_hist > 0))
	weighted_hist2 = weighted_hist2 if log_scale == False else np.log10(weighted_hist2, where = (weighted_hist2 > 0))

	if y_max is not None:
		y_axis_max = y_max
	else:
		y_axis_max = 1.05 * max(max(weighted_hist), max(weighted_hist2))

	axes = plt.axes()
	axes.xaxis.set_minor_locator(MultipleLocator(0.1))
	axes.xaxis.set_major_locator(MultipleLocator(0.2))
	axes.set_xlim(0.0, 1.0)
	axes.set_ylim(0.0, y_axis_max)

	plt.hist(bins[:-1], bins=bins, weights = weighted_hist, ec=lineColor, alpha=0.5, color=fillColorBelow)
	plt.hist(bins[:-1], bins=bins, weights = weighted_hist2, ec=lineColor2, alpha=0.5, color=fillColorAbove)

	axes.axvline(x = meanValue, color = meanLineColor, linewidth = 1.5)
	axes.axvline(x = meanValue2, color = meanLineColor2, linewidth = 1.5)

	if log_scale == True:
		axes.set(title = 'Weighted \uA7FBLIP Histogram', xlabel = '\uA7FBLIP error', ylabel = 'log(weighted \uA7FBLIP sum per megapixel)')
	else:
		axes.set(title = 'Weighted \uA7FBLIP Histogram', xlabel = '\uA7FBLIP error', ylabel = 'Weighted \uA7FBLIP sum per megapixel')

	plt.text(0.99, 1.0 - 1 * lineHeight, 'PPD: ' + str(f'{pixels_per_degree:.1f}'), ha = 'right', fontsize = fontSize, transform = axes.transAxes, color='black')

	name_test_image = test_images[0][test_images[0].rfind('/') + 1: test_images[0].rfind('.')]
	name_test_image2 = test_images[1][test_images[1].rfind('/') + 1: test_images[1].rfind('.')]
	plt.text(0.99, 1.0 - 2 * lineHeight, 'Mean (' + name_test_image + '): ' + str(f'{meanValue:.4f}'), ha = 'right', fontsize = fontSize, transform = axes.transAxes, color=meanLineColor)
	plt.text(0.99, 1.0 - 3 * lineHeight, 'Mean (' + name_test_image2 + '): ' + str(f'{meanValue2:.4f}'), ha = 'right', fontsize = fontSize, transform = axes.transAxes, color=meanLineColor2)

	if log_scale == True:
		plt.savefig(save_dir + '/log_weighted_flip_histogram.pdf')
	else:
		plt.savefig(save_dir + '/weighted_flip_histogram.pdf')

def set_pixels_per_degree(args):
	"""
	Determine the observer's number of pixels per degree of visual angle based on input arguments

	:param args: set of command line arguments (see main function)
	:return: float describing the observer's number of pixels per degree of visual angle
	"""
	if args.pixels_per_degree is not None:
		pixels_per_degree = args.pixels_per_degree
	else:
		monitor_distance = args.viewing_conditions[0]
		monitor_width = args.viewing_conditions[1]
		monitor_resolution_x = args.viewing_conditions[2]
		pixels_per_degree = monitor_distance * (monitor_resolution_x / monitor_width) * (np.pi / 180)
	return pixels_per_degree

def check_nans(reference, test, verbosity):
	"""
	Checks reference and test images for NaNs and sets NaNs to 0. Depending on verbosity level, warns if NaNs occur

	:param reference: float tensor
	:param test: float tensor
	:param verbosity: (optional) integer describing level of verbosity.
					  0: no printed output,
					  1: print mean FLIP error,
					  2: print pooled FLIP errors and (for HDR-FLIP) start and stop exposure,
					  3: print pooled FLIP errors, warnings, and runtime and (for HDR-FLIP) start and stop exposure and intermediate exposures
	:return: two float tensors
	"""
	if (np.isnan(reference)).any() or (np.isnan(test)).any():
		reference = np.nan_to_num(reference)
		test = np.nan_to_num(test)
		if verbosity == 3:
			print('=====================================================================')
			print('WARNING: either reference or test (or both) images contain NaNs.')
			print('Those values have been set to 0.')
			print('=====================================================================')
	return reference, test

def check_larger_than_one(value):
	"""
	Checks that value is larger than one. If so, return integer version of it, otherwise raises error

	:param value: float
	:raise: argparseArgumentTypeError if value <= 1
	:return: integer
	"""
	ivalue = int(value)
	if ivalue <= 1:
		raise argparse.ArgumentTypeError("Number of exposures must be greater than or equal to two. You entered %s." % value)
	return ivalue

#####################################################################################################################################################################################################################
#####################################################################################################################################################################################################################
# Main function
#####################################################################################################################################################################################################################
#####################################################################################################################################################################################################################

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description=("Compute FLIP between reference.<png, exr> and test.<png, exr>.\n"
												  "Reference and test(s) must have same layout and resolution.\n"
												  "If pngs are entered, LDR-FLIP will be evaluated. If exrs are entered, HDR-FLIP will be evaluated.\n"
												  "For HDR-FLIP, the reference is used to automatically calculate start and/or stop exposure, if they are not entered."),
									 formatter_class=argparse.RawTextHelpFormatter)
	group = parser.add_mutually_exclusive_group()
	parser.add_argument("-r", "--reference", help="Relative or absolute path (including file name and extension) for reference image", metavar='REFERENCE', type=str, required=True)
	parser.add_argument("-t", "--test", help="Relative or absolute paths (including file names and extensions) for one or more test images", metavar=('TEST1', 'TEST2 TEST3'), type=str, nargs='+', required=True)
	group.add_argument("-ppd", "--pixels_per_degree", help=("Observer's number of pixels per degree of visual angle. Default corresponds to\n"
															"viewing the images on a 0.7 meters wide 4K monitor at 0.7 meters from the display"),
													  type=float)
	group.add_argument("--viewing_conditions", help="Distance to monitor (in meters), width of monitor (in meters), width of monitor (in pixels)",
											   metavar=('MONITOR_DISTANCE', 'MONITOR_WIDTH_METERS', 'MONITOR_WIDTH_PIXELS'), type=float, default=[0.7, 0.7, 3840], nargs=3)
	parser.add_argument("-tm", "--tone_mapper", help="Tone mapper used for HDR-FLIP. Supported tone mappers are ACES, Hable, and Reinhard", metavar='ACES|HABLE|REINHARD', default="aces")
	parser.add_argument("--num_exposures", help="Number of exposures between (and including) start and stop exposure used to compute HDR-FLIP", type=check_larger_than_one)
	parser.add_argument("-cstart", "--start_exposure", help="Start exposure used to compute HDR-FLIP", metavar='C_START', type=float)
	parser.add_argument("-cstop", "--stop_exposure", help="Stop exposure used to compute HDR-FLIP", metavar='C_STOP', type=float)
	parser.add_argument("-v", "--verbosity", help=("Level of verbosity.\n"
													"0: no printed output,\n"
													"1: print mean FLIP error,\n"
													"2: print pooled FLIP errors and (for HDR-FLIP) start and stop exposure\n"
													"3: print pooled FLIP errors, warnings, and runtime and (for HDR-FLIP) start and stop exposure and intermediate exposures"),
											 choices=[0,1,2,3], type=int, default=2)
	parser.add_argument("--save_dir", help="Relative path or absolute path to save directory", metavar='path/to/save_directory', type=str, default='./output')
	parser.add_argument("--histogram", help="Store weighted histogram of the FLIP error map(s). Outputs overlapping histograms if exactly two test images are used", action="store_true")
	parser.add_argument("--y_max", help="Set upper limit of weighted histogram's y-axis", type=int)
	parser.add_argument("--log", help="Take logarithm of weighted histogram", action="store_true")
	parser.add_argument("--output_ldr_images", help="Save all exposure compensated and tone mapped images (png) used for HDR-FLIP", action="store_true")
	parser.add_argument("--output_ldrflip", help="Save all LDR-FLIP images used for HDR-FLIP", action="store_true")
	parser.add_argument("--output_textfile", help="Output text file (with same name as test image) with mean, median, 1st and 3rd quartile as well as minimum and maximum error", action="store_true")
	parser.add_argument("--no_magma", help="Save FLIP error map in grayscale instead of magma", action="store_true")
	parser.add_argument("--no_exposure_map", help="Do not store the HDR-FLIP exposure map", action="store_true")
	parser.add_argument("--no_error_map", help="Do not store the FLIP error map", action="store_true")
	args = parser.parse_args()

	# Create output directory if it doesn't exist and images should be saved
	if not os.path.isdir(args.save_dir) and (not args.no_error_map or not args.no_exposure_map or args.output_ldr_images or args.output_ldrflip or args.output_textfile or args.histogram):
		os.makedirs(args.save_dir)

	# Find out if we have HDR or LDR input and load reference
	image_format = args.reference.split('.')[-1]
	if image_format == "exr" or image_format == "EXR":
		hdr = True
		no_exposure_map = args.no_exposure_map
		reference = HWCtoCHW(read_exr(args.reference))
		tone_mapper = args.tone_mapper.lower()
		assert tone_mapper in ["aces", "hable", "reinhard"]
	elif image_format == "png" or image_format == "PNG":
		hdr = False
		no_exposure_map = True
		reference = load_image_array(args.reference)
	else:
		sys.exit("Error: Invalid image format. Please use png or exr.")

	# Find number of pixels per degree based on input arguments
	pixels_per_degree = set_pixels_per_degree(args)

	# Compute FLIP
	dim = reference.shape
	number_test_images = len(args.test)
	flip_array = np.zeros((dim[1], dim[2], number_test_images)).astype(np.float32)
	for idx, test in enumerate(args.test):
		test_file_name = test[test.rfind('/') + 1: test.rfind('.')]
		save_dir = args.save_dir + "/" + test_file_name

		if not os.path.isdir(save_dir) and (not args.no_error_map or not no_exposure_map or args.output_ldr_images or args.output_ldrflip or args.output_textfile or (args.histogram and (number_test_images != 2))):
			os.makedirs(save_dir)

		# Compute HDR or LDR FLIP depending on type of input
		if hdr == True:
			test = HWCtoCHW(read_exr(test))
			assert reference.shape == test.shape
			if (reference < 0).any() or (test < 0).any():
				reference = np.max(reference, 0.0)
				test = np.max(test, 0.0)
				if args.verbosity == 3:
					print('========================================================================================')
					print('WARNING: either reference or test (or both) images have negative pixel component values.')
					print('HDR-FLIP is defined only nonnegative values. Negative values have been set to 0.')
					print('========================================================================================')
			reference, test = check_nans(reference, test, args.verbosity)

			# Compute HDR-FLIP and exposure map
			flip, exposure_map, start_exposure, stop_exposure = compute_hdrflip(reference,
																				test,
																				save_dir=save_dir,
																				pixels_per_degree=pixels_per_degree,
																				tone_mapper=tone_mapper,
																				start_exp=args.start_exposure,
																				stop_exp=args.stop_exposure,
																				num_exposures=args.num_exposures,
																				output_ldr_images=args.output_ldr_images,
																				output_ldrflip=args.output_ldrflip,
																				verbosity=args.verbosity)

			# Store results
			start_exposure_sign = "m" if start_exposure < 0 else "p"
			stop_exposure_sign = "m" if stop_exposure < 0 else "p"
			error_map_filename = (save_dir + "/hdrflip_" + start_exposure_sign + "%.4f_to_" + stop_exposure_sign + "%.4f.png") % (abs(start_exposure), abs(stop_exposure))
			flip_array[:, :, idx] = flip

			# Save exposure map
			if args.no_exposure_map == False:
				save_image((save_dir + "/exposure_map_" + start_exposure_sign + "%.4f_to_" + stop_exposure_sign + "%.4f.png") % (abs(start_exposure), abs(stop_exposure)), exposure_map)

		else:
			test = load_image_array(test)
			assert reference.shape == test.shape
			if (reference < 0).any() or (reference > 1).any() or (test < 0).any() or (test > 1).any():
				reference = np.clip(reference, 0.0, 1.0)
				test = np.clip(test, 0.0, 1.0)
				if args.verbosity == 3:
					print('=============================================================================================')
					print('WARNING: either reference or test (or both) images have pixel component values outside [0,1].')
					print('LDR-FLIP is defined only for [0, 1]. Values have been clamped.')
					print('=============================================================================================')
			reference, test = check_nans(reference, test, args.verbosity)

			# Compute LDR-FLIP
			flip = compute_ldrflip(reference,
								   test,
								   pixels_per_degree=pixels_per_degree
								   ).squeeze(0)
			error_map_filename = save_dir + "/ldrflip.png"
			flip_array[:, :, idx] = flip

		# Output pooled values
		print_pooled_values(flip, args.output_textfile, save_dir, test_file_name, args.verbosity)

		# Save FLIP map
		if args.no_error_map == False:
			if args.no_magma == True:
				error_map = flip
				save_image(error_map_filename, flip)
			else:
				error_map = CHWtoHWC(index2color(np.floor(255.0 * flip), get_magma_map()))
			save_image(error_map_filename, error_map)

		# Save weighted histogram per test image
		if args.histogram == True and number_test_images != 2:
			weighted_flip_histogram(flip, save_dir, args.log, pixels_per_degree, args.y_max)

	# Overlapping histograms if we have exactly two test images
	if args.histogram == True and number_test_images == 2:
		overlapping_weighted_flip_histogram(flip_array, args.save_dir, args.log, pixels_per_degree, args.y_max, args.test)