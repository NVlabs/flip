""" FLIP metric tool """
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
import argparse
import time
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

def weighted_percentile(error_map, percentile):
	"""
	Computes a weighted percentile of an error map, i.e., the error which is such that the sum of
	the errors in the error map that are smaller than it and those that are larger than it
	are `percentile` percent of total error sum and `100 - percentile` percent of the total error sum, respectively.
	For example, if percentile = 50, the weighted percentile is the error satisfying that the errors in the
	error map that are smaller than it is 50% of the total error sum, and similar for the errors that are larger than it.

	:param error_map: matrix (with HxW layout) containing per-pixel FLIP values in the [0,1] range
	:param percentile: integer in the [0, 100] range describing which percentile is sought
	:return: float containing the weighted percentile
	"""
	error_sum = np.sum(error_map)
	percentile_equilibrium = error_sum * percentile / 100
	error_sorted = np.sort(error_map, axis=None)
	weighted_percentile_index = np.cumsum(error_sorted).searchsorted(percentile_equilibrium)
	weighted_percentile_error = 0.5 * (error_sorted[weighted_percentile_index + 1] + error_sorted[weighted_percentile_index])

	return weighted_percentile_error

def print_initialization_information(pixels_per_degree, hdr, tone_mapper=None, start_exposure=None, stop_exposure=None, num_exposures=None):
	"""
	Prints information about the metric invoked by FLIP

	:param pixels_per_degree: float indicating number of pixels per degree of visual angle
	:param hdr: bool indicating that HDR images are evaluated
	:param tone_mapper: string describing which tone mapper HDR-FLIP assumes
	:param start_exposure: (optional) float indicating the shortest exposure HDR-FLIP uses
	:param stop_exposure: (optional) float indicating the longest exposure HDR-FLIP uses
	:param number_exposures: (optional) integer indicating the number of exposure HDR-FLIP uses
	"""
	print("Invoking " + ("HDR" if hdr else "LDR") + "-FLIP")
	print("\tPixels per degree: %d" % round(pixels_per_degree))
	if hdr == True:
		tone_mapper = tone_mapper.lower()
		if tone_mapper == "hable":
			tm = "Hable"
		elif tone_mapper == "reinhard":
			tm = "Reinhard"
		else:
			tm = "ACES"
		print("\tAssumed tone mapper: %s" % tm)
		print("\tStart exposure: %.4f" % start_exposure)
		print("\tStop exposure: %.4f" % stop_exposure)
		print("\tNumber of exposures: %d" % num_exposures)
	print("")

def print_pooled_values(error_map, hdr, textfile, csvfile, directory, basename, default_basename, reference_filename, test_filename, evaluation_time, verbosity):
	"""
	Prints pooled values of the FLIP error map

	:param error_map: matrix (with HxW layout) containing per-pixel FLIP values in the [0,1] range
	:param output_text: bool describing if output should be written to file or to console
	:param directory: string describing relative or absolute path to directory where results should be saved
	:param basename: string describing basename of output text file
	:param default_basename: bool indicating that the default basename is used
	:param verbosity: (optional) integer describing level of verbosity.
					  0: no printed output,
					  1: print mean FLIP error,
					  "2: print pooled FLIP errors, PPD, and evaluation time and (for HDR-FLIP) start and stop exposure and number of exposures"
					  3: print pooled FLIP errors, PPD, warnings, and evaluation time and (for HDR-FLIP) start and stop exposure and number of exposures
	"""
	mean = "%.6f" % np.mean(error_map)
	weighted_median = "%.6f" % weighted_percentile(error_map, 50)
	weighted_quartile1 = "%.6f" % weighted_percentile(error_map, 25)
	weighted_quartile3 = "%.6f" % weighted_percentile(error_map, 75)
	minimum = "%.6f" % np.amin(error_map)
	maximum = "%.6f" % np.amax(error_map)
	evaluation_time = "%.4f" % evaluation_time

	if textfile == True:
		textfile_path = ("%s/%s%s.txt") % (directory, "pooled_values." if default_basename else "", basename)
		with open(textfile_path, 'w') as f:
			f.write("Mean: %s\n" % mean)
			f.write("Weighted median: %s\n" % weighted_median)
			f.write("1st weighted quartile: %s\n" % weighted_quartile1)
			f.write("3rd weighted quartile: %s\n" % weighted_quartile3)
			f.write("Min: %s\n" % minimum)
			f.write("Max: %s\n" % maximum)

	if csvfile is not None:
		csv_path = ("%s/%s") % (directory, csvfile)
		with open(csv_path, 'a') as f:
			if os.path.getsize(csv_path) == 0:
				f.write("\"Reference\",\"Test\",\"Mean\",\"Weighted median\",\"1st weighted quartile\",\"3rd weighted quartile\",\"Min\",\"Max\",\"Evaluation time\"\n")
			s  = "\"%s\"," % reference_filename
			s += "\"%s\"," % test_filename
			s += "\"%s\"," % mean
			s += "\"%s\"," % weighted_median
			s += "\"%s\"," % weighted_quartile1
			s += "\"%s\"," % weighted_quartile3
			s += "\"%s\"," % minimum
			s += "\"%s\"," % maximum
			s += "\"%s\"\n" % evaluation_time
			f.write(s)

	if verbosity > 0:
		print("\tMean: %s" % mean)
		if verbosity > 1:
			print("\tWeighted median: %s" % weighted_median)
			print("\t1st weighted quartile: %s" % weighted_quartile1)
			print("\t3rd weighted quartile: %s" % weighted_quartile3)
			print("\tMin: %s" % minimum)
			print("\tMax: %s" % maximum)

def weighted_flip_histogram(flip_error_map, directory, basename, default_basename, log_scale, y_max, exclude_pooled_values):
	"""
	Compute weighted FLIP histogram

	:param flip_error_map: matrix (with HxW layout) containing per-pixel FLIP values in the [0,1] range
	:param directory: string describing relative or absolute path to directory where results should be saved
	:param basename: string describing basename of output PDF file
	:param default_basename: bool indicating that the default basename is used
	:param log_scale: bool describing if histogram's y-axis should be in log-scale
	:param y_max: float indicating largest value on the histogram's y-axis
	:param exclude_pooled_values: bool indicating whether pooled FLIP values should be excluded in the weighted histogram
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
	weightedMedianValue = weighted_percentile(flip_error_map, 50)
	firstWeightedQuartileValue = weighted_percentile(flip_error_map, 25)
	thirdWeightedQuartileValue = weighted_percentile(flip_error_map, 75)
	maxValue = np.amax(flip_error_map)
	minValue = np.amin(flip_error_map)

	plt.hist(bins[:-1], bins=bins, weights = weighted_hist, ec=lineColor, color=fillColor)

	axes.set_xlim(0.0, 1.0)
	axes.set_ylim(0.0, y_axis_max)

	if exclude_pooled_values == False:
		plt.text(0.99, 1.0 - 1 * lineHeight, 'Mean: ' + str(f'{meanValue:.4f}'), ha = 'right', fontsize = fontSize, transform = axes.transAxes, color=meanLineColor)

		plt.text(0.99, 1.0 - 2 * lineHeight, 'Weighted median: ' + str(f'{weightedMedianValue:.4f}'), ha = 'right', fontsize = fontSize, transform = axes.transAxes, color=medianLineColor)

		plt.text(0.99, 1.0 - 3 * lineHeight, '1st weighted quartile: ' + str(f'{firstWeightedQuartileValue:.4f}'), ha = 'right', fontsize = fontSize, transform = axes.transAxes, color=quartileLineColor)

		plt.text(0.99, 1.0 - 4 * lineHeight, '3rd weighted quartile: ' + str(f'{thirdWeightedQuartileValue:.4f}'), ha = 'right', fontsize = fontSize, transform = axes.transAxes, color=quartileLineColor)

		plt.text(0.99, 1.0 - 5 * lineHeight, 'Min: ' + str(f'{minValue:.4f}'), ha = 'right', fontsize = fontSize, transform = axes.transAxes)

		plt.text(0.99, 1.0 - 6 * lineHeight, 'Max: ' + str(f'{maxValue:.4f}'), ha = 'right', fontsize = fontSize, transform = axes.transAxes)

		axes.axvline(x = meanValue, color = meanLineColor, linewidth = 1.5)

		axes.axvline(x = weightedMedianValue, color = medianLineColor, linewidth = 1.5)

		axes.axvline(x = firstWeightedQuartileValue, color = quartileLineColor, linewidth = 1.5)

		axes.axvline(x = thirdWeightedQuartileValue, color = quartileLineColor, linewidth = 1.5)

		axes.axvline(x = minValue, color='black', linestyle = ':', linewidth = 1.5)

		axes.axvline(x = maxValue, color='black', linestyle = ':', linewidth = 1.5)

	if log_scale == True:
		weighted_histogram_path = ("%s/%s%s.pdf") % (directory, "log_weighted_histogram." if default_basename else "", basename)
	else:
		weighted_histogram_path = ("%s/%s%s.pdf") % (directory, "weighted_histogram." if default_basename else "", basename)
	plt.savefig(weighted_histogram_path)

def overlapping_weighted_flip_histogram(flip_error_map_array, directory, basename, log_scale, y_max, test_images, exclude_pooled_values):
	"""
	Compute overlapping weighted FLIP histogram of two error maps

	:param flip_error_map_array: matrix array (with HxWx2 layout) containing per-pixel FLIP values in the [0,1] range for two test images
	:param directory: string describing relative path to directory where results should be saved
	:param basename: string describing basename of output PDF file
	:param log_scale: bool describing if histogram's y-axis should be in log-scale
	:param y_max: float indicating largest value on the histogram's y-axis
	:param test_images: string array describing names of the two test images
	:param exclude_pooled_values: bool indicating whether pooled FLIP values should be excluded in the weighted histogram
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

	if log_scale == True:
		axes.set(title = 'Weighted \uA7FBLIP Histogram', xlabel = '\uA7FBLIP error', ylabel = 'log(weighted \uA7FBLIP sum per megapixel)')
	else:
		axes.set(title = 'Weighted \uA7FBLIP Histogram', xlabel = '\uA7FBLIP error', ylabel = 'Weighted \uA7FBLIP sum per megapixel')

	name_test_image = test_images[0][test_images[0].rfind('/') + 1: test_images[0].rfind('.')]
	name_test_image2 = test_images[1][test_images[1].rfind('/') + 1: test_images[1].rfind('.')]
	if exclude_pooled_values == False:
		axes.axvline(x = meanValue, color = meanLineColor, linewidth = 1.5)
		axes.axvline(x = meanValue2, color = meanLineColor2, linewidth = 1.5)
		plt.text(0.99, 1.0 - 1 * lineHeight, 'Mean (' + name_test_image + '): ' + str(f'{meanValue:.4f}'), ha = 'right', fontsize = fontSize, transform = axes.transAxes, color=meanLineColor)
		plt.text(0.99, 1.0 - 2 * lineHeight, 'Mean (' + name_test_image2 + '): ' + str(f'{meanValue2:.4f}'), ha = 'right', fontsize = fontSize, transform = axes.transAxes, color=meanLineColor2)

	basename_split = basename.split('.')
	appended_basename = "%s.%s.%s.%s" % (basename_split[0], name_test_image, name_test_image2, '.'.join(basename_split[2:]))
	overlapping_weighted_histogram_path = ("%s/%soverlapping_weighted_histogram.%s.pdf") % (directory, "log_" if log_scale else "", appended_basename)
	plt.savefig(overlapping_weighted_histogram_path)

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

def set_start_stop_num_exposures(reference, start_exp=None, stop_exp=None, num_exposures=None, tone_mapper="aces"):
	# Set start and stop exposures
	if start_exp == None or stop_exp == None:
		start_exposure, stop_exposure = compute_exposure_params(reference, tone_mapper=tone_mapper)
		if start_exp is not None: start_exposure = start_exp
		if stop_exp is not None: stop_exposure = stop_exp
	else:
		start_exposure = start_exp
		stop_exposure = stop_exp
	assert start_exposure <= stop_exposure

	# Set number of exposures
	if start_exposure == stop_exposure:
		num_exposures = 1
	elif num_exposures is None:
		num_exposures = int(max(2, np.ceil(stop_exposure - start_exposure)))
	else:
		num_exposures = num_exposures

	return start_exposure, stop_exposure, num_exposures

def check_nans(reference, test, verbosity):
	"""
	Checks reference and test images for NaNs and sets NaNs to 0. Depending on verbosity level, warns if NaNs occur

	:param reference: float tensor
	:param test: float tensor
	:param verbosity: (optional) integer describing level of verbosity.
					  0: no printed output,
					  1: print mean FLIP error,
					  "2: print pooled FLIP errors, PPD, and evaluation time and (for HDR-FLIP) start and stop exposure and number of exposures"
					  3: print pooled FLIP errors, PPD, warnings, and evaluation time and (for HDR-FLIP) start and stop exposure and number of exposures
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
	major_version = 1
	minor_version = 2
	parser = argparse.ArgumentParser(description=("Compute FLIP between reference.<png|exr> and test.<png|exr>.\n"
												  "Reference and test(s) must have same resolution and format.\n"
												  "If pngs are entered, LDR-FLIP will be evaluated. If exrs are entered, HDR-FLIP will be evaluated.\n"
												  "For HDR-FLIP, the reference is used to automatically calculate start exposure and/or stop exposure and/or number of exposures, if they are not entered."),
									 formatter_class=argparse.RawTextHelpFormatter)
	group = parser.add_mutually_exclusive_group()
	parser.add_argument("-r", "--reference", help="Relative or absolute path (including file name and extension) for reference image", metavar='REFERENCE', type=str, required=True)
	parser.add_argument("-t", "--test", help="Relative or absolute paths (including file names and extensions) for one or more test images", metavar=('TEST1', 'TEST2 TEST3'), type=str, nargs='+', required=True)
	group.add_argument("-ppd", "--pixels_per_degree", help=("Observer's number of pixels per degree of visual angle. Default corresponds to\n"
															"viewing the images at 0.7 meters from a 0.7 meters wide 4K display"),
													  type=float)
	group.add_argument("-vc", "--viewing_conditions", help=("Distance to monitor (in meters), width of monitor (in meters), width of monitor (in pixels).\n"
															"Default corresponds to viewing the at 0.7 meters from a 0.7 meters wide 4K display"),
													  metavar=('MONITOR_DISTANCE', 'MONITOR_WIDTH_METERS', 'MONITOR_WIDTH_PIXELS'), type=float, default=[0.7, 0.7, 3840], nargs=3)
	parser.add_argument("-tm", "--tone_mapper", help="Tone mapper used for HDR-FLIP. Supported tone mappers are ACES, Hable, and Reinhard (default: %(default)s)", metavar='ACES|HABLE|REINHARD', default="ACES")
	parser.add_argument("-n", "--num_exposures", help="Number of exposures between (and including) start and stop exposure used to compute HDR-FLIP", type=check_larger_than_one)
	parser.add_argument("-cstart", "--start_exposure", help="Start exposure used to compute HDR-FLIP", metavar='C_START', type=float)
	parser.add_argument("-cstop", "--stop_exposure", help="Stop exposure used to compute HDR-FLIP", metavar='C_STOP', type=float)
	parser.add_argument("-v", "--verbosity", help=("Level of verbosity (default: %(default)s).\n"
													"0: no printed output,\n"
													"1: print mean FLIP error,\n"
													"2: print pooled FLIP errors, PPD, and evaluation time and (for HDR-FLIP) start and stop exposure and number of exposures\n"
													"3: print pooled FLIP errors, PPD, warnings, and evaluation time and (for HDR-FLIP) start and stop exposure and number of exposures\n"),
											 choices=[0,1,2,3], type=int, default=2)
	parser.add_argument("-d", "--directory", help="Relative or absolute path to save directory", metavar='path/to/save_directory', type=str, default='./')
	parser.add_argument("-b", "--basename", help="Basename used for output files", type=str)
	parser.add_argument("-txt", "--textfile", help="Save text file with pooled FLIP values (mean, weighted median and weighted 1st and 3rd quartiles as well as minimum and maximum error)", action="store_true")
	parser.add_argument("-c", "--csv", metavar='CSV_FILENAME', help="Write results to a csv file. Input is the desired file name (including .csv extension).\nResults are appended if the file already exists", type=str)
	parser.add_argument("-hist", "--histogram", help="Save weighted histogram of the FLIP error map(s). Outputs overlapping histograms if exactly two test images are evaluated", action="store_true")
	parser.add_argument("--y_max", help="Set upper limit of weighted histogram's y-axis", type=int)
	parser.add_argument("-lg", "--log", help="Take logarithm of weighted histogram", action="store_true")
	parser.add_argument("-epv", "--exclude_pooled_values", help="Do not include pooled FLIP values in the weighted histogram", action="store_true")
	parser.add_argument("-sli", "--save_ldr_images", help="Save all exposure compensated and tone mapped LDR images (png) used for HDR-FLIP", action="store_true")
	parser.add_argument("-slf", "--save_ldrflip", help="Save all LDR-FLIP maps used for HDR-FLIP", action="store_true")
	parser.add_argument("-nm", "--no_magma", help="Save FLIP error maps in grayscale instead of magma", action="store_true")
	parser.add_argument("-nexm", "--no_exposure_map", help="Do not save the HDR-FLIP exposure map", action="store_true")
	parser.add_argument("-nerm", "--no_error_map", help="Do not save the FLIP error map", action="store_true")

	# Print help string if flip.py is run without arguments
	if len(sys.argv) == 1:
		print("FLIP v%d.%d." % (major_version, minor_version))
		parser.print_help()
		sys.exit(1)

	args = parser.parse_args()

	# Check if there is risk of overwriting output because of same basename
	num_test_images = len(args.test)
	if args.basename is not None and num_test_images > 1:
		sys.exit("Error: Basename is the same for all test images. Results will overwrite each other. Change to multiple runs with different --directory for each test image.")

	# Create output directory if it doesn't exist and images or other files should be saved
	if not os.path.isdir(args.directory) and (not args.no_error_map or not args.no_exposure_map or args.save_ldr_images or args.save_ldrflip or args.textfile or args.csv or args.histogram):
		os.makedirs(args.directory)

	# Replace \ with / in reference and test paths
	args.reference = "".join(args.reference).replace("\\", "/")
	args.test = ["".join(test_name).replace("\\", "/") for test_name in args.test]

	# Find out if we have HDR or LDR input and load reference
	image_format = args.reference.split('.')[-1]
	reference_filename = args.reference[args.reference.rfind('/') + 1: args.reference.rfind('.')]
	if image_format == "exr" or image_format == "EXR":
		hdr = True
		tone_mapper = args.tone_mapper.lower()
		assert tone_mapper in ["aces", "hable", "reinhard"]
		no_exposure_map = args.no_exposure_map
		reference = HWCtoCHW(read_exr(args.reference))
	elif image_format == "png" or image_format == "PNG":
		hdr = False
		tone_mapper = None
		start_exposure = None
		stop_exposure = None
		num_exposures = None
		no_exposure_map = True
		reference = load_image_array(args.reference)
	else:
		sys.exit("Error: Invalid image format. Please use png or exr.")

	# Find number of pixels per degree based on input arguments
	pixels_per_degree = set_pixels_per_degree(args)

	if hdr == True:
		# Set start and stop exposures as well as number of exposures to be used by HDR-FLIP
		start_exposure, stop_exposure, num_exposures = set_start_stop_num_exposures(reference, start_exp=args.start_exposure, stop_exp=args.stop_exposure, num_exposures=args.num_exposures, tone_mapper=tone_mapper)
		start_exposure_sign = "m" if start_exposure < 0 else "p"
		stop_exposure_sign = "m" if stop_exposure < 0 else "p"

	# Print information about the metric to be invoked by FLIP:
	if args.verbosity > 1: print_initialization_information(pixels_per_degree, hdr, tone_mapper=tone_mapper, start_exposure=start_exposure, stop_exposure=stop_exposure, num_exposures=num_exposures)

	# Compute FLIP
	dim = reference.shape
	flip_array = np.zeros((dim[1], dim[2], num_test_images)).astype(np.float32)
	for idx, test_path in enumerate(args.test):
		test_filename = test_path[test_path.rfind('/') + 1: test_path.rfind('.')]

		# Set basename of output files
		if args.basename is not None:
			basename = args.basename
			default_basename = False
		elif hdr == True:
			basename = "%s.%s.%dppd.hdr.%s.%s%.4f_to_%s%.4f.%d" % (reference_filename, test_filename, pixels_per_degree, tone_mapper, start_exposure_sign, abs(start_exposure), stop_exposure_sign, abs(stop_exposure), num_exposures)
			default_basename = True
		else:
			basename = ("%s.%s.%dppd.ldr") % (reference_filename, test_filename, pixels_per_degree)
			default_basename = True

		# Compute HDR or LDR FLIP depending on type of input
		if hdr == True:
			test = HWCtoCHW(read_exr(test_path))
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
			t0 = time.time()
			flip, exposure_map = compute_hdrflip(reference,
												 test,
												 directory=args.directory,
												 reference_filename=reference_filename,
												 test_filename=test_filename,
												 basename=basename,
												 default_basename=default_basename,
												 pixels_per_degree=pixels_per_degree,
												 tone_mapper=tone_mapper,
												 start_exposure=start_exposure,
												 stop_exposure=stop_exposure,
												 num_exposures=num_exposures,
												 save_ldr_images=args.save_ldr_images,
												 save_ldrflip=args.save_ldrflip,
												 no_magma=args.no_magma)
			t = time.time()

			# Store results
			flip_array[:, :, idx] = flip

			# Save exposure map
			if args.no_exposure_map == False:
				exposure_map_path = ("%s/%s.png") % (args.directory, "exposure_map." + basename if default_basename else basename + ".exposure_map")
				save_image(exposure_map_path, exposure_map)

		else:
			test = load_image_array(test_path)
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
			t0 = time.time()
			flip = compute_ldrflip(reference,
								   test,
								   pixels_per_degree=pixels_per_degree
								   ).squeeze(0)
			t = time.time()
			flip_array[:, :, idx] = flip

		if args.verbosity > 0:
			print("FLIP between reference image <%s.%s> and test image <%s.%s>:" % (reference_filename, image_format, test_filename, image_format))

		# Output pooled values
		evaluation_time = t - t0
		print_pooled_values(flip, hdr, args.textfile, args.csv, args.directory, basename, default_basename, args.reference, test_path, evaluation_time, args.verbosity)

		# Print time spent computing FLIP
		if args.verbosity > 1: print(("\tEvaluation time: %.4f seconds") % evaluation_time)
		if (args.verbosity > 0 and idx < (num_test_images - 1)): print("")

		# Save FLIP map
		error_map_path = ("%s/%s%s.png") % (args.directory, "flip." if default_basename else "", basename)
		if args.no_error_map == False:
			if args.no_magma == True:
				error_map = flip
			else:
				error_map = CHWtoHWC(index2color(np.round(flip * 255.0), get_magma_map()))
			save_image(error_map_path, error_map)

		# Save weighted histogram per test image
		if args.histogram == True and num_test_images != 2:
			weighted_flip_histogram(flip, args.directory, basename, default_basename, args.log, args.y_max, args.exclude_pooled_values)

	# Overlapping histograms if we have exactly two test images
	if args.histogram == True and num_test_images == 2:
		overlapping_weighted_flip_histogram(flip_array, args.directory, basename, args.log, args.y_max, args.test, args.exclude_pooled_values)