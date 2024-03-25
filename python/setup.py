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

from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup
import os
import sys

__version__ = "1.4"

# Separate compiler options for Windows.
extra_compile_args = ["-DNDEBUG"]
if sys.platform.startswith("win"):
    extra_compile_args = ["-openmp"]
    extra_link_args = []
# Use OpenMP if environment variable is set or not on a Mac.
elif os.environ.get("USEOPENMP") or not sys.platform.startswith("darwin"):
    extra_compile_args = ["-fopenmp"]
    extra_link_args = ["-lgomp"]

ext_modules = [
    Pybind11Extension(
        "pbflip", # Name of pybind11 module.
        ["flip/main.cpp"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        cxx_std=20
    ),
]

# Get description from README.
with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Run setup.
setup(
    name="flip",
    version=__version__,
    author="NVIDIA",
    author_email="pandersson@nvidia.com",
    description="A Difference Evaluator for Alternating Images",
    url="https://github.com/nvlabs/flip",
    license="BSD",
    long_description=long_description,
    long_description_content_type='text/markdown',
    ext_modules=ext_modules,
    python_requires=">=3.7"
)