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

import flip
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Print docs for flip.evaluate.
    print("\n############################### DOCS ###############################\n")
    help(flip.evaluate)

    # Compute FLIP for reference and test image.
    ref = "../images/reference.exr"
    test = "../images/test.exr"
    flipErrorMap, meanFLIPError, parameters = flip.evaluate(ref, test, "HDR")

    # NOTE: An alternative to the above is to run flip.evaluate() with numpy arrays as input.
    #       Images may be loaded using flip.load(imagePath).
    # ref = flip.load("../images/reference.exr")
    # test = flip.load("../images/test.exr")
    # flipErrorMap, meanFLIPError, parameters = flip.evaluate(ref, test, "HDR")

    print("\n############################### FLIP OUTPUT ###############################\n")
    print("Mean FLIP error: ", round(meanFLIPError, 6), "\n")

    print("The following parameters were used:")
    for key in parameters:
        val = parameters[key]
        if isinstance(val, float):
            val = round(val, 4)
        print("\t%s: %s" % (key, str(val)))

    plt.imshow(flipErrorMap)
    plt.show()


