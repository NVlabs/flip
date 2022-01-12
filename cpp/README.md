# ꟻLIP: A Tool for Visualizing and Communicating Errors in Rendered Images
By
[Pontus Andersson](https://research.nvidia.com/person/pontus-andersson),
[Jim Nilsson](https://research.nvidia.com/person/jim-nilsson),
and
[Tomas Akenine-Möller](https://research.nvidia.com/person/tomas-akenine-m%C3%B6ller),
with
[Magnus Oskarsson](https://www1.maths.lth.se/matematiklth/personal/magnuso/),
[Kalle Åström](https://www.maths.lu.se/staff/kalleastrom/),
[Mark D. Fairchild](https://www.rit.edu/directory/mdfpph-mark-fairchild),
and
[Peter Shirley](https://research.nvidia.com/person/peter-shirley).

This [repository](https://github.com/NVlabs/flip) holds implementations of the [LDR-ꟻLIP](https://research.nvidia.com/publication/2020-07_FLIP)
and [HDR-ꟻLIP](https://research.nvidia.com/publication/2021-05_HDR-FLIP) image error metrics in C++ and CUDA.
It also holds code for the ꟻLIP tool, presented in [Ray Tracing Gems II](https://www.realtimerendering.com/raytracinggems/rtg2/index.html).

# License

Copyright © 2020-2022, NVIDIA Corporation & Affiliates. All rights reserved.

This work is made available under a [BSD 3-Clause License](../LICENSE.md).

The repository distributes code for `tinyexr`, which is subject to a [BSD 3-Clause License](../LICENSE-third-party.md#bsd-3-clause-license),<br>
and `stb_image`, which is subject to an [MIT License](../LICENSE-third-party.md#mit-license).

For individual contributions to the project, please confer the [Individual Contributor License Agreement](../CLA.md).

For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/).

# C++ and CUDA (API and Tool)
- The FLIP.sln solution contains one CUDA backend project and one pure C++ backend project.
- Compiling the CUDA project requires a CUDA compatible GPU. Instruction on how to install CUDA can be found [here](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html).
- Alternatively, a CMake build can be done by creating a build directory and invoking CMake on the source `cpp` dir:

  ```
  mkdir build
  cd build
  cmake ../cpp
  cmake --build .
  ```

  CUDA support is enabled via the `FLIP_ENABLE_CUDA`, which can be passed to CMake on the command line with `-DFLIP_ENABLE_CUDA=ON` or set interactively with `ccmake` or `cmake-gui`.
- Usage: `flip[-cuda].exe --reference reference.{exr|png} --test test.{exr|png} [options]`, where the list of options can be seen by `flip[-cuda].exe -h`.
- Tested on Windows 10 version 20H2 with CUDA 11.5. Compiled with Visual Studio 2019. If you use another version of CUDA, you will need to change the `CUDA 11.5` strings in the `CUDA.vcxproj` file accordingly.
- `../tests/test_{cpp|cuda}.py` contains simple tests used to test whether code updates alter results.
- Weighted histograms are output as Python scripts. Running the script will create a PDF version of the histogram.
- The naming convention used for the ꟻLIP tool's output is as follows (where `ppd` is the assumed number of pixels per degree,
  `tm` is the tone mapper assumed by HDR-ꟻLIP, `cstart` and `cstop` are the shortest and longest exposures, respectively, assumed by HDR-ꟻLIP,
  with `p` indicating a positive value and `m` indicating a negative value,
  `N` is the number of exposures used in the HDR-ꟻLIP calculation, `nnn` is a counter used to sort the intermediate results,
  and `exp` is the exposure used for the intermediate LDR image / ꟻLIP map):

  **Default:**

  *Low dynamic range images:*<br>

    LDR-ꟻLIP: `flip.<reference>.<test>.<ppd>ppd.ldr.png`<br>
    Weighted histogram: `weighted_histogram.reference>.<test>.<ppd>ppd.ldr.py`<br>

  *High dynamic range images:*<br>

    HDR-ꟻLIP: `flip.<reference>.<test>.<ppd>ppd.hdr.<tm>.<cstart>_to_<cstop>.<N>.png`<br>
    Exposure map: `exposure_map.<reference>.<test>.<ppd>ppd.hdr.<tm>.<cstart>_to_<cstop>.<N>.png`<br>
    Intermediate LDR-ꟻLIP maps: `flip.<reference>.<test>.<ppd>ppd.ldr.<tm>.<nnn>.<exp>.png`<br>
    Intermediate LDR images: `<reference|test>.<tm>.<nnn>.<exp>.png`<br>
    Weighted histogram: `weighted_histogram.<reference>.<test>.<ppd>ppd.hdr.<tm>.<cstart>_to_<cstop>.<N>.py`<br>

  **With** `--basename <name>` **(note: not applicable if more than one test image is evaluated):**

  *Low dynamic range images:*<br>
    
    LDR-ꟻLIP: `<name>.png`<br>
    Weighted histogram: `<name>.py`<br>

  *High dynamic range images:*<br>
    
    HDR-ꟻLIP: `<name>.png`<br>
    Exposure map: `<name>.exposure_map.png`<br>
    Intermediate LDR-ꟻLIP maps: `<name>.<nnn>.png`<br>
    Intermediate LDR images: `<name>.reference|test.<nnn>.png`<br>
    Weighted histogram: `<name>.py`<br>
    
 **Example usage:**
After compiling the `FLIP.sln` project, navigate to the `flip[-cuda].exe` executable and try:
  ```
  flip[-cuda].exe -r ../../../images/reference.exr -t ../../../images/test.exr
  ```
Assuming using the images in the source bundle, the result should be:
  ```
Invoking HDR-FLIP
        Pixels per degree: 67
        Assumed tone mapper: ACES
        Start exposure: -12.5423
        Stop exposure: 0.9427
        Number of exposures: 14

FLIP between reference image <reference.exr> and test image <test.exr>:
        Mean: 0.283478
        Weighted median: 0.339430
        1st weighted quartile: 0.251123
        3rd weighted quartile: 0.434673
        Min: 0.003118
        Max: 0.962022
        Evaluation time: <t> seconds 
  ```
where `<t>` is the time it took to evaluate HDR-ꟻLIP. In addition, you will now find the files `flip.reference.test.67ppd.hdr.aces.m12.5423_to_p0.9427.14.png` and `exposure_map.reference.test.67ppd.hdr.aces.m12.5423_to_p0.9427.14.png`
in the directory containing the `flip[-cuda].exe` executable, and we urge you to inspect those, which will reveal where the errors in the test image are located.
