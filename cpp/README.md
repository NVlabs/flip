# ꟻLIP: A Tool for Visualizing and Communicating Errors in Rendered Images (v1.4)

By
[Pontus Ebelin](https://research.nvidia.com/person/pontus-ebelin),
and
[Tomas Akenine-Möller](https://research.nvidia.com/person/tomas-akenine-m%C3%B6ller),
with
Jim Nilsson,
[Magnus Oskarsson](https://www1.maths.lth.se/matematiklth/personal/magnuso/),
[Kalle Åström](https://www.maths.lu.se/staff/kalleastrom/),
[Mark D. Fairchild](https://www.rit.edu/directory/mdfpph-mark-fairchild),
and
[Peter Shirley](https://research.nvidia.com/person/peter-shirley).

This [repository](https://github.com/NVlabs/flip) holds implementations of the [LDR-ꟻLIP](https://research.nvidia.com/publication/2020-07_FLIP)
and [HDR-ꟻLIP](https://research.nvidia.com/publication/2021-05_HDR-FLIP) image error metrics in C++ and CUDA.
It also holds code for the ꟻLIP tool, presented in [Ray Tracing Gems II](https://www.realtimerendering.com/raytracinggems/rtg2/index.html).

Note that since v1.2, we use separated convolutions for the C++ and CUDA versions of ꟻLIP. A note explaining those
can be found [here](misc/separatedConvolutions.pdf).

With v1.3, we have switched to a single header [FLIP.h](FLIP.h) for easier integration into other projects.

Since v1.4, the majority of the code for the tool is contained in [FLIPToolHelpers.h](FLIPToolHelpers.h), but the tool is still run through [FLIP-tool.cpp](FLIP-tool.cpp) and [FLIP-tool.cu](FLIP-tool.cu), respectively.


# License

Copyright © 2020-2024, NVIDIA Corporation & Affiliates. All rights reserved.

This work is made available under a [BSD 3-Clause License](../misc/LICENSE.md).

The repository distributes code for `tinyexr`, which is subject to a [BSD 3-Clause License](../misc/LICENSE-third-party.md#bsd-3-clause-license),<br>
and `stb_image`, which is subject to an [MIT License](../misc/LICENSE-third-party.md#mit-license).

For individual contributions to the project, please confer the [Individual Contributor License Agreement](../misc/CLA.md).

For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/).

# C++ and CUDA (API and Tool)
- If you want to use FLIP in your own project, it should suffice to use the header [FLIP.h](FLIP.h). Typical usage would be:
  ```
  #define FLIP_ENABLE_CUDA    // You need to define this if you want to run FLIP using CUDA. Otherwise, comment this out.
  #include "FLIP.h"           // See the bottom of FLIP.h for four different FLIP::evaluate(...) functions that can be used. 

  void someFunction()
  {
      FLIP::evaluate(...);  // See FLIP-tool.cpp for an example of how to use one of these overloaded functions.
  }  
  ```  
- The FLIP.sln solution contains one CUDA backend project and one pure C++ backend project for the FLIP tool.
- Compiling the CUDA project requires a CUDA compatible GPU. Instruction on how to install CUDA can be found [here](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html).
- Alternatively, a CMake build can be done by creating a build directory and invoking CMake on the source `cpp` dir (add `--config Release` to build release configuration on Windows):

  ```
  mkdir build
  cd build
  cmake ..
  cmake --build . [--config Release]
  ```

  CUDA support is enabled via the `FLIP_ENABLE_CUDA`, which can be passed to CMake on the command line with `-DFLIP_ENABLE_CUDA=ON` or set interactively with `ccmake` or `cmake-gui`.
  `FLIP_LIBRARY` option allows to output a library rather than an executable.
- Usage: `flip[-cuda].exe --reference reference.{exr|png} --test test.{exr|png} [options]`, where the list of options can be seen by `flip[-cuda].exe -h`.
- Tested on Windows 10 version 22H2 and Windows 11 version 23H2 with CUDA 12.3. Compiled with Visual Studio 2022. If you use another version of CUDA, you will need to change the `CUDA 12.3` strings in the `CUDA.vcxproj` file accordingly.
- `../tests/test.py` contains simple tests used to test whether code updates alter results.
- Weighted histograms are output as Python scripts. Running the script will create a PDF version of the histogram. Notice that those scripts require `numpy` and `matplotlib`, both of which may be installed using pip. These are automantically installed when installing the Python version of ꟻLIP (see [README.md](https://github.com/NVlabs/flip/blob/main/python/README.md)).
- The naming convention used for the ꟻLIP tool's output is as follows (where `ppd` is the assumed number of pixels per degree,
  `tm` is the tone mapper assumed by HDR-ꟻLIP, `cstart` and `cstop` are the shortest and longest exposures, respectively, assumed by HDR-ꟻLIP,
  with `p` indicating a positive value and `m` indicating a negative value,
  `N` is the number of exposures used in the HDR-ꟻLIP calculation, `nnn` is a counter used to sort the intermediate results,
  and `exp` is the exposure used for the intermediate LDR image / ꟻLIP map):

  **Default:**

  *Low dynamic range images:*<br>

    LDR-ꟻLIP: `flip.<reference>.<test>.<ppd>ppd.ldr.png`<br>
    Weighted histogram: `weighted_histogram.reference>.<test>.<ppd>ppd.ldr.py`<br>
    Overlapping weighted histogram: `overlapping_weighted_histogram.<reference>.<test1>.<test2>.<ppd>ppd.ldr.py`<br>
    Text file: `pooled_values.<reference>.<test>.<ppd>ppd.ldr.txt`<br>

  *High dynamic range images:*<br>

    HDR-ꟻLIP: `flip.<reference>.<test>.<ppd>ppd.hdr.<tm>.<cstart>_to_<cstop>.<N>.png`<br>
    Exposure map: `exposure_map.<reference>.<test>.<ppd>ppd.hdr.<tm>.<cstart>_to_<cstop>.<N>.png`<br>
    Intermediate LDR-ꟻLIP maps: `flip.<reference>.<test>.<ppd>ppd.ldr.<tm>.<nnn>.<exp>.png`<br>
    Intermediate LDR images: `<reference|test>.<tm>.<nnn>.<exp>.png`<br>
    Weighted histogram: `weighted_histogram.<reference>.<test>.<ppd>ppd.hdr.<tm>.<cstart>_to_<cstop>.<N>.py`<br>
    Overlapping weighted histogram: `overlapping_weighted_histogram.<reference>.<test1>.<test2>.<ppd>ppd.hdr.<tm>.<cstart>_to_<cstop>.<N>.py`<br>
    Text file: `pooled_values.<reference>.<test>.<ppd>ppd.hdr.<tm>.<cstart>_to_<cstop>.<N>.txt`<br>

  **With** `--basename <name>` **(note: not applicable if more than one test image is evaluated):**

  *Low dynamic range images:*<br>

    LDR-ꟻLIP: `<name>.png`<br>
    Weighted histogram: `<name>.py`<br>
    Overlapping weighted histogram: N/A<br>
    Text file: `<name>.txt`<br>

  *High dynamic range images:*<br>

    HDR-ꟻLIP: `<name>.png`<br>
    Exposure map: `<name>.exposure_map.png`<br>
    Intermediate LDR-ꟻLIP maps: `<name>.<nnn>.png`<br>
    Intermediate LDR images: `<name>.reference|test.<nnn>.png`<br>
    Weighted histogram: `<name>.py`<br>
    Overlapping weighted histogram: N/A<br>
    Text file: `<name>.txt`<br>

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
        1st weighted quartile: 0.251122
        3rd weighted quartile: 0.434673
        Min: 0.003123
        Max: 0.962022
        Evaluation time: <t> seconds
  ```
where `<t>` is the time it took to evaluate HDR-ꟻLIP. In addition, you will now find the files `flip.reference.test.67ppd.hdr.aces.m12.5423_to_p0.9427.14.png` and `exposure_map.reference.test.67ppd.hdr.aces.m12.5423_to_p0.9427.14.png`
in the directory containing the `flip[-cuda].exe` executable, and we urge you to inspect those, which will reveal where the errors in the test image are located.
