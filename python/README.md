# ꟻLIP: A Tool for Visualizing and Communicating Errors in Rendered Images (v1.3)
By
[Pontus Ebelin](https://research.nvidia.com/person/pontus-ebelin),
Jim Nilsson,
and
[Tomas Akenine-Möller](https://research.nvidia.com/person/tomas-akenine-m%C3%B6ller),
with
[Magnus Oskarsson](https://www1.maths.lth.se/matematiklth/personal/magnuso/),
[Kalle Åström](https://www.maths.lu.se/staff/kalleastrom/),
[Mark D. Fairchild](https://www.rit.edu/directory/mdfpph-mark-fairchild),
and
[Peter Shirley](https://research.nvidia.com/person/peter-shirley).

This [repository](https://github.com/NVlabs/flip) holds implementations of the [LDR-ꟻLIP](https://research.nvidia.com/publication/2020-07_FLIP)
and [HDR-ꟻLIP](https://research.nvidia.com/publication/2021-05_HDR-FLIP) image error metrics in Python.
It also holds code for the ꟻLIP tool, presented in [Ray Tracing Gems II](https://www.realtimerendering.com/raytracinggems/rtg2/index.html).

# License

Copyright © 2020-2024, NVIDIA Corporation & Affiliates. All rights reserved.

This work is made available under a [BSD 3-Clause License](../misc/LICENSE.md).

The repository distributes code for `tinyexr`, which is subject to a [BSD 3-Clause License](../misc/LICENSE-third-party.md#bsd-3-clause-license),<br>
and `stb_image`, which is subject to an [MIT License](../misc/LICENSE-third-party.md#mit-license).

For individual contributions to the project, please confer the [Individual Contributor License Agreement](../misc/CLA.md).

For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/).

# Python (API and Tool)
- **Setup** (with Anaconda3):
  ```
  conda create -n flip python numpy matplotlib
  conda activate flip
  conda install -c conda-forge opencv
  conda install -c conda-forge openexr-python
  ```
- Remember to activate the `flip` environment through `conda activate flip` before using the tool.
- Usage: `python flip.py --reference reference.{exr|png} --test test.{exr|png} [--options]`, where the list of options can be seen by `python flip.py -h`.
- Tested with Conda 4.10.0, Python 3.8.3, NumPy 1.19.0, OpenCV 4.0.1, and OpenEXR b1.3.2.
- The ꟻLIP tool is provided in `flip.py`, which also contains several tool-specific utility functions.
  The API is provided in `flip-api.py` and image loading/saving/manipulation functions in `data.py`.
  `../tests/test.py` contains simple tests used to test whether code updates alter results.
- The naming convention used for the ꟻLIP tool's output is as follows (where `ppd` is the assumed number of pixels per degree,
  `tm` is the tone mapper assumed by HDR-ꟻLIP, `cstart` and `cstop` are the shortest and longest exposures, respectively, assumed by HDR-ꟻLIP,
  with `p` indicating a positive value and `m` indicating a negative value,
  `N` is the number of exposures used in the HDR-ꟻLIP calculation, `nnn` is a counter used to sort the intermediate results,
  and `exp` is the exposure used for the intermediate LDR image / ꟻLIP map):

  **Default:**

  *Low dynamic range images:*<br>

    LDR-ꟻLIP: `flip.<reference>.<test>.<ppd>ppd.ldr.png`<br>
    Weighted histogram: `weighted_histogram.reference>.<test>.<ppd>ppd.ldr.pdf`<br>
    Overlapping weighted histogram: `overlapping_weighted_histogram.<reference>.<test1>.<test2>.<ppd>ppd.ldr.pdf`<br>
    Text file: `pooled_values.<reference>.<test>.<ppd>ppd.ldr.txt`<br>

  *High dynamic range images:*<br>

    HDR-ꟻLIP: `flip.<reference>.<test>.<ppd>ppd.hdr.<tm>.<cstart>_to_<cstop>.<N>.png`<br>
    Exposure map: `exposure_map.<reference>.<test>.<ppd>ppd.hdr.<tm>.<cstart>_to_<cstop>.<N>.png`<br>
    Intermediate LDR-ꟻLIP maps: `flip.<reference>.<test>.<ppd>ppd.ldr.<tm>.<nnn>.<exp>.png`<br>
    Intermediate LDR images: `<reference|test>.<tm>.<nnn>.<exp>.png`<br>
    Weighted histogram: `weighted_histogram.<reference>.<test>.<ppd>ppd.hdr.<tm>.<cstart>_to_<cstop>.<N>.pdf`<br>
    Overlapping weighted histogram: `overlapping_weighted_histogram.<reference>.<test1>.<test2>.<ppd>ppd.hdr.<tm>.<cstart>_to_<cstop>.<N>.pdf`<br>
    Text file: `pooled_values.<reference>.<test>.<ppd>ppd.hdr.<tm>.<cstart>_to_<cstop>.<N>.txt`<br>

  **With** `--basename <name>` **(note: not applicable if more than one test image is evaluated):**

  *Low dynamic range images:*<br>
    
    LDR-ꟻLIP: `<name>.png`<br>
    Weighted histogram: `<name>.pdf`<br>
    Text file: `<name>.txt`<br>

  *High dynamic range images:*<br>
    
    HDR-ꟻLIP: `<name>.png`<br>
    Exposure map: `<name>.exposure_map.png`<br>
    Intermediate LDR-ꟻLIP maps: `<name>.<nnn>.png`<br>
    Intermediate LDR images: `<name>.reference|test.<nnn>.png`<br>
    Weighted histogram: `<name>.pdf`<br>
    Overlapping weighted histogram: N/A<br>
    Text file: `<name>.txt`<br>

**Example usage:**
First navigate to the directory containing the `flip.py` script and the rest of the Python files. Then start an Ananconda prompt and try:
  ```
  conda activate flip
  python flip.py -r ../images/reference.exr -t ../images/test.exr
  ```
The result should be:
  ```
Invoking HDR-FLIP
        Pixels per degree: 67
        Assumed tone mapper: ACES
        Start exposure: -12.5423
        Stop exposure: 0.9427
        Number of exposures: 14

FLIP between reference image <reference.exr> and test image <test.exr>:
        Mean: 0.283547
        Weighted median: 0.339469
        1st weighted quartile: 0.251148
        3rd weighted quartile: 0.434763
        Min: 0.003120
        Max: 0.962022
        Evaluation time: <t> seconds
  ```
where `<t>` is the time it took to evaluate HDR-ꟻLIP. In addition, you will now find the files `flip.reference.test.67ppd.hdr.aces.m12.5423_to_p0.9427.14.png` and `exposure_map.reference.test.67ppd.hdr.aces.m12.5423_to_p0.9427.14.png`
in the directory containing the `flip.py` script, and we urge you to inspect those, which will reveal where the errors in the test image are located.
