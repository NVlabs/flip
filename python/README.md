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

This [repository](https://github.com/NVlabs/flip) implements the [LDR-ꟻLIP](https://research.nvidia.com/publication/2020-07_FLIP)
and [HDR-ꟻLIP](https://research.nvidia.com/publication/2021-05_HDR-FLIP) image error metrics in Python, using the C++ implementation through [pybind11](https://github.com/pybind/pybind11).
Similarly, it implements the ꟻLIP tool, presented in [Ray Tracing Gems II](https://www.realtimerendering.com/raytracinggems/rtg2/index.html).

# License

Copyright © 2020-2024, NVIDIA Corporation & Affiliates. All rights reserved.

This work is made available under a [BSD 3-Clause License](../misc/LICENSE.md).

The repository distributes code for `tinyexr`, which is subject to a [BSD 3-Clause License](../misc/LICENSE-third-party.md#bsd-3-clause-license),<br>
and `stb_image`, which is subject to an [MIT License](../misc/LICENSE-third-party.md#mit-license).

For individual contributions to the project, please confer the [Individual Contributor License Agreement](../misc/CLA.md).

For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/).

# Python (API and Tool)
- **Setup** (with pip, from root directory):
  ```
  cd python
  pip install -r requirements.txt .
  ```
  After this, you may `import flip` from any Python script.
- Usage (API): See example in the script `flip/api_example.py`. Note that the script requires `matplotlib`. 
- Usage (tool): `python flip.py --reference reference.{exr|png} --test test.{exr|png} [--options]`, where the list of options can be seen by `python flip.py -h`.
- Tested with pip 24.0, Python 3.11.8, pybind11 2.11.1, and C++20.
- The code that implements ꟻLIP metrics and the ꟻLIP tool is available in [FLIP.h](https://github.com/NVlabs/flip/blob/main/cpp/FLIP.h) and [flip/cpp/tool](https://github.com/NVlabs/flip/blob/main/cpp/tool), respectively.
  The Python API is provided in `flip/python/flip/main.py`.
  `../tests/test.py` contains simple tests used to test whether code updates alter results.
- Weighted histograms are output as Python scripts. Running the script will create a PDF version of the histogram. Notice that those scripts require `numpy` and `matplotlib`, both of which are automatically installed during setup.
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
To test the API, please inspect the `flip/api_example.py` script. This shows how the available API commands may be used. Note that the script requires `matplotlib`.
Please note that not all capabilities of the tool is available through the Python API. For example, the exposure map is not output when running HDR-ꟻLIP. For that, use the tool or the C++ API in [FLIP.h](https://github.com/NVlabs/flip/blob/main/cpp/FLIP.h).

To test the tool, first navigate to the directory containing the `flip.py` script. Then start a shell and try:
  ```
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
        Mean: 0.283478
        Weighted median: 0.339430
        1st weighted quartile: 0.251122
        3rd weighted quartile: 0.434673
        Min: 0.003123
        Max: 0.962022
        Evaluation time: <t> seconds
  ```
where `<t>` is the time it took to evaluate HDR-ꟻLIP. In addition, you will now find the files `flip.reference.test.67ppd.hdr.aces.m12.5423_to_p0.9427.14.png` and `exposure_map.reference.test.67ppd.hdr.aces.m12.5423_to_p0.9427.14.png`
in the directory containing the `flip.py` script, and we urge you to inspect those, which will reveal where the errors in the test image are located.
