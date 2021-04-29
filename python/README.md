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
and [HDR-ꟻLIP](https://research.nvidia.com/publication/2021-05_HDR-FLIP) image error metrics in Python.
It also holds code for the ꟻLIP tool, to be presented in [Ray Tracing Gems II](https://developer.nvidia.com/blog/ray-tracing-gems-ii-available-august-4th/).

### License ###

Copyright © 2020-2021, NVIDIA Corporation. All rights reserved.

This work is made available under the [NVIDIA Source Code License](../LICENSE.txt).

For business inquiries, please contact researchinquiries@nvidia.com.

For press and other inquiries, please contact Hector Marinez at hmarinez@nvidia.com.

### Python (API and Tool) ###
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
  `tests.py` contains simple tests used to test whether code updates alter results.
- The naming convention used for HDR-ꟻLIP-related results is either
  1. `typeOfImage_X_to_Y`, where `typeOfImage` says
  what type of image it is (e.g., `hdrflip` or `exposure_map`),
  `X` is the start exposure and `Y` is the stop
  exposure (a p in front of `X` or `Y`
  implies plus and an m implies minus), or
  2. `typeOfImage.xxx.X_to_Y.exposure`, where `typeOfImage`,
  `X`, and `Y` are as in 1, `xxx` is an enumeration
  of the `N = ceil(max(2, (Y - X)))` exposures used to compute HDR-FLIP,
  and `exposure` indicates the exposure compensation factor used.
