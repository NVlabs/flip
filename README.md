![Teaser image](images/teaser.png "Teaser image")

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

This repository holds implementations of the [LDR-ꟻLIP](https://research.nvidia.com/publication/2020-07_FLIP)
and [HDR-ꟻLIP](https://research.nvidia.com/publication/2021-05_HDR-FLIP) image error metrics.
It also holds code for the ꟻLIP tool, to be presented in [Ray Tracing Gems II](https://developer.nvidia.com/blog/ray-tracing-gems-ii-available-august-4th/).

[A list of papers](papersUsingFLIP.md) that use ꟻLIP.

# License

Copyright © 2020-2021, NVIDIA Corporation. All rights reserved.

This work is made available under the [NVIDIA Source Code License](LICENSE.txt).

For business inquiries, please contact researchinquiries@nvidia.com.

For press and other inquiries, please contact Hector Marinez at hmarinez@nvidia.com.

# Python (API and Tool)
**Setup** (with Anaconda3):
```
conda create -n flip python numpy matplotlib
conda activate flip
conda install -c conda-forge opencv
conda install -c conda-forge openexr-python
```

**Usage:**

*Remember to activate the* `flip` *environment through* `conda activate flip` *before using the tool.*

```
python flip.py <reference.{exr|png}> <test.{exr|png}> [--options]
```

See the [README](python/README.md) in the `python` folder and run `python flip.py -h` for further information and usage instructions.

# C++/CUDA (API and Tool)
C++/CUDA code will be released in the week of May 5.

# PyTorch (Loss Function)
PyTorch code will be released in the week of May 5.
