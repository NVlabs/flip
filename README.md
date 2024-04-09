![Teaser image](images/teaser.png "Teaser image")

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

This repository holds implementations of the [LDR-ꟻLIP](https://research.nvidia.com/publication/2020-07_FLIP)
and [HDR-ꟻLIP](https://research.nvidia.com/publication/2021-05_HDR-FLIP) image error metrics.
It also holds code for the ꟻLIP tool, presented in [Ray Tracing Gems II](https://www.realtimerendering.com/raytracinggems/rtg2/index.html).

The changes made for the different versions of ꟻLIP are summarized in the [version list](misc/versionList.md).

[A list of papers](misc/papersUsingFLIP.md) that use/cite ꟻLIP.

[A note](misc/precision.md) about the precision of ꟻLIP.

[An image gallery](https://research.nvidia.com/node/3525) displaying a large quantity of reference/test images and corresponding error maps from
different metrics.

**Note**: in v1.3, we switched to a *single header* ([FLIP.h](https://github.com/NVlabs/flip/blob/singleheader_WIP/cpp/FLIP.h)) for C++/CUDA for easier integration.

# License

Copyright © 2020-2024, NVIDIA Corporation & Affiliates. All rights reserved.

This work is made available under a [BSD 3-Clause License](misc/LICENSE.md).

The repository distributes code for `tinyexr`, which is subject to a [BSD 3-Clause License](misc/LICENSE-third-party.md#bsd-3-clause-license),<br>
and `stb_image`, which is subject to an [MIT License](misc/LICENSE-third-party.md#mit-license).

For individual contributions to the project, please confer the [Individual Contributor License Agreement](misc/CLA.md).

For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/).

# Python (API and Tool)
**Setup** (with pip):
```
cd python
pip install -r requirements.txt .
```

**Usage:**<br>

API:<br>
See the example script `python/api_example.py`. Note that the script requires `matplotlib`.

Tool:
```
python flip.py --reference reference.{exr|png} --test test.{exr|png} [--options]
```

See the [README](python/README.md) in the `python` folder and run `python flip.py -h` for further information and usage instructions.

# C++ and CUDA (API and Tool)
**Setup:**

The `FLIP.sln` solution contains one CUDA backend project and one pure C++ backend project.

Compiling the CUDA project requires a CUDA compatible GPU. Instruction on how to install CUDA can be found [here](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html).

Alternatively, a CMake build can be done by creating a build directory and invoking CMake on the source dir (add `--config Release` to build release configuration on Windows):

```
mkdir build
cd build
cmake ..
cmake --build . [--config Release]
```

CUDA support is enabled via the `FLIP_ENABLE_CUDA`, which can be passed to CMake on the command line with
`-DFLIP_ENABLE_CUDA=ON` or set interactively with `ccmake` or `cmake-gui`.
`FLIP_LIBRARY` option allows to output a library rather than an executable.

**Usage:**<br>

API:<br>
See the [README](cpp/README.md).

Tool:
```
flip[-cuda].exe --reference reference.{exr|png} --test test.{exr|png} [options]
```

See the [README](cpp/README.md) in the `cpp` folder and run `flip[-cuda].exe -h` for further information and usage instructions.

# PyTorch (Loss Function)
**Setup** (with Anaconda3):
```
conda create -n flip_dl python numpy matplotlib
conda activate flip_dl
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
conda install -c conda-forge openexr-python
```

**Usage:**

*Remember to activate the* `flip_dl` *environment through* `conda activate flip_dl` *before using the loss function.*

LDR- and HDR-ꟻLIP are implemented as loss modules in `flip_loss.py`. An example where the loss function is used to train a simple autoencoder is provided in `train.py`.

See the [README](pytorch/README.md) in the `pytorch` folder for further information and usage instructions.

# Citation
If your work uses the ꟻLIP tool to find the errors between *low dynamic range* images,
please cite the LDR-ꟻLIP paper:<br>
[Paper](https://research.nvidia.com/publication/2020-07_FLIP) | [BibTeX](misc/LDRFLIP.txt)

If it uses the ꟻLIP tool to find the errors between *high dynamic range* images,
instead cite the HDR-ꟻLIP paper:<br>
[Paper](https://research.nvidia.com/publication/2021-05_HDR-FLIP) | [BibTeX](misc/HDRFLIP.txt)

Should your work use the ꟻLIP tool in a more general fashion, please cite the Ray Tracing Gems II article:<br>
[Chapter](https://link.springer.com/chapter/10.1007%2F978-1-4842-7185-8_19) | [BibTeX](misc/FLIP.txt)

# Acknowledgements
We appreciate the following peoples' contributions to this repository:
Jonathan Granskog, Jacob Munkberg, Jon Hasselgren, Jefferson Amstutz, Alan Wolfe, Killian Herveau, Vinh Truong, Philippe Dagobert, Hannes Hergeth, Matt Pharr, and Tizian Zeltner.
