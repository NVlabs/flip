![Teaser image](flip_evaluator/images/teaser.png "Teaser image")

# FLIP: A Tool for Visualizing and Communicating Errors in Rendered Images (v1.5)

By
[Pontus Ebelin](https://research.nvidia.com/person/pontus-ebelin)
and
[Tomas Akenine-Möller](https://research.nvidia.com/person/tomas-akenine-m%C3%B6ller),
with
Jim Nilsson,
[Magnus Oskarsson](https://www1.maths.lth.se/matematiklth/personal/magnuso/),
[Kalle Åström](https://www.maths.lu.se/staff/kalleastrom/),
[Mark D. Fairchild](https://www.rit.edu/directory/mdfpph-mark-fairchild),
and
[Peter Shirley](https://research.nvidia.com/person/peter-shirley).

This repository holds implementations of the [LDR-FLIP](https://research.nvidia.com/publication/2020-07_FLIP)
and [HDR-FLIP](https://research.nvidia.com/publication/2021-05_HDR-FLIP) image error metrics.
It also holds code for the FLIP tool, presented in [Ray Tracing Gems II](https://www.realtimerendering.com/raytracinggems/rtg2/index.html).

The changes made for the different versions of FLIP are summarized in the [version list](https://github.com/NVlabs/flip/blob/main/flip_evaluator/misc/versionList.md).

[A list of papers](https://github.com/NVlabs/flip/blob/main/flip_evaluator/misc/papersUsingFLIP.md) that use/cite FLIP.

[A note](https://github.com/NVlabs/flip/blob/main/flip_evaluator/misc/precision.md) about the precision of FLIP.

[An image gallery](https://research.nvidia.com/node/3525) displaying a large quantity of reference/test images and corresponding error maps from
different metrics.

**Note**: since v1.5, the Python version of FLIP can now be installed via `pip install flip-evaluator`.

**Note**: in v1.3, we switched to a *single header* ([FLIP.h](flip_evaluator/cpp/FLIP.h)) for C++/CUDA for easier integration.

# License

Copyright © 2020-2024, NVIDIA Corporation & Affiliates. All rights reserved.

This work is made available under a [BSD 3-Clause License](LICENSE).

The repository distributes code for `tinyexr`, which is subject to a [BSD 3-Clause License](https://github.com/NVlabs/flip/blob/main/flip_evaluator/misc/LICENSE-third-party.md#bsd-3-clause-license),<br>
and `stb_image`, which is subject to an [MIT License](https://github.com/NVlabs/flip/blob/main/flip_evaluator/misc/LICENSE-third-party.md#mit-license).

For individual contributions to the project, please confer the [Individual Contributor License Agreement](https://github.com/NVlabs/flip/blob/main/flip_evaluator/misc/CLA.md).

For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/).

# Simplest Way To Get Started
The simplest way to run FLIP to compare a test image `testImage.png` to a reference image `referenceImage.png` is as follows:
```
pip install flip-evaluator
flip -r referenceImage.png -t testImage.png
```
For more information about the tool's capabilities, try running `flip -h`.

If you wish to use FLIP in your Python or C++ evaluation scripts, please read the next sections.

# Python (API and Tool)
**Setup** (with pip):
```
pip install flip-evaluator
```

**Usage:**<br>

API:<br>
See the example script `flip_evaluator/python/api_example.py`.

Tool:
```
flip --reference reference.{exr|png} --test test.{exr|png} [--options]
```

See the [README](https://github.com/NVlabs/flip/blob/main/flip_evaluator/python/README.md) in the `python` folder and run `flip -h` for further information and usage instructions.

# C++ and CUDA (API and Tool)
**Setup:**

The `flip_evaluator/cpp/FLIP.sln` solution contains one CUDA backend project and one pure C++ backend project.

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
See the [README](https://github.com/NVlabs/flip/blob/main/flip_evaluator/cpp/README.md).

Tool:
```
flip[-cuda].exe --reference reference.{exr|png} --test test.{exr|png} [options]
```

See the [README](https://github.com/NVlabs/flip/blob/main/flip_evaluator/cpp/README.md) in the `flip_evaluator/cpp` folder and run `flip[-cuda].exe -h` for further information and usage instructions.

# PyTorch (Loss Function)
**Setup** (with Anaconda3 or Miniconda):
```
conda create -n flip_dl python numpy matplotlib
conda activate flip_dl
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
conda install -c conda-forge openexr-python
```

**Usage:**

*Remember to activate the* `flip_dl` *environment through* `conda activate flip_dl` *before using the loss function.*

LDR- and HDR-FLIP are implemented as loss modules in `flip_evaluator/pytorch/flip_loss.py`. An example where the loss function is used to train a simple autoencoder is provided in `flip_evaluator/pytorch/train.py`.

See the [README](https://github.com/NVlabs/flip/blob/main/flip_evaluator/pytorch/README.md) in the `pytorch` folder for further information and usage instructions.

# Citation
If your work uses the FLIP tool to find the errors between *low dynamic range* images,
please cite the LDR-FLIP paper:<br>
[Paper](https://research.nvidia.com/publication/2020-07_FLIP) | [BibTeX](https://github.com/NVlabs/flip/blob/main/flip_evaluator/misc/LDRFLIP.txt)

If it uses the FLIP tool to find the errors between *high dynamic range* images,
instead cite the HDR-FLIP paper:<br>
[Paper](https://research.nvidia.com/publication/2021-05_HDR-FLIP) | [BibTeX](https://github.com/NVlabs/flip/blob/main/flip_evaluator/misc/HDRFLIP.txt)

Should your work use the FLIP tool in a more general fashion, please cite the Ray Tracing Gems II article:<br>
[Chapter](https://link.springer.com/chapter/10.1007%2F978-1-4842-7185-8_19) | [BibTeX](https://github.com/NVlabs/flip/blob/main/flip_evaluator/misc/FLIP.txt)

# Acknowledgements
We appreciate the following peoples' contributions to this repository:
Jonathan Granskog, Jacob Munkberg, Jon Hasselgren, Jefferson Amstutz, Alan Wolfe, Killian Herveau, Vinh Truong, Philippe Dagobert, Hannes Hergeth, Matt Pharr, Tizian Zeltner, Jan Honsbrok, and Chris Zhang.
