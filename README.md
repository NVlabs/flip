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

[A list of papers](papersUsingFLIP.md) that use/cite ꟻLIP.

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
python flip.py --reference reference.{exr|png} --test test.{exr|png} [--options]
```

See the [README](python/README.md) in the `python` folder and run `python flip.py -h` for further information and usage instructions.

# C++ and CUDA (API and Tool)
**Setup:**

The `FLIP.sln` solution contains one CUDA backend project and one pure C++ backend project.

Compiling the CUDA project requires a CUDA compatible GPU. Instruction on how to install CUDA can be found [here](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html).

Alternatively, a CMake build can be done by creating a build directory and invoking CMake on the source `cpp` dir:

```
mkdir build
cd build
cmake ../cpp
cmake --build .
```

CUDA support is enabled via the `FLIP_ENABLE_CUDA`, which can be passed to CMake on the command line with
`-DFLIP_ENABLE_CUDA=ON` or set interactively with `ccmake` or `cmake-gui`.

**Usage:**
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
[Paper](https://research.nvidia.com/publication/2020-07_FLIP) | [BibTeX](LDRFLIP.txt)

If it uses the ꟻLIP tool to find the errors between *high dynamic range* images,
instead cite the HDR-ꟻLIP paper:<br>
[Paper](https://research.nvidia.com/publication/2021-05_HDR-FLIP) | [BibTeX](HDRFLIP.txt)

Should your work use the ꟻLIP tool in a more general fashion, please cite the Ray Tracing Gems II article:<br>
Article (to be published in August 2021) | [BibTeX](FLIP.txt)
