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
and [HDR-ꟻLIP](https://research.nvidia.com/publication/2021-05_HDR-FLIP) image error metrics as loss modules in PyTorch.

# License

Copyright © 2020-2024, NVIDIA Corporation & Affiliates. All rights reserved.

This work is made available under a [BSD 3-Clause License](../misc/LICENSE.md).

The repository distributes code for `tinyexr`, which is subject to a [BSD 3-Clause License](../misc/LICENSE-third-party.md#bsd-3-clause-license),<br>
and `stb_image`, which is subject to an [MIT License](../misc/LICENSE-third-party.md#mit-license).

For individual contributions to the project, please confer the [Individual Contributor License Agreement](../misc/CLA.md).

For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/).

# PyTorch (Loss Function)

- **Setup** (with Anaconda3):
  ```
  conda create -n flip_dl python numpy matplotlib
  conda activate flip_dl
  conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
  conda install -c conda-forge openexr-python
  ```
- *Remember to activate the* `flip_dl` *environment through* `conda activate flip_dl` *before using the loss function.*
- LDR- and HDR-ꟻLIP are implemented as loss modules in `flip_loss.py`.
  An example where the loss function is used to train a simple autoencoder is provided in `train.py`.
- Tested on Windows with Conda 4.10.0, CUDA 11.2, Python 3.9.4, PyTorch 1.8.1, NumPy 1.20.1, and OpenEXR b1.3.2.
- Per default, the loss function returns the mean of the error maps. To return the full error maps,
  remove `torch.mean()` from the `forward()` function.
- For LDR-ꟻLIP, the images are assumed to be in sRGB space
  (change the color space transform in `LDRFLIPLoss`'s `forward()` function to `linrgb2ycxcz` if your network's output is in linear RGB),
  in the [0,1] range.
- Both LDR- and HDR-ꟻLIP takes an optional argument describing the assumed number of pixels per
  degree of the observer. Per default, it is assume that the images are viewed at a distance 0.7 m from
  a 0.7 m wide 4K monitor.
- The `HDRFLIPLoss` can take three additional, optional arguments: `tone_mapper`, `start_exposure`, and `stop_exposure`.
  `tone_mapper` is a string describing the tone mapper that HDR-ꟻLIP should assume, for which the choices are `aces` (default), `hable`, and `reinhard`. The default assumption is the [ACES](https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/) tone mapper.
  `start_exposure`, and `stop_exposure` should have `Nx1x1x1` layout and hold the
  start and stop exposures, respectively, used for each of the `N` reference/test
  pairs in the batch. Per default, `HDRFLIPLoss` computes start and stop exposures as described
  in the [paper](https://d1qx31qr3h6wln.cloudfront.net/publications/HDRFLIP-paper.pdf).
  **NOTE:** When start and/or stop exposures are not provided, HDR-ꟻLIP is not symmetric. The
  user should therefore make sure to input the test images as the *first* argument and the reference image
  as the *second* argument to the `HDRFLIPLoss`'s `forward()` function.
- `../tests/test_pytorch.py` contains simple tests used to test whether code updates alter results and
  `data.py` contains image loading/saving functions.
