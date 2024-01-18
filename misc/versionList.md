# ꟻLIP Version List

In addition to various minor changes, the following was
changed for the different versions of ꟻLIP:

# Version 1.3 (commit a00bc7d)
- Changed to CUDA 12.3.
- Rewrote C++ code so that ꟻLIP is in a single header (both CPP/CUDA).
- Rewrote `FLIP-tool.cpp` to use many more local functions to make the code easier to read.
- Some of the `tests/correct_*.png` images have been update due to minor changes in output that occurred as part of switching to CUDA 12.3 and changing the order of some transforms.

# Version 1.2 (commit dde1eca)
- Changed to CUDA 11.5 (was done after v1.1, but before v1.2).
- Adds tests for C++ and CUDA in the tests-directory.
  Additionally, the Python and PyTorch tests were moved to that directory.
- Performance optimizations for C++ and CUDA implementations:
    - Uses separable filters.
    - Merges several functions/kernels into fewer.
    - Uses OpenMP for the CPU.
    - Results (not incuding file load/save):
        - 111-124x faster for LDR/HDR CPU (depends on CPU setup, though).
        - 2.4-2.8x faster LDR/HDR CUDA (depends on CPU/GPU setup)
		- Timings for 1920x1080 images:
			- CPP/LDR: 63 ms
			- CPP/HDR: 1050 ms
			- CUDA/LDR: 13 ms
			- CUDA/HDR: 136 ms

# Version 1.1 (commit 4ed59e9)
- NVIDIA Source Code License changed to a 3-Clause BSD License
- Precision updates:
    - Constants use nine decimal digits (a float32 number has the same
      bit representation if stored with nine or more decimals in NumPy
      and C++)
    - Increased accuracy in the XYZ2CIELab transform and its inverse in
      C++ and CUDA
    - Changed reference_illuminant to a float32 array in Python
      (reducing LDR-FLIP runtime by almost 30% for a 1080p image)
- Magma and Viridis are indexed into using round instead of floor
- Introduced the constant 1.0 / referenceIlluminant to avoid unnecessary
  divisions during color transforms
- Updated the Python reference error maps based on the changes above
- Updated the PyTorch test script based on the changes above
- Expected CUDA version updated from 11.2 to 11.3
- Removed erroneous abs() from the XYZ2CIELab transform in C++ and CUDA
- Added "Acknowledgements" section to the main README file
- A cross platform CMake build was added recently (commit 6bdbbaa)

# Version 1.0
- Initial release
