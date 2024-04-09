# ꟻLIP Version List

In addition to various minor changes, the following was
changed for the different versions of ꟻLIP:

# Version 1.4 (commits 6265f80 to 0349494)
- Changed the Python version of ꟻLIP so that it leverages the C++ code through [pybind11](https://github.com/pybind/pybind11).
	- Results (only evaluation, not including file load/save, etc; measured on an AMD Ryzen Threadripper 3970X 32-Core Processor, 3693 MHz, with 32 Cores and 64 Logical Processors):
		- 20-47x faster for LDR/HDR CPU.
		- Timings for 1920x1080 images:
			- Python/LDR: 77 ms
			- Python/HDR: 1007 ms
	- **NOTE**: The Python version can currently _not_ run the CUDA version of ꟻLIP (see issue [#22](https://github.com/NVlabs/flip/issues/22)).
	- **NOTE**: The Python tool now uses the C++ tool. Compared to before, you will need to change `_` to `-` when calling flip.py (e.g., `python flip.py -r reference.exr -t test.exr --start_exposure 3` is now `python flip.py -r reference.exr -t test.exr --start-exposure 3`; see `python flip.py -h`).
- The Python version of ꟻLIP can now be installed using `pip` (run `pip install -r requirements.txt .` from the `python` folder).
- The code for the C++/CUDA tool is now in `FLIPToolHelpers.h`.
- **NOTE**: The fourth `evaluate()` function in `FLIP.h` now takes two additional arguments: `computeMeanError` and `meanError`. Furthermore, its list of arguments has been partly reordered.
- **NOTE**: The median computation (used for automatic start and stop expsoure computations in HDR-ꟻLIP) in the C++/CUDA code has been changed, sometimes causing a minor change in results but always resulting in a significant speedup. The tests have been updated following this change.
  - Timings for 1920x1080 images (only evaluation, not including file load/save, etc, *but* measured with another GPU and including more code than the numbers presented in the v1.2 update, so the numbers are not directly comparable; measured on an AMD Ryzen Threadripper 3970X 32-Core Processor, 3693 MHz, with 32 Cores and 64 Logical Processors and an NVIDIA RTX 4090 GPU):
    - CPP/LDR: 86 ms
    - CPP/HDR: 1179 ms
    - CUDA/LDR: 8 ms
    - CUDA/HDR: 131 ms
- Added check for OpenMP for CMake build.
- Overlapped histograms are now available in the C++ tool code. These are created when one reference and _two_ test images are input, together with the `--histogram` flag.
- Text file output are now available in the C++ tool code. These are created when the `--textfile` flag is input.
- The Python and C++ tests now use the same targets.

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
    - Results (not including file load/save):
        - 111-124x faster for LDR/HDR CPU (measured on an AMD Ryzen Threadripper 3970X 32-Core Processor, 3693 MHz, with 32 Cores and 64 Logical Processors).
        - 2.4-2.8x faster LDR/HDR CUDA (measured on an AMD Ryzen Threadripper 3970X 32-Core Processor, 3693 MHz, with 32 Cores and 64 Logical Processors, together with an NVIDIA RTX 3090).
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
