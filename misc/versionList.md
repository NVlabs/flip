# ꟻLIP Version List

In addition to various minor changes, the following was
changed for the different versions of ꟻLIP:

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