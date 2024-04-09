/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: BSD-3-Clause
 */

 // Visualizing and Communicating Errors in Rendered Images
 // Ray Tracing Gems II, 2021,
 // by Pontus Andersson, Jim Nilsson, and Tomas Akenine-Moller.
 // Pointer to the chapter: https://research.nvidia.com/publication/2021-08_Visualizing-and-Communicating.

 // Visualizing Errors in Rendered High Dynamic Range Images
 // Eurographics 2021,
 // by Pontus Andersson, Jim Nilsson, Peter Shirley, and Tomas Akenine-Moller.
 // Pointer to the paper: https://research.nvidia.com/publication/2021-05_HDR-FLIP.

 // FLIP: A Difference Evaluator for Alternating Images
 // High Performance Graphics 2020,
 // by Pontus Andersson, Jim Nilsson, Tomas Akenine-Moller,
 // Magnus Oskarsson, Kalle Astrom, and Mark D. Fairchild.
 // Pointer to the paper: https://research.nvidia.com/publication/2020-07_FLIP.

// Single header code by Pontus Ebelin (formerly Andersson) and Tomas Akenine-Moller.
//
// We provide the following FLIP::evaluate() functions with different in/out parameters (see bottom of this file for more explanations):
//
// 1. FLIP::evaluate(const bool useHDR, FLIP::Parameters& parameters, FLIP::image<FLIP::color3>& referenceImageInput, FLIP::image<FLIP::color3>& testImageInput,
//                  FLIP::image<float>& errorMapFLIPOutput, FLIP::image<float>& maxErrorExposureMapOutput,
//                  const bool returnLDRFLIPImages, std::vector<FLIP::image<float>*>& hdrOutputFlipLDRImages,
//                  const bool returnLDRImages, std::vector<FLIP::image<FLIP::color3>*>& hdrOutputLDRImages)
//
//    # This is the one with most parameters and is used by FLIP-tool.cpp in main().
//    # See the function at the bottom of this file for detailed description of the parameters.
//
// 2. FLIP::evaluate(const bool useHDR, FLIP::Parameters& parameters, FLIP::image<FLIP::color3>& referenceImageInput, FLIP::image<FLIP::color3>& testImageInput,
//                  FLIP::image<float>& errorMapFLIPOutput, FLIP::image<float>& maxErrorExposureMap);
//
//    # We do not expect that many user will want the LDR-FLIP images and the tonemappe LDR images computed during HDR-FLIP, so provide this simpler function.
//
// 3.FLIP::evaluate(const bool useHDR, FLIP::Parameters& parameters, FLIP::image<FLIP::color3>& referenceImageInput, FLIP::image<FLIP::color3>& testImageInput,
//                 FLIP::image<float>& errorMapFLIPOutput);
//
//    # This one also excludes the exposure map for HDR-FLIP, in case it is not used.
//
// 4. FLIP::evaluate(const bool useHDR, FLIP::Parameters& parameters, const int imageWidth, const int imageHeight,
//                  const float* referenceThreeChannelImage, const float* testThreeChannelImage, const bool applyMagmaMapToOutput, float** errorMapFLIPOutput)
//
//    # An even simpler function that does not use any of our image classes to input the images.

#pragma once
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <sstream>
#include <fstream>
#include <limits>
#include "tool/pooling.h"

#ifdef FLIP_ENABLE_CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif

#ifdef FLIP_ENABLE_CUDA
#define HOST_DEVICE_FOR_CUDA __host__ __device__
#else
#define HOST_DEVICE_FOR_CUDA
#endif

namespace FLIP
{
    const float PI = 3.14159265358979f;

#define Max(x, y) ((x) > (y) ? (x) : (y))
#define Min(x, y) ((x) > (y) ? (y) : (x))

#define DEFAULT_ILLUMINANT { 0.950428545f, 1.000000000f, 1.088900371f }
#define INV_DEFAULT_ILLUMINANT { 1.052156925f, 1.000000000f, 0.918357670f }

    //  Pixels per degree (PPD).
    inline float calculatePPD(const float dist, const float resolutionX, const float monitorWidth)
    {
        return dist * (resolutionX / monitorWidth) * (float(FLIP::PI) / 180.0f);
    }

    struct Parameters
    {
        Parameters() = default;
        float PPD = FLIP::calculatePPD(0.7f, 3840.0f, 0.7f);            // Populate PPD with default values based on 0.7 meters = distance to screen, 3840 pixels screen width, 0.7 meters monitor width.
        float startExposure = std::numeric_limits<float>::infinity();   // Used when the input is HDR.
        float stopExposure = std::numeric_limits<float>::infinity();    // Used when the input is HDR.
        int numExposures = -1;                                          // Used when the input is HDR.
        std::string tonemapper = "aces";                                // Default tonemapper (used for HDR).
    };

    static const struct xFLIPConstants
    {
        xFLIPConstants() = default;
        float gqc = 0.7f;
        float gpc = 0.4f;
        float gpt = 0.95f;
        float gw = 0.082f;
        float gqf = 0.5f;
    } FLIPConstants;



#ifndef FLIP_ENABLE_CUDA
    static const float ToneMappingCoefficients[3][6] =
#else
    __device__ const float ToneMappingCoefficients[3][6] =
#endif
    {
        { 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f },                                                 // Reinhard.
        { 0.6f * 0.6f * 2.51f, 0.6f * 0.03f, 0.0f, 0.6f * 0.6f * 2.43f, 0.6f * 0.59f, 0.14f },  // ACES, 0.6 is pre-exposure cancellation.
        { 0.231683f, 0.013791f, 0.0f, 0.18f, 0.3f, 0.018f },                                    // Hable.
    };


#ifndef FLIP_ENABLE_CUDA
    union int3
    {
        struct { int x, y, z; };
    };
#else // FLIP_ENABLE_CUDA
    __constant__ struct
    {
        float gqc = 0.7f;
        float gpc = 0.4f;
        float gpt = 0.95f;
        float gw = 0.082f;
        float gqf = 0.5f;
    } DeviceFLIPConstants;

    const dim3 DEFAULT_KERNEL_BLOCK_DIM = { 32, 32, 1 };
    enum class CudaTensorState
    {
        UNINITIALIZED,
        ALLOCATED,
        HOST_ONLY,
        DEVICE_ONLY,
        SYNCHRONIZED
    };
#endif

    class color3
    {
    public:
        union
        {
            struct { float r, g, b; };
            struct { float x, y, z; };
            struct { float h, s, v; };
        };

    public:
        HOST_DEVICE_FOR_CUDA color3(void)
        {
            this->x = 0.0f;
            this->y = 0.0f;
            this->z = 0.0f;
        }

        HOST_DEVICE_FOR_CUDA color3(float v)
        {
            this->x = v;
            this->y = v;
            this->z = v;
        }

        HOST_DEVICE_FOR_CUDA color3(const float* pColor)
        {
            this->x = pColor[0];
            this->y = pColor[1];
            this->z = pColor[2];
        }

        HOST_DEVICE_FOR_CUDA color3(const unsigned char* pColor)
        {
            this->x = float(pColor[0]);
            this->y = float(pColor[1]);
            this->z = float(pColor[2]);
            *this /= 255.0f;
        }

        HOST_DEVICE_FOR_CUDA color3(float _x, float _y, float _z)
        {
            this->x = _x;
            this->y = _y;
            this->z = _z;
        }

        HOST_DEVICE_FOR_CUDA color3(const color3& c)
        {
            this->x = c.x;
            this->y = c.y;
            this->z = c.z;
        }

        HOST_DEVICE_FOR_CUDA bool operator==(const color3 c) const
        {
            return this->x == c.x && this->y == c.y && this->z == c.z;
        }

        HOST_DEVICE_FOR_CUDA bool operator!=(const color3 c) const
        {
            return !(*this == c);
        }

        HOST_DEVICE_FOR_CUDA color3 operator+(const color3 c) const
        {
            return color3(this->x + c.x, this->y + c.y, this->z + c.z);
        }

        HOST_DEVICE_FOR_CUDA color3 operator-(const color3 c) const
        {
            return color3(this->x - c.x, this->y - c.y, this->z - c.z);
        }

        HOST_DEVICE_FOR_CUDA color3 operator*(const float c) const
        {
            return color3(this->x * c, this->y * c, this->z * c);
        }

        HOST_DEVICE_FOR_CUDA color3 operator*(const color3 c) const
        {
            return color3(this->x * c.x, this->y * c.y, this->z * c.z);
        }

        HOST_DEVICE_FOR_CUDA color3 operator/(const float c) const
        {
            return color3(this->x / c, this->y / c, this->z / c);
        }

        HOST_DEVICE_FOR_CUDA color3 operator/(const color3 c) const
        {
            return color3(this->x / c.x, this->y / c.y, this->z / c.z);
        }

        HOST_DEVICE_FOR_CUDA color3 operator+=(const color3 c)
        {
            this->x += c.x;
            this->y += c.y;
            this->z += c.z;
            return *this;
        }

        HOST_DEVICE_FOR_CUDA color3 operator*=(const color3 c)
        {
            this->x *= c.x;
            this->y *= c.y;
            this->z *= c.z;
            return *this;
        }

        HOST_DEVICE_FOR_CUDA color3 operator/=(const color3 c)
        {
            this->x /= c.x;
            this->y /= c.y;
            this->z /= c.z;
            return *this;
        }

        HOST_DEVICE_FOR_CUDA void clear(const color3 c = { 0.0f, 0.0f, 0.0f })
        {
            this->x = c.x;
            this->y = c.y;
            this->z = c.z;
        }

        HOST_DEVICE_FOR_CUDA static inline color3 min(color3 v0, color3 v1)
        {
            return color3(Min(v0.x, v1.x), Min(v0.y, v1.y), Min(v0.z, v1.z));
        }

        HOST_DEVICE_FOR_CUDA static inline color3 max(color3 v0, color3 v1)
        {
            return color3(Max(v0.x, v1.x), Max(v0.y, v1.y), Max(v0.z, v1.z));
        }

        HOST_DEVICE_FOR_CUDA static inline color3 abs(color3 v)
        {
            return color3(std::abs(v.x), std::abs(v.y), std::abs(v.z));
        }

        HOST_DEVICE_FOR_CUDA static inline color3 sqrt(color3 v)
        {
            return color3(std::sqrt(v.x), std::sqrt(v.y), std::sqrt(v.z));
        }

        HOST_DEVICE_FOR_CUDA static inline color3 clamp(color3 v, float _min = 0.0f, float _max = 1.0f)
        {
            return color3(Min(Max(v.x, _min), _max), Min(Max(v.y, _min), _max), Min(Max(v.z, _min), _max));
        }

        HOST_DEVICE_FOR_CUDA static inline float linearRGB2Luminance(color3 linearRGB)
        {
            return 0.2126f * linearRGB.r + 0.7152f * linearRGB.g + 0.0722f * linearRGB.b;
        }

        HOST_DEVICE_FOR_CUDA static inline float sRGB2LinearRGB(float sC)
        {
            if (sC <= 0.04045f)
            {
                return sC / 12.92f;
            }
            return powf((sC + 0.055f) / 1.055f, 2.4f);
        }

        HOST_DEVICE_FOR_CUDA static inline float LinearRGB2sRGB(float lC)
        {
            if (lC <= 0.0031308f)
            {
                return lC * 12.92f;
            }

            return 1.055f * powf(lC, 1.0f / 2.4f) - 0.055f;
        }

        HOST_DEVICE_FOR_CUDA static inline color3 sRGB2LinearRGB(color3 sRGB)
        {
            float R = sRGB2LinearRGB(sRGB.x);
            float G = sRGB2LinearRGB(sRGB.y);
            float B = sRGB2LinearRGB(sRGB.z);

            return color3(R, G, B);
        }

        HOST_DEVICE_FOR_CUDA static inline color3 LinearRGB2sRGB(color3 RGB)
        {
            float sR = LinearRGB2sRGB(RGB.x);
            float sG = LinearRGB2sRGB(RGB.y);
            float sB = LinearRGB2sRGB(RGB.z);

            return color3(sR, sG, sB);
        }

        HOST_DEVICE_FOR_CUDA static inline color3 LinearRGB2XYZ(color3 RGB)
        {
            // Source: https://www.image-engineering.de/library/technotes/958-how-to-convert-between-srgb-and-ciexyz
            // Assumes D65 standard illuminant.
            const float a11 = 10135552.0f / 24577794.0f;
            const float a12 = 8788810.0f / 24577794.0f;
            const float a13 = 4435075.0f / 24577794.0f;
            const float a21 = 2613072.0f / 12288897.0f;
            const float a22 = 8788810.0f / 12288897.0f;
            const float a23 = 887015.0f / 12288897.0f;
            const float a31 = 1425312.0f / 73733382.0f;
            const float a32 = 8788810.0f / 73733382.0f;
            const float a33 = 70074185.0f / 73733382.0f;

            color3 XYZ;
            XYZ.x = a11 * RGB.x + a12 * RGB.y + a13 * RGB.z;
            XYZ.y = a21 * RGB.x + a22 * RGB.y + a23 * RGB.z;
            XYZ.z = a31 * RGB.x + a32 * RGB.y + a33 * RGB.z;
            return XYZ;
        }

        HOST_DEVICE_FOR_CUDA static inline color3 XYZ2LinearRGB(color3 XYZ)
        {
            // Return values in linear RGB, assuming D65 standard illuminant.
            const float a11 = 3.241003275f;
            const float a12 = -1.537398934f;
            const float a13 = -0.498615861f;
            const float a21 = -0.969224334f;
            const float a22 = 1.875930071f;
            const float a23 = 0.041554224f;
            const float a31 = 0.055639423f;
            const float a32 = -0.204011202f;
            const float a33 = 1.057148933f;

            color3 RGB;
            RGB.x = a11 * XYZ.x + a12 * XYZ.y + a13 * XYZ.z;
            RGB.y = a21 * XYZ.x + a22 * XYZ.y + a23 * XYZ.z;
            RGB.z = a31 * XYZ.x + a32 * XYZ.y + a33 * XYZ.z;
            return RGB;
        }

        HOST_DEVICE_FOR_CUDA static inline color3 XYZ2CIELab(color3 XYZ, const color3 invReferenceIlluminant = INV_DEFAULT_ILLUMINANT)
        {
            const float delta = 6.0f / 29.0f;
            const float deltaSquare = delta * delta;
            const float deltaCube = delta * deltaSquare;
            const float factor = 1.0f / (3.0f * deltaSquare);
            const float term = 4.0f / 29.0f;

            // The default illuminant is D65.
            XYZ = XYZ * invReferenceIlluminant;
            XYZ.x = (XYZ.x > deltaCube ? powf(XYZ.x, 1.0f / 3.0f) : factor * XYZ.x + term);
            XYZ.y = (XYZ.y > deltaCube ? powf(XYZ.y, 1.0f / 3.0f) : factor * XYZ.y + term);
            XYZ.z = (XYZ.z > deltaCube ? powf(XYZ.z, 1.0f / 3.0f) : factor * XYZ.z + term);
            float L = 116.0f * XYZ.y - 16.0f;
            float a = 500.0f * (XYZ.x - XYZ.y);
            float b = 200.0f * (XYZ.y - XYZ.z);
            return color3(L, a, b);
        }

        HOST_DEVICE_FOR_CUDA static inline color3 CIELab2XYZ(color3 Lab, const color3 referenceIlluminant = DEFAULT_ILLUMINANT)
        {
            // The default illuminant is D65.
            float Y = (Lab.x + 16.0f) / 116.0f;
            float X = Lab.y / 500.0f + Y;
            float Z = Y - Lab.z / 200.0f;

            const float delta = 6.0f / 29.0f;
            const float factor = 3.0f * delta * delta;
            const float term = 4.0f / 29.0f;
            X = (X > delta ? X * X * X : (X - term) * factor);
            Y = (Y > delta ? Y * Y * Y : (Y - term) * factor);
            Z = (Z > delta ? Z * Z * Z : (Z - term) * factor);
            return color3(X, Y, Z) * referenceIlluminant;
        }

        HOST_DEVICE_FOR_CUDA static inline color3 XYZ2YCxCz(color3 XYZ, const color3 invReferenceIlluminant = INV_DEFAULT_ILLUMINANT)
        {
            // The default illuminant is D65.
            XYZ = XYZ * invReferenceIlluminant;
            float Y = 116.0f * XYZ.y - 16.0f;
            float Cx = 500.0f * (XYZ.x - XYZ.y);
            float Cz = 200.0f * (XYZ.y - XYZ.z);
            return color3(Y, Cx, Cz);
        }

        HOST_DEVICE_FOR_CUDA static inline color3 YCxCz2XYZ(color3 YCxCz, const color3 referenceIlluminant = DEFAULT_ILLUMINANT)
        {
            // The default illuminant is D65.
            const float Y = (YCxCz.x + 16.0f) / 116.0f;
            const float Cx = YCxCz.y / 500.0f;
            const float Cz = YCxCz.z / 200.0f;
            float X = Y + Cx;
            float Z = Y - Cz;
            return color3(X, Y, Z) * referenceIlluminant;
        }

        HOST_DEVICE_FOR_CUDA static inline float YCxCz2Gray(color3 YCxCz)
        {
            return (YCxCz.x + 16.0f) / 116.0f; // Make it [0,1].
        }

        // FLIP-specific functions below.
        HOST_DEVICE_FOR_CUDA static inline float Hunt(const float luminance, const float chrominance)
        {
            return 0.01f * luminance * chrominance;
        }

        HOST_DEVICE_FOR_CUDA static inline float HyAB(color3& refPixel, color3& testPixel)
        {
            float cityBlockDistanceL = std::fabs(refPixel.x - testPixel.x);
            float euclideanDistanceAB = std::sqrt((refPixel.y - testPixel.y) * (refPixel.y - testPixel.y) + (refPixel.z - testPixel.z) * (refPixel.z - testPixel.z));
            return cityBlockDistanceL + euclideanDistanceAB;
        }

        HOST_DEVICE_FOR_CUDA static inline float computeMaxDistance(float gqc)
        {
            color3 greenLab = color3::XYZ2CIELab(color3::LinearRGB2XYZ(color3(0.0f, 1.0f, 0.0f)));
            color3 blueLab = color3::XYZ2CIELab(color3::LinearRGB2XYZ(color3(0.0f, 0.0f, 1.0f)));
            color3 greenLabHunt = color3(greenLab.x, Hunt(greenLab.x, greenLab.y), Hunt(greenLab.x, greenLab.z));
            color3 blueLabHunt = color3(blueLab.x, Hunt(blueLab.x, blueLab.y), Hunt(blueLab.x, blueLab.z));
            return powf(HyAB(greenLabHunt, blueLabHunt), gqc);
        }

        std::string toString(void) const { return "(" + std::to_string(this->x) + ", " + std::to_string(this->y) + ", " + std::to_string(this->z) + ")"; }
        friend std::ostream& operator<<(std::ostream& os, const color3& c) { os << c.toString(); return os; }
    };

    static const color3 MapMagma[256] =
    {
        {0.001462f, 0.000466f, 0.013866f}, {0.002258f, 0.001295f, 0.018331f}, {0.003279f, 0.002305f, 0.023708f}, {0.004512f, 0.003490f, 0.029965f}, {0.005950f, 0.004843f, 0.037130f}, {0.007588f, 0.006356f, 0.044973f}, {0.009426f, 0.008022f, 0.052844f}, {0.011465f, 0.009828f, 0.060750f},
        {0.013708f, 0.011771f, 0.068667f}, {0.016156f, 0.013840f, 0.076603f}, {0.018815f, 0.016026f, 0.084584f}, {0.021692f, 0.018320f, 0.092610f}, {0.024792f, 0.020715f, 0.100676f}, {0.028123f, 0.023201f, 0.108787f}, {0.031696f, 0.025765f, 0.116965f}, {0.035520f, 0.028397f, 0.125209f},
        {0.039608f, 0.031090f, 0.133515f}, {0.043830f, 0.033830f, 0.141886f}, {0.048062f, 0.036607f, 0.150327f}, {0.052320f, 0.039407f, 0.158841f}, {0.056615f, 0.042160f, 0.167446f}, {0.060949f, 0.044794f, 0.176129f}, {0.065330f, 0.047318f, 0.184892f}, {0.069764f, 0.049726f, 0.193735f},
        {0.074257f, 0.052017f, 0.202660f}, {0.078815f, 0.054184f, 0.211667f}, {0.083446f, 0.056225f, 0.220755f}, {0.088155f, 0.058133f, 0.229922f}, {0.092949f, 0.059904f, 0.239164f}, {0.097833f, 0.061531f, 0.248477f}, {0.102815f, 0.063010f, 0.257854f}, {0.107899f, 0.064335f, 0.267289f},
        {0.113094f, 0.065492f, 0.276784f}, {0.118405f, 0.066479f, 0.286321f}, {0.123833f, 0.067295f, 0.295879f}, {0.129380f, 0.067935f, 0.305443f}, {0.135053f, 0.068391f, 0.315000f}, {0.140858f, 0.068654f, 0.324538f}, {0.146785f, 0.068738f, 0.334011f}, {0.152839f, 0.068637f, 0.343404f},
        {0.159018f, 0.068354f, 0.352688f}, {0.165308f, 0.067911f, 0.361816f}, {0.171713f, 0.067305f, 0.370771f}, {0.178212f, 0.066576f, 0.379497f}, {0.184801f, 0.065732f, 0.387973f}, {0.191460f, 0.064818f, 0.396152f}, {0.198177f, 0.063862f, 0.404009f}, {0.204935f, 0.062907f, 0.411514f},
        {0.211718f, 0.061992f, 0.418647f}, {0.218512f, 0.061158f, 0.425392f}, {0.225302f, 0.060445f, 0.431742f}, {0.232077f, 0.059889f, 0.437695f}, {0.238826f, 0.059517f, 0.443256f}, {0.245543f, 0.059352f, 0.448436f}, {0.252220f, 0.059415f, 0.453248f}, {0.258857f, 0.059706f, 0.457710f},
        {0.265447f, 0.060237f, 0.461840f}, {0.271994f, 0.060994f, 0.465660f}, {0.278493f, 0.061978f, 0.469190f}, {0.284951f, 0.063168f, 0.472451f}, {0.291366f, 0.064553f, 0.475462f}, {0.297740f, 0.066117f, 0.478243f}, {0.304081f, 0.067835f, 0.480812f}, {0.310382f, 0.069702f, 0.483186f},
        {0.316654f, 0.071690f, 0.485380f}, {0.322899f, 0.073782f, 0.487408f}, {0.329114f, 0.075972f, 0.489287f}, {0.335308f, 0.078236f, 0.491024f}, {0.341482f, 0.080564f, 0.492631f}, {0.347636f, 0.082946f, 0.494121f}, {0.353773f, 0.085373f, 0.495501f}, {0.359898f, 0.087831f, 0.496778f},
        {0.366012f, 0.090314f, 0.497960f}, {0.372116f, 0.092816f, 0.499053f}, {0.378211f, 0.095332f, 0.500067f}, {0.384299f, 0.097855f, 0.501002f}, {0.390384f, 0.100379f, 0.501864f}, {0.396467f, 0.102902f, 0.502658f}, {0.402548f, 0.105420f, 0.503386f}, {0.408629f, 0.107930f, 0.504052f},
        {0.414709f, 0.110431f, 0.504662f}, {0.420791f, 0.112920f, 0.505215f}, {0.426877f, 0.115395f, 0.505714f}, {0.432967f, 0.117855f, 0.506160f}, {0.439062f, 0.120298f, 0.506555f}, {0.445163f, 0.122724f, 0.506901f}, {0.451271f, 0.125132f, 0.507198f}, {0.457386f, 0.127522f, 0.507448f},
        {0.463508f, 0.129893f, 0.507652f}, {0.469640f, 0.132245f, 0.507809f}, {0.475780f, 0.134577f, 0.507921f}, {0.481929f, 0.136891f, 0.507989f}, {0.488088f, 0.139186f, 0.508011f}, {0.494258f, 0.141462f, 0.507988f}, {0.500438f, 0.143719f, 0.507920f}, {0.506629f, 0.145958f, 0.507806f},
        {0.512831f, 0.148179f, 0.507648f}, {0.519045f, 0.150383f, 0.507443f}, {0.525270f, 0.152569f, 0.507192f}, {0.531507f, 0.154739f, 0.506895f}, {0.537755f, 0.156894f, 0.506551f}, {0.544015f, 0.159033f, 0.506159f}, {0.550287f, 0.161158f, 0.505719f}, {0.556571f, 0.163269f, 0.505230f},
        {0.562866f, 0.165368f, 0.504692f}, {0.569172f, 0.167454f, 0.504105f}, {0.575490f, 0.169530f, 0.503466f}, {0.581819f, 0.171596f, 0.502777f}, {0.588158f, 0.173652f, 0.502035f}, {0.594508f, 0.175701f, 0.501241f}, {0.600868f, 0.177743f, 0.500394f}, {0.607238f, 0.179779f, 0.499492f},
        {0.613617f, 0.181811f, 0.498536f}, {0.620005f, 0.183840f, 0.497524f}, {0.626401f, 0.185867f, 0.496456f}, {0.632805f, 0.187893f, 0.495332f}, {0.639216f, 0.189921f, 0.494150f}, {0.645633f, 0.191952f, 0.492910f}, {0.652056f, 0.193986f, 0.491611f}, {0.658483f, 0.196027f, 0.490253f},
        {0.664915f, 0.198075f, 0.488836f}, {0.671349f, 0.200133f, 0.487358f}, {0.677786f, 0.202203f, 0.485819f}, {0.684224f, 0.204286f, 0.484219f}, {0.690661f, 0.206384f, 0.482558f}, {0.697098f, 0.208501f, 0.480835f}, {0.703532f, 0.210638f, 0.479049f}, {0.709962f, 0.212797f, 0.477201f},
        {0.716387f, 0.214982f, 0.475290f}, {0.722805f, 0.217194f, 0.473316f}, {0.729216f, 0.219437f, 0.471279f}, {0.735616f, 0.221713f, 0.469180f}, {0.742004f, 0.224025f, 0.467018f}, {0.748378f, 0.226377f, 0.464794f}, {0.754737f, 0.228772f, 0.462509f}, {0.761077f, 0.231214f, 0.460162f},
        {0.767398f, 0.233705f, 0.457755f}, {0.773695f, 0.236249f, 0.455289f}, {0.779968f, 0.238851f, 0.452765f}, {0.786212f, 0.241514f, 0.450184f}, {0.792427f, 0.244242f, 0.447543f}, {0.798608f, 0.247040f, 0.444848f}, {0.804752f, 0.249911f, 0.442102f}, {0.810855f, 0.252861f, 0.439305f},
        {0.816914f, 0.255895f, 0.436461f}, {0.822926f, 0.259016f, 0.433573f}, {0.828886f, 0.262229f, 0.430644f}, {0.834791f, 0.265540f, 0.427671f}, {0.840636f, 0.268953f, 0.424666f}, {0.846416f, 0.272473f, 0.421631f}, {0.852126f, 0.276106f, 0.418573f}, {0.857763f, 0.279857f, 0.415496f},
        {0.863320f, 0.283729f, 0.412403f}, {0.868793f, 0.287728f, 0.409303f}, {0.874176f, 0.291859f, 0.406205f}, {0.879464f, 0.296125f, 0.403118f}, {0.884651f, 0.300530f, 0.400047f}, {0.889731f, 0.305079f, 0.397002f}, {0.894700f, 0.309773f, 0.393995f}, {0.899552f, 0.314616f, 0.391037f},
        {0.904281f, 0.319610f, 0.388137f}, {0.908884f, 0.324755f, 0.385308f}, {0.913354f, 0.330052f, 0.382563f}, {0.917689f, 0.335500f, 0.379915f}, {0.921884f, 0.341098f, 0.377376f}, {0.925937f, 0.346844f, 0.374959f}, {0.929845f, 0.352734f, 0.372677f}, {0.933606f, 0.358764f, 0.370541f},
        {0.937221f, 0.364929f, 0.368567f}, {0.940687f, 0.371224f, 0.366762f}, {0.944006f, 0.377643f, 0.365136f}, {0.947180f, 0.384178f, 0.363701f}, {0.950210f, 0.390820f, 0.362468f}, {0.953099f, 0.397563f, 0.361438f}, {0.955849f, 0.404400f, 0.360619f}, {0.958464f, 0.411324f, 0.360014f},
        {0.960949f, 0.418323f, 0.359630f}, {0.963310f, 0.425390f, 0.359469f}, {0.965549f, 0.432519f, 0.359529f}, {0.967671f, 0.439703f, 0.359810f}, {0.969680f, 0.446936f, 0.360311f}, {0.971582f, 0.454210f, 0.361030f}, {0.973381f, 0.461520f, 0.361965f}, {0.975082f, 0.468861f, 0.363111f},
        {0.976690f, 0.476226f, 0.364466f}, {0.978210f, 0.483612f, 0.366025f}, {0.979645f, 0.491014f, 0.367783f}, {0.981000f, 0.498428f, 0.369734f}, {0.982279f, 0.505851f, 0.371874f}, {0.983485f, 0.513280f, 0.374198f}, {0.984622f, 0.520713f, 0.376698f}, {0.985693f, 0.528148f, 0.379371f},
        {0.986700f, 0.535582f, 0.382210f}, {0.987646f, 0.543015f, 0.385210f}, {0.988533f, 0.550446f, 0.388365f}, {0.989363f, 0.557873f, 0.391671f}, {0.990138f, 0.565296f, 0.395122f}, {0.990871f, 0.572706f, 0.398714f}, {0.991558f, 0.580107f, 0.402441f}, {0.992196f, 0.587502f, 0.406299f},
        {0.992785f, 0.594891f, 0.410283f}, {0.993326f, 0.602275f, 0.414390f}, {0.993834f, 0.609644f, 0.418613f}, {0.994309f, 0.616999f, 0.422950f}, {0.994738f, 0.624350f, 0.427397f}, {0.995122f, 0.631696f, 0.431951f}, {0.995480f, 0.639027f, 0.436607f}, {0.995810f, 0.646344f, 0.441361f},
        {0.996096f, 0.653659f, 0.446213f}, {0.996341f, 0.660969f, 0.451160f}, {0.996580f, 0.668256f, 0.456192f}, {0.996775f, 0.675541f, 0.461314f}, {0.996925f, 0.682828f, 0.466526f}, {0.997077f, 0.690088f, 0.471811f}, {0.997186f, 0.697349f, 0.477182f}, {0.997254f, 0.704611f, 0.482635f},
        {0.997325f, 0.711848f, 0.488154f}, {0.997351f, 0.719089f, 0.493755f}, {0.997351f, 0.726324f, 0.499428f}, {0.997341f, 0.733545f, 0.505167f}, {0.997285f, 0.740772f, 0.510983f}, {0.997228f, 0.747981f, 0.516859f}, {0.997138f, 0.755190f, 0.522806f}, {0.997019f, 0.762398f, 0.528821f},
        {0.996898f, 0.769591f, 0.534892f}, {0.996727f, 0.776795f, 0.541039f}, {0.996571f, 0.783977f, 0.547233f}, {0.996369f, 0.791167f, 0.553499f}, {0.996162f, 0.798348f, 0.559820f}, {0.995932f, 0.805527f, 0.566202f}, {0.995680f, 0.812706f, 0.572645f}, {0.995424f, 0.819875f, 0.579140f},
        {0.995131f, 0.827052f, 0.585701f}, {0.994851f, 0.834213f, 0.592307f}, {0.994524f, 0.841387f, 0.598983f}, {0.994222f, 0.848540f, 0.605696f}, {0.993866f, 0.855711f, 0.612482f}, {0.993545f, 0.862859f, 0.619299f}, {0.993170f, 0.870024f, 0.626189f}, {0.992831f, 0.877168f, 0.633109f},
        {0.992440f, 0.884330f, 0.640099f}, {0.992089f, 0.891470f, 0.647116f}, {0.991688f, 0.898627f, 0.654202f}, {0.991332f, 0.905763f, 0.661309f}, {0.990930f, 0.912915f, 0.668481f}, {0.990570f, 0.920049f, 0.675675f}, {0.990175f, 0.927196f, 0.682926f}, {0.989815f, 0.934329f, 0.690198f},
        {0.989434f, 0.941470f, 0.697519f}, {0.989077f, 0.948604f, 0.704863f}, {0.988717f, 0.955742f, 0.712242f}, {0.988367f, 0.962878f, 0.719649f}, {0.988033f, 0.970012f, 0.727077f}, {0.987691f, 0.977154f, 0.734536f}, {0.987387f, 0.984288f, 0.742002f}, {0.987053f, 0.991438f, 0.749504f}
    };

    static const color3 MapViridis[256] =
    {
        {0.267004f, 0.004874f, 0.329415f}, {0.268510f, 0.009605f, 0.335427f}, {0.269944f, 0.014625f, 0.341379f}, {0.271305f, 0.019942f, 0.347269f}, {0.272594f, 0.025563f, 0.353093f}, {0.273809f, 0.031497f, 0.358853f}, {0.274952f, 0.037752f, 0.364543f}, {0.276022f, 0.044167f, 0.370164f},
        {0.277018f, 0.050344f, 0.375715f}, {0.277941f, 0.056324f, 0.381191f}, {0.278791f, 0.062145f, 0.386592f}, {0.279566f, 0.067836f, 0.391917f}, {0.280267f, 0.073417f, 0.397163f}, {0.280894f, 0.078907f, 0.402329f}, {0.281446f, 0.084320f, 0.407414f}, {0.281924f, 0.089666f, 0.412415f},
        {0.282327f, 0.094955f, 0.417331f}, {0.282656f, 0.100196f, 0.422160f}, {0.282910f, 0.105393f, 0.426902f}, {0.283091f, 0.110553f, 0.431554f}, {0.283197f, 0.115680f, 0.436115f}, {0.283229f, 0.120777f, 0.440584f}, {0.283187f, 0.125848f, 0.444960f}, {0.283072f, 0.130895f, 0.449241f},
        {0.282884f, 0.135920f, 0.453427f}, {0.282623f, 0.140926f, 0.457517f}, {0.282290f, 0.145912f, 0.461510f}, {0.281887f, 0.150881f, 0.465405f}, {0.281412f, 0.155834f, 0.469201f}, {0.280868f, 0.160771f, 0.472899f}, {0.280255f, 0.165693f, 0.476498f}, {0.279574f, 0.170599f, 0.479997f},
        {0.278826f, 0.175490f, 0.483397f}, {0.278012f, 0.180367f, 0.486697f}, {0.277134f, 0.185228f, 0.489898f}, {0.276194f, 0.190074f, 0.493001f}, {0.275191f, 0.194905f, 0.496005f}, {0.274128f, 0.199721f, 0.498911f}, {0.273006f, 0.204520f, 0.501721f}, {0.271828f, 0.209303f, 0.504434f},
        {0.270595f, 0.214069f, 0.507052f}, {0.269308f, 0.218818f, 0.509577f}, {0.267968f, 0.223549f, 0.512008f}, {0.266580f, 0.228262f, 0.514349f}, {0.265145f, 0.232956f, 0.516599f}, {0.263663f, 0.237631f, 0.518762f}, {0.262138f, 0.242286f, 0.520837f}, {0.260571f, 0.246922f, 0.522828f},
        {0.258965f, 0.251537f, 0.524736f}, {0.257322f, 0.256130f, 0.526563f}, {0.255645f, 0.260703f, 0.528312f}, {0.253935f, 0.265254f, 0.529983f}, {0.252194f, 0.269783f, 0.531579f}, {0.250425f, 0.274290f, 0.533103f}, {0.248629f, 0.278775f, 0.534556f}, {0.246811f, 0.283237f, 0.535941f},
        {0.244972f, 0.287675f, 0.537260f}, {0.243113f, 0.292092f, 0.538516f}, {0.241237f, 0.296485f, 0.539709f}, {0.239346f, 0.300855f, 0.540844f}, {0.237441f, 0.305202f, 0.541921f}, {0.235526f, 0.309527f, 0.542944f}, {0.233603f, 0.313828f, 0.543914f}, {0.231674f, 0.318106f, 0.544834f},
        {0.229739f, 0.322361f, 0.545706f}, {0.227802f, 0.326594f, 0.546532f}, {0.225863f, 0.330805f, 0.547314f}, {0.223925f, 0.334994f, 0.548053f}, {0.221989f, 0.339161f, 0.548752f}, {0.220057f, 0.343307f, 0.549413f}, {0.218130f, 0.347432f, 0.550038f}, {0.216210f, 0.351535f, 0.550627f},
        {0.214298f, 0.355619f, 0.551184f}, {0.212395f, 0.359683f, 0.551710f}, {0.210503f, 0.363727f, 0.552206f}, {0.208623f, 0.367752f, 0.552675f}, {0.206756f, 0.371758f, 0.553117f}, {0.204903f, 0.375746f, 0.553533f}, {0.203063f, 0.379716f, 0.553925f}, {0.201239f, 0.383670f, 0.554294f},
        {0.199430f, 0.387607f, 0.554642f}, {0.197636f, 0.391528f, 0.554969f}, {0.195860f, 0.395433f, 0.555276f}, {0.194100f, 0.399323f, 0.555565f}, {0.192357f, 0.403199f, 0.555836f}, {0.190631f, 0.407061f, 0.556089f}, {0.188923f, 0.410910f, 0.556326f}, {0.187231f, 0.414746f, 0.556547f},
        {0.185556f, 0.418570f, 0.556753f}, {0.183898f, 0.422383f, 0.556944f}, {0.182256f, 0.426184f, 0.557120f}, {0.180629f, 0.429975f, 0.557282f}, {0.179019f, 0.433756f, 0.557430f}, {0.177423f, 0.437527f, 0.557565f}, {0.175841f, 0.441290f, 0.557685f}, {0.174274f, 0.445044f, 0.557792f},
        {0.172719f, 0.448791f, 0.557885f}, {0.171176f, 0.452530f, 0.557965f}, {0.169646f, 0.456262f, 0.558030f}, {0.168126f, 0.459988f, 0.558082f}, {0.166617f, 0.463708f, 0.558119f}, {0.165117f, 0.467423f, 0.558141f}, {0.163625f, 0.471133f, 0.558148f}, {0.162142f, 0.474838f, 0.558140f},
        {0.160665f, 0.478540f, 0.558115f}, {0.159194f, 0.482237f, 0.558073f}, {0.157729f, 0.485932f, 0.558013f}, {0.156270f, 0.489624f, 0.557936f}, {0.154815f, 0.493313f, 0.557840f}, {0.153364f, 0.497000f, 0.557724f}, {0.151918f, 0.500685f, 0.557587f}, {0.150476f, 0.504369f, 0.557430f},
        {0.149039f, 0.508051f, 0.557250f}, {0.147607f, 0.511733f, 0.557049f}, {0.146180f, 0.515413f, 0.556823f}, {0.144759f, 0.519093f, 0.556572f}, {0.143343f, 0.522773f, 0.556295f}, {0.141935f, 0.526453f, 0.555991f}, {0.140536f, 0.530132f, 0.555659f}, {0.139147f, 0.533812f, 0.555298f},
        {0.137770f, 0.537492f, 0.554906f}, {0.136408f, 0.541173f, 0.554483f}, {0.135066f, 0.544853f, 0.554029f}, {0.133743f, 0.548535f, 0.553541f}, {0.132444f, 0.552216f, 0.553018f}, {0.131172f, 0.555899f, 0.552459f}, {0.129933f, 0.559582f, 0.551864f}, {0.128729f, 0.563265f, 0.551229f},
        {0.127568f, 0.566949f, 0.550556f}, {0.126453f, 0.570633f, 0.549841f}, {0.125394f, 0.574318f, 0.549086f}, {0.124395f, 0.578002f, 0.548287f}, {0.123463f, 0.581687f, 0.547445f}, {0.122606f, 0.585371f, 0.546557f}, {0.121831f, 0.589055f, 0.545623f}, {0.121148f, 0.592739f, 0.544641f},
        {0.120565f, 0.596422f, 0.543611f}, {0.120092f, 0.600104f, 0.542530f}, {0.119738f, 0.603785f, 0.541400f}, {0.119512f, 0.607464f, 0.540218f}, {0.119423f, 0.611141f, 0.538982f}, {0.119483f, 0.614817f, 0.537692f}, {0.119699f, 0.618490f, 0.536347f}, {0.120081f, 0.622161f, 0.534946f},
        {0.120638f, 0.625828f, 0.533488f}, {0.121380f, 0.629492f, 0.531973f}, {0.122312f, 0.633153f, 0.530398f}, {0.123444f, 0.636809f, 0.528763f}, {0.124780f, 0.640461f, 0.527068f}, {0.126326f, 0.644107f, 0.525311f}, {0.128087f, 0.647749f, 0.523491f}, {0.130067f, 0.651384f, 0.521608f},
        {0.132268f, 0.655014f, 0.519661f}, {0.134692f, 0.658636f, 0.517649f}, {0.137339f, 0.662252f, 0.515571f}, {0.140210f, 0.665859f, 0.513427f}, {0.143303f, 0.669459f, 0.511215f}, {0.146616f, 0.673050f, 0.508936f}, {0.150148f, 0.676631f, 0.506589f}, {0.153894f, 0.680203f, 0.504172f},
        {0.157851f, 0.683765f, 0.501686f}, {0.162016f, 0.687316f, 0.499129f}, {0.166383f, 0.690856f, 0.496502f}, {0.170948f, 0.694384f, 0.493803f}, {0.175707f, 0.697900f, 0.491033f}, {0.180653f, 0.701402f, 0.488189f}, {0.185783f, 0.704891f, 0.485273f}, {0.191090f, 0.708366f, 0.482284f},
        {0.196571f, 0.711827f, 0.479221f}, {0.202219f, 0.715272f, 0.476084f}, {0.208030f, 0.718701f, 0.472873f}, {0.214000f, 0.722114f, 0.469588f}, {0.220124f, 0.725509f, 0.466226f}, {0.226397f, 0.728888f, 0.462789f}, {0.232815f, 0.732247f, 0.459277f}, {0.239374f, 0.735588f, 0.455688f},
        {0.246070f, 0.738910f, 0.452024f}, {0.252899f, 0.742211f, 0.448284f}, {0.259857f, 0.745492f, 0.444467f}, {0.266941f, 0.748751f, 0.440573f}, {0.274149f, 0.751988f, 0.436601f}, {0.281477f, 0.755203f, 0.432552f}, {0.288921f, 0.758394f, 0.428426f}, {0.296479f, 0.761561f, 0.424223f},
        {0.304148f, 0.764704f, 0.419943f}, {0.311925f, 0.767822f, 0.415586f}, {0.319809f, 0.770914f, 0.411152f}, {0.327796f, 0.773980f, 0.406640f}, {0.335885f, 0.777018f, 0.402049f}, {0.344074f, 0.780029f, 0.397381f}, {0.352360f, 0.783011f, 0.392636f}, {0.360741f, 0.785964f, 0.387814f},
        {0.369214f, 0.788888f, 0.382914f}, {0.377779f, 0.791781f, 0.377939f}, {0.386433f, 0.794644f, 0.372886f}, {0.395174f, 0.797475f, 0.367757f}, {0.404001f, 0.800275f, 0.362552f}, {0.412913f, 0.803041f, 0.357269f}, {0.421908f, 0.805774f, 0.351910f}, {0.430983f, 0.808473f, 0.346476f},
        {0.440137f, 0.811138f, 0.340967f}, {0.449368f, 0.813768f, 0.335384f}, {0.458674f, 0.816363f, 0.329727f}, {0.468053f, 0.818921f, 0.323998f}, {0.477504f, 0.821444f, 0.318195f}, {0.487026f, 0.823929f, 0.312321f}, {0.496615f, 0.826376f, 0.306377f}, {0.506271f, 0.828786f, 0.300362f},
        {0.515992f, 0.831158f, 0.294279f}, {0.525776f, 0.833491f, 0.288127f}, {0.535621f, 0.835785f, 0.281908f}, {0.545524f, 0.838039f, 0.275626f}, {0.555484f, 0.840254f, 0.269281f}, {0.565498f, 0.842430f, 0.262877f}, {0.575563f, 0.844566f, 0.256415f}, {0.585678f, 0.846661f, 0.249897f},
        {0.595839f, 0.848717f, 0.243329f}, {0.606045f, 0.850733f, 0.236712f}, {0.616293f, 0.852709f, 0.230052f}, {0.626579f, 0.854645f, 0.223353f}, {0.636902f, 0.856542f, 0.216620f}, {0.647257f, 0.858400f, 0.209861f}, {0.657642f, 0.860219f, 0.203082f}, {0.668054f, 0.861999f, 0.196293f},
        {0.678489f, 0.863742f, 0.189503f}, {0.688944f, 0.865448f, 0.182725f}, {0.699415f, 0.867117f, 0.175971f}, {0.709898f, 0.868751f, 0.169257f}, {0.720391f, 0.870350f, 0.162603f}, {0.730889f, 0.871916f, 0.156029f}, {0.741388f, 0.873449f, 0.149561f}, {0.751884f, 0.874951f, 0.143228f},
        {0.762373f, 0.876424f, 0.137064f}, {0.772852f, 0.877868f, 0.131109f}, {0.783315f, 0.879285f, 0.125405f}, {0.793760f, 0.880678f, 0.120005f}, {0.804182f, 0.882046f, 0.114965f}, {0.814576f, 0.883393f, 0.110347f}, {0.824940f, 0.884720f, 0.106217f}, {0.835270f, 0.886029f, 0.102646f},
        {0.845561f, 0.887322f, 0.099702f}, {0.855810f, 0.888601f, 0.097452f}, {0.866013f, 0.889868f, 0.095953f}, {0.876168f, 0.891125f, 0.095250f}, {0.886271f, 0.892374f, 0.095374f}, {0.896320f, 0.893616f, 0.096335f}, {0.906311f, 0.894855f, 0.098125f}, {0.916242f, 0.896091f, 0.100717f},
        {0.926106f, 0.897330f, 0.104071f}, {0.935904f, 0.898570f, 0.108131f}, {0.945636f, 0.899815f, 0.112838f}, {0.955300f, 0.901065f, 0.118128f}, {0.964894f, 0.902323f, 0.123941f}, {0.974417f, 0.903590f, 0.130215f}, {0.983868f, 0.904867f, 0.136897f}, {0.993248f, 0.906157f, 0.143936f}
    };

    static const struct xGaussianConstants
    {
        xGaussianConstants() = default;
        color3 a1 = { 1.0f, 1.0f, 34.1f };
        color3 b1 = { 0.0047f, 0.0053f, 0.04f };
        color3 a2 = { 0.0f, 0.0f, 13.5f };
        color3 b2 = { 1.0e-5f, 1.0e-5f, 0.025f };
    } GaussianConstants;  // Constants for Gaussians -- see paper for details.

    static inline float Gaussian(const float x, const float sigma) // 1D Gaussian (without normalization factor).
    {
        return std::exp(-(x * x) / (2.0f * sigma * sigma));
    }

    static inline float Gaussian(const float x2, const float a, const float b) // 1D Gaussian in alternative form (see FLIP paper).
    {
        const float pi = float(PI);
        const float pi_sq = float(PI * PI);
        return a * std::sqrt(pi / b) * std::exp(-pi_sq * x2 / b);
    }

    // This function is needed to separate sum of Gaussians filters See separatedConvolutions.pdf in the FLIP repository:
    // https://github.com/NVlabs/flip/blob/main/misc/separatedConvolutions.pdf
    static inline float GaussianSqrt(const float x2, const float a, const float b)
    {
        const float pi = float(PI);
        const float pi_sq = float(PI * PI);
        return std::sqrt(a * std::sqrt(pi / b)) * std::exp(-pi_sq * x2 / b);
    }

    static inline void solveSecondDegree(float& xMin, float& xMax, float a, float b, float c)
    {
        //  a * x^2 + b * x + c = 0
        if (a == 0.0f)
        {
            xMin = xMax = -c / b;
            return;
        }

        float d1 = -0.5f * (b / a);
        float d2 = sqrtf((d1 * d1) - (c / a));
        xMin = d1 - d2;
        xMax = d1 + d2;
    }

    static int calculateSpatialFilterRadius(const float ppd)
    {
        const float pi_sq = float(PI * PI);

        float maxScaleParameter = std::max(std::max(std::max(GaussianConstants.b1.x, GaussianConstants.b1.y), std::max(GaussianConstants.b1.z, GaussianConstants.b2.x)), std::max(GaussianConstants.b2.y, GaussianConstants.b2.z));
        int radius = int(std::ceil(3.0f * std::sqrt(maxScaleParameter / (2.0f * pi_sq)) * ppd)); // Set radius based on largest scale parameter.

        return radius;
    }

    // CUDA kernels.
#ifdef FLIP_ENABLE_CUDA
    __global__ static void kernelColorMap(color3* pDstImage, const float* pSrcImage, const color3* pColorMap, const int3 dim, const int mapSize)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int i = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y || z >= dim.z) return;
        pDstImage[i] = pColorMap[int(pSrcImage[i] * 255.0f + 0.5f) % mapSize];
    }

    __global__ static void kernelFloat2Color3(color3* pDstImage, float* pSrcImage, const int3 dim)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int i = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y || z >= dim.z) return;
        pDstImage[i] = color3(pSrcImage[i]);
    }

    __global__ static void kernelFinalError(float* pDstImage, color3* pColorFeatureDifference, const int3 dim)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int i = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y || z >= dim.z) return;

        const float cdiff = pColorFeatureDifference[i].x;
        const float fdiff = pColorFeatureDifference[i].y;
        const float errorFLIP = std::pow(cdiff, 1.0f - fdiff);
        pDstImage[i] = errorFLIP;
    }

    __global__ static void kernelSetMaxExposure(float* pDstErrorMap, float* pSrcErrorMap, float* pExposureMap, const int3 dim, float exposure)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int i = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y || z >= dim.z) return;
        float srcValue = pSrcErrorMap[i];
        float dstValue = pDstErrorMap[i];

        if (srcValue > dstValue)
        {
            pExposureMap[i] = exposure;
            pDstErrorMap[i] = srcValue;
        }
    }

    __global__ static void kernelsRGB2YCxCz(color3* pImage, const int3 dim)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int i = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y || z >= dim.z) return;
        pImage[i] = color3::XYZ2YCxCz(color3::LinearRGB2XYZ(color3::sRGB2LinearRGB(pImage[i])));
    }

    __global__ static void kernelsRGB2LinearRGB(color3* pImage, const int3 dim)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int i = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y || z >= dim.z) return;
        pImage[i] = color3::sRGB2LinearRGB(pImage[i]);
    }


    __global__ static void kernelLinearRGB2YCxCz(color3* pImage, const int3 dim)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int i = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y || z >= dim.z) return;
        pImage[i] = color3::XYZ2YCxCz(color3::LinearRGB2XYZ(pImage[i]));
    }

    __global__ static void kernelLinearRGB2sRGB(color3* pImage, const int3 dim)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int i = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y || z >= dim.z) return;
        pImage[i] = color3::LinearRGB2sRGB(pImage[i]);
    }

    //  General kernels.
    __global__ static void kernelClear(color3* pImage, const int3 dim, const color3 color = { 0.0f, 0.0f, 0.0f })
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int i = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y || z >= dim.z) return;
        pImage[i] = color;
    }

    __global__ static void kernelClear(float* pImage, const int3 dim, const float color = 0.0f)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int i = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y || z >= dim.z) return;
        pImage[i] = color;
    }

    __global__ static void kernelMultiplyAndAdd(color3* pImage, const int3 dim, color3 m, color3 a)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int i = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y || z >= dim.z) return;
        pImage[i] = pImage[i] * m + a;
    }

    __global__ static void kernelMultiplyAndAdd(color3* pImage, const int3 dim, float m, float a)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int i = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y || z >= dim.z) return;
        pImage[i] = pImage[i] * m + a;
    }

    __global__ static void kernelMultiplyAndAdd(float* pImage, const int3 dim, float m, float a)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int i = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y || z >= dim.z) return;

        pImage[i] = pImage[i] * m + a;
    }

    __global__ static void kernelToneMap(color3* pImage, const int3 dim, int toneMapper)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int i = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y || z >= dim.z) return;

        if (toneMapper == 0)
        {
            color3 color = pImage[i];
            float luminance = color3::linearRGB2Luminance(color);
            pImage[i] /= (1.0f + luminance);
            return;
        }

        const float* tc = ToneMappingCoefficients[toneMapper];
        color3 color = pImage[i];
        pImage[i] = color3(((color * color) * tc[0] + color * tc[1] + tc[2]) / (color * color * tc[3] + color * tc[4] + tc[5]));
    }

    __global__ static void kernelClamp(color3* pImage, const int3 dim, float low, float high)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int i = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y || z >= dim.z) return;
        pImage[i] = color3::clamp(pImage[i], low, high);
    }

    // Convolve in x direction (1st and 2nd derivative for filter in x direction, Gaussian in y direction).
    // For details on the convolution, see separatedConvolutions.pdf in the FLIP repository:
    // https://github.com/NVlabs/flip/blob/main/misc/separatedConvolutions.pdf.
    // We filter both reference and test image simultaneously (for better performance).
    // referenceImage and testImage are expected to be in YCxCz space.
    __global__ static void kernelFeatureFilterFirstDir(color3* intermediateFeaturesImageReference, color3* referenceImage, color3* intermediateFeaturesImageTest, color3* testImage, color3* pFilter, const int3 dim, const int3 filterDim)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int dstIndex = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y) return;

        const int halfFilterWidth = filterDim.x / 2;

        float dxReference = 0.0f, dxTest = 0.0f, ddxReference = 0.0f, ddxTest = 0.0f;
        float gaussianFilteredReference = 0.0f, gaussianFilteredTest = 0.0f;

        const float oneOver116 = 1.0f / 116.0f;
        const float sixteenOver116 = 16.0f / 116.0f;
        for (int ix = -halfFilterWidth; ix <= halfFilterWidth; ix++)
        {
            int xx = Min(Max(0, x + ix), dim.x - 1);

            int filterIndex = ix + halfFilterWidth;
            int srcIndex = y * dim.x + xx;
            const color3 featureWeights = pFilter[filterIndex];
            float yReference = referenceImage[srcIndex].x;
            float yTest = testImage[srcIndex].x;

            // Normalize the Y values to [0,1].
            float yReferenceNormalized = yReference * oneOver116 + sixteenOver116;
            float yTestNormalized = yTest * oneOver116 + sixteenOver116;

            // Image multiplied by 1st and 2nd x-derivatives of Gaussian.
            dxReference += featureWeights.y * yReferenceNormalized;
            dxTest += featureWeights.y * yTestNormalized;
            ddxReference += featureWeights.z * yReferenceNormalized;
            ddxTest += featureWeights.z * yTestNormalized;

            // Image multiplied by Gaussian.
            gaussianFilteredReference += featureWeights.x * yReferenceNormalized;
            gaussianFilteredTest += featureWeights.x * yTestNormalized;
        }
        intermediateFeaturesImageReference[dstIndex] = color3(dxReference, ddxReference, gaussianFilteredReference);
        intermediateFeaturesImageTest[dstIndex] = color3(dxTest, ddxTest, gaussianFilteredTest);
    }

    // Convolve in y direction (1st and 2nd derivative for filter in y direction, Gaussian in x direction), then compute difference.
    // For details on the convolution, see separatedConvolutions.pdf in the FLIP repository:
    // https://github.com/NVlabs/flip/blob/main/misc/separatedConvolutions.pdf.
    // We filter both reference and test image simultaneously (for better performance).
    __global__ static void kernelFeatureFilterSecondDirAndFeatureDifference(color3* featureDifferenceImage, color3* intermediateFeaturesImageReference, color3* intermediateFeaturesImageTest, color3* pFilter, const int3 dim, const int3 filterDim)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int dstIndex = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y) return;

        const float normalizationFactor = 1.0f / std::sqrt(2.0f);
        const int halfFilterWidth = filterDim.x / 2;

        float dxReference = 0.0f, dxTest = 0.0f, ddxReference = 0.0f, ddxTest = 0.0f;
        float dyReference = 0.0f, dyTest = 0.0f, ddyReference = 0.0f, ddyTest = 0.0f;

        for (int iy = -halfFilterWidth; iy <= halfFilterWidth; iy++)
        {
            int yy = Min(Max(0, y + iy), dim.y - 1);

            int filterIndex = iy + halfFilterWidth;
            int srcIndex = yy * dim.x + x;
            const color3 featureWeights = pFilter[filterIndex];
            const color3 intermediateFeaturesReference = intermediateFeaturesImageReference[srcIndex];
            const color3 intermediateFeatureTest = intermediateFeaturesImageTest[srcIndex];

            // Intermediate images (1st and 2nd derivative in x) multiplied by Gaussian.
            dxReference += featureWeights.x * intermediateFeaturesReference.x;
            dxTest += featureWeights.x * intermediateFeatureTest.x;
            ddxReference += featureWeights.x * intermediateFeaturesReference.y;
            ddxTest += featureWeights.x * intermediateFeatureTest.y;

            // Intermediate image (Gaussian) multiplied by 1st and 2nd y-derivatives of Gaussian.
            dyReference += featureWeights.y * intermediateFeaturesReference.z;
            dyTest += featureWeights.y * intermediateFeatureTest.z;
            ddyReference += featureWeights.z * intermediateFeaturesReference.z;
            ddyTest += featureWeights.z * intermediateFeatureTest.z;
        }

        const float edgeValueRef = std::sqrt(dxReference * dxReference + dyReference * dyReference);
        const float edgeValueTest = std::sqrt(dxTest * dxTest + dyTest * dyTest);
        const float pointValueRef = std::sqrt(ddxReference * ddxReference + ddyReference * ddyReference);
        const float pointValueTest = std::sqrt(ddxTest * ddxTest + ddyTest * ddyTest);
        const float edgeDifference = std::abs(edgeValueRef - edgeValueTest);
        const float pointDifference = std::abs(pointValueRef - pointValueTest);
        const float featureDifference = std::pow(normalizationFactor * Max(edgeDifference, pointDifference), DeviceFLIPConstants.gqf);
        featureDifferenceImage[dstIndex].y = featureDifference;
    }

    // Performs spatial filtering in the x direction on both the reference and test image at the same time (for better performance).
    // Filtering has been changed to using separable filtering for better performance.
    // For details on the convolution, see separatedConvolutions.pdf in the FLIP repository:
    // https://github.com/NVlabs/flip/blob/main/misc/separatedConvolutions.pdf.
    // referenceImage and testImage are expected to be in YCxCz space.
    __global__ static void kernelSpatialFilterFirstDir(color3* intermediateYCxImageReference, color3* intermediateCzImageReference, color3* referenceImage, color3* intermediateYCxImageTest, color3* intermediateCzImageTest, color3* testImage, color3* pFilterYCx, color3* pFilterCz, const int3 dim, const int3 filterDim)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int dstIndex = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y) return;

        const int halfFilterWidth = filterDim.x / 2;

        // Filter in x direction.
        color3 intermediateYCxReference = { 0.0f, 0.0f, 0.0f };
        color3 intermediateYCxTest = { 0.0f, 0.0f, 0.0f };
        color3 intermediateCzReference = { 0.0f, 0.0f, 0.0f };
        color3 intermediateCzTest = { 0.0f, 0.0f, 0.0f };

        for (int ix = -halfFilterWidth; ix <= halfFilterWidth; ix++)
        {
            int xx = Min(Max(0, x + ix), dim.x - 1);

            int filterIndex = ix + halfFilterWidth;
            int srcIndex = y * dim.x + xx;
            const color3 weightsYCx = pFilterYCx[filterIndex];
            const color3 weightsCz = pFilterCz[filterIndex];
            const color3 referenceColor = referenceImage[srcIndex];
            const color3 testColor = testImage[srcIndex];

            intermediateYCxReference += color3(weightsYCx.x * referenceColor.x, weightsYCx.y * referenceColor.y, 0.0f);
            intermediateYCxTest += color3(weightsYCx.x * testColor.x, weightsYCx.y * testColor.y, 0.0f);
            intermediateCzReference += color3(weightsCz.x * referenceColor.z, weightsCz.y * referenceColor.z, 0.0f);
            intermediateCzTest += color3(weightsCz.x * testColor.z, weightsCz.y * testColor.z, 0.0f);
        }

        intermediateYCxImageReference[dstIndex] = color3(intermediateYCxReference.x, intermediateYCxReference.y, 0.0f);
        intermediateYCxImageTest[dstIndex] = color3(intermediateYCxTest.x, intermediateYCxTest.y, 0.0f);
        intermediateCzImageReference[dstIndex] = color3(intermediateCzReference.x, intermediateCzReference.y, 0.0f);
        intermediateCzImageTest[dstIndex] = color3(intermediateCzTest.x, intermediateCzTest.y, 0.0f);
    }

    // Performs spatial filtering in the y direction (and clamps the results) on both the reference and test image at the same time (for better performance).
    // Filtering has been changed to using separable filtering for better performance. For details on the convolution, see separatedConvolutions.pdf in the FLIP repository:
    // https://github.com/NVlabs/flip/blob/main/misc/separatedConvolutions.pdf.
    // After filtering, compute color differences.
    __global__ static void kernelSpatialFilterSecondDirAndColorDifference(color3* colorDifferenceImage, color3* intermediateYCxImageReference, color3* intermediateCzImageReference, color3* intermediateYCxImageTest, color3* intermediateCzImageTest, color3* pFilterYCx, color3* pFilterCz, const int3 dim, const int3 filterDim, const float cmax, const float pccmax)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int dstIndex = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y) return;

        const int halfFilterWidth = filterDim.x / 2;

        // Filter in y direction.
        color3 filteredYCxReference = { 0.0f, 0.0f, 0.0f };
        color3 filteredYCxTest = { 0.0f, 0.0f, 0.0f };
        color3 filteredCzReference = { 0.0f, 0.0f, 0.0f };
        color3 filteredCzTest = { 0.0f, 0.0f, 0.0f };

        for (int iy = -halfFilterWidth; iy <= halfFilterWidth; iy++)
        {
            int yy = Min(Max(0, y + iy), dim.y - 1);

            int filterIndex = iy + halfFilterWidth;
            int srcIndex = yy * dim.x + x;
            const color3 weightsYCx = pFilterYCx[filterIndex];
            const color3 weightsCz = pFilterCz[filterIndex];
            const color3 intermediateYCxReference = intermediateYCxImageReference[srcIndex];
            const color3 intermediateYCxTest = intermediateYCxImageTest[srcIndex];
            const color3 intermediateCzReference = intermediateCzImageReference[srcIndex];
            const color3 intermediateCzTest = intermediateCzImageTest[srcIndex];

            filteredYCxReference += color3(weightsYCx.x * intermediateYCxReference.x, weightsYCx.y * intermediateYCxReference.y, 0.0f);
            filteredYCxTest += color3(weightsYCx.x * intermediateYCxTest.x, weightsYCx.y * intermediateYCxTest.y, 0.0f);
            filteredCzReference += color3(weightsCz.x * intermediateCzReference.x, weightsCz.y * intermediateCzReference.y, 0.0f);
            filteredCzTest += color3(weightsCz.x * intermediateCzTest.x, weightsCz.y * intermediateCzTest.y, 0.0f);
        }

        // Clamp to [0,1] in linear RGB.
        color3 filteredYCxCzReference = color3(filteredYCxReference.x, filteredYCxReference.y, filteredCzReference.x + filteredCzReference.y);
        color3 filteredYCxCzTest = color3(filteredYCxTest.x, filteredYCxTest.y, filteredCzTest.x + filteredCzTest.y);
        filteredYCxCzReference = FLIP::color3::clamp(FLIP::color3::XYZ2LinearRGB(FLIP::color3::YCxCz2XYZ(filteredYCxCzReference)));
        filteredYCxCzTest = FLIP::color3::clamp(FLIP::color3::XYZ2LinearRGB(FLIP::color3::YCxCz2XYZ(filteredYCxCzTest)));

        // Move from linear RGB to CIELab.
        filteredYCxCzReference = color3::XYZ2CIELab(color3::LinearRGB2XYZ(filteredYCxCzReference));
        filteredYCxCzTest = color3::XYZ2CIELab(color3::LinearRGB2XYZ(filteredYCxCzTest));

        // Apply Hunt adjustment.
        filteredYCxCzReference.y = color3::Hunt(filteredYCxCzReference.x, filteredYCxCzReference.y);
        filteredYCxCzReference.z = color3::Hunt(filteredYCxCzReference.x, filteredYCxCzReference.z);
        filteredYCxCzTest.y = color3::Hunt(filteredYCxCzTest.x, filteredYCxCzTest.y);
        filteredYCxCzTest.z = color3::Hunt(filteredYCxCzTest.x, filteredYCxCzTest.z);

        float colorDifference = color3::HyAB(filteredYCxCzReference, filteredYCxCzTest);
        colorDifference = powf(colorDifference, DeviceFLIPConstants.gqc);

        // Re-map error to the [0, 1] range. Values between 0 and pccmax are mapped to the range [0, gpt],
        // while the rest are mapped to the range (gpt, 1].
        if (colorDifference < pccmax)
        {
            colorDifference *= DeviceFLIPConstants.gpt / pccmax;
        }
        else
        {
            colorDifference = DeviceFLIPConstants.gpt + ((colorDifference - pccmax) / (cmax - pccmax)) * (1.0f - DeviceFLIPConstants.gpt);
        }

        colorDifferenceImage[dstIndex] = color3(colorDifference, 0.0f, 0.0f);
    }
#endif

    template<typename T = color3>
    class tensor
    {
    protected:
        int3 mDim = {0, 0, 0};
        int mArea = 0, mVolume = 0;
        T* mvpHostData = nullptr;
#ifdef FLIP_ENABLE_CUDA
        T* mvpDeviceData;
        CudaTensorState mState = CudaTensorState::UNINITIALIZED;
        dim3 mBlockDim, mGridDim;
#endif
    protected:
        bool allocateHost(void)
        {
            this->mvpHostData = (T*)malloc(this->mVolume * sizeof(T));

            if (this->mvpHostData == nullptr)
            {
                return false;
            }

            return true;
        }

#ifdef FLIP_ENABLE_CUDA
        bool allocateDevice(void)
        {
            int deviceVolume = this->mGridDim.x * this->mGridDim.y * this->mGridDim.z * this->mBlockDim.x * this->mBlockDim.y * this->mBlockDim.z;
            cudaError cudaError = cudaMalloc((void**)&(this->mvpDeviceData), deviceVolume * sizeof(T));

            if (cudaError != cudaSuccess)
            {
                std::cerr << "cudaMalloc() failed: " << cudaGetErrorString(cudaError) << "\n";
                this->~tensor();
                return false;
            }

            return true;
        }
#endif
        void init(const int3 dim, bool bClear = false, T clearColor = T(0.0f))
        {
            this->mDim = dim;
            this->mArea = dim.x * dim.y;
            this->mVolume = dim.x * dim.y * dim.z;

#ifdef FLIP_ENABLE_CUDA
            this->mGridDim.x = (this->mDim.x + this->mBlockDim.x - 1) / this->mBlockDim.x;
            this->mGridDim.y = (this->mDim.y + this->mBlockDim.y - 1) / this->mBlockDim.y;
            this->mGridDim.z = (this->mDim.z + this->mBlockDim.z - 1) / this->mBlockDim.z;

            cudaError_t cudaError = cudaSetDevice(0);
            if (cudaError != cudaSuccess)
            {
                std::cerr << "cudaSetDevice() failed: " << cudaGetErrorString(cudaError) << "\n";
                this->~tensor();
                exit(-1);
            }

            allocateDevice();
            this->mState = CudaTensorState::ALLOCATED;
#endif
            allocateHost();

            if (bClear)
            {
                this->clear(clearColor);
            }
        }

    public:

#ifndef FLIP_ENABLE_CUDA
        // Constructors for the CPU side.
        tensor()
        {
        }

        tensor(const int width, const int height, const int depth)
        {
            this->init({ width, height, depth });
        }

        tensor(const int width, const int height, const int depth, const T clearColor)
        {
            this->init({ width, height, depth }, true, clearColor);
        }

        tensor(const int3 dim, const T clearColor)
        {
            this->init(dim, true, clearColor);
        }

        tensor(tensor& image)
        {
            this->init(image.mDim);
            this->copy(image);
        }

        tensor(const color3* pColorMap, int size)
        {
            this->init({ size, 1, 1 });
            if (this->mvpHostData != nullptr)
            {
                memcpy(this->mvpHostData, pColorMap, size * sizeof(color3));
            }
        }
#else   // FLIP_ENABLE_CUDA
        // Constructors for the CUDA side.
        tensor(const dim3 blockDim = DEFAULT_KERNEL_BLOCK_DIM)
        {
            this->mBlockDim = blockDim;
        }

        tensor(const int width, const int height, const int depth, const dim3 blockDim = DEFAULT_KERNEL_BLOCK_DIM)
        {
            this->mBlockDim = blockDim;
            this->init({ width, height, depth });
        }

        tensor(const int width, const int height, const int depth, const T clearColor, const dim3 blockDim = DEFAULT_KERNEL_BLOCK_DIM)
        {
            this->mBlockDim = blockDim;
            this->init({ width, height, depth }, true, clearColor);
        }

        tensor(const int3 dim, const T clearColor, const dim3 blockDim = DEFAULT_KERNEL_BLOCK_DIM)
        {
            this->mBlockDim = blockDim;
            this->init(dim, true, clearColor);
        }

        tensor(tensor& image, const dim3 blockDim = DEFAULT_KERNEL_BLOCK_DIM)
        {
            this->mBlockDim = blockDim;
            this->init(image.mDim);
            this->copy(image);
        }

        tensor(const color3* pColorMap, int size, const dim3 blockDim = DEFAULT_KERNEL_BLOCK_DIM)
        {
            this->mBlockDim = blockDim;
            this->init({ size, 1, 1 });

            cudaError_t cudaError = cudaMemcpy(this->mvpDeviceData, pColorMap, size * sizeof(color3), cudaMemcpyHostToDevice);
            if (cudaError != cudaSuccess)
            {
                std::cout << "copy() failed: " << cudaGetErrorString(cudaError) << "\n";
                exit(-1);
            }
            this->mState = CudaTensorState::DEVICE_ONLY;
        }
#endif
        ~tensor(void)
        {
            free(this->mvpHostData);
#ifdef FLIP_ENABLE_CUDA
            cudaFree(this->mvpDeviceData);
#endif
        }

#ifdef FLIP_ENABLE_CUDA
        T* getDeviceData(const int z = 0)
        {
            return this->mvpDeviceData + z * this->mArea;
        }

        inline dim3 getBlockDim() const
        {
            return mBlockDim;
        }

        inline dim3 getGridDim() const
        {
            return mGridDim;
        }
#endif

        T* getHostData(void)
        {
            return this->mvpHostData;
        }

        inline int index(int x, int y = 0, int z = 0) const
        {
            return (z * this->mDim.y + y) * mDim.x + x;
        }

#ifndef FLIP_ENABLE_CUDA
        T get(int x, int y, int z) const
        {
            return this->mvpHostData[this->index(x, y, z)];
        }

        void set(int x, int y, int z, T value)
        {
            this->mvpHostData[this->index(x, y, z)] = value;
        }

        void setPixels(const float* pPixels, const int width, const int height)
        {
            this->init({ width, height, 1 });
            memcpy(this->mvpHostData, pPixels, size_t(width) * height * sizeof(T));
        }
#else
        T get(int x, int y, int z)
        {
            this->synchronizeHost();
            return this->mvpHostData[this->index(x, y, z)];
        }

        void set(int x, int y, int z, T value)
        {
            this->synchronizeHost();
            this->mvpHostData[this->index(x, y, z)] = value;
            this->mState = CudaTensorState::HOST_ONLY;
        }

        inline void setState(CudaTensorState state)
        {
            this->mState = state;
        }

        void setPixels(const float* pPixels, const int width, const int height)  // This assume that T is a color3.
        {
            if (this->mState == CudaTensorState::UNINITIALIZED)
            {
                this->init({ width, height, 1});
            }
            memcpy(this->mvpHostData, pPixels, size_t(width) * height * sizeof(T));
            this->mState = CudaTensorState::HOST_ONLY;
        }
#endif

        int3 getDimensions(void) const
        {
            return this->mDim;
        }

        int getWidth(void) const
        {
            return this->mDim.x;
        }

        int getHeight(void) const
        {
            return this->mDim.y;
        }

        int getDepth(void) const
        {
            return this->mDim.z;
        }

#ifndef FLIP_ENABLE_CUDA
        void colorMap(tensor<float>& srcImage, tensor<color3>& colorMap)
        {
            for (int z = 0; z < this->getDepth(); z++)
            {
#pragma omp parallel for
                for (int y = 0; y < this->getHeight(); y++)
                {
                    for (int x = 0; x < this->getWidth(); x++)
                    {
                        this->set(x, y, z, colorMap.get(int(srcImage.get(x, y, z) * 255.0f + 0.5f) % colorMap.getWidth(), 0, 0));
                    }
                }
            }
        }

        void sRGB2YCxCz(void)
        {
            for (int z = 0; z < this->getDepth(); z++)
            {
#pragma omp parallel for
                for (int y = 0; y < this->getHeight(); y++)
                {
                    for (int x = 0; x < this->getWidth(); x++)
                    {
                        this->set(x, y, z, color3::XYZ2YCxCz(color3::LinearRGB2XYZ(color3::sRGB2LinearRGB(this->get(x, y, z)))));
                    }
                }
            }
        }

        void sRGB2LinearRGB(void)
        {
            for (int z = 0; z < this->getDepth(); z++)
            {
#pragma omp parallel for
                for (int y = 0; y < this->getHeight(); y++)
                {
                    for (int x = 0; x < this->getWidth(); x++)
                    {
                        this->set(x, y, z, color3::sRGB2LinearRGB(this->get(x, y, z)));
                    }
                }
            }
        }

        void LinearRGB2YCxCz(void)
        {
            for (int z = 0; z < this->getDepth(); z++)
            {
#pragma omp parallel for
                for (int y = 0; y < this->getHeight(); y++)
                {
                    for (int x = 0; x < this->getWidth(); x++)
                    {
                        this->set(x, y, z, color3::XYZ2YCxCz(color3::LinearRGB2XYZ(this->get(x, y, z))));
                    }
                }
            }
        }

        void LinearRGB2sRGB(void)
        {
            for (int z = 0; z < this->getDepth(); z++)
            {
#pragma omp parallel for
                for (int y = 0; y < this->getHeight(); y++)
                {
                    for (int x = 0; x < this->getWidth(); x++)
                    {
                        this->set(x, y, z, color3::LinearRGB2sRGB(this->get(x, y, z)));
                    }
                }
            }
        }

        void clear(const T color = T(0.0f))
        {
            for (int z = 0; z < this->getDepth(); z++)
            {
#pragma omp parallel for
                for (int y = 0; y < this->getHeight(); y++)
                {
                    for (int x = 0; x < this->getWidth(); x++)
                    {
                        this->set(x, y, z, color);
                    }
                }
            }
        }

        void clamp(float low = 0.0f, float high = 1.0f)
        {
            for (int z = 0; z < this->getDepth(); z++)
            {
#pragma omp parallel for
                for (int y = 0; y < this->getHeight(); y++)
                {
                    for (int x = 0; x < this->getWidth(); x++)
                    {
                        this->set(x, y, z, color3::clamp(this->get(x, y, z), low, high));
                    }
                }
            }
        }

        void toneMap(std::string tm)
        {
            int toneMapper = 1;
            if (tm == "reinhard")
            {
                for (int z = 0; z < this->getDepth(); z++)
                {
#pragma omp parallel for
                    for (int y = 0; y < this->getHeight(); y++)
                    {
                        for (int x = 0; x < this->getWidth(); x++)
                        {
                            color3 color = this->get(x, y, z);
                            float luminance = color3::linearRGB2Luminance(color);
                            float factor = 1.0f / (1.0f + luminance);
                            this->set(x, y, z, color * factor);
                        }
                    }
                }
                return;
            }

            if (tm == "aces")
                toneMapper = 1;
            if (tm == "hable")
                toneMapper = 2;

            for (int z = 0; z < this->getDepth(); z++)
            {
#pragma omp parallel for
                for (int y = 0; y < this->getHeight(); y++)
                {
                    for (int x = 0; x < this->getWidth(); x++)
                    {
                        const float* tc = ToneMappingCoefficients[toneMapper];
                        color3 color = this->get(x, y, z);
                        this->set(x, y, z, color3(((color * color) * tc[0] + color * tc[1] + tc[2]) / (color * color * tc[3] + color * tc[4] + tc[5])));
                    }
                }
            }
        }

        void copy(tensor<T>& srcImage)
        {
            if (this->mDim.x == srcImage.getWidth() && this->mDim.y == srcImage.getHeight() && this->mDim.z == srcImage.getDepth())
            {
                memcpy(this->mvpHostData, srcImage.getHostData(), this->mVolume * sizeof(T));
            }
        }

        void copyFloat2Color3(tensor<float>& srcImage)
        {
            for (int z = 0; z < this->getDepth(); z++)
            {
#pragma omp parallel for
                for (int y = 0; y < this->getHeight(); y++)
                {
                    for (int x = 0; x < this->getWidth(); x++)
                    {
                        this->set(x, y, z, color3(srcImage.get(x, y, z)));
                    }
                }
            }
        }
#else  // FLIP_ENABLE_CUDA
        void synchronizeHost(void)
        {
            if (this->mState == CudaTensorState::DEVICE_ONLY)
            {
                cudaError_t cudaError = cudaMemcpy(this->mvpHostData, this->mvpDeviceData, this->mVolume * sizeof(T), cudaMemcpyDeviceToHost);
                if (cudaError != cudaSuccess)
                {
                    std::cout << "cudaMemcpy(), DEVICE -> HOST, failed: " << cudaGetErrorString(cudaError) << "\n";
                    exit(-1);
                }
                this->mState = CudaTensorState::SYNCHRONIZED;
            }
        }

        void synchronizeDevice(void)
        {
            if (this->mState == CudaTensorState::HOST_ONLY)
            {
                cudaError_t cudaError = cudaMemcpy(this->mvpDeviceData, this->mvpHostData, this->mVolume * sizeof(T), cudaMemcpyHostToDevice);
                if (cudaError != cudaSuccess)
                {
                    std::cout << "cudaMemcpy(), HOST -> DEVICE, failed: " << cudaGetErrorString(cudaError) << "\n";
                    exit(-1);
                }
                this->mState = CudaTensorState::SYNCHRONIZED;
            }
        }

        void colorMap(tensor<float>& srcImage, tensor<color3>& colorMap)
        {
            srcImage.synchronizeDevice();
            FLIP::kernelColorMap <<<this->mGridDim, this->mBlockDim >>> (this->getDeviceData(), srcImage.getDeviceData(), colorMap.getDeviceData(), this->mDim, colorMap.getWidth());
            checkStatus("kernelColorMap");
            this->mState = CudaTensorState::DEVICE_ONLY;
        }

        void sRGB2YCxCz(void)
        {
            this->synchronizeDevice();
            kernelsRGB2YCxCz << <this->mGridDim, this->mBlockDim >> > (this->mvpDeviceData, this->mDim);
            checkStatus("kernelsRGB2YCxCz");
            this->mState = CudaTensorState::DEVICE_ONLY;
        }

        void sRGB2LinearRGB(void)
        {
            this->synchronizeDevice();
            kernelsRGB2LinearRGB << <this->mGridDim, this->mBlockDim >> > (this->mvpDeviceData, this->mDim);
            checkStatus("kernelsRGB2LinearRGB");
            this->mState = CudaTensorState::DEVICE_ONLY;
        }

        void LinearRGB2YCxCz(void)
        {
            this->synchronizeDevice();
            kernelLinearRGB2YCxCz << <this->mGridDim, this->mBlockDim >> > (this->mvpDeviceData, this->mDim);
            checkStatus("kernelLinearRGB2YCxCz");
            this->mState = CudaTensorState::DEVICE_ONLY;
        }

        void LinearRGB2sRGB(void)
        {
            this->synchronizeDevice();
            FLIP::kernelLinearRGB2sRGB << <this->mGridDim, this->mBlockDim >> > (this->mvpDeviceData, this->mDim);
            checkStatus("kernelLinearRGB2sRGB");
            this->mState = CudaTensorState::DEVICE_ONLY;
        }

        static void checkStatus(std::string kernelName)
        {
            cudaError_t cudaError = cudaGetLastError();
            if (cudaError != cudaSuccess)
            {
                std::cerr << kernelName << "() failed: " << cudaGetErrorString(cudaError) << "\n";
                exit(-1);
            }

            // Used if debugging.
            if (true)
            {
                deviceSynchronize(kernelName);
            }
        }

        // Used if debugging.
        static void deviceSynchronize(std::string kernelName)
        {
            cudaError_t cudaError = cudaDeviceSynchronize();
            if (cudaError != cudaSuccess)
            {
                std::cerr << kernelName << "(): cudeDeviceSynchronize: " << cudaGetErrorString(cudaError) << "\n";
                exit(-1);
            }
        }

        void clear(const T color = T(0.0f))
        {
            FLIP::kernelClear << <this->mGridDim, this->mBlockDim >> > (this->mvpDeviceData, this->mDim, color);
            checkStatus("kernelClear");
            this->mState = CudaTensorState::DEVICE_ONLY;
        }

        void clamp(float low = 0.0f, float high = 1.0f)
        {
            this->synchronizeDevice();
            FLIP::kernelClamp << <this->mGridDim, this->mBlockDim >> > (this->mvpDeviceData, this->mDim, low, high);
            checkStatus("kernelClamp");
            this->mState = CudaTensorState::DEVICE_ONLY;
        }

        void toneMap(std::string tm)
        {
            int toneMapper = 1;
            if (tm == "reinhard")
                toneMapper = 0;
            if (tm == "aces")
                toneMapper = 1;
            if (tm == "hable")
                toneMapper = 2;

            FLIP::kernelToneMap << <this->mGridDim, this->mBlockDim >> > (this->mvpDeviceData, this->mDim, toneMapper);
            checkStatus("kernelToneMap");
            this->mState = CudaTensorState::DEVICE_ONLY;
        }

        void copy(tensor<T>& srcImage)
        {
            srcImage.synchronizeDevice();
            if (this->mDim.x == srcImage.getWidth() && this->mDim.y == srcImage.getHeight() && this->mDim.z == srcImage.getDepth())
            {
                cudaError_t cudaError = cudaMemcpy(this->mvpDeviceData, srcImage.getDeviceData(), this->mVolume * sizeof(T), cudaMemcpyDeviceToDevice);
                if (cudaError != cudaSuccess)
                {
                    std::cout << "copy() failed: " << cudaGetErrorString(cudaError) << "\n";
                    exit(-1);
                }
            }
            else
            {
                kernelBilinearCopy << <this->mGridDim, this->mBlockDim >> > (this->mvpDeviceData, srcImage.getDeviceData(), this->mDim, srcImage.getDimensions());
            }
            this->mState = CudaTensorState::DEVICE_ONLY;
        }

        void copyFloat2Color3(tensor<float>& srcImage)
        {
            srcImage.synchronizeDevice();
            FLIP::kernelFloat2Color3 << <this->mGridDim, this->mBlockDim >> > (this->mvpDeviceData, srcImage.getDeviceData(), this->mDim);
            checkStatus("kernelFloat2Color3");
            this->mState = CudaTensorState::DEVICE_ONLY;
        }
#endif
    };

    template<typename T>
    class image: public tensor<T>
    {
    public:
        image()
        {}
#ifndef FLIP_ENABLE_CUDA

        image(const int width, const int height)
            : tensor<T>(width, height, 1)
        {
        }

        image(const int width, const int height, const T clearColor)
            : tensor<T>(width, height, 1, clearColor)
        {
        }

        image(const int3 dim, const T clearColor)
            : tensor<T>(dim.x, dim.y, 1)
        {
        }

        image(const int3 dim)
            : tensor<T>(dim.x, dim.y, 1)
        {
        }

        image(image& image)
        {
            this->init(image.mDim);
            this->copy(image);
        }

        image(tensor<T>& tensor, const int offset)
        {
            this->init(tensor.getDimensions());
            this->copy(tensor, offset);
        }

        image(const color3* pColorMap, int size)
            : tensor<T>(pColorMap, size)
        {
        }
#else // FLIP_ENABLE_CUDA
        image(std::string fileName, const dim3 blockDim = DEFAULT_KERNEL_BLOCK_DIM)
        {
            this->mBlockDim = blockDim;
            bool bOk = this->load(fileName);
            if (!bOk)
            {
                std::cout << "Failed to load image <" << fileName << ">" << std::endl;
                exit(-1);
            }
            this->setState(CudaTensorState::HOST_ONLY);
        }

        image(const int width, const int height, const dim3 blockDim = DEFAULT_KERNEL_BLOCK_DIM)
            : tensor<T>(width, height, 1, blockDim)
        {
        }

        image(const int width, const int height, const T clearColor, const dim3 blockDim = DEFAULT_KERNEL_BLOCK_DIM)
            : tensor<T>(width, height, 1, clearColor, blockDim)
        {
        }

        image(const int3 dim, const T clearColor, const dim3 blockDim = DEFAULT_KERNEL_BLOCK_DIM)
            : tensor<T>(dim.x, dim.y, 1, blockDim)
        {
        }

        image(const int3 dim, const dim3 blockDim = DEFAULT_KERNEL_BLOCK_DIM)
            : tensor<T>(dim.x, dim.y, 1, blockDim)
        {
        }

        image(image& image, const dim3 blockDim = DEFAULT_KERNEL_BLOCK_DIM)
        {
            this->mBlockDim = blockDim;
            this->init(image.mDim);
            this->copy(image);
        }

        image(tensor<T>& tensor, const int offset, const dim3 blockDim = DEFAULT_KERNEL_BLOCK_DIM)
        {
            this->mBlockDim = blockDim;
            this->init(tensor.getDimensions());
            this->copy(tensor, offset);
        }

        image(const color3* pColorMap, int size, const dim3 blockDim = DEFAULT_KERNEL_BLOCK_DIM)
            : tensor<T>(pColorMap, size, blockDim)
        {
        }
#endif
        ~image(void)
        {
        }

#ifndef FLIP_ENABLE_CUDA
        T get(int x, int y) const
        {
            return this->mvpHostData[this->index(x, y)];
        }

        void set(int x, int y, T value)
        {
            this->mvpHostData[this->index(x, y)] = value;
        }
#else
        T get(int x, int y)
        {
            this->synchronizeHost();
            return this->mvpHostData[this->index(x, y)];
        }

        void set(int x, int y, T value)
        {
            this->synchronizeHost();
            this->mvpHostData[this->index(x, y)] = value;
            this->setState(CudaTensorState::HOST_ONLY);
        }
#endif

        // For details, see separatedConvolutions.pdf in the FLIP repository:
        // https://github.com/NVlabs/flip/blob/main/misc/separatedConvolutions.pdf.
        static void setSpatialFilters(image<color3>& filterYCx, image<color3>& filterCz, float ppd, int filterRadius)
        {
            float deltaX = 1.0f / ppd;
            color3 filterSumYCx = { 0.0f, 0.0f, 0.0f };
            color3 filterSumCz = { 0.0f, 0.0f, 0.0f };
            int filterWidth = 2 * filterRadius + 1;

            for (int x = 0; x < filterWidth; x++)
            {
                float ix = (x - filterRadius) * deltaX;

                float ix2 = ix * ix;
                float gY = Gaussian(ix2, GaussianConstants.a1.x, GaussianConstants.b1.x);
                float gCx = Gaussian(ix2, GaussianConstants.a1.y, GaussianConstants.b1.y);
                float gCz1 = GaussianSqrt(ix2, GaussianConstants.a1.z, GaussianConstants.b1.z);
                float gCz2 = GaussianSqrt(ix2, GaussianConstants.a2.z, GaussianConstants.b2.z);
                color3 valueYCx = color3(gY, gCx, 0.0f);
                color3 valueCz = color3(gCz1, gCz2, 0.0f);
                filterYCx.set(x, 0, valueYCx);
                filterCz.set(x, 0, valueCz);
                filterSumYCx += valueYCx;
                filterSumCz += valueCz;
            }

            // Normalize weights.
            color3 normFactorYCx = { 1.0f / filterSumYCx.x, 1.0f / filterSumYCx.y, 1.0f };
            float normFactorCz = 1.0f / std::sqrt(filterSumCz.x * filterSumCz.x + filterSumCz.y * filterSumCz.y);
            for (int x = 0; x < filterWidth; x++)
            {
                color3 pYCx = filterYCx.get(x, 0);
                color3 pCz = filterCz.get(x, 0);

                filterYCx.set(x, 0, color3(pYCx.x * normFactorYCx.x, pYCx.y * normFactorYCx.y, 0.0f));
                filterCz.set(x, 0, color3(pCz.x * normFactorCz, pCz.y * normFactorCz, 0.0f));
            }
        }

        // For details, see separatedConvolutions.pdf in the FLIP repository:
        // https://github.com/NVlabs/flip/blob/main/misc/separatedConvolutions.pdf
        static void setFeatureFilter(image<color3>& filter, const float ppd)
        {
            const float stdDev = 0.5f * FLIPConstants.gw * ppd;
            const int radius = int(std::ceil(3.0f * stdDev));
            const int width = 2 * radius + 1;

            float gSum = 0.0f;
            float dgSumNegative = 0.0f;
            float dgSumPositive = 0.0f;
            float ddgSumNegative = 0.0f;
            float ddgSumPositive = 0.0f;

            for (int x = 0; x < width; x++)
            {
                int xx = x - radius;

                float g = Gaussian(float(xx), stdDev);
                gSum += g;

                // 1st derivative.
                float dg = -float(xx) * g;
                if (dg > 0.0f)
                    dgSumPositive += dg;
                else
                    dgSumNegative -= dg;

                // 2nd derivative.
                float ddg = (float(xx) * float(xx) / (stdDev * stdDev) - 1.0f) * g;
                if (ddg > 0.0f)
                    ddgSumPositive += ddg;
                else
                    ddgSumNegative -= ddg;

                filter.set(x, 0, color3(g, dg, ddg));
            }

            // Normalize weights (Gaussian weights should sum to 1; postive and negative weights of 1st and 2nd derivative should sum to 1 and -1, respectively).
            for (int x = 0; x < width; x++)
            {
                color3 p = filter.get(x, 0);

                filter.set(x, 0, color3(p.x / gSum, p.y / (p.y > 0.0f ? dgSumPositive : dgSumNegative), p.z / (p.z > 0.0f ? ddgSumPositive : ddgSumNegative)));
            }
        }

#ifndef FLIP_ENABLE_CUDA
        // Performs spatial filtering (and clamps the results) on both the reference and test image at the same time (for better performance).
        // Filtering has been changed to separable filtering for better performance. For details on the convolution, see separatedConvolutions.pdf in the FLIP repository:
        // https://github.com/NVlabs/flip/blob/main/misc/separatedConvolutions.pdf
        // After filtering, compute color differences. referenceImage and testImage are expected to be in YCxCz space.
        void computeColorDifference(const FLIP::image<color3>& referenceImage, const FLIP::image<color3>& testImage, const FLIP::image<color3>& filterYCx, const FLIP::image<color3>& filterCz)
        {
            // Color difference constants
            const float cmax = color3::computeMaxDistance(FLIPConstants.gqc);
            const float pccmax = FLIPConstants.gpc * cmax;

            const int halfFilterWidth = filterYCx.getWidth() / 2; // YCx and Cz filters are the same size.

            const int w = referenceImage.getWidth();
            const int h = referenceImage.getHeight();

            image<color3> intermediateYCxImageReference(w, h);
            image<color3> intermediateYCxImageTest(w, h);
            image<color3> intermediateCzImageReference(w, h);
            image<color3> intermediateCzImageTest(w, h);

            // Filter in x direction.
#pragma omp parallel for
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    color3 intermediateYCxReference = { 0.0f, 0.0f, 0.0f };
                    color3 intermediateYCxTest = { 0.0f, 0.0f, 0.0f };
                    color3 intermediateCzReference = { 0.0f, 0.0f, 0.0f };
                    color3 intermediateCzTest = { 0.0f, 0.0f, 0.0f };

                    for (int ix = -halfFilterWidth; ix <= halfFilterWidth; ix++)
                    {
                        int xx = Min(Max(0, x + ix), w - 1);

                        const color3 weightsYCx = filterYCx.get(ix + halfFilterWidth, 0);
                        const color3 weightsCz = filterCz.get(ix + halfFilterWidth, 0);
                        const color3 referenceColor = referenceImage.get(xx, y);
                        const color3 testColor = testImage.get(xx, y);

                        intermediateYCxReference += color3(weightsYCx.x * referenceColor.x, weightsYCx.y * referenceColor.y, 0.0f);
                        intermediateYCxTest += color3(weightsYCx.x * testColor.x, weightsYCx.y * testColor.y, 0.0f);
                        intermediateCzReference += color3(weightsCz.x * referenceColor.z, weightsCz.y * referenceColor.z, 0.0f);
                        intermediateCzTest += color3(weightsCz.x * testColor.z, weightsCz.y * testColor.z, 0.0f);
                    }

                    intermediateYCxImageReference.set(x, y, intermediateYCxReference);
                    intermediateYCxImageTest.set(x, y, intermediateYCxTest);
                    intermediateCzImageReference.set(x, y, intermediateCzReference);
                    intermediateCzImageTest.set(x, y, intermediateCzTest);
                }
            }

            // Filter in y direction.
#pragma omp parallel for
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    color3 filteredYCxReference = { 0.0f, 0.0f, 0.0f };
                    color3 filteredYCxTest = { 0.0f, 0.0f, 0.0f };
                    color3 filteredCzReference = { 0.0f, 0.0f, 0.0f };
                    color3 filteredCzTest = { 0.0f, 0.0f, 0.0f };

                    for (int iy = -halfFilterWidth; iy <= halfFilterWidth; iy++)
                    {
                        int yy = Min(Max(0, y + iy), h - 1);

                        const color3 weightsYCx = filterYCx.get(iy + halfFilterWidth, 0);
                        const color3 weightsCz = filterCz.get(iy + halfFilterWidth, 0);
                        const color3 intermediateYCxReference = intermediateYCxImageReference.get(x, yy);
                        const color3 intermediateYCxTest = intermediateYCxImageTest.get(x, yy);
                        const color3 intermediateCzReference = intermediateCzImageReference.get(x, yy);
                        const color3 intermediateCzTest = intermediateCzImageTest.get(x, yy);

                        filteredYCxReference += color3(weightsYCx.x * intermediateYCxReference.x, weightsYCx.y * intermediateYCxReference.y, 0.0f);
                        filteredYCxTest += color3(weightsYCx.x * intermediateYCxTest.x, weightsYCx.y * intermediateYCxTest.y, 0.0f);
                        filteredCzReference += color3(weightsCz.x * intermediateCzReference.x, weightsCz.y * intermediateCzReference.y, 0.0f);
                        filteredCzTest += color3(weightsCz.x * intermediateCzTest.x, weightsCz.y * intermediateCzTest.y, 0.0f);
                    }

                    // Clamp to [0,1] in linear RGB.
                    color3 filteredYCxCzReference = color3(filteredYCxReference.x, filteredYCxReference.y, filteredCzReference.x + filteredCzReference.y);
                    color3 filteredYCxCzTest = color3(filteredYCxTest.x, filteredYCxTest.y, filteredCzTest.x + filteredCzTest.y);
                    filteredYCxCzReference = FLIP::color3::clamp(FLIP::color3::XYZ2LinearRGB(FLIP::color3::YCxCz2XYZ(filteredYCxCzReference)));
                    filteredYCxCzTest = FLIP::color3::clamp(FLIP::color3::XYZ2LinearRGB(FLIP::color3::YCxCz2XYZ(filteredYCxCzTest)));

                    // Move from linear RGB to CIELab.
                    filteredYCxCzReference = color3::XYZ2CIELab(color3::LinearRGB2XYZ(filteredYCxCzReference));
                    filteredYCxCzTest = color3::XYZ2CIELab(color3::LinearRGB2XYZ(filteredYCxCzTest));

                    // Apply Hunt adjustment.
                    filteredYCxCzReference.y = color3::Hunt(filteredYCxCzReference.x, filteredYCxCzReference.y);
                    filteredYCxCzReference.z = color3::Hunt(filteredYCxCzReference.x, filteredYCxCzReference.z);
                    filteredYCxCzTest.y = color3::Hunt(filteredYCxCzTest.x, filteredYCxCzTest.y);
                    filteredYCxCzTest.z = color3::Hunt(filteredYCxCzTest.x, filteredYCxCzTest.z);

                    float colorDifference = color3::HyAB(filteredYCxCzReference, filteredYCxCzTest);

                    colorDifference = powf(colorDifference, FLIPConstants.gqc);

                    // Re-map error to the [0, 1] range. Values between 0 and pccmax are mapped to the range [0, gpt],
                    // while the rest are mapped to the range (gpt, 1].
                    if (colorDifference < pccmax)
                    {
                        colorDifference *= FLIPConstants.gpt / pccmax;
                    }
                    else
                    {
                        colorDifference = FLIPConstants.gpt + ((colorDifference - pccmax) / (cmax - pccmax)) * (1.0f - FLIPConstants.gpt);
                    }
                    this->set(x, y, colorDifference);
                }
            }
        }

        // This includes convolution (using separable filtering) of grayRefImage and grayTestImage for both edge and point filtering.
        // In addition, it computes the final FLIP error and stores in "this". referenceImage and testImage are expected to be in YCxCz space.
        void computeFeatureDifferenceAndFinalError(const image<color3>& referenceImage, const image<color3>& testImage, const image<color3>& featureFilter)
        {
            const float normalizationFactor = 1.0f / std::sqrt(2.0f);
            const int halfFilterWidth = featureFilter.getWidth() / 2;      // The edge and point filters are of the same size.
            const int w = referenceImage.getWidth();
            const int h = referenceImage.getHeight();

            image<color3> intermediateFeaturesImageReference(w, h);
            image<color3> intermediateFeaturesImageTest(w, h);

            // Convolve in x direction (1st and 2nd derivative for filter in x direction, Gaussian in y direction).
            // For details, see separatedConvolutions.pdf in the FLIP repository:
            // https://github.com/NVlabs/flip/blob/main/misc/separatedConvolutions.pdf
            // We filter both reference and test image simultaneously (for better performance).
            const float oneOver116 = 1.0f / 116.0f;
            const float sixteenOver116 = 16.0f / 116.0f;
#pragma omp parallel for
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    float dxReference = 0.0f, dxTest = 0.0f, ddxReference = 0.0f, ddxTest = 0.0f;
                    float gaussianFilteredReference = 0.0f, gaussianFilteredTest = 0.0f;

                    for (int ix = -halfFilterWidth; ix <= halfFilterWidth; ix++)
                    {
                        int xx = Min(Max(0, x + ix), w - 1);

                        const color3 featureWeights = featureFilter.get(ix + halfFilterWidth, 0);
                        float yReference = referenceImage.get(xx, y).x;
                        float yTest = testImage.get(xx, y).x;

                        // Normalize the Y values to [0,1].
                        float yReferenceNormalized = yReference * oneOver116 + sixteenOver116;
                        float yTestNormalized = yTest * oneOver116 + sixteenOver116;

                        // Image multiplied by 1st and 2nd x-derivatives of Gaussian.
                        dxReference += featureWeights.y * yReferenceNormalized;
                        dxTest += featureWeights.y * yTestNormalized;
                        ddxReference += featureWeights.z * yReferenceNormalized;
                        ddxTest += featureWeights.z * yTestNormalized;

                        // Image multiplied by Gaussian.
                        gaussianFilteredReference += featureWeights.x * yReferenceNormalized;
                        gaussianFilteredTest += featureWeights.x * yTestNormalized;
                    }
                    intermediateFeaturesImageReference.set(x, y, color3(dxReference, ddxReference, gaussianFilteredReference));
                    intermediateFeaturesImageTest.set(x, y, color3(dxTest, ddxTest, gaussianFilteredTest));
                }
            }

            // Convolve in y direction (1st and 2nd derivative for filter in y direction, Gaussian in x direction), then compute difference.
            // For details on the convolution, see separatedConvolutions.pdf in the FLIP repository:
            // https://github.com/NVlabs/flip/blob/main/misc/separatedConvolutions.pdf
            // We filter both reference and test image simultaneously (for better performance).
#pragma omp parallel for
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    float dxReference = 0.0f, dxTest = 0.0f, ddxReference = 0.0f, ddxTest = 0.0f;
                    float dyReference = 0.0f, dyTest = 0.0f, ddyReference = 0.0f, ddyTest = 0.0f;

                    for (int iy = -halfFilterWidth; iy <= halfFilterWidth; iy++)
                    {
                        int yy = Min(Max(0, y + iy), h - 1);

                        const color3 featureWeights = featureFilter.get(iy + halfFilterWidth, 0);
                        const color3 intermediateFeaturesReference = intermediateFeaturesImageReference.get(x, yy);
                        const color3 intermediateFeatureTest = intermediateFeaturesImageTest.get(x, yy);

                        // Intermediate images (1st and 2nd derivative in x) multiplied by Gaussian.
                        dxReference += featureWeights.x * intermediateFeaturesReference.x;
                        dxTest += featureWeights.x * intermediateFeatureTest.x;
                        ddxReference += featureWeights.x * intermediateFeaturesReference.y;
                        ddxTest += featureWeights.x * intermediateFeatureTest.y;

                        // Intermediate image (Gaussian) multiplied by 1st and 2nd y-derivatives of Gaussian.
                        dyReference += featureWeights.y * intermediateFeaturesReference.z;
                        dyTest += featureWeights.y * intermediateFeatureTest.z;
                        ddyReference += featureWeights.z * intermediateFeaturesReference.z;
                        ddyTest += featureWeights.z * intermediateFeatureTest.z;
                    }

                    const float edgeValueRef = std::sqrt(dxReference * dxReference + dyReference * dyReference);
                    const float edgeValueTest = std::sqrt(dxTest * dxTest + dyTest * dyTest);
                    const float pointValueRef = std::sqrt(ddxReference * ddxReference + ddyReference * ddyReference);
                    const float pointValueTest = std::sqrt(ddxTest * ddxTest + ddyTest * ddyTest);

                    const float edgeDifference = std::abs(edgeValueRef - edgeValueTest);
                    const float pointDifference = std::abs(pointValueRef - pointValueTest);

                    const float featureDifference = std::pow(normalizationFactor * Max(edgeDifference, pointDifference), FLIPConstants.gqf);
                    const float colorDifference = this->get(x, y);

                    const float errorFLIP = std::pow(colorDifference, 1.0f - featureDifference);

                    this->set(x, y, errorFLIP);
                }
            }
        }

        void setMaxExposure(image<float>& errorMap, image<float>& exposureMap, float exposure)
        {
#pragma omp parallel for
            for (int y = 0; y < this->getHeight(); y++)
            {
                for (int x = 0; x < this->getWidth(); x++)
                {
                    float srcValue = errorMap.get(x, y);
                    float dstValue = this->get(x, y);

                    if (srcValue > dstValue)
                    {
                        exposureMap.set(x, y, exposure);
                        this->set(x, y, srcValue);
                    }
                }
            }
        }

        void expose(float level)
        {
            const float m = std::pow(2.0f, level);
#pragma omp parallel for
            for (int y = 0; y < this->getHeight(); y++)
            {
                for (int x = 0; x < this->getWidth(); x++)
                {
                    this->set(x, y, this->get(x, y) * m);
                }
            }
        }
#else // FLIP_ENABLE_CUDA
        // Perform the x-component of separable spatial filtering of both the reference and the test.
        // referenceImage and testImage are expected to be in YCxCz space.
        static void spatialFilterFirstDir(image& referenceImage, image& intermediateYCxImageReference, image& intermediateCzImageReference, image& testImage, image& intermediateYCxImageTest, image& intermediateCzImageTest, image& filterYCx, image& filterCz)
        {
            referenceImage.synchronizeDevice();
            testImage.synchronizeDevice();
            filterYCx.synchronizeDevice();
            filterCz.synchronizeDevice();
            FLIP::kernelSpatialFilterFirstDir << <intermediateYCxImageReference.getGridDim(), intermediateYCxImageReference.getBlockDim() >> > (intermediateYCxImageReference.mvpDeviceData, intermediateCzImageReference.mvpDeviceData, referenceImage.mvpDeviceData, intermediateYCxImageTest.mvpDeviceData, intermediateCzImageTest.mvpDeviceData, testImage.mvpDeviceData, filterYCx.mvpDeviceData, filterCz.mvpDeviceData, intermediateYCxImageReference.mDim, filterYCx.mDim); // Filter sizes are the same.
            image<T>::checkStatus("kernelSpatialFilterFirstDir");
            intermediateYCxImageReference.setState(CudaTensorState::DEVICE_ONLY);
            intermediateCzImageReference.setState(CudaTensorState::DEVICE_ONLY);
            intermediateYCxImageTest.setState(CudaTensorState::DEVICE_ONLY);
            intermediateCzImageTest.setState(CudaTensorState::DEVICE_ONLY);
        }

        // Perform the y-component of separable spatial filtering of both the reference and the test and compute color difference.
        static void spatialFilterSecondDirAndColorDifference(image& intermediateYCxImageReference, image& intermediateCzImageReference, image& intermediateYCxImageTest, image& intermediateCzImageTest, image& colorDifferenceImage, image& filterYCx, image& filterCz)
        {
            intermediateYCxImageReference.synchronizeDevice();
            intermediateCzImageReference.synchronizeDevice();
            intermediateYCxImageTest.synchronizeDevice();
            intermediateCzImageTest.synchronizeDevice();
            filterYCx.synchronizeDevice();
            filterCz.synchronizeDevice();

            // Set color difference constants.
            const float cmax = color3::computeMaxDistance(FLIPConstants.gqc);
            const float pccmax = FLIPConstants.gpc * cmax;

            FLIP::kernelSpatialFilterSecondDirAndColorDifference << <colorDifferenceImage.getGridDim(), colorDifferenceImage.getBlockDim() >> > (colorDifferenceImage.mvpDeviceData, intermediateYCxImageReference.mvpDeviceData, intermediateCzImageReference.mvpDeviceData, intermediateYCxImageTest.mvpDeviceData, intermediateCzImageTest.mvpDeviceData, filterYCx.mvpDeviceData, filterCz.mvpDeviceData, colorDifferenceImage.mDim, filterYCx.mDim, cmax, pccmax); // Filter sizes are the same.
            image<T>::checkStatus("kernelSpatialFilterSecondDirAndColorDifference");
            colorDifferenceImage.setState(CudaTensorState::DEVICE_ONLY);
        }

        // Perform the x-component of separable feature detection filtering of both the reference and the test.
        // referenceImage and testImage are expected to be in YCxCz space.
        static void featureFilterFirstDir(image& referenceImage, image& intermediateFeaturesImageReference, image& testImage, image& intermediateFeaturesImageTest, image& featureFilter)
        {
            referenceImage.synchronizeDevice();
            testImage.synchronizeDevice();
            featureFilter.synchronizeDevice();
            FLIP::kernelFeatureFilterFirstDir << <intermediateFeaturesImageReference.mGridDim, intermediateFeaturesImageReference.mBlockDim >> > (intermediateFeaturesImageReference.mvpDeviceData, referenceImage.mvpDeviceData, intermediateFeaturesImageTest.mvpDeviceData, testImage.mvpDeviceData, featureFilter.mvpDeviceData, intermediateFeaturesImageReference.mDim, featureFilter.mDim);
            image<T>::checkStatus("kernelFeatureFilterFirstDir");
            intermediateFeaturesImageReference.setState(CudaTensorState::DEVICE_ONLY);
            intermediateFeaturesImageTest.setState(CudaTensorState::DEVICE_ONLY);
        }

        // Perform the y-component of separable feature detection filtering of both the reference and the test and compute feature difference.
        static void featureFilterSecondDirAndFeatureDifference(image& intermediateFeaturesImageReference, image& intermediateFeaturesImageTest, image& featureDifferenceImage, image& featureFilter)
        {
            intermediateFeaturesImageReference.synchronizeDevice();
            intermediateFeaturesImageTest.synchronizeDevice();
            featureFilter.synchronizeDevice();
            FLIP::kernelFeatureFilterSecondDirAndFeatureDifference << <featureDifferenceImage.mGridDim, featureDifferenceImage.mBlockDim >> > (featureDifferenceImage.mvpDeviceData, intermediateFeaturesImageReference.mvpDeviceData, intermediateFeaturesImageTest.mvpDeviceData, featureFilter.mvpDeviceData, featureDifferenceImage.mDim, featureFilter.mDim);
            image<T>::checkStatus("kernelFeatureFilterSecondDirAndFeatureDifference");
            featureDifferenceImage.setState(CudaTensorState::DEVICE_ONLY);
        }

        void finalError(image<color3>& colorFeatureDifference)
        {
            FLIP::kernelFinalError << <this->mGridDim, this->mBlockDim >> > (this->getDeviceData(), colorFeatureDifference.getDeviceData(), this->mDim);
            image<T>::checkStatus("kernelFinalError");
            this->setState(CudaTensorState::DEVICE_ONLY);
        }

        void setMaxExposure(tensor<float>& errorMap, tensor<float>& exposureMap, float exposure)
        {
            errorMap.synchronizeDevice();
            exposureMap.synchronizeDevice();
            FLIP::kernelSetMaxExposure << <this->mGridDim, this->mBlockDim >> > (this->getDeviceData(), errorMap.getDeviceData(), exposureMap.getDeviceData(), this->mDim, exposure);
            image<T>::checkStatus("kernelSetMaxExposure");
            exposureMap.setState(CudaTensorState::DEVICE_ONLY);
            this->setState(CudaTensorState::DEVICE_ONLY);
        }

        void copy(image& srcImage)
        {
            srcImage.synchronizeDevice();
            cudaError_t cudaStatus = cudaMemcpy(this->getDeviceData(), srcImage.getDeviceData(), this->mVolume * sizeof(T), cudaMemcpyDeviceToDevice);
            if (cudaStatus != cudaSuccess)
            {
                std::cout << "copy() failed: " << cudaGetErrorString(cudaStatus) << "\n";
                exit(-1);
            }
            this->setState(CudaTensorState::DEVICE_ONLY);
        }

        void copy(tensor<T>& srcTensor, const int z)
        {
            srcTensor.synchronizeDevice();
            cudaError_t cudaStatus = cudaMemcpy(this->getDeviceData(), srcTensor.getDeviceData(z), this->mArea * sizeof(T), cudaMemcpyDeviceToDevice);
            if (cudaStatus != cudaSuccess)
            {
                std::cout << "copy() failed: " << cudaGetErrorString(cudaStatus) << "\n";
                exit(-1);
            }
            this->setState(CudaTensorState::DEVICE_ONLY);
        }

        void expose(float level)
        {
            float m = std::pow(2.0f, level);
            FLIP::kernelMultiplyAndAdd << <this->mGridDim, this->mBlockDim >> > (this->mvpDeviceData, this->mDim, m, 0.0f);
            image<T>::checkStatus("kernelExpose");
            this->setState(CudaTensorState::DEVICE_ONLY);
        }
#endif
        void computeExposures(const std::string& tm, float& startExposure, float& stopExposure)
        {
            int toneMapper = 1;
            if (tm == "reinhard")
                toneMapper = 0;
            if (tm == "aces")
                toneMapper = 1;
            if (tm == "hable")
                toneMapper = 2;

#ifndef FLIP_ENABLE_CUDA
            const float* tc = FLIP::ToneMappingCoefficients[toneMapper];
#else
            const float* tc = ToneMappingCoefficients[toneMapper];
#endif
            float t = 0.85f;
            float a = tc[0] - t * tc[3];
            float b = tc[1] - t * tc[4];
            float c = tc[2] - t * tc[5];

            float xMin = 0.0f;
            float xMax = 0.0f;
            solveSecondDegree(xMin, xMax, a, b, c);

            float Ymin = 1e30f;
            float Ymax = -1e30f;
            std::vector<float> luminances;
            for (int y = 0; y < this->mDim.y; y++)
            {
                for (int x = 0; x < this->mDim.x; x++)
                {
                    float luminance = color3::linearRGB2Luminance(this->get(x, y));
                    luminances.push_back(luminance);
                    if (luminance != 0.0f)
                    {
                        Ymin = std::min(luminance, Ymin);
                    }
                    Ymax = std::max(luminance, Ymax);
                }
            }

            size_t medianLocation = luminances.size() / 2;
            std::nth_element(luminances.begin(), luminances.begin() + medianLocation, luminances.end());
            float Ymedian = luminances[medianLocation];

            startExposure = log2(xMax / Ymax);
            stopExposure = log2(xMax / Ymedian);
        }

#ifndef FLIP_ENABLE_CUDA
        void LDR_FLIP(image<color3>& reference, image<color3>& test, float ppd)     // Both reference and test are assumed to be in linear RGB.
        {
            // Transform from linear RGB to YCxCz.
            reference.LinearRGB2YCxCz();
            test.LinearRGB2YCxCz();

            // Prepare separated spatial filters. Because the filter for the Blue-Yellow channel is a sum of two Gaussians, we need to separate the spatial filter into two
            // (YCx for the Achromatic and Red-Green channels and Cz for the Blue-Yellow channel).
            int spatialFilterRadius = calculateSpatialFilterRadius(ppd);
            int spatialFilterWidth = 2 * spatialFilterRadius + 1;
            image<color3> spatialFilterYCx(spatialFilterWidth, 1);
            image<color3> spatialFilterCz(spatialFilterWidth, 1);
            setSpatialFilters(spatialFilterYCx, spatialFilterCz, ppd, spatialFilterRadius);

            // The next call performs spatial filtering on both the reference and test image at the same time (for better performance).
            // It then computes the color difference between the images. "this" is an image<float> here, so we store the color difference in that image.
            this->computeColorDifference(reference, test, spatialFilterYCx, spatialFilterCz);

            // Prepare separated feature (edge/point) detection filters.
            const float stdDev = 0.5f * FLIPConstants.gw * ppd;
            const int featureFilterRadius = int(std::ceil(3.0f * stdDev));
            int featureFilterWidth = 2 * featureFilterRadius + 1;
            image<color3> featureFilter(featureFilterWidth, 1);
            setFeatureFilter(featureFilter, ppd);

            // The following call convolves referenceImage and testImage with the edge and point detection filters and performs additional
            // computations for the final feature differences, and then computes the final FLIP error and stores in "this".
            this->computeFeatureDifferenceAndFinalError(reference, test, featureFilter);
        }
#else
        void LDR_FLIP(image<color3>& reference, image<color3>& test, float ppd)     // Both reference and test are assumed to be in linear RGB.
        {
            int width = reference.getWidth();
            int height = reference.getHeight();

            // Temporary images (on device).
            image<color3> intermediateReferenceYCx(width, height), intermediateReferenceCz(width, height), intermediateTestYCx(width, height), intermediateTestCz(width, height);
            image<color3> intermediateFeaturesReference(width, height), intermediateFeaturesTest(width, height);
            image<color3> colorFeatureDifference(width, height);

            // Transform from linear RGB to YCxCz.
            reference.LinearRGB2YCxCz();
            test.LinearRGB2YCxCz();

            // Prepare separated spatial filters. Because the filter for the Blue-Yellow channel is a sum of two Gaussians, we need to separate the spatial filter into two
            // (YCx for the Achromatic and Red-Green channels and Cz for the Blue-Yellow channel). For details, see separatedConvolutions.pdf in the FLIP repository:
            // https://github.com/NVlabs/flip/blob/main/misc/separatedConvolutions.pdf
            int spatialFilterRadius = calculateSpatialFilterRadius(ppd);
            int spatialFilterWidth = 2 * spatialFilterRadius + 1;
            image<color3> spatialFilterYCx(spatialFilterWidth, 1);
            image<color3> spatialFilterCz(spatialFilterWidth, 1);
            setSpatialFilters(spatialFilterYCx, spatialFilterCz, ppd, spatialFilterRadius);

            // The next two calls perform separable spatial filtering on both the reference and test image at the same time (for better performance).
            // The second call also computes the color difference between the images.
            FLIP::image<color3>::spatialFilterFirstDir(reference, intermediateReferenceYCx, intermediateReferenceCz, test, intermediateTestYCx, intermediateTestCz, spatialFilterYCx, spatialFilterCz);
            FLIP::image<color3>::spatialFilterSecondDirAndColorDifference(intermediateReferenceYCx, intermediateReferenceCz, intermediateTestYCx, intermediateTestCz, colorFeatureDifference, spatialFilterYCx, spatialFilterCz);

            // Prepare separated feature (edge/point) detection filters.
            const float stdDev = 0.5f * FLIPConstants.gw * ppd;
            const int featureFilterRadius = int(std::ceil(3.0f * stdDev));
            int featureFilterWidth = 2 * featureFilterRadius + 1;
            image<color3> featureFilter(featureFilterWidth, 1);
            setFeatureFilter(featureFilter, ppd);

            // The following two calls convolve (separably) referenceImage and testImage with the edge and point detection filters and performs additional computations for the feature differences.
            FLIP::image<color3>::featureFilterFirstDir(reference, intermediateFeaturesReference, test, intermediateFeaturesTest, featureFilter);
            FLIP::image<color3>::featureFilterSecondDirAndFeatureDifference(intermediateFeaturesReference, intermediateFeaturesTest, colorFeatureDifference, featureFilter);

            this->finalError(colorFeatureDifference);
        }
#endif
    };

    static FLIP::image<FLIP::color3> magmaMap = FLIP::image<FLIP::color3>(FLIP::MapMagma, 256);
    static FLIP::image<FLIP::color3> viridisMap = FLIP::image<FLIP::color3>(FLIP::MapViridis, 256);

    /** Main function for computing (the image metric called) FLIP between a reference image and a test image.
     *  See FLIP-tool.cpp for usage of this function.
     *
     * @param[in] referenceImageInput Reference input image. For LDR, the content should be in [0,1]. Input is expected in linear RGB.
     * @param[in] testImageInput Test input image. For LDR, the content should be in [0,1]. Input is expected in linear RGB.
     * @param[in] useHDR Set to true if the input images are to be considered containing HDR content, i.e., not necessarily in [0,1].
     * @param[in,out] parameters Contains parameters (e.g., PPD, exposure settings,etc). If the exposures have not been set by the user, then those will be computed (and returned).
     * @param[out] errorMapFLIPOutput The FLIP error image in [0,1], a single channel (grayscale).
                   The user should map it using MapMagma if that is desired (with: errorMapWithMagma.colorMap(errorMapFLIP, FLIP::magmaMap);)
     * @param[out] maxErrorExposureMapOutput Exposure map output (only for HDR content).
     * @param[in] returnLDRFLIPImages True if the next argument should be filled in by FLIP::evaluate().
     * @param[out] hdrOutputFlipLDRImages A list of temporary output LDR-FLIP error maps (in grayscale) from HDR-FLIP.
                   See explanation of the errorMapFLIPOutput parameter for how to convert the maps to magma.
     * @param[in] returnLDRImages True if the next argument should be filled in by FLIP::evaluate().
     * @param[out] hdrOutputLDRImages A list of temporary tonemapped output LDR images (in linear RGB) from HDR-FLIP. Images in this order: Ref0, Test0, Ref1, Test1,...
     */
    static void evaluate(FLIP::image<FLIP::color3>& referenceImageInput, FLIP::image<FLIP::color3>& testImageInput,
        const bool useHDR, FLIP::Parameters& parameters, FLIP::image<float>& errorMapFLIPOutput, FLIP::image<float>& maxErrorExposureMapOutput,
        const bool returnLDRFLIPImages, std::vector<FLIP::image<float>*>& hdrOutputFlipLDRImages,
        const bool returnLDRImages, std::vector<FLIP::image<FLIP::color3>*>& hdrOutputLDRImages)
    {
        FLIP::image<FLIP::color3> referenceImage(referenceImageInput.getWidth(), referenceImageInput.getHeight());
        FLIP::image<FLIP::color3> testImage(referenceImageInput.getWidth(), referenceImageInput.getHeight());
        referenceImage.copy(referenceImageInput);               // Make a copy, since image::LDR_FLIP() destroys the input images.
        testImage.copy(testImageInput);

        if (useHDR)         // Set parameters for HDR-FLIP.
        {
            // If startExposure/stopExposure are inf, they have not been set by the user. If so, compute from referenceImage.
            // See our paper about HDR-FLIP about the details.
            if (parameters.startExposure == std::numeric_limits<float>::infinity() || parameters.stopExposure == std::numeric_limits<float>::infinity())
            {
                float startExp, stopExp;
                referenceImage.computeExposures(parameters.tonemapper, startExp, stopExp);
                if (parameters.startExposure == std::numeric_limits<float>::infinity())
                {
                    parameters.startExposure = startExp;
                }
                if (parameters.stopExposure == std::numeric_limits<float>::infinity())
                {
                    parameters.stopExposure = stopExp;
                }
            }
            if (parameters.startExposure > parameters.stopExposure)
            {
                std::cout << "Start exposure must be smaller than stop exposure!\n";
                exit(-1);
            }
            if (parameters.numExposures == -1)  // -1 means it has not been set by the user, so then we compute it.
            {
                parameters.numExposures = int(std::max(2.0f, std::ceil(parameters.stopExposure - parameters.startExposure)));
            }
        }

        if (useHDR)     // Compute HDR-FLIP.
        {
            FLIP::image<FLIP::color3> rImage(referenceImage.getWidth(), referenceImage.getHeight());
            FLIP::image<FLIP::color3> tImage(referenceImage.getWidth(), referenceImage.getHeight());
            FLIP::image<float> tmpErrorMap(referenceImage.getWidth(), referenceImage.getHeight(), 0.0f);

            float exposureStepSize = (parameters.stopExposure - parameters.startExposure) / (parameters.numExposures - 1);
            for (int i = 0; i < parameters.numExposures; i++)
            {
                float exposure = parameters.startExposure + i * exposureStepSize;
                rImage.copy(referenceImage);
                tImage.copy(testImage);
                rImage.expose(exposure);
                tImage.expose(exposure);
                rImage.toneMap(parameters.tonemapper);
                tImage.toneMap(parameters.tonemapper);
                rImage.clamp();
                tImage.clamp();
                if (returnLDRImages)
                {
                    hdrOutputLDRImages.push_back(new FLIP::image<FLIP::color3>(rImage));
                    hdrOutputLDRImages.push_back(new FLIP::image<FLIP::color3>(tImage));
                }
                tmpErrorMap.LDR_FLIP(rImage, tImage, parameters.PPD);
                if (returnLDRFLIPImages)
                {
                    hdrOutputFlipLDRImages.push_back(new FLIP::image<float>(tmpErrorMap));
                }
                errorMapFLIPOutput.setMaxExposure(tmpErrorMap, maxErrorExposureMapOutput, float(i) / (parameters.numExposures - 1));
            }
        }
        else    // Compute LDR-FLIP.
        {
            referenceImage.clamp();     // The input images should always be in [0,1], but we clamp them here to avoid any problems.
            testImage.clamp();
            errorMapFLIPOutput.LDR_FLIP(referenceImage, testImage, parameters.PPD);
        }
    }

    // This variant does not return any LDR images computed by HDR-FLIP and thus avoids two parameters (since using those is a rare use case).
    static void evaluate(FLIP::image<FLIP::color3>& referenceImageInput, FLIP::image<FLIP::color3>& testImageInput,
         const bool useHDR, FLIP::Parameters& parameters, FLIP::image<float>& errorMapFLIPOutput, FLIP::image<float>& maxErrorExposureMapOutput)
    {
        std::vector<FLIP::image<float>*> hdrOutputFlipLDRImages;
        std::vector<FLIP::image<FLIP::color3>*> hdrOutputLDRImages;
        FLIP::evaluate(referenceImageInput, testImageInput, useHDR, parameters, errorMapFLIPOutput, maxErrorExposureMapOutput, false, hdrOutputFlipLDRImages, false, hdrOutputLDRImages);
    }

    // This variant does not return the exposure map, which may also be used quite seldom.
    static void evaluate(FLIP::image<FLIP::color3>& referenceImageInput, FLIP::image<FLIP::color3>& testImageInput, const bool useHDR, FLIP::Parameters& parameters,
         FLIP::image<float>& errorMapFLIPOutput)
    {
        FLIP::image<float> maxErrorExposureMapOutput(referenceImageInput.getWidth(), referenceImageInput.getHeight());
        FLIP::evaluate(referenceImageInput, testImageInput, useHDR, parameters, errorMapFLIPOutput, maxErrorExposureMapOutput);
    }

    /** A simplified function for computing (the image metric called) FLIP between a reference image and a test image, without the input images being defined using FLIP::image, etc.
     *
     * Note that the user is responsible for deallocating the output image in the varible errorMapFLIPOutput. See the desciption of errorMapFLIPOutput below.
     *
     * @param[in] referenceThreeChannelImage Reference input image. For LDR, the content should be in [0,1]. The image is expected to have 3 floats per pixel and they
     *            are interleaved, i.e., they come in the order: R0G0B0, R1G1B1, etc. Input is expected to be in linear RGB.
     * @param[in] testThreeChannelImage Test input image. For LDR, the content should be in [0,1]. The image is expected to have 3 floats per pixel and they are interleaved.
                  Input is expected to be in linear RGB.
     * @param[in] imageWidth Width of the reference and test images.
     * @param[in] imageHeight Height of the reference and test images.
     * @param[in,out] parameters Contains parameters (e.g., PPD, exposure settings,etc). If the exposures have not been set by the user, then those will be computed (and returned).
     * @param[in] useHDR Set to true if the input images are to be considered containing HDR content, i.e., not necessarily in [0,1].
     * @param[in] applyMagmaMapToOutput A boolean indicating whether the output should have the MagmaMap applied to it before the image is returned.
     * @param[in] computeMeanFLIPError Set to true if the mean FLIP error should be computed. If false, mean error is set to -1.
     * @param[out] meanFLIPError Mean FLIP error in the test (testThreeChannelImage) compared to the reference (referenceThreeChannelImage).
     * @param[out] errorMapFLIPOutput The computed FLIP error image is returned in this variable. If applyMagmaMapToOutput is true, the function will allocate
     *             three channels (and store the magma-mapped FLIP images in sRGB), and
     *             if it is false, only one channel will be allocated (and the FLIP error is returned in that grayscale image).
     *             Note that the user is responsible for deallocating the errorMapFLIPOutput image.
     */
    static void evaluate(const float* referenceThreeChannelImage, const float* testThreeChannelImage,
        const int imageWidth, const int imageHeight, const bool useHDR, FLIP::Parameters& parameters,
        const bool applyMagmaMapToOutput, const bool computeMeanFLIPError, float& meanFLIPError, float** errorMapFLIPOutput)
    {
        FLIP::image<FLIP::color3> referenceImage;
        FLIP::image<FLIP::color3> testImage;
        FLIP::image<float> errorMapFLIPOutputImage(imageWidth, imageHeight);
        referenceImage.setPixels(referenceThreeChannelImage, imageWidth, imageHeight);
        testImage.setPixels(testThreeChannelImage, imageWidth, imageHeight);

        FLIP::evaluate(referenceImage, testImage, useHDR, parameters, errorMapFLIPOutputImage);

#ifdef FLIP_ENABLE_CUDA
        errorMapFLIPOutputImage.synchronizeHost();
#endif

    // Compute mean FLIP error, if desired.
    if (computeMeanFLIPError)
    {
        FLIPPooling::pooling<float> pooledValues;
        for (int y = 0; y < errorMapFLIPOutputImage.getHeight(); y++)
        {
            for (int x = 0; x < errorMapFLIPOutputImage.getWidth(); x++)
            {
                pooledValues.update(x, y, errorMapFLIPOutputImage.get(x, y));
            }
        }
        meanFLIPError = pooledValues.getMean();
    }

        if (applyMagmaMapToOutput)
        {
            *errorMapFLIPOutput = new float[imageWidth * imageHeight * 3];
            FLIP::image<FLIP::color3> magmaMappedFLIPImage(imageWidth, imageHeight);
            magmaMappedFLIPImage.colorMap(errorMapFLIPOutputImage, FLIP::magmaMap);
#ifdef FLIP_ENABLE_CUDA
            magmaMappedFLIPImage.synchronizeHost();
#endif
            memcpy(*errorMapFLIPOutput, magmaMappedFLIPImage.getHostData(), size_t(imageWidth) * imageHeight * sizeof(float) * 3);

        }
        else    // No MagmaMap applied, which means that we will return the gray scale image.
        {
            *errorMapFLIPOutput = new float[imageWidth * imageHeight];
            memcpy(*errorMapFLIPOutput, errorMapFLIPOutputImage.getHostData(), size_t(imageWidth) * imageHeight * sizeof(float));
        }
    }
}
