/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2020-2021 NVIDIA CORPORATION & AFFILIATES
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

// Code by Pontus Andersson, Jim Nilsson, and Tomas Akenine-Moller.

#pragma once

#include <iostream>
#include <string>

namespace FLIP
{

#define Max(x, y) ((x) > (y) ? (x) : (y))
#define Min(x, y) ((x) > (y) ? (y) : (x))

#define DEFAULT_ILLUMINANT { 0.950428545f, 1.000000000f, 1.088900371f }
#define INV_DEFAULT_ILLUMINANT { 1.052156925f, 1.000000000f, 0.918357670f }

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
        color3(void)
        {
            this->x = 0.0f;
            this->y = 0.0f;
            this->z = 0.0f;
        }

        color3(float v)
        {
            this->x = v;
            this->y = v;
            this->z = v;
        }

        color3(const float* pColor)
        {
            this->x = pColor[0];
            this->y = pColor[1];
            this->z = pColor[2];
        }

        color3(const unsigned char* pColor)
        {
            this->x = float(pColor[0]);
            this->y = float(pColor[1]);
            this->z = float(pColor[2]);
            *this /= 255.0f;
        }

        color3(float _x, float _y, float _z)
        {
            this->x = _x;
            this->y = _y;
            this->z = _z;
        }

        color3(const color3& c)
        {
            this->x = c.x;
            this->y = c.y;
            this->z = c.z;
        }

        bool operator==(const color3 v) const
        {
            return this->x == v.x && this->y == v.y && this->z == v.z;
        }

        bool operator!=(const color3 v) const
        {
            return !(*this == v);
        }

        color3 operator+(const color3 v) const
        {
            return color3(this->x + v.x, this->y + v.y, this->z + v.z);
        }

        color3 operator-(const color3 v) const
        {
            return color3(this->x - v.x, this->y - v.y, this->z - v.z);
        }

        color3 operator*(const float v) const
        {
            return color3(this->x * v, this->y * v, this->z * v);
        }

        color3 operator*(const color3 v) const
        {
            return color3(this->x * v.x, this->y * v.y, this->z * v.z);
        }

        color3 operator/(const float v) const
        {
            return color3(this->x / v, this->y / v, this->z / v);
        }

        color3 operator/(const color3 v) const
        {
            return color3(this->x / v.x, this->y / v.y, this->z / v.z);
        }

        color3 operator+=(const color3 v)
        {
            this->x += v.x;
            this->y += v.y;
            this->z += v.z;
            return *this;
        }

        color3 operator*=(const color3 v)
        {
            this->x *= v.x;
            this->y *= v.y;
            this->z *= v.z;
            return *this;
        }

        color3 operator/=(const color3 v)
        {
            this->x /= v.x;
            this->y /= v.y;
            this->z /= v.z;
            return *this;
        }

        void clear(const color3 v = { 0.0f, 0.0f, 0.0f })
        {
            this->x = v.x;
            this->y = v.y;
            this->z = v.z;
        }

        static inline color3 min(color3 v0, color3 v1)
        {
            return color3(Min(v0.x, v1.x), Min(v0.y, v1.y), Min(v0.z, v1.z));
        }

        static inline color3 max(color3 v0, color3 v1)
        {
            return color3(Max(v0.x, v1.x), Max(v0.y, v1.y), Max(v0.z, v1.z));
        }

        static inline color3 abs(color3 v)
        {
            return color3(std::abs(v.x), std::abs(v.y), std::abs(v.z));
        }

        static inline color3 sqrt(color3 v)
        {
            return color3(std::sqrt(v.x), std::sqrt(v.y), std::sqrt(v.z));
        }

        static inline color3 clamp(color3 v, float _min = 0.0f, float _max = 1.0f)
        {
            return color3(Min(Max(v.x, _min), _max), Min(Max(v.y, _min), _max), Min(Max(v.z, _min), _max));
        }

        static inline float linearRGB2Luminance(color3 linearRGB)
        {
            return 0.2126f * linearRGB.r + 0.7152f * linearRGB.g + 0.0722f * linearRGB.b;
        }

        static inline float sRGB2LinearRGB(float sC)
        {
            if (sC <= 0.04045f)
            {
                return sC / 12.92f;
            }
            return powf((sC + 0.055f) / 1.055f, 2.4f);
        }

        static inline float LinearRGB2sRGB(float lC)
        {
            if (lC <= 0.0031308f)
            {
                return lC * 12.92f;
            }

            return 1.055f * powf(lC, 1.0f / 2.4f) - 0.055f;
        }

        static inline color3 sRGB2LinearRGB(color3 sRGB)
        {
            float R = sRGB2LinearRGB(sRGB.x);
            float G = sRGB2LinearRGB(sRGB.y);
            float B = sRGB2LinearRGB(sRGB.z);

            return color3(R, G, B);
        }

        static inline color3 LinearRGB2sRGB(color3 RGB)
        {
            float sR = LinearRGB2sRGB(RGB.x);
            float sG = LinearRGB2sRGB(RGB.y);
            float sB = LinearRGB2sRGB(RGB.z);

            return color3(sR, sG, sB);
        }

        static inline color3 LinearRGB2XYZ(color3 RGB)
        {
            // Source: https://www.image-engineering.de/library/technotes/958-how-to-convert-between-srgb-and-ciexyz
            // Assumes D65 standard illuminant
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

        static inline color3 XYZ2LinearRGB(color3 XYZ)
        {
            // Return values in linear RGB, assuming D65 standard illuminant
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

        static inline color3 XYZ2CIELab(color3 XYZ, const color3 invReferenceIlluminant = INV_DEFAULT_ILLUMINANT)
        {
            const float delta = 6.0f / 29.0f;
            const float deltaSquare = delta * delta;
            const float deltaCube = delta * deltaSquare;
            const float factor = 1.0f / (3.0f * deltaSquare);
            const float term = 4.0f / 29.0f;

            // the default illuminant is D65
            XYZ = XYZ * invReferenceIlluminant;
            XYZ.x = (XYZ.x > deltaCube ? powf(XYZ.x, 1.0f / 3.0f) : factor * XYZ.x + term);
            XYZ.y = (XYZ.y > deltaCube ? powf(XYZ.y, 1.0f / 3.0f) : factor * XYZ.y + term);
            XYZ.z = (XYZ.z > deltaCube ? powf(XYZ.z, 1.0f / 3.0f) : factor * XYZ.z + term);
            float L = 116.0f * XYZ.y - 16.0f;
            float a = 500.0f * (XYZ.x - XYZ.y);
            float b = 200.0f * (XYZ.y - XYZ.z);

            return color3(L, a, b);
        }

        static inline color3 CIELab2XYZ(color3 Lab, const color3 referenceIlluminant = DEFAULT_ILLUMINANT)
        {
            // the default illuminant is D65
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

        static inline color3 XYZ2YCxCz(color3 XYZ, const color3 invReferenceIlluminant = INV_DEFAULT_ILLUMINANT)
        {
            // the default illuminant is D65
            XYZ = XYZ * invReferenceIlluminant;
            float Y = 116.0f * XYZ.y - 16.0f;
            float Cx = 500.0f * (XYZ.x - XYZ.y);
            float Cz = 200.0f * (XYZ.y - XYZ.z);

            return color3(Y, Cx, Cz);
        }

        static inline color3 YCxCz2XYZ(color3 YCxCz, const color3 referenceIlluminant = DEFAULT_ILLUMINANT)
        {
            // the default illuminant is D65
            const float Y = (YCxCz.x + 16.0f) / 116.0f;
            const float Cx = YCxCz.y / 500.0f;
            const float Cz = YCxCz.z / 200.0f;
            float X = Y + Cx;
            float Z = Y - Cz;

            return color3(X, Y, Z) * referenceIlluminant;
        }

        static inline float YCxCz2Gray(color3 YCxCz)
        {
            return (YCxCz.x + 16.0f) / 116.0f; //  make it [0,1]
        }

        //  FLIP-specific functions below

        static inline float Hunt(const float luminance, const float chrominance)
        {
            return 0.01f * luminance * chrominance;
        }

        static inline float HyAB(color3& refPixel, color3& testPixel)
        {
            float cityBlockDistanceL = std::fabs(refPixel.x - testPixel.x);
            float euclideanDistanceAB = std::sqrt((refPixel.y - testPixel.y) * (refPixel.y - testPixel.y) + (refPixel.z - testPixel.z) * (refPixel.z - testPixel.z));
            return cityBlockDistanceL + euclideanDistanceAB;
        }

        static inline float computeMaxDistance(float gqc)
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

}
