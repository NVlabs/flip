#pragma once
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>

#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#ifdef FLIP_USE_CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif

#ifdef FLIP_USE_CUDA
#define HOST_DEVICE_FOR_CUDA __host__ __device__
#else
#define HOST_DEVICE_FOR_CUDA
#endif 

// color.h
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
#ifdef FLIP_USE_CUDA
            float3 float3;
#endif
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

        HOST_DEVICE_FOR_CUDA bool operator==(const color3 v) const
        {
            return this->x == v.x && this->y == v.y && this->z == v.z;
        }

        HOST_DEVICE_FOR_CUDA bool operator!=(const color3 v) const
        {
            return !(*this == v);
        }

        HOST_DEVICE_FOR_CUDA color3 operator+(const color3 v) const
        {
            return color3(this->x + v.x, this->y + v.y, this->z + v.z);
        }

        HOST_DEVICE_FOR_CUDA color3 operator-(const color3 v) const
        {
            return color3(this->x - v.x, this->y - v.y, this->z - v.z);
        }

        HOST_DEVICE_FOR_CUDA color3 operator*(const float v) const
        {
            return color3(this->x * v, this->y * v, this->z * v);
        }

        HOST_DEVICE_FOR_CUDA color3 operator*(const color3 v) const
        {
            return color3(this->x * v.x, this->y * v.y, this->z * v.z);
        }

        HOST_DEVICE_FOR_CUDA color3 operator/(const float v) const
        {
            return color3(this->x / v, this->y / v, this->z / v);
        }

        HOST_DEVICE_FOR_CUDA color3 operator/(const color3 v) const
        {
            return color3(this->x / v.x, this->y / v.y, this->z / v.z);
        }

        HOST_DEVICE_FOR_CUDA color3 operator+=(const color3 v)
        {
            this->x += v.x;
            this->y += v.y;
            this->z += v.z;
            return *this;
        }

        HOST_DEVICE_FOR_CUDA color3 operator*=(const color3 v)
        {
            this->x *= v.x;
            this->y *= v.y;
            this->z *= v.z;
            return *this;
        }

        HOST_DEVICE_FOR_CUDA color3 operator/=(const color3 v)
        {
            this->x /= v.x;
            this->y /= v.y;
            this->z /= v.z;
            return *this;
        }

        HOST_DEVICE_FOR_CUDA void clear(const color3 v = { 0.0f, 0.0f, 0.0f })
        {
            this->x = v.x;
            this->y = v.y;
            this->z = v.z;
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

}



// sharedflip.h
namespace FLIP
{
    const float PI = 3.14159265358979f;

    static const struct xFLIPConstants
    {
        xFLIPConstants() = default;
        float gqc = 0.7f;
        float gpc = 0.4f;
        float gpt = 0.95f;
        float gw = 0.082f;
        float gqf = 0.5f;
    } FLIPConstants;

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

    static inline float GaussianSqrt(const float x2, const float a, const float b) // Needed to separate sum of Gaussians filters (see separatedConvolutions.pdf in the FLIP repository).
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
        const float deltaX = 1.0f / ppd;
        const float pi_sq = float(PI * PI);

        float maxScaleParameter = std::max(std::max(std::max(GaussianConstants.b1.x, GaussianConstants.b1.y), std::max(GaussianConstants.b1.z, GaussianConstants.b2.x)), std::max(GaussianConstants.b2.y, GaussianConstants.b2.z));
        int radius = int(std::ceil(3.0f * std::sqrt(maxScaleParameter / (2.0f * pi_sq)) * ppd)); // Set radius based on largest scale parameter.

        return radius;
    }
}

// tensor.h
namespace FLIP
{

    static const float ToneMappingCoefficients[3][6] =
    {
        { 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f },                                                 // Reinhard.
        { 0.6f * 0.6f * 2.51f, 0.6f * 0.03f, 0.0f, 0.6f * 0.6f * 2.43f, 0.6f * 0.59f, 0.14f },  // ACES, 0.6 is pre-exposure cancellation.
        { 0.231683f, 0.013791f, 0.0f, 0.18f, 0.3f, 0.018f },                                    // Hable.
    };

    union int3
    {
        struct { int x, y, z; };
    };

    template<typename T = color3>
    class tensor
    {
    protected:
        int3 mDim;
        int mArea, mVolume;
        T* mvpHostData;

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

        void init(const int3 dim, bool bClear = false, T clearColor = T(0.0f))
        {
            this->mDim = dim;
            this->mArea = dim.x * dim.y;
            this->mVolume = dim.x * dim.y * dim.z;

            allocateHost();

            if (bClear)
            {
                this->clear(clearColor);
            }
        }

    public:

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
            memcpy(this->mvpHostData, pColorMap, size * sizeof(color3));
        }

        ~tensor(void)
        {
            free(this->mvpHostData);
        }

        T* getHostData(void)
        {
            return this->mvpHostData;
        }

        inline int index(int x, int y = 0, int z = 0) const
        {
            return (z * this->mDim.y + y) * mDim.x + x;
        }

        T get(int x, int y, int z) const
        {
            return this->mvpHostData[this->index(x, y, z)];
        }

        void set(int x, int y, int z, T value)
        {
            this->mvpHostData[this->index(x, y, z)] = value;
        }

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

        bool load(const std::string& fileName, const int z = 0)
        {
            bool bOk = false;
            std::string extension = fileName.substr(fileName.find_last_of(".") + 1);
            if (extension == "png" || extension == "bmp" || extension == "tga")
            {
                bOk = this->imageLoad(fileName, z);
            }
            else if (extension == "exr")
            {
                bOk = this->exrLoad(fileName, z);
            }

            return bOk;
        }

        bool imageLoad(const std::string& filename, const int z = 0)
        {
            int width, height, bpp;
            unsigned char* pixels = stbi_load(filename.c_str(), &width, &height, &bpp, 3);
            if (!pixels)
            {
                return false;
            }

            this->init({ width, height, z + 1 });

#pragma omp parallel for
            for (int y = 0; y < this->mDim.y; y++)
            {
                for (int x = 0; x < this->mDim.x; x++)
                {
                    this->set(x, y, z, color3(&pixels[3 * this->index(x, y)]));
                }
            }
            delete[] pixels;

            return true;
        }

        inline static float fClamp(float value) { return std::max(0.0f, std::min(1.0f, value)); }

        bool pngSave(const std::string& filename, int z = 0)
        {
            unsigned char* pixels = new unsigned char[3 * this->mDim.x * this->mDim.y];

#pragma omp parallel for
            for (int y = 0; y < this->mDim.y; y++)
            {
                for (int x = 0; x < this->mDim.x; x++)
                {
                    int index = this->index(x, y);
                    color3 color = this->mvpHostData[this->index(x, y, z)];
                    pixels[3 * index + 0] = (unsigned char)(255.0f * fClamp(color.x) + 0.5f);
                    pixels[3 * index + 1] = (unsigned char)(255.0f * fClamp(color.y) + 0.5f);
                    pixels[3 * index + 2] = (unsigned char)(255.0f * fClamp(color.z) + 0.5f);
                }
            }

            int ok = stbi_write_png(filename.c_str(), this->mDim.x, this->mDim.y, 3, pixels, 3 * this->mDim.x);
            delete[] pixels;

            return (ok != 0);
        }

        bool exrLoad(const std::string& fileName, const int z = 0)
        {
            EXRVersion exrVersion;
            EXRImage exrImage;
            EXRHeader exrHeader;
            InitEXRHeader(&exrHeader);
            InitEXRImage(&exrImage);
            int width, height;

            {
                int ret;
                const char* errorString;

                ret = ParseEXRVersionFromFile(&exrVersion, fileName.c_str());
                if (ret != TINYEXR_SUCCESS || exrVersion.multipart || exrVersion.non_image)
                {
                    std::cerr << "Unsupported EXR version or type!" << std::endl;
                    return false;
                }

                ret = ParseEXRHeaderFromFile(&exrHeader, &exrVersion, fileName.c_str(), &errorString);
                if (ret != TINYEXR_SUCCESS)
                {
                    std::cerr << "Error loading EXR header: " << errorString << std::endl;
                    return false;
                }

                for (int i = 0; i < exrHeader.num_channels; i++)
                {
                    if (exrHeader.pixel_types[i] == TINYEXR_PIXELTYPE_HALF)
                    {
                        exrHeader.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
                    }
                }

                ret = LoadEXRImageFromFile(&exrImage, &exrHeader, fileName.c_str(), &errorString);
                if (ret != TINYEXR_SUCCESS)
                {
                    std::cerr << "Error loading EXR file: " << errorString << std::endl;
                    return false;
                }
            }

            width = exrImage.width;
            height = exrImage.height;

            this->init({ width, height, z + 1 });

            int idxR = -1;
            int idxG = -1;
            int idxB = -1;
            int numRecognizedChannels = 0;
            for (int c = 0; c < exrHeader.num_channels; c++)
            {
                std::string channelName = exrHeader.channels[c].name;
                std::transform(channelName.begin(), channelName.end(), channelName.begin(), ::tolower);
                if (channelName == "r")
                {
                    idxR = c;
                    ++numRecognizedChannels;
                }
                else if (channelName == "g")
                {
                    idxG = c;
                    ++numRecognizedChannels;
                }
                else if (channelName == "b")
                {
                    idxB = c;
                    ++numRecognizedChannels;
                }
                else if (channelName == "a")
                {
                    ++numRecognizedChannels;
                }
                else
                {
                    std::cerr << "Undefined EXR channel name: " << exrHeader.channels[c].name << std::endl;
                }
            }
            if (numRecognizedChannels < exrHeader.num_channels)
            {
                std::cerr << "EXR channels may be loaded in the wrong order." << std::endl;
                idxR = 0;
                idxG = 1;
                idxB = 2;
            }

            auto rawImgChn = reinterpret_cast<float**>(exrImage.images);
            bool loaded = false;

            // 1 channel images can be loaded into either scalar or vector formats.
            if (exrHeader.num_channels == 1)
            {
                for (int y = 0; y < this->mDim.y; y++)
                {
                    for (int x = 0; x < this->mDim.x; x++)
                    {
                        float color(rawImgChn[0][this->index(x, y)]);
                        this->set(x, y, z, color3(color));
                    }
                }
                loaded = true;
            }

            // 2 channel images can only be loaded into vector2/3/4 formats.
            if (exrHeader.num_channels == 2)
            {
                assert(idxR != -1 && idxG != -1);

#pragma omp parallel for
                for (int y = 0; y < this->mDim.y; y++)
                {
                    for (int x = 0; x < this->mDim.x; x++)
                    {
                        size_t linearIdx = this->index(x, y);
                        color3 color;
                        color.x = rawImgChn[idxR][linearIdx];
                        color.y = rawImgChn[idxG][linearIdx];
                        this->set(x, y, z, color);
                    }
                }
                loaded = true;
            }

            // 3 channel images can only be loaded into vector3/4 formats.
            if (exrHeader.num_channels == 3)
            {
                assert(idxR != -1 && idxG != -1 && idxB != -1);

#pragma omp parallel for
                for (int y = 0; y < this->mDim.y; y++)
                {
                    for (int x = 0; x < this->mDim.x; x++)
                    {
                        size_t linearIdx = this->index(x, y);
                        color3 color;
                        color.x = rawImgChn[idxR][linearIdx];
                        color.y = rawImgChn[idxG][linearIdx];
                        color.z = rawImgChn[idxB][linearIdx];
                        this->set(x, y, z, color);
                    }
                }
                loaded = true;
            }

            // 4 channel images can only be loaded into vector4 formats.
            if (exrHeader.num_channels == 4)
            {
                assert(idxR != -1 && idxG != -1 && idxB != -1);

#pragma omp parallel for
                for (int y = 0; y < this->mDim.y; y++)
                {
                    for (int x = 0; x < this->mDim.x; x++)
                    {
                        size_t linearIdx = this->index(x, y);
                        color3 color;
                        color.x = rawImgChn[idxR][linearIdx];
                        color.y = rawImgChn[idxG][linearIdx];
                        color.z = rawImgChn[idxB][linearIdx];
                        this->set(x, y, z, color);
                    }
                }
                loaded = true;
            }

            FreeEXRHeader(&exrHeader);
            FreeEXRImage(&exrImage);

            if (!loaded)
            {
                std::cerr << "Insufficient target channels when loading EXR: need " << exrHeader.num_channels << std::endl;
                return false;
            }
            else
            {
                return true;
            }
        }

    };
}


// image.h
namespace FLIP
{

    template<typename T>
    class image :
        public tensor<T>
    {
    public:

        image(std::string fileName)
        {
            bool bOk = this->load(fileName);
            if (!bOk)
            {
                std::cout << "Failed to load image <" << fileName << ">" << std::endl;
                exit(-1);
            }
        }

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

        ~image(void)
        {
        }

        T get(int x, int y) const
        {
            return this->mvpHostData[this->index(x, y)];
        }

        void set(int x, int y, T value)
        {
            this->mvpHostData[this->index(x, y)] = value;
        }


        //////////////////////////////////////////////////////////////////////////////////

        static void setSpatialFilters(image<color3>& filterYCx, image<color3>& filterCz, float ppd, int filterRadius) // For details, see separatedConvolutions.pdf in the FLIP repository.
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

        static void setFeatureFilter(image<color3>& filter, const float ppd) // For details, see separatedConvolutions.pdf in the FLIP repository.
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

        void FLIP(image<color3>& reference, image<color3>& test, float ppd)
        {
            int width = reference.getWidth();
            int height = reference.getHeight();

            // Transform from sRGB to YCxCz.
            reference.sRGB2YCxCz();
            test.sRGB2YCxCz();

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

        // Performs spatial filtering (and clamps the results) on both the reference and test image at the same time (for better performance).
        // Filtering has been changed to separable filtering for better performance. For details on the convolution, see separatedConvolutions.pdf in the FLIP repository.
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
            // For details, see separatedConvolutions.pdf in the FLIP repository.
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
            // For details on the convolution, see separatedConvolutions.pdf in the FLIP repository.
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

        void computeExposures(std::string tm, float& startExposure, float& stopExposure)
        {
            int toneMapper = 1;
            if (tm == "reinhard")
                toneMapper = 0;
            if (tm == "aces")
                toneMapper = 1;
            if (tm == "hable")
                toneMapper = 2;

            const float* tc = FLIP::ToneMappingCoefficients[toneMapper];

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
                    float luminance = color3::linearRGB2Luminance(this->mvpHostData[this->index(x, y)]);
                    luminances.push_back(luminance);
                    if (luminance != 0.0f)
                    {
                        Ymin = std::min(luminance, Ymin);
                    }
                    Ymax = std::max(luminance, Ymax);
                }
            }
            std::sort(luminances.begin(), luminances.end());
            size_t medianLocation = luminances.size() / 2 - 1;
            float Ymedian = (luminances[medianLocation] + luminances[medianLocation + 1]) * 0.5f;

            startExposure = log2(xMax / Ymax);
            stopExposure = log2(xMax / Ymedian);
        }

        bool exrSave(const std::string& fileName)
        {
            constexpr int channels = 3;

            float* vpImage[channels];
            std::vector<float> vImages[channels];
            for (int i = 0; i < channels; ++i)
            {
                vImages[i].resize(this->mDim.x * this->mDim.y);
            }
            int pixelIndex = 0;
            for (int y = 0; y < this->mDim.y; y++)
            {
                for (int x = 0; x < this->mDim.x; x++)
                {
                    color3 p = this->get(x, y);
                    vImages[0][pixelIndex] = p.r;
                    vImages[1][pixelIndex] = p.g;
                    vImages[2][pixelIndex] = p.b;
                    pixelIndex++;
                }
            }
            vpImage[0] = &(vImages[2].at(0));  // B
            vpImage[1] = &(vImages[1].at(0));  // G
            vpImage[2] = &(vImages[0].at(0));  // R

            EXRHeader exrHeader;
            InitEXRHeader(&exrHeader);
            exrHeader.num_channels = channels;
            exrHeader.channels = (EXRChannelInfo*)malloc(channels * sizeof(EXRChannelInfo));
            exrHeader.pixel_types = (int*)malloc(channels * sizeof(int));
            exrHeader.requested_pixel_types = (int*)malloc(channels * sizeof(int));
            exrHeader.compression_type = TINYEXR_COMPRESSIONTYPE_ZIP;
            for (int i = 0; i < channels; i++)
            {
                exrHeader.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
                exrHeader.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
                exrHeader.channels[i].name[1] = '\0';
            }
            exrHeader.channels[0].name[0] = 'B';
            exrHeader.channels[1].name[0] = 'G';
            exrHeader.channels[2].name[0] = 'R';

            EXRImage exrImage;
            InitEXRImage(&exrImage);
            exrImage.num_channels = channels;
            exrImage.images = (unsigned char**)vpImage;
            exrImage.width = this->mWidth;
            exrImage.height = this->mHeight;

            const char* error;
            int ret = SaveEXRImageToFile(&exrImage, &exrHeader, fileName.c_str(), &error);
            if (ret != TINYEXR_SUCCESS)
            {
                std::cerr << "Failed to save EXR file <" << fileName << ">: " << error << "\n";
                return false;
            }

            free(exrHeader.channels);
            free(exrHeader.pixel_types);
            free(exrHeader.requested_pixel_types);

            return true;
        }

    };

    template <>
    bool image<float>::exrSave(const std::string& fileName)
    {
        EXRHeader exrHeader;
        InitEXRHeader(&exrHeader);
        exrHeader.num_channels = 1;
        exrHeader.channels = (EXRChannelInfo*)malloc(sizeof(EXRChannelInfo));
        exrHeader.pixel_types = (int*)malloc(sizeof(int));
        exrHeader.requested_pixel_types = (int*)malloc(sizeof(int));
        exrHeader.compression_type = TINYEXR_COMPRESSIONTYPE_ZIP;
        exrHeader.pixel_types[0] = TINYEXR_PIXELTYPE_FLOAT;
        exrHeader.requested_pixel_types[0] = TINYEXR_PIXELTYPE_FLOAT;
        exrHeader.channels[0].name[0] = 'R';
        exrHeader.channels[0].name[1] = '\0';

        EXRImage exrImage;
        InitEXRImage(&exrImage);
        exrImage.num_channels = 1;
        exrImage.images = (unsigned char**)&(this->mvpHostData);
        exrImage.width = this->mDim.x;
        exrImage.height = this->mDim.y;

        const char* error;
        int ret = SaveEXRImageToFile(&exrImage, &exrHeader, fileName.c_str(), &error);
        if (ret != TINYEXR_SUCCESS)
        {
            std::cerr << "Failed to save EXR file <" << fileName << ">: " << error << "\n";
            return false;
        }

        free(exrHeader.channels);
        free(exrHeader.pixel_types);
        free(exrHeader.requested_pixel_types);

        return true;
    }
}
