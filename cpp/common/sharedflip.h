#pragma once

#ifdef USING_CUDA
#include "../CUDA/color.cuh"
#else
#include "../CPP/color.h"
#endif

namespace FLIP
{
    const float PI = 3.14159265358979f;

    static const struct
    {
        float gqc = 0.7f;
        float gpc = 0.4f;
        float gpt = 0.95f;
        float gw = 0.082f;
        float gqf = 0.5f;
    } FLIPConstants;

    static const struct
    {
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

    static inline float GaussianSqrt(const float x2, const float a, const float b) // Needed to separate sum of Gaussians filters (see separated_convolutions.pdf in the FLIP repository).
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