//
// std.cpp
//
// Code generation for function 'std'
//

// Include files
#include "std.h"
#include "blockedSummation.h"
#include "extractAllFeatures_data.h"
#include "rt_nonfinite.h"
#include "xnrm2.h"
#include "coder_array.h"
#include "omp.h"
#include <cmath>
#include <cstring>
#include <xmmintrin.h>

// Function Definitions
namespace coder {
float b_std(const array<creal32_T, 1U> &x)
{
  array<float, 1U> absdiff;
  creal32_T b_x;
  creal32_T xbar;
  float a;
  float b;
  float y;
  int n;
  n = x.size(0);
  xbar = blockedSummation(x, x.size(0));
  if (xbar.im == 0.0F) {
    xbar.re /= static_cast<float>(x.size(0));
    xbar.im = 0.0F;
  } else if (xbar.re == 0.0F) {
    xbar.re = 0.0F;
    xbar.im /= static_cast<float>(x.size(0));
  } else {
    xbar.re /= static_cast<float>(x.size(0));
    xbar.im /= static_cast<float>(x.size(0));
  }
  absdiff.set_size(x.size(0));
  if (static_cast<int>(x.size(0) < 800)) {
    for (int k{0}; k < n; k++) {
      a = std::abs(x[k].re - xbar.re);
      b = std::abs(x[k].im - xbar.im);
      if (a < b) {
        a /= b;
        absdiff[k] = b * std::sqrt(a * a + 1.0F);
      } else if (a > b) {
        b /= a;
        absdiff[k] = a * std::sqrt(b * b + 1.0F);
      } else if (std::isnan(b)) {
        absdiff[k] = rtNaNF;
      } else {
        absdiff[k] = a * 1.41421354F;
      }
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(b, a, b_x)

    for (int k = 0; k < n; k++) {
      b_x.re = x[k].re - xbar.re;
      b_x.im = x[k].im - xbar.im;
      a = std::abs(b_x.re);
      b = std::abs(b_x.im);
      if (a < b) {
        a /= b;
        absdiff[k] = b * std::sqrt(a * a + 1.0F);
      } else if (a > b) {
        b /= a;
        absdiff[k] = a * std::sqrt(b * b + 1.0F);
      } else if (std::isnan(b)) {
        absdiff[k] = rtNaNF;
      } else {
        absdiff[k] = a * 1.41421354F;
      }
    }
  }
  y = internal::blas::xnrm2(x.size(0), absdiff) /
      std::sqrt(static_cast<float>(x.size(0)) - 1.0F);
  return y;
}

float b_std(const float x_data[], int x_size)
{
  array<float, 1U> b_absdiff_data;
  array<float, 1U> b_x_data;
  float absdiff_data[16383];
  float fv[4];
  float fv1[4];
  float y;
  if (x_size == 0) {
    y = rtNaNF;
  } else if (x_size == 1) {
    if ((!std::isinf(x_data[0])) && (!std::isnan(x_data[0]))) {
      y = 0.0F;
    } else {
      y = rtNaNF;
    }
  } else {
    int scalarLB;
    int vectorUB;
    b_x_data.set(const_cast<float *>(&x_data[0]), x_size);
    y = blockedSummation(b_x_data, x_size) / static_cast<float>(x_size);
    scalarLB = (x_size / 4) << 2;
    vectorUB = scalarLB - 4;
    for (int k{0}; k <= vectorUB; k += 4) {
      __m128 r;
      _mm_storeu_ps(&fv[0],
                    _mm_sub_ps(_mm_loadu_ps(&x_data[k]), _mm_set1_ps(y)));
      fv1[0] = std::abs(fv[0]);
      fv1[1] = std::abs(fv[1]);
      fv1[2] = std::abs(fv[2]);
      fv1[3] = std::abs(fv[3]);
      r = _mm_loadu_ps(&fv1[0]);
      _mm_storeu_ps(&absdiff_data[k], r);
    }
    for (int k{scalarLB}; k < x_size; k++) {
      absdiff_data[k] = std::abs(x_data[k] - y);
    }
    b_absdiff_data.set(&absdiff_data[0], x_size);
    y = internal::blas::xnrm2(x_size, b_absdiff_data) /
        std::sqrt(static_cast<float>(x_size) - 1.0F);
  }
  return y;
}

float b_std(const float x[4096])
{
  float bsum;
  float scale;
  float y;
  y = x[0];
  for (int k{0}; k < 1023; k++) {
    y += x[k + 1];
  }
  for (int k{0}; k < 3; k++) {
    int xblockoffset;
    xblockoffset = (k + 1) << 10;
    bsum = x[xblockoffset];
    for (int b_k{0}; b_k < 1023; b_k++) {
      bsum += x[(xblockoffset + b_k) + 1];
    }
    y += bsum;
  }
  bsum = y / 4096.0F;
  y = 0.0F;
  scale = 1.29246971E-26F;
  for (int k{0}; k < 4096; k++) {
    float f;
    f = std::abs(x[k] - bsum);
    if (f > scale) {
      float t;
      t = scale / f;
      y = y * t * t + 1.0F;
      scale = f;
    } else {
      float t;
      t = f / scale;
      y += t * t;
    }
  }
  return scale * std::sqrt(y) / 63.9921875F;
}

} // namespace coder

// End of code generation (std.cpp)
