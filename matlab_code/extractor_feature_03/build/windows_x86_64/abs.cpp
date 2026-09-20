//
// abs.cpp
//
// Code generation for function 'abs'
//

// Include files
#include "abs.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include "omp.h"
#include <cmath>
#include <cstring>

// Function Definitions
namespace coder {
float b_abs(const creal32_T x)
{
  float b;
  float y;
  y = std::abs(x.re);
  b = std::abs(x.im);
  if (y < b) {
    y /= b;
    y = b * std::sqrt(y * y + 1.0F);
  } else if (y > b) {
    b /= y;
    y *= std::sqrt(b * b + 1.0F);
  } else if (std::isnan(b)) {
    y = rtNaNF;
  } else {
    y *= 1.41421354F;
  }
  return y;
}

void b_abs(const array<creal32_T, 1U> &x, array<float, 1U> &y)
{
  float a;
  float b;
  int i;
  i = x.size(0);
  y.set_size(x.size(0));
  if (static_cast<int>(x.size(0) < 800)) {
    for (int k{0}; k < i; k++) {
      a = std::abs(x[k].re);
      b = std::abs(x[k].im);
      if (a < b) {
        a /= b;
        y[k] = b * std::sqrt(a * a + 1.0F);
      } else if (a > b) {
        b /= a;
        y[k] = a * std::sqrt(b * b + 1.0F);
      } else if (std::isnan(b)) {
        y[k] = rtNaNF;
      } else {
        y[k] = a * 1.41421354F;
      }
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(b, a)

    for (int k = 0; k < i; k++) {
      a = std::abs(x[k].re);
      b = std::abs(x[k].im);
      if (a < b) {
        a /= b;
        y[k] = b * std::sqrt(a * a + 1.0F);
      } else if (a > b) {
        b /= a;
        y[k] = a * std::sqrt(b * b + 1.0F);
      } else if (std::isnan(b)) {
        y[k] = rtNaNF;
      } else {
        y[k] = a * 1.41421354F;
      }
    }
  }
}

void b_abs(const array<creal32_T, 2U> &x, array<float, 2U> &y)
{
  float a;
  float b;
  int nx;
  nx = x.size(1) << 9;
  y.set_size(512, x.size(1));
  if (static_cast<int>(nx < 800)) {
    for (int k{0}; k < nx; k++) {
      a = std::abs(x[k].re);
      b = std::abs(x[k].im);
      if (a < b) {
        a /= b;
        y[k] = b * std::sqrt(a * a + 1.0F);
      } else if (a > b) {
        b /= a;
        y[k] = a * std::sqrt(b * b + 1.0F);
      } else if (std::isnan(b)) {
        y[k] = rtNaNF;
      } else {
        y[k] = a * 1.41421354F;
      }
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(b, a)

    for (int k = 0; k < nx; k++) {
      a = std::abs(x[k].re);
      b = std::abs(x[k].im);
      if (a < b) {
        a /= b;
        y[k] = b * std::sqrt(a * a + 1.0F);
      } else if (a > b) {
        b /= a;
        y[k] = a * std::sqrt(b * b + 1.0F);
      } else if (std::isnan(b)) {
        y[k] = rtNaNF;
      } else {
        y[k] = a * 1.41421354F;
      }
    }
  }
}

void b_abs(const array<creal_T, 2U> &x, array<double, 2U> &y)
{
  double a;
  double b;
  int nx;
  nx = x.size(0) * x.size(1);
  y.set_size(x.size(0), x.size(1));
  if (static_cast<int>(nx < 800)) {
    for (int k{0}; k < nx; k++) {
      a = std::abs(x[k].re);
      b = std::abs(x[k].im);
      if (a < b) {
        a /= b;
        y[k] = b * std::sqrt(a * a + 1.0);
      } else if (a > b) {
        b /= a;
        y[k] = a * std::sqrt(b * b + 1.0);
      } else if (std::isnan(b)) {
        y[k] = rtNaN;
      } else {
        y[k] = a * 1.4142135623730951;
      }
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(b, a)

    for (int k = 0; k < nx; k++) {
      a = std::abs(x[k].re);
      b = std::abs(x[k].im);
      if (a < b) {
        a /= b;
        y[k] = b * std::sqrt(a * a + 1.0);
      } else if (a > b) {
        b /= a;
        y[k] = a * std::sqrt(b * b + 1.0);
      } else if (std::isnan(b)) {
        y[k] = rtNaN;
      } else {
        y[k] = a * 1.4142135623730951;
      }
    }
  }
}

void c_abs(const array<creal32_T, 2U> &x, array<float, 2U> &y)
{
  float a;
  float b;
  int nx;
  nx = x.size(0) * x.size(1);
  y.set_size(static_cast<int>(static_cast<short>(x.size(0))),
             static_cast<int>(static_cast<short>(x.size(1))));
  if (static_cast<int>(nx < 800)) {
    for (int k{0}; k < nx; k++) {
      a = std::abs(x[k].re);
      b = std::abs(x[k].im);
      if (a < b) {
        a /= b;
        y[k] = b * std::sqrt(a * a + 1.0F);
      } else if (a > b) {
        b /= a;
        y[k] = a * std::sqrt(b * b + 1.0F);
      } else if (std::isnan(b)) {
        y[k] = rtNaNF;
      } else {
        y[k] = a * 1.41421354F;
      }
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(b, a)

    for (int k = 0; k < nx; k++) {
      a = std::abs(x[k].re);
      b = std::abs(x[k].im);
      if (a < b) {
        a /= b;
        y[k] = b * std::sqrt(a * a + 1.0F);
      } else if (a > b) {
        b /= a;
        y[k] = a * std::sqrt(b * b + 1.0F);
      } else if (std::isnan(b)) {
        y[k] = rtNaNF;
      } else {
        y[k] = a * 1.41421354F;
      }
    }
  }
}

} // namespace coder

// End of code generation (abs.cpp)
