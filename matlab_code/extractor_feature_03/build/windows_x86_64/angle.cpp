//
// angle.cpp
//
// Code generation for function 'angle'
//

// Include files
#include "angle.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include "omp.h"
#include "rt_defines.h"
#include <cmath>
#include <cstring>

// Function Definitions
namespace coder {
int angle(const array<creal32_T, 1U> &x, float y_data[])
{
  float b_y;
  float c_x;
  int i1;
  int i3;
  int y_size;
  y_size = x.size(0);
  if (static_cast<int>(x.size(0) < 800)) {
    for (int k{0}; k < y_size; k++) {
      float b_x;
      float y;
      y = x[k].im;
      b_x = x[k].re;
      if (std::isnan(y) || std::isnan(b_x)) {
        y = rtNaNF;
      } else if (std::isinf(y) && std::isinf(b_x)) {
        int i;
        int i2;
        if (y > 0.0F) {
          i = 1;
        } else {
          i = -1;
        }
        if (b_x > 0.0F) {
          i2 = 1;
        } else {
          i2 = -1;
        }
        y = std::atan2(static_cast<float>(i), static_cast<float>(i2));
      } else if (b_x == 0.0F) {
        if (y > 0.0F) {
          y = RT_PIF / 2.0F;
        } else if (y < 0.0F) {
          y = -(RT_PIF / 2.0F);
        } else {
          y = 0.0F;
        }
      } else {
        y = std::atan2(y, b_x);
      }
      y_data[k] = y;
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        b_y, c_x, i1, i3)

    for (int k = 0; k < y_size; k++) {
      b_y = x[k].im;
      c_x = x[k].re;
      if (std::isnan(b_y) || std::isnan(c_x)) {
        b_y = rtNaNF;
      } else if (std::isinf(b_y) && std::isinf(c_x)) {
        if (b_y > 0.0F) {
          i1 = 1;
        } else {
          i1 = -1;
        }
        if (c_x > 0.0F) {
          i3 = 1;
        } else {
          i3 = -1;
        }
        b_y = std::atan2(static_cast<float>(i1), static_cast<float>(i3));
      } else if (c_x == 0.0F) {
        if (b_y > 0.0F) {
          b_y = RT_PIF / 2.0F;
        } else if (b_y < 0.0F) {
          b_y = -(RT_PIF / 2.0F);
        } else {
          b_y = 0.0F;
        }
      } else {
        b_y = std::atan2(b_y, c_x);
      }
      y_data[k] = b_y;
    }
  }
  return y_size;
}

} // namespace coder

// End of code generation (angle.cpp)
