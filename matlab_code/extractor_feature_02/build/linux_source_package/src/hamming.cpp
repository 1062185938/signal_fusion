//
// hamming.cpp
//
// Code generation for function 'hamming'
//

// Include files
#include "hamming.h"
#include "extractAllFeatures_rtwutil.h"
#include "gencoswin.h"
#include "rt_nonfinite.h"
#include <algorithm>
#include <cmath>
#include <cstring>

// Function Definitions
namespace coder {
int hamming(double varargin_1, double w_data[])
{
  double b_w_data[1024];
  double L;
  double r;
  int w_size;
  if (varargin_1 == std::floor(varargin_1)) {
    L = varargin_1;
  } else {
    L = varargin_1;
    if (std::abs(varargin_1) < 4.503599627370496E+15) {
      if (varargin_1 >= 0.5) {
        L = std::floor(varargin_1 + 0.5);
      } else if (varargin_1 > -0.5) {
        L = varargin_1 * 0.0;
      } else {
        L = std::ceil(varargin_1 - 0.5);
      }
    }
  }
  if (std::isnan(L + 1.0) || std::isinf(L + 1.0)) {
    r = rtNaN;
  } else {
    r = std::fmod(L + 1.0, 2.0);
    if (r == 0.0) {
      r = 0.0;
    } else if (r < 0.0) {
      r += 2.0;
    }
  }
  if (r == 0.0) {
    int b_loop_ub;
    int i;
    int i1;
    int loop_ub;
    w_size = calc_window((L + 1.0) / 2.0, L + 1.0, w_data);
    if (w_size < 2) {
      i = 0;
      i1 = 1;
      loop_ub = -1;
    } else {
      i = w_size - 1;
      i1 = -1;
      loop_ub = 1;
    }
    loop_ub = div_s32(loop_ub - i, i1);
    b_loop_ub = (w_size + loop_ub) + 1;
    if (w_size - 1 >= 0) {
      std::copy(&w_data[0], &w_data[w_size], &b_w_data[0]);
    }
    for (int i2{0}; i2 <= loop_ub; i2++) {
      b_w_data[i2 + w_size] = w_data[i + i1 * i2];
    }
    w_size = b_loop_ub;
    if (b_loop_ub - 1 >= 0) {
      std::copy(&b_w_data[0], &b_w_data[b_loop_ub], &w_data[0]);
    }
  } else {
    int b_loop_ub;
    int i;
    int i1;
    int loop_ub;
    w_size = calc_window(((L + 1.0) + 1.0) / 2.0, L + 1.0, w_data);
    if (w_size - 1 < 2) {
      i = 0;
      i1 = 1;
      loop_ub = -1;
    } else {
      i = w_size - 2;
      i1 = -1;
      loop_ub = 1;
    }
    loop_ub = div_s32(loop_ub - i, i1);
    b_loop_ub = (w_size + loop_ub) + 1;
    if (w_size - 1 >= 0) {
      std::copy(&w_data[0], &w_data[w_size], &b_w_data[0]);
    }
    for (int i2{0}; i2 <= loop_ub; i2++) {
      b_w_data[i2 + w_size] = w_data[i + i1 * i2];
    }
    w_size = b_loop_ub;
    if (b_loop_ub - 1 >= 0) {
      std::copy(&b_w_data[0], &b_w_data[b_loop_ub], &w_data[0]);
    }
  }
  return w_size;
}

} // namespace coder

// End of code generation (hamming.cpp)
