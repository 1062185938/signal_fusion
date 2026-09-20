//
// std.cpp
//
// Code generation for function 'std'
//

// Include files
#include "std.h"
#include "blockedSummation.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include <cmath>
#include <cstring>

// Function Definitions
namespace coder {
float b_std(const array<float, 1U> &x)
{
  float y;
  int n;
  n = x.size(0);
  if (x.size(0) == 0) {
    y = rtNaNF;
  } else if (x.size(0) == 1) {
    if ((!std::isinf(x[0])) && (!std::isnan(x[0]))) {
      y = 0.0F;
    } else {
      y = rtNaNF;
    }
  } else {
    float scale;
    float xbar;
    xbar = blockedSummation(x, x.size(0)) / static_cast<float>(x.size(0));
    y = 0.0F;
    scale = 1.29246971E-26F;
    for (int k{0}; k < n; k++) {
      float f;
      f = std::abs(x[k] - xbar);
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
    y = scale * std::sqrt(y) / std::sqrt(static_cast<float>(x.size(0)) - 1.0F);
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
