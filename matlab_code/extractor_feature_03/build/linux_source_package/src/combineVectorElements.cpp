//
// combineVectorElements.cpp
//
// Code generation for function 'combineVectorElements'
//

// Include files
#include "combineVectorElements.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include <cstring>

// Function Definitions
namespace coder {
int b_combineVectorElements(const boolean_T x_data[], int x_size)
{
  int y;
  y = x_data[0];
  for (int k{2}; k <= x_size; k++) {
    y += x_data[k - 1];
  }
  return y;
}

float combineVectorElements(const float x[4096])
{
  float y;
  y = x[0];
  for (int k{0}; k < 1023; k++) {
    y += x[k + 1];
  }
  for (int k{0}; k < 3; k++) {
    float bsum;
    int xblockoffset;
    xblockoffset = (k + 1) << 10;
    bsum = x[xblockoffset];
    for (int b_k{0}; b_k < 1023; b_k++) {
      bsum += x[(xblockoffset + b_k) + 1];
    }
    y += bsum;
  }
  return y;
}

creal32_T combineVectorElements(const array<creal32_T, 1U> &x)
{
  creal32_T y;
  if (x.size(0) == 0) {
    y.re = 0.0F;
    y.im = 0.0F;
  } else {
    int firstBlockLength;
    int lastBlockLength;
    int nblocks;
    if (x.size(0) <= 1024) {
      firstBlockLength = x.size(0);
      lastBlockLength = 0;
      nblocks = 1;
    } else {
      firstBlockLength = 1024;
      nblocks = static_cast<unsigned short>(x.size(0)) >> 10;
      lastBlockLength = x.size(0) - (nblocks << 10);
      if (lastBlockLength > 0) {
        nblocks++;
      } else {
        lastBlockLength = 1024;
      }
    }
    y = x[0];
    for (int k{2}; k <= firstBlockLength; k++) {
      y.re += x[k - 1].re;
      y.im += x[k - 1].im;
    }
    for (int k{2}; k <= nblocks; k++) {
      float bsum_im;
      float bsum_re;
      int hi;
      firstBlockLength = (k - 1) << 10;
      bsum_re = x[firstBlockLength].re;
      bsum_im = x[firstBlockLength].im;
      if (k == nblocks) {
        hi = lastBlockLength;
      } else {
        hi = 1024;
      }
      for (int b_k{2}; b_k <= hi; b_k++) {
        int bsum_re_tmp;
        bsum_re_tmp = (firstBlockLength + b_k) - 1;
        bsum_re += x[bsum_re_tmp].re;
        bsum_im += x[bsum_re_tmp].im;
      }
      y.re += bsum_re;
      y.im += bsum_im;
    }
  }
  return y;
}

} // namespace coder

// End of code generation (combineVectorElements.cpp)
