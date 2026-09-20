//
// blockedSummation.cpp
//
// Code generation for function 'blockedSummation'
//

// Include files
#include "blockedSummation.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include <cstring>

// Function Definitions
namespace coder {
creal32_T blockedSummation(const array<creal32_T, 1U> &x, int vlen)
{
  creal32_T y;
  if ((x.size(0) == 0) || (vlen == 0)) {
    y.re = 0.0F;
    y.im = 0.0F;
  } else {
    int firstBlockLength;
    int lastBlockLength;
    int nblocks;
    if (vlen <= 1024) {
      firstBlockLength = vlen;
      lastBlockLength = 0;
      nblocks = 1;
    } else {
      firstBlockLength = 1024;
      nblocks = static_cast<int>(static_cast<unsigned int>(vlen) >> 10);
      lastBlockLength = vlen - (nblocks << 10);
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

float blockedSummation(const array<float, 1U> &x, int vlen)
{
  float y;
  if ((x.size(0) == 0) || (vlen == 0)) {
    y = 0.0F;
  } else {
    int firstBlockLength;
    int lastBlockLength;
    int nblocks;
    if (vlen <= 1024) {
      firstBlockLength = vlen;
      lastBlockLength = 0;
      nblocks = 1;
    } else {
      firstBlockLength = 1024;
      nblocks = static_cast<int>(static_cast<unsigned int>(vlen) >> 10);
      lastBlockLength = vlen - (nblocks << 10);
      if (lastBlockLength > 0) {
        nblocks++;
      } else {
        lastBlockLength = 1024;
      }
    }
    y = x[0];
    for (int k{2}; k <= firstBlockLength; k++) {
      y += x[k - 1];
    }
    for (int k{2}; k <= nblocks; k++) {
      float bsum;
      int hi;
      firstBlockLength = (k - 1) << 10;
      bsum = x[firstBlockLength];
      if (k == nblocks) {
        hi = lastBlockLength;
      } else {
        hi = 1024;
      }
      for (int b_k{2}; b_k <= hi; b_k++) {
        bsum += x[(firstBlockLength + b_k) - 1];
      }
      y += bsum;
    }
  }
  return y;
}

} // namespace coder

// End of code generation (blockedSummation.cpp)
