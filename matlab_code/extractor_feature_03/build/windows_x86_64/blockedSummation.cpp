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
