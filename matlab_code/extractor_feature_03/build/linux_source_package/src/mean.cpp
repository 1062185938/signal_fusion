//
// mean.cpp
//
// Code generation for function 'mean'
//

// Include files
#include "mean.h"
#include "rt_nonfinite.h"
#include <cstring>

// Function Definitions
namespace coder {
float mean(const float x[512])
{
  float y;
  y = x[0];
  for (int k{0}; k < 511; k++) {
    y += x[k + 1];
  }
  y /= 512.0F;
  return y;
}

} // namespace coder

// End of code generation (mean.cpp)
