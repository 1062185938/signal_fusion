//
// combineVectorElements.cpp
//
// Code generation for function 'combineVectorElements'
//

// Include files
#include "combineVectorElements.h"
#include "rt_nonfinite.h"
#include <cstring>

// Function Definitions
namespace coder {
float b_combineVectorElements(const float x[4096])
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

int combineVectorElements(const boolean_T x_data[], int x_size)
{
  int y;
  y = x_data[0];
  for (int k{2}; k <= x_size; k++) {
    y += x_data[k - 1];
  }
  return y;
}

} // namespace coder

// End of code generation (combineVectorElements.cpp)
