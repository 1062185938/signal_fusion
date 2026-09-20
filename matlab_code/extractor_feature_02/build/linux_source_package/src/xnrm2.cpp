//
// xnrm2.cpp
//
// Code generation for function 'xnrm2'
//

// Include files
#include "xnrm2.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include <cmath>
#include <cstring>

// Function Definitions
namespace coder {
namespace internal {
namespace blas {
float xnrm2(int n, const array<float, 1U> &x)
{
  float scale;
  float y;
  int i;
  y = 0.0F;
  scale = 1.29246971E-26F;
  i = static_cast<unsigned short>(n);
  for (int k{0}; k < i; k++) {
    float absxk;
    absxk = std::abs(x[k]);
    if (absxk > scale) {
      float t;
      t = scale / absxk;
      y = y * t * t + 1.0F;
      scale = absxk;
    } else {
      float t;
      t = absxk / scale;
      y += t * t;
    }
  }
  return scale * std::sqrt(y);
}

} // namespace blas
} // namespace internal
} // namespace coder

// End of code generation (xnrm2.cpp)
