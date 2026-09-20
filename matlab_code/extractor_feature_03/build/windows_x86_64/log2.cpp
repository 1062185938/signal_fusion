//
// log2.cpp
//
// Code generation for function 'log2'
//

// Include files
#include "log2.h"
#include "rt_nonfinite.h"
#include <cmath>
#include <cstring>

// Function Definitions
namespace coder {
float b_log2(float x)
{
  float f;
  int eint;
  if ((!std::isinf(x)) && (!std::isnan(x))) {
    f = std::frexp(x, &eint);
    if (f == 0.5F) {
      f = static_cast<float>(eint) - 1.0F;
    } else if ((eint == 1) && (f < 0.75F)) {
      f = std::log(2.0F * f) / 0.693147182F;
    } else {
      f = std::log(f) / 0.693147182F + static_cast<float>(eint);
    }
  } else {
    f = x;
  }
  return f;
}

} // namespace coder

// End of code generation (log2.cpp)
