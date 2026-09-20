//
// log21.cpp
//
// Code generation for function 'log21'
//

// Include files
#include "log21.h"
#include "rt_nonfinite.h"
#include <cmath>
#include <cstring>

// Function Definitions
namespace coder {
namespace internal {
namespace scalar {
double scalar_real_log2(double x)
{
  double y;
  int eint;
  if (x == 0.0) {
    y = rtMinusInf;
  } else if (x < 0.0) {
    y = rtNaN;
  } else if ((!std::isinf(x)) && (!std::isnan(x))) {
    y = std::frexp(x, &eint);
    if (y == 0.5) {
      y = static_cast<double>(eint) - 1.0;
    } else if ((eint == 1) && (y < 0.75)) {
      y = std::log(2.0 * y) / 0.69314718055994529;
    } else {
      y = std::log(y) / 0.69314718055994529 + static_cast<double>(eint);
    }
  } else {
    y = x;
  }
  return y;
}

} // namespace scalar
} // namespace internal
} // namespace coder

// End of code generation (log21.cpp)
