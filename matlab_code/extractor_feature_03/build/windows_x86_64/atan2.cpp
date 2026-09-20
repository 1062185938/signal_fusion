//
// atan2.cpp
//
// Code generation for function 'atan2'
//

// Include files
#include "atan2.h"
#include "rt_nonfinite.h"
#include "rt_defines.h"
#include <cmath>
#include <cstring>

// Function Definitions
namespace coder {
namespace internal {
namespace scalar {
double b_atan2(double y, double x)
{
  double r;
  if (std::isnan(y) || std::isnan(x)) {
    r = rtNaN;
  } else if (std::isinf(y) && std::isinf(x)) {
    int i;
    int i1;
    if (y > 0.0) {
      i = 1;
    } else {
      i = -1;
    }
    if (x > 0.0) {
      i1 = 1;
    } else {
      i1 = -1;
    }
    r = std::atan2(static_cast<double>(i), static_cast<double>(i1));
  } else if (x == 0.0) {
    if (y > 0.0) {
      r = RT_PI / 2.0;
    } else if (y < 0.0) {
      r = -(RT_PI / 2.0);
    } else {
      r = 0.0;
    }
  } else {
    r = std::atan2(y, x);
  }
  return r;
}

} // namespace scalar
} // namespace internal
} // namespace coder

// End of code generation (atan2.cpp)
