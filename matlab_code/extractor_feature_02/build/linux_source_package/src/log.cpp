//
// log.cpp
//
// Code generation for function 'log'
//

// Include files
#include "log.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include "omp.h"
#include "rt_defines.h"
#include <cmath>
#include <cstring>

// Function Definitions
namespace coder {
void b_log(creal32_T &x)
{
  if (x.im == 0.0F) {
    if (x.re < 0.0F) {
      x.re = std::log(std::abs(x.re));
      x.im = 3.14159274F;
    } else {
      x.re = std::log(std::abs(x.re));
      x.im = 0.0F;
    }
  } else {
    float a;
    float b_a;
    int i;
    int i1;
    boolean_T guard1;
    a = std::abs(x.re);
    guard1 = false;
    if (a > 1.70141173E+38F) {
      guard1 = true;
    } else {
      float b;
      b = std::abs(x.im);
      if (b > 1.70141173E+38F) {
        guard1 = true;
      } else {
        if (a < b) {
          b_a = a / b;
          b_a = b * std::sqrt(b_a * b_a + 1.0F);
        } else if (a > b) {
          b_a = b / a;
          b_a = a * std::sqrt(b_a * b_a + 1.0F);
        } else if (std::isnan(b)) {
          b_a = rtNaNF;
        } else {
          b_a = a * 1.41421354F;
        }
        a = x.re;
        x.re = std::log(b_a);
        b_a = x.im;
        if (std::isnan(b_a) || std::isnan(a)) {
          b_a = rtNaNF;
        } else if (std::isinf(b_a) && std::isinf(a)) {
          if (b_a > 0.0F) {
            i = 1;
          } else {
            i = -1;
          }
          if (a > 0.0F) {
            i1 = 1;
          } else {
            i1 = -1;
          }
          b_a = std::atan2(static_cast<float>(i), static_cast<float>(i1));
        } else if (a == 0.0F) {
          if (b_a > 0.0F) {
            b_a = RT_PIF / 2.0F;
          } else if (b_a < 0.0F) {
            b_a = -(RT_PIF / 2.0F);
          } else {
            b_a = 0.0F;
          }
        } else {
          b_a = std::atan2(b_a, a);
        }
        x.im = b_a;
      }
    }
    if (guard1) {
      b_a = std::abs(x.re / 2.0F);
      a = std::abs(x.im / 2.0F);
      if (b_a < a) {
        b_a /= a;
        b_a = a * std::sqrt(b_a * b_a + 1.0F);
      } else if (b_a > a) {
        a /= b_a;
        b_a *= std::sqrt(a * a + 1.0F);
      } else if (std::isnan(a)) {
        b_a = rtNaNF;
      } else {
        b_a *= 1.41421354F;
      }
      a = x.re;
      x.re = std::log(b_a) + 0.693147182F;
      b_a = x.im;
      if (std::isnan(b_a) || std::isnan(a)) {
        b_a = rtNaNF;
      } else if (std::isinf(b_a) && std::isinf(a)) {
        if (b_a > 0.0F) {
          i = 1;
        } else {
          i = -1;
        }
        if (a > 0.0F) {
          i1 = 1;
        } else {
          i1 = -1;
        }
        b_a = std::atan2(static_cast<float>(i), static_cast<float>(i1));
      } else if (a == 0.0F) {
        if (b_a > 0.0F) {
          b_a = RT_PIF / 2.0F;
        } else if (b_a < 0.0F) {
          b_a = -(RT_PIF / 2.0F);
        } else {
          b_a = 0.0F;
        }
      } else {
        b_a = std::atan2(b_a, a);
      }
      x.im = b_a;
    }
  }
}

void b_log(creal_T &x)
{
  if (x.im == 0.0) {
    if (x.re < 0.0) {
      x.re = std::log(std::abs(x.re));
      x.im = 3.1415926535897931;
    } else {
      x.re = std::log(std::abs(x.re));
      x.im = 0.0;
    }
  } else {
    double a;
    double b_a;
    int i;
    int i1;
    boolean_T guard1;
    a = std::abs(x.re);
    guard1 = false;
    if (a > 8.9884656743115785E+307) {
      guard1 = true;
    } else {
      double b;
      b = std::abs(x.im);
      if (b > 8.9884656743115785E+307) {
        guard1 = true;
      } else {
        if (a < b) {
          b_a = a / b;
          b_a = b * std::sqrt(b_a * b_a + 1.0);
        } else if (a > b) {
          b_a = b / a;
          b_a = a * std::sqrt(b_a * b_a + 1.0);
        } else if (std::isnan(b)) {
          b_a = rtNaN;
        } else {
          b_a = a * 1.4142135623730951;
        }
        a = x.re;
        x.re = std::log(b_a);
        b_a = x.im;
        if (std::isnan(b_a) || std::isnan(a)) {
          b_a = rtNaN;
        } else if (std::isinf(b_a) && std::isinf(a)) {
          if (b_a > 0.0) {
            i = 1;
          } else {
            i = -1;
          }
          if (a > 0.0) {
            i1 = 1;
          } else {
            i1 = -1;
          }
          b_a = std::atan2(static_cast<double>(i), static_cast<double>(i1));
        } else if (a == 0.0) {
          if (b_a > 0.0) {
            b_a = RT_PI / 2.0;
          } else if (b_a < 0.0) {
            b_a = -(RT_PI / 2.0);
          } else {
            b_a = 0.0;
          }
        } else {
          b_a = std::atan2(b_a, a);
        }
        x.im = b_a;
      }
    }
    if (guard1) {
      b_a = std::abs(x.re / 2.0);
      a = std::abs(x.im / 2.0);
      if (b_a < a) {
        b_a /= a;
        b_a = a * std::sqrt(b_a * b_a + 1.0);
      } else if (b_a > a) {
        a /= b_a;
        b_a *= std::sqrt(a * a + 1.0);
      } else if (std::isnan(a)) {
        b_a = rtNaN;
      } else {
        b_a *= 1.4142135623730951;
      }
      a = x.re;
      x.re = std::log(b_a) + 0.69314718055994529;
      b_a = x.im;
      if (std::isnan(b_a) || std::isnan(a)) {
        b_a = rtNaN;
      } else if (std::isinf(b_a) && std::isinf(a)) {
        if (b_a > 0.0) {
          i = 1;
        } else {
          i = -1;
        }
        if (a > 0.0) {
          i1 = 1;
        } else {
          i1 = -1;
        }
        b_a = std::atan2(static_cast<double>(i), static_cast<double>(i1));
      } else if (a == 0.0) {
        if (b_a > 0.0) {
          b_a = RT_PI / 2.0;
        } else if (b_a < 0.0) {
          b_a = -(RT_PI / 2.0);
        } else {
          b_a = 0.0;
        }
      } else {
        b_a = std::atan2(b_a, a);
      }
      x.im = b_a;
    }
  }
}

void b_log(array<float, 1U> &x)
{
  int nx;
  nx = x.size(0);
  if (static_cast<int>(nx < 800)) {
    for (int k{0}; k < nx; k++) {
      x[k] = std::log(x[k]);
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

    for (int k = 0; k < nx; k++) {
      x[k] = std::log(x[k]);
    }
  }
}

} // namespace coder

// End of code generation (log.cpp)
