//
// rms.cpp
//
// Code generation for function 'rms'
//

// Include files
#include "rms.h"
#include "blockedSummation.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include <cmath>
#include <cstring>
#include <xmmintrin.h>

// Function Definitions
namespace coder {
float rms(const array<creal32_T, 1U> &xIn)
{
  array<float, 1U> r;
  array<float, 1U> r1;
  array<float, 1U> x;
  int loop_ub;
  int scalarLB;
  int vectorUB;
  loop_ub = xIn.size(0);
  r.set_size(xIn.size(0));
  r1.set_size(xIn.size(0));
  for (int i{0}; i < loop_ub; i++) {
    r[i] = xIn[i].re;
    r1[i] = xIn[i].im;
  }
  x.set_size(xIn.size(0));
  scalarLB = (r.size(0) / 4) << 2;
  vectorUB = scalarLB - 4;
  for (int i{0}; i <= vectorUB; i += 4) {
    __m128 r2;
    __m128 r3;
    r2 = _mm_loadu_ps(&r[i]);
    r3 = _mm_loadu_ps(&r1[i]);
    _mm_storeu_ps(&x[i], _mm_add_ps(_mm_mul_ps(r2, r2), _mm_mul_ps(r3, r3)));
  }
  for (int i{scalarLB}; i < loop_ub; i++) {
    x[i] = r[i] * r[i] + r1[i] * r1[i];
  }
  return std::sqrt(blockedSummation(x, x.size(0)) /
                   static_cast<float>(x.size(0)));
}

} // namespace coder

// End of code generation (rms.cpp)
