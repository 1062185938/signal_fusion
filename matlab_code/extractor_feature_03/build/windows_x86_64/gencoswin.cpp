//
// gencoswin.cpp
//
// Code generation for function 'gencoswin'
//

// Include files
#include "gencoswin.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include <cmath>
#include <cstring>
#include <emmintrin.h>

// Function Definitions
namespace coder {
int calc_window(double m, double n, double w_data[])
{
  __m128d r;
  array<double, 2U> y;
  int scalarLB;
  int vectorUB;
  int w_size;
  if (std::isnan(m - 1.0)) {
    w_size = 1;
    y.set_size(1, 1);
    y[0] = rtNaN;
  } else if (m - 1.0 < 0.0) {
    w_size = 0;
    y.set_size(1, 0);
  } else {
    w_size = static_cast<int>(m - 1.0) + 1;
    y.set_size(1, static_cast<int>(m - 1.0) + 1);
    vectorUB = static_cast<int>(m - 1.0);
    for (int k{0}; k <= vectorUB; k++) {
      y[k] = k;
    }
  }
  scalarLB = (w_size / 2) << 1;
  vectorUB = scalarLB - 2;
  for (int k{0}; k <= vectorUB; k += 2) {
    r = _mm_loadu_pd(&y[k]);
    _mm_storeu_pd(&w_data[k], _mm_mul_pd(_mm_set1_pd(6.2831853071795862),
                                         _mm_div_pd(r, _mm_set1_pd(n - 1.0))));
  }
  for (int k{scalarLB}; k < w_size; k++) {
    w_data[k] = 6.2831853071795862 * (y[k] / (n - 1.0));
  }
  for (int k{0}; k < w_size; k++) {
    w_data[k] = std::cos(w_data[k]);
  }
  vectorUB = scalarLB - 2;
  for (int k{0}; k <= vectorUB; k += 2) {
    r = _mm_loadu_pd(&w_data[k]);
    _mm_storeu_pd(&w_data[k], _mm_sub_pd(_mm_set1_pd(0.54),
                                         _mm_mul_pd(_mm_set1_pd(0.46), r)));
  }
  for (int k{scalarLB}; k < w_size; k++) {
    w_data[k] = 0.54 - 0.46 * w_data[k];
  }
  return w_size;
}

} // namespace coder

// End of code generation (gencoswin.cpp)
