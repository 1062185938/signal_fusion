//
// ifft.cpp
//
// Code generation for function 'ifft'
//

// Include files
#include "ifft.h"
#include "FFTImplementationCallback.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include "omp.h"
#include <algorithm>
#include <cmath>
#include <cstring>

// Function Definitions
namespace coder {
void ifft(const array<creal_T, 2U> &x, array<creal_T, 2U> &y)
{
  array<creal_T, 2U> xPerm;
  array<creal_T, 2U> yPerm;
  array<double, 2U> costab;
  array<double, 2U> costab1q;
  array<double, 2U> sintab;
  array<double, 2U> sintabinv;
  int nd2;
  if ((x.size(0) == 0) || (x.size(1) == 0)) {
    y.set_size(x.size(0), x.size(1));
    nd2 = x.size(0) * x.size(1);
    for (int k{0}; k < nd2; k++) {
      y[k].re = 0.0;
      y[k].im = 0.0;
    }
  } else {
    double e;
    int N2blue;
    int b_n;
    int n;
    boolean_T useRadix2;
    nd2 = x.size(1);
    n = x.size(0);
    xPerm.set_size(x.size(1), x.size(0));
    for (int k{0}; k < n; k++) {
      for (int i{0}; i < nd2; i++) {
        xPerm[i + xPerm.size(0) * k] = x[k + x.size(0) * i];
      }
    }
    useRadix2 =
        (static_cast<int>(static_cast<unsigned int>(x.size(1)) &
                          static_cast<unsigned int>(x.size(1) - 1)) == 0);
    N2blue = internal::fft::FFTImplementationCallback::get_algo_sizes(
        x.size(1), useRadix2, nd2);
    e = 6.2831853071795862 / static_cast<double>(nd2);
    b_n = static_cast<int>((static_cast<unsigned int>(nd2) >> 1) >> 1);
    costab1q.set_size(1, b_n + 1);
    costab1q[0] = 1.0;
    nd2 = static_cast<int>(static_cast<unsigned int>(b_n) >> 1) - 1;
    if (static_cast<int>(nd2 + 1 < 800)) {
      for (int b_k{0}; b_k <= nd2; b_k++) {
        costab1q[b_k + 1] = std::cos(e * (static_cast<double>(b_k) + 1.0));
      }
    } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

      for (int b_k = 0; b_k <= nd2; b_k++) {
        costab1q[b_k + 1] = std::cos(e * (static_cast<double>(b_k) + 1.0));
      }
    }
    n = nd2 + 2;
    if (static_cast<int>((b_n - nd2) - 2 < 800)) {
      for (int c_k{n}; c_k < b_n; c_k++) {
        costab1q[c_k] = std::sin(e * static_cast<double>(b_n - c_k));
      }
    } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

      for (int c_k = n; c_k < b_n; c_k++) {
        costab1q[c_k] = std::sin(e * static_cast<double>(b_n - c_k));
      }
    }
    costab1q[b_n] = 0.0;
    if (!useRadix2) {
      int n2;
      b_n = costab1q.size(1) - 1;
      n2 = (costab1q.size(1) - 1) << 1;
      costab.set_size(1, n2 + 1);
      sintab.set_size(1, n2 + 1);
      costab[0] = 1.0;
      sintab[0] = 0.0;
      sintabinv.set_size(1, n2 + 1);
      nd2 = (costab1q.size(1) - 1 < 800);
      if (nd2) {
        for (int e_k{0}; e_k < b_n; e_k++) {
          sintabinv[e_k + 1] = costab1q[(b_n - e_k) - 1];
        }
      } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

        for (int e_k = 0; e_k < b_n; e_k++) {
          sintabinv[e_k + 1] = costab1q[(b_n - e_k) - 1];
        }
      }
      n = costab1q.size(1);
      for (int k{n}; k <= n2; k++) {
        sintabinv[k] = costab1q[k - b_n];
      }
      if (nd2) {
        for (int g_k{0}; g_k < b_n; g_k++) {
          costab[g_k + 1] = costab1q[g_k + 1];
          sintab[g_k + 1] = -costab1q[(b_n - g_k) - 1];
        }
      } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

        for (int g_k = 0; g_k < b_n; g_k++) {
          costab[g_k + 1] = costab1q[g_k + 1];
          sintab[g_k + 1] = -costab1q[(b_n - g_k) - 1];
        }
      }
      if (static_cast<int>((n2 - costab1q.size(1)) + 1 < 800)) {
        for (int h_k{n}; h_k <= n2; h_k++) {
          costab[h_k] = -costab1q[n2 - h_k];
          sintab[h_k] = -costab1q[h_k - b_n];
        }
      } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

        for (int h_k = n; h_k <= n2; h_k++) {
          costab[h_k] = -costab1q[n2 - h_k];
          sintab[h_k] = -costab1q[h_k - b_n];
        }
      }
      internal::fft::FFTImplementationCallback::dobluesteinfft(
          xPerm, N2blue, x.size(1), costab, sintab, sintabinv, yPerm);
    } else {
      n = costab1q.size(1) - 1;
      b_n = (costab1q.size(1) - 1) << 1;
      costab.set_size(1, b_n + 1);
      sintab.set_size(1, b_n + 1);
      costab[0] = 1.0;
      sintab[0] = 0.0;
      if (static_cast<int>(costab1q.size(1) - 1 < 800)) {
        if (n - 1 >= 0) {
          std::copy(&costab1q[1], &costab1q[1 + n], &costab[1]);
        }
        for (int d_k{0}; d_k < n; d_k++) {
          sintab[d_k + 1] = costab1q[(n - d_k) - 1];
        }
      } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

        for (int d_k = 0; d_k < n; d_k++) {
          costab[d_k + 1] = costab1q[d_k + 1];
          sintab[d_k + 1] = costab1q[(n - d_k) - 1];
        }
      }
      nd2 = costab1q.size(1);
      if (static_cast<int>((b_n - costab1q.size(1)) + 1 < 800)) {
        for (int f_k{nd2}; f_k <= b_n; f_k++) {
          costab[f_k] = -costab1q[b_n - f_k];
          sintab[f_k] = costab1q[f_k - n];
        }
      } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

        for (int f_k = nd2; f_k <= b_n; f_k++) {
          costab[f_k] = -costab1q[b_n - f_k];
          sintab[f_k] = costab1q[f_k - n];
        }
      }
      internal::fft::FFTImplementationCallback::r2br_r2dit_trig(
          xPerm, x.size(1), costab, sintab, yPerm);
    }
    nd2 = yPerm.size(1);
    n = yPerm.size(0);
    y.set_size(yPerm.size(1), yPerm.size(0));
    for (int k{0}; k < n; k++) {
      for (int i{0}; i < nd2; i++) {
        y[i + y.size(0) * k] = yPerm[k + yPerm.size(0) * i];
      }
    }
  }
}

} // namespace coder

// End of code generation (ifft.cpp)
