//
// computeDFT.cpp
//
// Code generation for function 'computeDFT'
//

// Include files
#include "computeDFT.h"
#include "FFTImplementationCallback.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include "omp.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <emmintrin.h>

// Function Definitions
namespace coder {
int computeDFTviaFFT(const array<creal32_T, 2U> &xin, double nx, double nfft,
                     double Fs, array<creal32_T, 2U> &Xx, double f_data[])
{
  array<creal32_T, 2U> wrappedData;
  array<creal32_T, 2U> xw;
  array<double, 2U> y;
  double w1_data[258];
  double Fs1;
  double Nyq;
  double freq_res;
  double halfNPTS;
  double half_res;
  float sintabinv_data[517];
  float costab1q_data[259];
  int b_loop_ub;
  int b_remainder;
  int costab1q_size_idx_1;
  int f_size;
  int i;
  int i1;
  int loop_ub;
  int n2;
  int nd2;
  int remainder_tmp;
  boolean_T useRadix2;
  if (nx > nfft) {
    loop_ub = static_cast<int>(nfft);
    costab1q_size_idx_1 = xin.size(1);
    xw.set_size(static_cast<int>(nfft), xin.size(1));
    f_size = static_cast<int>(nfft) * xin.size(1);
    if (f_size - 1 >= 0) {
      std::memset(&xw[0], 0,
                  static_cast<unsigned int>(f_size) * sizeof(creal32_T));
    }
    if (costab1q_size_idx_1 - 1 >= 0) {
      b_loop_ub = xin.size(0);
      if (static_cast<unsigned short>(static_cast<int>(nfft)) == 0) {
        i = MAX_int32_T;
      } else {
        i = static_cast<int>(
            static_cast<unsigned int>(xin.size(0)) /
            static_cast<unsigned short>(static_cast<int>(nfft)));
      }
      remainder_tmp = i * static_cast<int>(nfft);
      b_remainder = (xin.size(0) - remainder_tmp) - 1;
      i1 = b_remainder + 2;
    }
    for (int j{0}; j < costab1q_size_idx_1; j++) {
      creal32_T x_data[258];
      for (int k{0}; k < b_loop_ub; k++) {
        x_data[k] = xin[k + xin.size(0) * j];
      }
      if (xin.size(0) == 1) {
        wrappedData.set_size(1, static_cast<int>(nfft));
        if (loop_ub - 1 >= 0) {
          std::memset(&wrappedData[0], 0,
                      static_cast<unsigned int>(loop_ub) * sizeof(creal32_T));
        }
      } else {
        wrappedData.set_size(static_cast<int>(nfft), 1);
        if (loop_ub - 1 >= 0) {
          std::memset(&wrappedData[0], 0,
                      static_cast<unsigned int>(loop_ub) * sizeof(creal32_T));
        }
      }
      for (int k{0}; k <= b_remainder; k++) {
        wrappedData[k] = x_data[remainder_tmp + k];
      }
      if (i1 <= loop_ub) {
        std::memset(&wrappedData[i1 + -1], 0,
                    static_cast<unsigned int>((loop_ub - i1) + 1) *
                        sizeof(creal32_T));
      }
      for (int k{0}; k < i; k++) {
        f_size = k * static_cast<int>(nfft);
        nd2 = static_cast<unsigned short>(static_cast<int>(nfft));
        for (int c_k{0}; c_k < nd2; c_k++) {
          n2 = f_size + c_k;
          wrappedData[c_k].re = wrappedData[c_k].re + x_data[n2].re;
          wrappedData[c_k].im = wrappedData[c_k].im + x_data[n2].im;
        }
      }
      for (int k{0}; k < loop_ub; k++) {
        xw[k + xw.size(0) * j] = wrappedData[k];
      }
    }
  } else {
    xw.set_size(xin.size(0), xin.size(1));
    f_size = xin.size(0) * xin.size(1);
    for (int k{0}; k < f_size; k++) {
      xw[k] = xin[k];
    }
  }
  if ((xw.size(0) == 0) || (xw.size(1) == 0) || (static_cast<int>(nfft) == 0)) {
    Xx.set_size(static_cast<int>(nfft), xw.size(1));
    f_size = static_cast<int>(nfft) * xw.size(1);
    for (int k{0}; k < f_size; k++) {
      Xx[k].re = 0.0F;
      Xx[k].im = 0.0F;
    }
  } else {
    float e;
    useRadix2 =
        ((static_cast<int>(nfft) > 0) &&
         ((static_cast<int>(nfft) & (static_cast<int>(nfft) - 1)) == 0));
    loop_ub = internal::fft::FFTImplementationCallback::get_algo_sizes(
        static_cast<int>(nfft), useRadix2, f_size);
    e = 6.28318548F / static_cast<float>(f_size);
    n2 = static_cast<int>((static_cast<unsigned int>(f_size) >> 1) >> 1);
    costab1q_size_idx_1 = n2 + 1;
    costab1q_data[0] = 1.0F;
    nd2 = static_cast<int>(static_cast<unsigned int>(n2) >> 1);
    f_size = static_cast<unsigned char>(nd2);
    for (int k{0}; k < f_size; k++) {
      costab1q_data[k + 1] = std::cos(e * (static_cast<float>(k) + 1.0F));
    }
    f_size = nd2 + 1;
    if (static_cast<int>((n2 - nd2) - 1 < 800)) {
      for (int b_k{f_size}; b_k < n2; b_k++) {
        costab1q_data[b_k] = std::sin(e * static_cast<float>(n2 - b_k));
      }
    } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

      for (int b_k = f_size; b_k < n2; b_k++) {
        costab1q_data[b_k] = std::sin(e * static_cast<float>(n2 - b_k));
      }
    }
    costab1q_data[n2] = 0.0F;
    if (!useRadix2) {
      float costab_data[517];
      float sintab_data[517];
      f_size = costab1q_size_idx_1 - 1;
      n2 = (costab1q_size_idx_1 - 1) << 1;
      costab_data[0] = 1.0F;
      sintab_data[0] = 0.0F;
      for (int k{0}; k < f_size; k++) {
        sintabinv_data[k + 1] = costab1q_data[(costab1q_size_idx_1 - k) - 2];
      }
      if (costab1q_size_idx_1 <= n2) {
        std::copy(&costab1q_data[1],
                  &costab1q_data[(n2 - costab1q_size_idx_1) + 2],
                  &sintabinv_data[costab1q_size_idx_1]);
      }
      for (int k{0}; k < f_size; k++) {
        costab_data[k + 1] = costab1q_data[k + 1];
        sintab_data[k + 1] = -costab1q_data[(costab1q_size_idx_1 - k) - 2];
      }
      for (int k{costab1q_size_idx_1}; k <= n2; k++) {
        costab_data[k] = -costab1q_data[n2 - k];
        sintab_data[k] = -costab1q_data[(k - costab1q_size_idx_1) + 1];
      }
      internal::fft::FFTImplementationCallback::dobluesteinfft(
          xw, loop_ub, static_cast<int>(nfft), costab_data, sintab_data,
          sintabinv_data, Xx);
    } else {
      float costab_data[517];
      float sintab_data[517];
      f_size = costab1q_size_idx_1 - 1;
      nd2 = (costab1q_size_idx_1 - 1) << 1;
      costab_data[0] = 1.0F;
      sintab_data[0] = 0.0F;
      for (int k{0}; k < f_size; k++) {
        costab_data[k + 1] = costab1q_data[k + 1];
        sintab_data[k + 1] = -costab1q_data[(costab1q_size_idx_1 - k) - 2];
      }
      for (int k{costab1q_size_idx_1}; k <= nd2; k++) {
        costab_data[k] = -costab1q_data[nd2 - k];
        sintab_data[k] = -costab1q_data[(k - costab1q_size_idx_1) + 1];
      }
      internal::fft::FFTImplementationCallback::r2br_r2dit_trig(
          xw, static_cast<int>(nfft), costab_data, sintab_data, Xx);
    }
  }
  if (std::isnan(Fs)) {
    Fs1 = 6.2831853071795862;
  } else {
    Fs1 = Fs;
  }
  freq_res = Fs1 / nfft;
  if (std::isnan(nfft - 1.0)) {
    y.set_size(1, 1);
    y[0] = rtNaN;
  } else if (nfft - 1.0 < 0.0) {
    y.set_size(1, 0);
  } else {
    y.set_size(1, static_cast<int>(nfft - 1.0) + 1);
    f_size = static_cast<int>(nfft - 1.0);
    for (int k{0}; k <= f_size; k++) {
      y[k] = k;
    }
  }
  f_size = y.size(1);
  nd2 = (y.size(1) / 2) << 1;
  n2 = nd2 - 2;
  for (int k{0}; k <= n2; k += 2) {
    __m128d r;
    r = _mm_loadu_pd(&y[k]);
    _mm_storeu_pd(&w1_data[k], _mm_mul_pd(_mm_set1_pd(freq_res), r));
  }
  for (int k{nd2}; k < f_size; k++) {
    w1_data[k] = freq_res * y[k];
  }
  Nyq = Fs1 / 2.0;
  half_res = freq_res / 2.0;
  if (std::isnan(nfft) || std::isinf(nfft)) {
    halfNPTS = rtNaN;
  } else {
    halfNPTS = std::fmod(nfft, 2.0);
  }
  useRadix2 = (halfNPTS != 0.0);
  if (useRadix2) {
    halfNPTS = (nfft + 1.0) / 2.0;
    w1_data[static_cast<int>(halfNPTS) - 1] = Nyq - half_res;
    w1_data[static_cast<int>(static_cast<unsigned int>(halfNPTS))] =
        Nyq + half_res;
  } else {
    halfNPTS = nfft / 2.0 + 1.0;
    w1_data[static_cast<int>(halfNPTS) - 1] = Nyq;
  }
  w1_data[static_cast<int>(nfft) - 1] = Fs1 - freq_res;
  if (f_size - 1 >= 0) {
    std::copy(&w1_data[0], &w1_data[f_size], &f_data[0]);
  }
  return f_size;
}

} // namespace coder

// End of code generation (computeDFT.cpp)
