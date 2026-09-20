//
// cwtfilterbank.cpp
//
// Code generation for function 'cwtfilterbank'
//

// Include files
#include "cwtfilterbank.h"
#include "cwtfreqlimits.h"
#include "extractAllFeatures_data.h"
#include "extractAllFeatures_rtwutil.h"
#include "log21.h"
#include "rt_nonfinite.h"
#include "wavCFandSD.h"
#include "wavbpfilters.h"
#include "coder_array.h"
#include "omp.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <emmintrin.h>

// Function Definitions
namespace coder {
cwtfilterbank *cwtfilterbank::setProperties(double varargin_2,
                                            const double varargin_8[2],
                                            double varargin_12)
{
  static const char b_cv[8]{'p', 'e', 'r', 'i', 'o', 'd', 'i', 'c'};
  static const char a[4]{'b', 'u', 'm', 'p'};
  static const char b_a[4]{'a', 'm', 'o', 'r'};
  cwtfilterbank *self;
  array<char, 2U> b_str;
  array<char, 2U> str;
  double NyquistRange_idx_1;
  int nbytes;
  boolean_T b[2];
  boolean_T exitg1;
  boolean_T freqsep;
  boolean_T guard1;
  self = this;
  self->CutOff = 50.0;
  self->Gamma = 3.0;
  self->Beta = 20.0;
  self->Wavelet[0] = 'a';
  self->Wavelet[1] = 'm';
  self->Wavelet[2] = 'o';
  self->Wavelet[3] = 'r';
  self->TimeBandwidth = rtNaN;
  self->SignalLength = varargin_2;
  NyquistRange_idx_1 = self->SignalLength / 2.0;
  NyquistRange_idx_1 = std::floor(NyquistRange_idx_1);
  self->SignalPad = NyquistRange_idx_1;
  self->VoicesPerOctave = 32.0;
  self->SamplingFrequency = varargin_12;
  self->WaveletParameters[0] = rtNaN;
  self->FrequencyLimits[0] = varargin_8[0];
  self->WaveletParameters[1] = rtNaN;
  self->FrequencyLimits[1] = varargin_8[1];
  for (int i{0}; i < 8; i++) {
    self->Boundary[i] = b_cv[i];
  }
  NyquistRange_idx_1 = self->TimeBandwidth;
  guard1 = false;
  if (!std::isnan(NyquistRange_idx_1)) {
    NyquistRange_idx_1 = self->WaveletParameters[0];
    b[0] = std::isnan(NyquistRange_idx_1);
    NyquistRange_idx_1 = self->WaveletParameters[1];
    b[1] = std::isnan(NyquistRange_idx_1);
    freqsep = true;
    nbytes = 0;
    exitg1 = false;
    while ((!exitg1) && (nbytes < 2)) {
      if (!b[nbytes]) {
        freqsep = false;
        exitg1 = true;
      } else {
        nbytes++;
      }
    }
    if (freqsep) {
      self->Beta = self->TimeBandwidth / self->Gamma;
    } else {
      guard1 = true;
    }
  } else {
    guard1 = true;
  }
  if (guard1) {
    NyquistRange_idx_1 = self->WaveletParameters[0];
    b[0] = std::isnan(NyquistRange_idx_1);
    NyquistRange_idx_1 = self->WaveletParameters[1];
    b[1] = std::isnan(NyquistRange_idx_1);
    freqsep = true;
    nbytes = 0;
    exitg1 = false;
    while ((!exitg1) && (nbytes < 2)) {
      if (!b[nbytes]) {
        freqsep = false;
        exitg1 = true;
      } else {
        nbytes++;
      }
    }
    if (!freqsep) {
      NyquistRange_idx_1 = self->TimeBandwidth;
      if (std::isnan(NyquistRange_idx_1)) {
        self->Gamma = self->WaveletParameters[0];
        self->Beta = self->WaveletParameters[1] / self->Gamma;
      }
    }
  }
  self->SignalPad = 0.0;
  NyquistRange_idx_1 = self->FrequencyLimits[0];
  b[0] = std::isnan(NyquistRange_idx_1);
  NyquistRange_idx_1 = self->FrequencyLimits[1];
  b[1] = std::isnan(NyquistRange_idx_1);
  freqsep = true;
  nbytes = 0;
  exitg1 = false;
  while ((!exitg1) && (nbytes < 2)) {
    if (!b[nbytes]) {
      freqsep = false;
      exitg1 = true;
    } else {
      nbytes++;
    }
  }
  if (!freqsep) {
    double FourierFactor;
    double be;
    double cf;
    double cutoff;
    double freqrange_idx_0;
    double freqrange_idx_1;
    double fs;
    double ga;
    double omegac;
    double sigmat;
    double varargin_1;
    double varargin_3;
    char b_wav[4];
    char wav[4];
    char c;
    freqrange_idx_0 = self->FrequencyLimits[0];
    freqrange_idx_1 = self->FrequencyLimits[1];
    NyquistRange_idx_1 = self->SamplingFrequency / 2.0;
    if ((freqrange_idx_1 <= 0.0) || (freqrange_idx_0 >= NyquistRange_idx_1)) {
      nbytes = (int)std::snprintf(nullptr, 0, "%f", NyquistRange_idx_1) + 1;
      str.set_size(1, nbytes);
      std::snprintf(&str[0], (size_t)nbytes, "%f", NyquistRange_idx_1);
    }
    fs = self->SamplingFrequency;
    ga = self->Gamma;
    be = self->Beta;
    NyquistRange_idx_1 = self->SignalLength;
    varargin_3 = self->VoicesPerOctave;
    cutoff = self->CutOff;
    varargin_1 = self->SamplingFrequency;
    c = self->Wavelet[0];
    wav[0] = cv[static_cast<int>(static_cast<unsigned char>(c) & 127U)];
    c = self->Wavelet[1];
    wav[1] = cv[static_cast<int>(static_cast<unsigned char>(c) & 127U)];
    c = self->Wavelet[2];
    wav[2] = cv[static_cast<int>(static_cast<unsigned char>(c) & 127U)];
    c = self->Wavelet[3];
    wav[3] = cv[static_cast<int>(static_cast<unsigned char>(c) & 127U)];
    omegac = 3.1415926535897931;
    cutoff /= 100.0;
    b_wav[0] = wav[0];
    b_wav[1] = wav[1];
    b_wav[2] = wav[2];
    b_wav[3] = wav[3];
    FourierFactor =
        wavelet::internal::cwt::wavCFandSD(b_wav, ga, be, sigmat, cf);
    sigmat = NyquistRange_idx_1 / (sigmat * 2.0);
    nbytes = std::memcmp(&a[0], &wav[0], 4);
    if (nbytes == 0) {
      nbytes = 1;
    } else {
      nbytes = std::memcmp(&b_a[0], &wav[0], 4);
      if (nbytes == 0) {
        nbytes = 2;
      } else {
        nbytes = -1;
      }
    }
    switch (nbytes) {
    case 0:
      omegac =
          wavelet::internal::cwt::getFreqFromCutoffMorse(cutoff, cf, ga, be);
      break;
    case 1:
      omegac = wavelet::internal::cwt::getFreqFromCutoffBump(cutoff, cf);
      break;
    case 2:
      omegac = wavelet::internal::cwt::getFreqFromCutoffAmor(cutoff, cf);
      break;
    }
    NyquistRange_idx_1 =
        omegac / 3.1415926535897931 * rt_powd_snf(2.0, 1.0 / varargin_3);
    if (sigmat < NyquistRange_idx_1) {
      sigmat = NyquistRange_idx_1;
    }
    NyquistRange_idx_1 = 1.0 / (sigmat * FourierFactor) * varargin_1;
    if (freqrange_idx_0 < NyquistRange_idx_1) {
      self->FrequencyLimits[0] = NyquistRange_idx_1;
      freqrange_idx_0 = self->FrequencyLimits[0];
    }
    NyquistRange_idx_1 = fs / 2.0;
    if (freqrange_idx_1 > NyquistRange_idx_1) {
      self->FrequencyLimits[1] = NyquistRange_idx_1;
      freqrange_idx_1 = self->FrequencyLimits[1];
    }
    freqsep = (internal::scalar::scalar_real_log2(freqrange_idx_1) -
                   internal::scalar::scalar_real_log2(freqrange_idx_0) >=
               1.0 / self->VoicesPerOctave);
    if (!freqsep) {
      NyquistRange_idx_1 = 1.0 / self->VoicesPerOctave;
      nbytes = (int)std::snprintf(nullptr, 0, "%2.2f", NyquistRange_idx_1) + 1;
      b_str.set_size(1, nbytes);
      std::snprintf(&b_str[0], (size_t)nbytes, "%2.2f", NyquistRange_idx_1);
    }
  }
  self->CutOff = 10.0;
  return self;
}

cwtfilterbank *cwtfilterbank::init(double varargin_2,
                                   const double varargin_8[2],
                                   double varargin_12)
{
  static const char a[4]{'b', 'u', 'm', 'p'};
  static const char b_a[4]{'a', 'm', 'o', 'r'};
  __m128d r;
  cwtfilterbank *self;
  array<double, 2U> b_y;
  array<double, 2U> expnt;
  array<double, 2U> f;
  array<double, 2U> omega;
  array<double, 2U> somega;
  array<boolean_T, 2U> r1;
  double b_dv[2];
  double N;
  double b;
  double d;
  int b_loop_ub;
  int bcoef;
  int fc;
  int loop_ub;
  int nx;
  int ret;
  char b_self[4];
  boolean_T b_b[2];
  boolean_T exitg1;
  boolean_T y;
  self = this;
  self = self->setProperties(varargin_2, varargin_8, varargin_12);
  N = self->SignalLength + 2.0 * self->SignalPad;
  b = std::trunc(N / 2.0);
  if (std::isnan(b)) {
    omega.set_size(1, 1);
    omega[0] = rtNaN;
  } else if (b < 1.0) {
    omega.set_size(omega.size(0), 0);
  } else {
    omega.set_size(1, static_cast<int>(b - 1.0) + 1);
    ret = static_cast<int>(b - 1.0);
    nx = ((static_cast<int>(b - 1.0) + 1) / 2) << 1;
    bcoef = nx - 2;
    for (int k{0}; k <= bcoef; k += 2) {
      b_dv[0] = k;
      b_dv[1] = k + 1;
      r = _mm_loadu_pd(&b_dv[0]);
      _mm_storeu_pd(&omega[k], _mm_add_pd(_mm_set1_pd(1.0), r));
    }
    for (int k{nx}; k <= ret; k++) {
      omega[k] = static_cast<double>(k) + 1.0;
    }
  }
  omega.set_size(1, omega.size(1));
  b = 6.2831853071795862 / N;
  ret = omega.size(1) - 1;
  bcoef = (omega.size(1) / 2) << 1;
  nx = bcoef - 2;
  for (int k{0}; k <= nx; k += 2) {
    r = _mm_loadu_pd(&omega[k]);
    _mm_storeu_pd(&omega[k], _mm_mul_pd(r, _mm_set1_pd(b)));
  }
  for (int k{bcoef}; k <= ret; k++) {
    omega[k] = omega[k] * b;
  }
  b = std::trunc((N - 1.0) / 2.0);
  if (b < 1.0) {
    bcoef = 0;
    loop_ub = 1;
    ret = -1;
  } else {
    bcoef = static_cast<int>(b) - 1;
    loop_ub = -1;
    ret = 0;
  }
  nx = div_s32(ret - bcoef, loop_ub);
  fc = (omega.size(1) + nx) + 2;
  f.set_size(1, fc);
  f[0] = 0.0;
  ret = omega.size(1);
  if (ret - 1 >= 0) {
    std::copy(&omega[0], &omega[ret], &f[1]);
  }
  if (static_cast<int>(nx + 1 < 800)) {
    for (int i{0}; i <= nx; i++) {
      f[(i + omega.size(1)) + 1] = -omega[bcoef + loop_ub * i];
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

    for (int i = 0; i <= nx; i++) {
      f[(i + omega.size(1)) + 1] = -omega[bcoef + loop_ub * i];
    }
  }
  omega.set_size(1, fc);
  if (fc - 1 >= 0) {
    std::copy(&f[0], &f[fc], &omega[0]);
  }
  self->Omega.set_size(1, fc);
  for (int k{0}; k < fc; k++) {
    self->Omega[k] = omega[k];
  }
  b = self->FrequencyLimits[0];
  b_b[0] = std::isnan(b);
  b = self->FrequencyLimits[1];
  b_b[1] = std::isnan(b);
  y = true;
  ret = 0;
  exitg1 = false;
  while ((!exitg1) && (ret < 2)) {
    if (!b_b[ret]) {
      y = false;
      exitg1 = true;
    } else {
      ret++;
    }
  }
  if (!y) {
    double b_nv;
    double frange_idx_0;
    double omega_psi;
    double s0;
    b = self->FrequencyLimits[0];
    b /= self->SamplingFrequency;
    b = b * 2.0 * 3.1415926535897931;
    frange_idx_0 = b;
    b = self->FrequencyLimits[1];
    b /= self->SamplingFrequency;
    b = b * 2.0 * 3.1415926535897931;
    b_nv = self->VoicesPerOctave;
    b_self[0] = self->Wavelet[0];
    b_self[1] = self->Wavelet[1];
    b_self[2] = self->Wavelet[2];
    b_self[3] = self->Wavelet[3];
    wavelet::internal::cwt::wavCFandSD(b_self, self->Gamma, self->Beta, N,
                                       omega_psi);
    s0 = omega_psi / b;
    b = b_nv *
        internal::scalar::scalar_real_log2(omega_psi / frange_idx_0 / s0);
    if (std::isnan(b)) {
      omega.set_size(1, 1);
      omega[0] = rtNaN;
    } else if (b < 0.0) {
      omega.set_size(1, 0);
    } else {
      omega.set_size(1, static_cast<int>(b) + 1);
      ret = static_cast<int>(b);
      for (int k{0}; k <= ret; k++) {
        omega[k] = k;
      }
    }
    b = rt_powd_snf(2.0, 1.0 / b_nv);
    self->Scales.set_size(1, self->Scales.size(1));
    ret = omega.size(1);
    self->Scales.set_size(self->Scales.size(0), omega.size(1));
    for (int k{0}; k < ret; k++) {
      N = omega[k];
      self->Scales[k] = s0 * rt_powd_snf(b, N);
    }
  } else {
    double b_nv;
    double cf;
    double frange_idx_0;
    double maxScale;
    double nv;
    double omega_psi;
    double s0;
    char wname[4];
    char c;
    b = self->SignalLength;
    frange_idx_0 = self->Gamma;
    omega_psi = self->Beta;
    nv = self->VoicesPerOctave;
    b_nv = self->CutOff;
    c = self->Wavelet[0];
    b_self[0] = cv[static_cast<int>(static_cast<unsigned char>(c) & 127U)];
    c = self->Wavelet[1];
    b_self[1] = cv[static_cast<int>(static_cast<unsigned char>(c) & 127U)];
    c = self->Wavelet[2];
    b_self[2] = cv[static_cast<int>(static_cast<unsigned char>(c) & 127U)];
    c = self->Wavelet[3];
    b_self[3] = cv[static_cast<int>(static_cast<unsigned char>(c) & 127U)];
    s0 = 3.1415926535897931;
    b_nv /= 100.0;
    wname[0] = b_self[0];
    wname[1] = b_self[1];
    wname[2] = b_self[2];
    wname[3] = b_self[3];
    wavelet::internal::cwt::wavCFandSD(wname, frange_idx_0, omega_psi, N, cf);
    maxScale = b / (N * 2.0);
    ret = std::memcmp(&a[0], &b_self[0], 4);
    if (ret == 0) {
      ret = 1;
    } else {
      ret = std::memcmp(&b_a[0], &b_self[0], 4);
      if (ret == 0) {
        ret = 2;
      } else {
        ret = -1;
      }
    }
    switch (ret) {
    case 0:
      s0 = wavelet::internal::cwt::getFreqFromCutoffMorse(
          b_nv, cf, frange_idx_0, omega_psi);
      break;
    case 1:
      s0 = wavelet::internal::cwt::getFreqFromCutoffBump(b_nv, cf);
      break;
    case 2:
      s0 = wavelet::internal::cwt::getFreqFromCutoffAmor(b_nv, cf);
      break;
    }
    frange_idx_0 = s0 / 3.1415926535897931;
    b = 1.0 / nv;
    omega_psi = rt_powd_snf(2.0, b);
    N = frange_idx_0 * omega_psi;
    if (maxScale < N) {
      maxScale = N;
    }
    b = std::fmax(internal::scalar::scalar_real_log2(maxScale / frange_idx_0),
                  b) *
        nv;
    if (std::isnan(b)) {
      omega.set_size(1, 1);
      omega[0] = rtNaN;
    } else if (b < 0.0) {
      omega.set_size(1, 0);
    } else {
      omega.set_size(1, static_cast<int>(b) + 1);
      ret = static_cast<int>(b);
      for (int k{0}; k <= ret; k++) {
        omega[k] = k;
      }
    }
    self->Scales.set_size(1, self->Scales.size(1));
    ret = omega.size(1);
    self->Scales.set_size(self->Scales.size(0), omega.size(1));
    for (int k{0}; k < ret; k++) {
      b = omega[k];
      self->Scales[k] = frange_idx_0 * rt_powd_snf(omega_psi, b);
    }
  }
  b_self[0] = self->Wavelet[0];
  b_self[1] = self->Wavelet[1];
  b_self[2] = self->Wavelet[2];
  b_self[3] = self->Wavelet[3];
  nx = self->Omega.size(1);
  omega.set_size(1, nx);
  ret = self->Omega.size(1);
  for (int k{0}; k < ret; k++) {
    omega[k] = self->Omega[k];
  }
  b_loop_ub = self->Scales.size(1);
  f.set_size(1, b_loop_ub);
  ret = self->Scales.size(1);
  for (int k{0}; k < ret; k++) {
    f[k] = self->Scales[k];
  }
  if (f.size(1) == 1) {
    if (omega.size(1) == 1) {
      ret = 1;
    } else {
      ret = omega.size(1);
    }
    somega.set_size(1, ret);
    if (somega.size(1) != 0) {
      bcoef = (omega.size(1) != 1);
      for (int k{0}; k < ret; k++) {
        somega[k] = f[0] * omega[bcoef * k];
      }
    }
  } else {
    somega.set_size(b_loop_ub, nx);
    for (int k{0}; k < nx; k++) {
      ret = (b_loop_ub / 2) << 1;
      bcoef = ret - 2;
      for (int i1{0}; i1 <= bcoef; i1 += 2) {
        r = _mm_loadu_pd(&f[i1]);
        _mm_storeu_pd(&somega[i1 + somega.size(0) * k],
                      _mm_mul_pd(r, _mm_set1_pd(omega[k])));
      }
      for (int i1{ret}; i1 < b_loop_ub; i1++) {
        somega[i1 + somega.size(0) * k] = f[i1] * omega[k];
      }
    }
  }
  ret = std::memcmp(&b_a[0], &b_self[0], 4);
  if (ret == 0) {
    ret = 0;
  } else {
    ret = -1;
  }
  if (ret == 0) {
    fc = 6;
    ret = somega.size(0);
    bcoef = somega.size(1);
    r1.set_size(somega.size(0), somega.size(1));
    loop_ub = somega.size(0) * somega.size(1);
    nx = (loop_ub < 800);
    if (nx) {
      for (int i2{0}; i2 < loop_ub; i2++) {
        r1[i2] = (somega[i2] > 0.0);
      }
    } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

      for (int i2 = 0; i2 < loop_ub; i2++) {
        r1[i2] = (somega[i2] > 0.0);
      }
    }
    if ((somega.size(0) == r1.size(0)) && (somega.size(1) == r1.size(1))) {
      expnt.set_size(ret, bcoef);
      if (nx) {
        for (int i3{0}; i3 < loop_ub; i3++) {
          b = somega[i3] - 6.0;
          expnt[i3] = -(b * b) / 2.0 * static_cast<double>(r1[i3]);
        }
      } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(d)

        for (int i3 = 0; i3 < loop_ub; i3++) {
          d = somega[i3] - 6.0;
          expnt[i3] = -(d * d) / 2.0 * static_cast<double>(r1[i3]);
        }
      }
    } else {
      binary_expand_op_15(expnt, somega, r1);
    }
    nx = expnt.size(0) * expnt.size(1);
    ret = (nx < 800);
    if (ret) {
      for (int b_k{0}; b_k < nx; b_k++) {
        expnt[b_k] = std::exp(expnt[b_k]);
      }
    } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

      for (int b_k = 0; b_k < nx; b_k++) {
        expnt[b_k] = std::exp(expnt[b_k]);
      }
    }
    if ((expnt.size(0) == r1.size(0)) && (expnt.size(1) == r1.size(1))) {
      if (ret) {
        for (int i4{0}; i4 < nx; i4++) {
          expnt[i4] = 2.0 * expnt[i4] * static_cast<double>(r1[i4]);
        }
      } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

        for (int i4 = 0; i4 < nx; i4++) {
          expnt[i4] = 2.0 * expnt[i4] * static_cast<double>(r1[i4]);
        }
      }
    } else {
      binary_expand_op_14(expnt, r1);
    }
  } else {
    fc = 5;
    ret = somega.size(0) * somega.size(1);
    bcoef = (ret / 2) << 1;
    nx = bcoef - 2;
    for (int k{0}; k <= nx; k += 2) {
      r = _mm_loadu_pd(&somega[k]);
      _mm_storeu_pd(&somega[k], _mm_div_pd(_mm_sub_pd(r, _mm_set1_pd(5.0)),
                                           _mm_set1_pd(0.6)));
    }
    for (int k{bcoef}; k < ret; k++) {
      somega[k] = (somega[k] - 5.0) / 0.6;
    }
    expnt.set_size(somega.size(0), somega.size(1));
    nx = somega.size(0) * somega.size(1);
    ret = (nx / 2) << 1;
    bcoef = ret - 2;
    for (int k{0}; k <= bcoef; k += 2) {
      __m128d r2;
      r = _mm_loadu_pd(&somega[k]);
      r2 = _mm_loadu_pd(&somega[k]);
      _mm_storeu_pd(&expnt[k], _mm_div_pd(_mm_set1_pd(-1.0),
                                          _mm_sub_pd(_mm_set1_pd(1.0),
                                                     _mm_mul_pd(r, r2))));
    }
    for (int k{ret}; k < nx; k++) {
      expnt[k] = -1.0 / (1.0 - somega[k] * somega[k]);
    }
    bcoef = expnt.size(0) * expnt.size(1);
    ret = (bcoef < 800);
    if (ret) {
      for (int c_k{0}; c_k < bcoef; c_k++) {
        expnt[c_k] = std::exp(expnt[c_k]);
      }
    } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

      for (int c_k = 0; c_k < bcoef; c_k++) {
        expnt[c_k] = std::exp(expnt[c_k]);
      }
    }
    b_y.set_size(somega.size(0), somega.size(1));
    if (static_cast<int>(nx < 800)) {
      for (int d_k{0}; d_k < nx; d_k++) {
        b_y[d_k] = std::abs(somega[d_k]);
      }
    } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

      for (int d_k = 0; d_k < nx; d_k++) {
        b_y[d_k] = std::abs(somega[d_k]);
      }
    }
    if ((expnt.size(0) == b_y.size(0)) && (expnt.size(1) == b_y.size(1))) {
      if (ret) {
        for (int i5{0}; i5 < bcoef; i5++) {
          expnt[i5] = 5.43656365691809 * expnt[i5] *
                      static_cast<double>(b_y[i5] < 0.99999999999999978);
        }
      } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

        for (int i5 = 0; i5 < bcoef; i5++) {
          expnt[i5] = 5.43656365691809 * expnt[i5] *
                      static_cast<double>(b_y[i5] < 0.99999999999999978);
        }
      }
    } else {
      binary_expand_op_16(expnt, b_y);
    }
    ret = expnt.size(0) * expnt.size(1);
    if (static_cast<int>(ret < 800)) {
      for (int b_i{0}; b_i < ret; b_i++) {
        if (std::isnan(expnt[b_i])) {
          expnt[b_i] = 0.0;
        }
      }
    } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

      for (int b_i = 0; b_i < ret; b_i++) {
        if (std::isnan(expnt[b_i])) {
          expnt[b_i] = 0.0;
        }
      }
    }
  }
  b = static_cast<double>(fc) / 6.2831853071795862;
  f.set_size(1, f.size(1));
  ret = f.size(1) - 1;
  for (int k{0}; k <= ret; k++) {
    f[k] = b / f[k] * self->SamplingFrequency;
  }
  self->PsiDFT.set_size(expnt.size(0), expnt.size(1));
  ret = expnt.size(0) * expnt.size(1);
  for (int k{0}; k < ret; k++) {
    self->PsiDFT[k] = expnt[k];
  }
  self->WaveletCenterFrequencies.set_size(b_loop_ub);
  for (int k{0}; k < b_loop_ub; k++) {
    self->WaveletCenterFrequencies[k] = f[k];
  }
  return self;
}

} // namespace coder

// End of code generation (cwtfilterbank.cpp)
