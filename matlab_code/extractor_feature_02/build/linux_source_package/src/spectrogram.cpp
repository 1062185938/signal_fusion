//
// spectrogram.cpp
//
// Code generation for function 'spectrogram'
//

// Include files
#include "spectrogram.h"
#include "FFTImplementationCallback.h"
#include "bsxfun.h"
#include "extractAllFeatures_data.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include <cmath>
#include <cstring>
#include <emmintrin.h>
#include <xmmintrin.h>

// Function Definitions
namespace coder {
void spectrogram(const array<creal32_T, 1U> &x, const float varargin_1_data[],
                 int varargin_1_size, double varargin_2, double varargin_4,
                 array<creal32_T, 2U> &varargout_1, float varargout_2[512],
                 array<float, 2U> &varargout_3)
{
  __m128 r;
  array<creal32_T, 2U> b_xin;
  array<creal32_T, 2U> xin;
  double hopSize;
  double nCol;
  float Fs1;
  float freq_res;
  int b_eint;
  int eint;
  int i;
  int pageroot;
  std::frexp(static_cast<double>(varargin_1_size), &eint);
  std::frexp(static_cast<double>(x.size(0)), &b_eint);
  hopSize = static_cast<double>(varargin_1_size) - varargin_2;
  nCol = std::trunc((static_cast<double>(x.size(0)) - varargin_2) / hopSize);
  i = static_cast<int>(nCol);
  xin.set_size(varargin_1_size, static_cast<int>(nCol));
  eint = varargin_1_size * static_cast<int>(nCol);
  if (eint - 1 >= 0) {
    std::memset(&xin[0], 0,
                static_cast<unsigned int>(eint) * sizeof(creal32_T));
  }
  if (nCol - 1.0 < 0.0) {
    varargout_3.set_size(varargout_3.size(0), 0);
  } else {
    varargout_3.set_size(1, static_cast<int>(nCol - 1.0) + 1);
    eint = static_cast<int>(nCol - 1.0);
    b_eint = ((static_cast<int>(nCol - 1.0) + 1) / 4) << 2;
    pageroot = b_eint - 4;
    for (int iCol{0}; iCol <= pageroot; iCol += 4) {
      _mm_storeu_ps(
          &varargout_3[iCol],
          _mm_cvtepi32_ps(_mm_add_epi32(
              _mm_set1_epi32(iCol), _mm_loadu_si128((const __m128i *)&iv[0]))));
    }
    for (int iCol{b_eint}; iCol <= eint; iCol++) {
      varargout_3[iCol] = static_cast<float>(iCol);
    }
  }
  for (int iCol{0}; iCol < i; iCol++) {
    nCol = hopSize * ((static_cast<double>(iCol) + 1.0) - 1.0);
    if (nCol + 1.0 > static_cast<double>(varargin_1_size) + nCol) {
      eint = 1;
    } else {
      eint = static_cast<int>(nCol + 1.0);
    }
    for (int b_i{0}; b_i < varargin_1_size; b_i++) {
      xin[b_i + xin.size(0) * iCol] = x[(eint + b_i) - 1];
    }
  }
  bsxfun(varargin_1_data, (*(int(*)[1]) & varargin_1_size)[0], xin, b_xin);
  if (b_xin.size(1) == 0) {
    varargout_1.set_size(512, 0);
  } else {
    internal::fft::FFTImplementationCallback::r2br_r2dit_trig(b_xin,
                                                              varargout_1);
  }
  if (std::isnan(static_cast<float>(varargin_4))) {
    Fs1 = 6.28318548F;
  } else {
    Fs1 = static_cast<float>(varargin_4);
  }
  freq_res = Fs1 / 512.0F;
  for (int iCol{0}; iCol <= 508; iCol += 4) {
    _mm_storeu_ps(&varargout_2[iCol],
                  _mm_mul_ps(_mm_set1_ps(freq_res),
                             _mm_cvtepi32_ps(_mm_add_epi32(
                                 _mm_set1_epi32(iCol),
                                 _mm_loadu_si128((const __m128i *)&iv[0])))));
  }
  varargout_2[256] = Fs1 / 2.0F;
  varargout_2[511] = Fs1 - freq_res;
  if (varargout_1.size(1) != 0) {
    b_eint = varargout_1.size(1);
    for (int b_i{0}; b_i < b_eint; b_i++) {
      creal32_T a__1[255];
      pageroot = b_i << 9;
      for (int iCol{0}; iCol < 255; iCol++) {
        a__1[iCol] = varargout_1[(pageroot + iCol) + 257];
      }
      for (int iCol{256}; iCol >= 0; iCol--) {
        eint = pageroot + iCol;
        varargout_1[eint + 255] = varargout_1[eint];
      }
      for (int iCol{0}; iCol < 255; iCol++) {
        varargout_1[pageroot + iCol] = a__1[iCol];
      }
    }
  }
  Fs1 = varargout_2[255];
  for (int iCol{0}; iCol <= 508; iCol += 4) {
    r = _mm_loadu_ps(&varargout_2[iCol]);
    _mm_storeu_ps(&varargout_2[iCol], _mm_sub_ps(r, _mm_set1_ps(Fs1)));
  }
  varargout_3.set_size(1, varargout_3.size(1));
  Fs1 = static_cast<float>(static_cast<double>(varargin_1_size) / 2.0);
  eint = varargout_3.size(1) - 1;
  b_eint = (varargout_3.size(1) / 4) << 2;
  pageroot = b_eint - 4;
  for (int iCol{0}; iCol <= pageroot; iCol += 4) {
    r = _mm_loadu_ps(&varargout_3[iCol]);
    _mm_storeu_ps(
        &varargout_3[iCol],
        _mm_div_ps(
            _mm_add_ps(_mm_mul_ps(r, _mm_set1_ps(static_cast<float>(hopSize))),
                       _mm_set1_ps(Fs1)),
            _mm_set1_ps(static_cast<float>(varargin_4))));
  }
  for (int iCol{b_eint}; iCol <= eint; iCol++) {
    varargout_3[iCol] =
        (varargout_3[iCol] * static_cast<float>(hopSize) + Fs1) /
        static_cast<float>(varargin_4);
  }
}

} // namespace coder

// End of code generation (spectrogram.cpp)
