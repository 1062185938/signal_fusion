//
// fsst.cpp
//
// Code generation for function 'fsst'
//

// Include files
#include "fsst.h"
#include "bsxfun.h"
#include "centerest.h"
#include "computeDFT.h"
#include "dtwin.h"
#include "extractAllFeatures_data.h"
#include "fsstParser.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include "omp.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <xmmintrin.h>

// Function Declarations
static void binary_expand_op_10(coder::array<float, 2U> &in1,
                                const coder::array<creal32_T, 2U> &in2,
                                const coder::array<creal32_T, 2U> &in3);

// Function Definitions
static void binary_expand_op_10(coder::array<float, 2U> &in1,
                                const coder::array<creal32_T, 2U> &in2,
                                const coder::array<creal32_T, 2U> &in3)
{
  int aux_0_1;
  int aux_1_1;
  int b_loop_ub;
  int loop_ub;
  int stride_0_0;
  int stride_0_1;
  int stride_1_0;
  int stride_1_1;
  if (in3.size(0) == 1) {
    loop_ub = in2.size(0);
  } else {
    loop_ub = in3.size(0);
  }
  in1.set_size(loop_ub, in1.size(1));
  if (in3.size(1) == 1) {
    b_loop_ub = in2.size(1);
  } else {
    b_loop_ub = in3.size(1);
  }
  in1.set_size(in1.size(0), b_loop_ub);
  stride_0_0 = (in2.size(0) != 1);
  stride_0_1 = (in2.size(1) != 1);
  stride_1_0 = (in3.size(0) != 1);
  stride_1_1 = (in3.size(1) != 1);
  aux_0_1 = 0;
  aux_1_1 = 0;
  for (int i{0}; i < b_loop_ub; i++) {
    for (int i1{0}; i1 < loop_ub; i1++) {
      float ai;
      float ar;
      float bi;
      float br;
      float in2_im;
      int ar_tmp;
      ar_tmp = i1 * stride_0_0 + in2.size(0) * aux_0_1;
      ar = in2[ar_tmp].re;
      ai = in2[ar_tmp].im;
      ar_tmp = i1 * stride_1_0 + in3.size(0) * aux_1_1;
      br = in3[ar_tmp].re;
      bi = in3[ar_tmp].im;
      if (bi == 0.0F) {
        if (ai == 0.0F) {
          in2_im = 0.0F;
        } else {
          in2_im = ai / br;
        }
      } else if (br == 0.0F) {
        if (ar == 0.0F) {
          in2_im = 0.0F;
        } else {
          in2_im = -(ar / bi);
        }
      } else {
        float brm;
        brm = std::abs(br);
        in2_im = std::abs(bi);
        if (brm > in2_im) {
          in2_im = bi / br;
          in2_im = (ai - in2_im * ar) / (br + in2_im * bi);
        } else if (in2_im == brm) {
          if (br > 0.0F) {
            br = 0.5F;
          } else {
            br = -0.5F;
          }
          if (bi > 0.0F) {
            in2_im = 0.5F;
          } else {
            in2_im = -0.5F;
          }
          in2_im = (ai * br - ar * in2_im) / brm;
        } else {
          in2_im = br / bi;
          in2_im = (in2_im * ai - ar) / (bi + in2_im * br);
        }
      }
      in1[i1 + in1.size(0) * i] = -in2_im;
    }
    aux_1_1 += stride_1_1;
    aux_0_1 += stride_0_1;
  }
}

namespace coder {
int fsst(const array<creal32_T, 1U> &x, double varargin_1,
         const double varargin_2_data[], int varargin_2_size,
         array<creal32_T, 2U> &sst, float f_data[], array<float, 2U> &t)
{
  __m128 r1;
  array<creal32_T, 2U> Xx;
  array<creal32_T, 2U> b_xin;
  array<creal32_T, 2U> xin;
  array<creal32_T, 1U> xp;
  array<double, 2U> r;
  array<float, 2U> fcorr;
  array<float, 2U> tcorr;
  array<float, 2U> tout;
  array<float, 1U> colIdx;
  array<float, 1U> rowIdx;
  array<short, 2U> y;
  creal_T b_z_data[258];
  creal_T z_data[258];
  double b_f_data[258];
  double win_data[258];
  double Fs;
  double y_im;
  float tmp_data[258];
  float win1_data[258];
  float b_ai;
  float b_ar;
  float b_bi;
  float b_br;
  float b_brm;
  float b_varargin_1;
  float b_x;
  float b_xin_im;
  float bi;
  float br;
  float c_x;
  float xin_im;
  int b_bcoef;
  int bcoef;
  int csz_idx_0;
  int f_size;
  int i2;
  int loop_ub;
  int nCol;
  int nx;
  int win_size;
  win_size = varargin_2_size;
  if (varargin_2_size - 1 >= 0) {
    std::copy(&varargin_2_data[0], &varargin_2_data[varargin_2_size],
              &win_data[0]);
  }
  Fs = b_signal::internal::fsst::fsstParser(x, varargin_1, win_data, win_size);
  if (x.size(0) - 1 < 0) {
    y.set_size(1, 0);
  } else {
    y.set_size(1, x.size(0));
    bcoef = x.size(0) - 1;
    for (int iCol{0}; iCol <= bcoef; iCol++) {
      y[iCol] = static_cast<short>(iCol);
    }
  }
  loop_ub = y.size(1);
  tout.set_size(1, y.size(1));
  bcoef = y.size(1);
  if (static_cast<int>(y.size(1) < 800)) {
    for (int i{0}; i < loop_ub; i++) {
      tout[i] = static_cast<float>(static_cast<double>(y[i]) / Fs);
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

    for (int i = 0; i < bcoef; i++) {
      tout[i] = static_cast<float>(static_cast<double>(y[i]) / Fs);
    }
  }
  for (int iCol{0}; iCol < win_size; iCol++) {
    win1_data[iCol] = static_cast<float>(win_data[iCol]);
  }
  if (std::fmod(static_cast<double>(win_size), 2.0) == 1.0) {
    nCol = static_cast<int>((static_cast<double>(win_size) - 1.0) / 2.0);
    xp.set_size((nCol + x.size(0)) + nCol);
    if (nCol - 1 >= 0) {
      std::memset(&xp[0], 0,
                  static_cast<unsigned int>(nCol) * sizeof(creal32_T));
    }
    bcoef = x.size(0);
    for (int iCol{0}; iCol < bcoef; iCol++) {
      xp[iCol + nCol] = x[iCol];
    }
    for (int iCol{0}; iCol < nCol; iCol++) {
      bcoef = (iCol + nCol) + x.size(0);
      xp[bcoef].re = 0.0F;
      xp[bcoef].im = 0.0F;
    }
  } else {
    nCol = static_cast<int>(static_cast<double>(win_size) / 2.0);
    b_bcoef = static_cast<int>((static_cast<double>(win_size) - 2.0) / 2.0);
    xp.set_size((nCol + x.size(0)) + b_bcoef);
    if (nCol - 1 >= 0) {
      std::memset(&xp[0], 0,
                  static_cast<unsigned int>(nCol) * sizeof(creal32_T));
    }
    bcoef = x.size(0);
    for (int iCol{0}; iCol < bcoef; iCol++) {
      xp[iCol + nCol] = x[iCol];
    }
    for (int iCol{0}; iCol < b_bcoef; iCol++) {
      bcoef = (iCol + nCol) + x.size(0);
      xp[bcoef].re = 0.0F;
      xp[bcoef].im = 0.0F;
    }
  }
  nCol = (xp.size(0) - win_size) + 1;
  xin.set_size(win_size, nCol);
  bcoef = win_size * nCol;
  if (bcoef - 1 >= 0) {
    std::memset(&xin[0], 0,
                static_cast<unsigned int>(bcoef) * sizeof(creal32_T));
  }
  for (int iCol{0}; iCol < nCol; iCol++) {
    if (((static_cast<double>(iCol) + 1.0) - 1.0) + 1.0 >
        static_cast<double>(win_size) +
            ((static_cast<double>(iCol) + 1.0) - 1.0)) {
      bcoef = 1;
    } else {
      bcoef = iCol + 1;
    }
    for (int k{0}; k < win_size; k++) {
      xin[k + xin.size(0) * iCol] = xp[(bcoef + k) - 1];
    }
  }
  bsxfun(win1_data, win_size, xin, b_xin);
  f_size = computeDFTviaFFT(b_xin, static_cast<double>(b_xin.size(0)),
                            static_cast<double>(win_size), Fs, Xx, b_f_data);
  for (int iCol{0}; iCol < f_size; iCol++) {
    f_data[iCol] = static_cast<float>(b_f_data[iCol]);
  }
  bcoef =
      b_signal::internal::spectral::dtwin(win1_data, win_size, Fs, tmp_data);
  bsxfun(tmp_data, bcoef, xin, b_xin);
  computeDFTviaFFT(b_xin, static_cast<double>(b_xin.size(0)),
                   static_cast<double>(win_size), Fs, xin, b_f_data);
  if ((xin.size(0) == Xx.size(0)) && (xin.size(1) == Xx.size(1))) {
    fcorr.set_size(xin.size(0), xin.size(1));
    bcoef = xin.size(0) * xin.size(1);
    if (static_cast<int>(bcoef < 800)) {
      for (int i1{0}; i1 < bcoef; i1++) {
        float ai;
        float ar;
        ar = xin[i1].re;
        ai = xin[i1].im;
        br = Xx[i1].re;
        bi = Xx[i1].im;
        if (bi == 0.0F) {
          if (ai == 0.0F) {
            xin_im = 0.0F;
          } else {
            xin_im = ai / br;
          }
        } else if (br == 0.0F) {
          if (ar == 0.0F) {
            xin_im = 0.0F;
          } else {
            xin_im = -(ar / bi);
          }
        } else {
          float brm;
          brm = std::abs(br);
          xin_im = std::abs(bi);
          if (brm > xin_im) {
            xin_im = bi / br;
            xin_im = (ai - xin_im * ar) / (br + xin_im * bi);
          } else if (xin_im == brm) {
            if (br > 0.0F) {
              br = 0.5F;
            } else {
              br = -0.5F;
            }
            if (bi > 0.0F) {
              xin_im = 0.5F;
            } else {
              xin_im = -0.5F;
            }
            xin_im = (ai * br - ar * xin_im) / brm;
          } else {
            xin_im = br / bi;
            xin_im = (xin_im * ai - ar) / (bi + xin_im * br);
          }
        }
        fcorr[i1] = -xin_im;
      }
    } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        b_ar, b_ai, b_br, b_bi, b_xin_im, b_brm)

      for (int i1 = 0; i1 < bcoef; i1++) {
        b_ar = xin[i1].re;
        b_ai = xin[i1].im;
        b_br = Xx[i1].re;
        b_bi = Xx[i1].im;
        if (b_bi == 0.0F) {
          if (b_ai == 0.0F) {
            b_xin_im = 0.0F;
          } else {
            b_xin_im = b_ai / b_br;
          }
        } else if (b_br == 0.0F) {
          if (b_ar == 0.0F) {
            b_xin_im = 0.0F;
          } else {
            b_xin_im = -(b_ar / b_bi);
          }
        } else {
          b_brm = std::abs(b_br);
          b_xin_im = std::abs(b_bi);
          if (b_brm > b_xin_im) {
            b_xin_im = b_bi / b_br;
            b_xin_im = (b_ai - b_xin_im * b_ar) / (b_br + b_xin_im * b_bi);
          } else if (b_xin_im == b_brm) {
            if (b_br > 0.0F) {
              b_br = 0.5F;
            } else {
              b_br = -0.5F;
            }
            if (b_bi > 0.0F) {
              b_xin_im = 0.5F;
            } else {
              b_xin_im = -0.5F;
            }
            b_xin_im = (b_ai * b_br - b_ar * b_xin_im) / b_brm;
          } else {
            b_xin_im = b_br / b_bi;
            b_xin_im = (b_xin_im * b_ai - b_ar) / (b_bi + b_xin_im * b_br);
          }
        }
        fcorr[i1] = -b_xin_im;
      }
    }
  } else {
    binary_expand_op_10(fcorr, xin, Xx);
  }
  bcoef = fcorr.size(0) * fcorr.size(1);
  if (static_cast<int>(bcoef < 800)) {
    for (int b_i{0}; b_i < bcoef; b_i++) {
      if (std::isinf(fcorr[b_i]) || std::isnan(fcorr[b_i])) {
        fcorr[b_i] = 0.0F;
      }
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

    for (int b_i = 0; b_i < bcoef; b_i++) {
      if (std::isinf(fcorr[b_i]) || std::isnan(fcorr[b_i])) {
        fcorr[b_i] = 0.0F;
      }
    }
  }
  i2 = fcorr.size(1);
  tcorr.set_size(fcorr.size(0), fcorr.size(1));
  if (bcoef - 1 >= 0) {
    std::copy(&fcorr[0], &fcorr[bcoef], &tcorr[0]);
  }
  bcoef = fcorr.size(0);
  csz_idx_0 = f_size;
  if (bcoef <= f_size) {
    csz_idx_0 = bcoef;
  }
  if (fcorr.size(0) == 1) {
    csz_idx_0 = f_size;
  } else if (f_size == 1) {
    csz_idx_0 = fcorr.size(0);
  } else if (f_size == fcorr.size(0)) {
    csz_idx_0 = f_size;
  }
  fcorr.set_size(csz_idx_0, fcorr.size(1));
  if ((csz_idx_0 != 0) && (i2 != 0)) {
    bcoef = (tcorr.size(1) != 1);
    nCol = (f_size != 1);
    b_bcoef = (tcorr.size(0) != 1);
    for (int iCol{0}; iCol < i2; iCol++) {
      nx = bcoef * iCol;
      for (int k{0}; k < csz_idx_0; k++) {
        fcorr[k + fcorr.size(0) * iCol] =
            f_data[nCol * k] + tcorr[b_bcoef * k + tcorr.size(0) * nx];
      }
    }
  }
  r.set_size(fcorr.size(0), fcorr.size(1));
  b_bcoef = fcorr.size(0) * fcorr.size(1);
  if (b_bcoef - 1 >= 0) {
    std::memset(&r[0], 0, static_cast<unsigned int>(b_bcoef) * sizeof(double));
  }
  bsxfun(tout, r, tcorr);
  if (win_size - 1 < 0) {
    y.set_size(1, 0);
  } else {
    y.set_size(1, win_size);
    bcoef = win_size - 1;
    for (int iCol{0}; iCol <= bcoef; iCol++) {
      y[iCol] = static_cast<short>(iCol);
    }
  }
  y_im = std::floor(static_cast<double>(win_size) / 2.0) * -6.2831853071795862;
  bcoef = y.size(1);
  for (int iCol{0}; iCol < bcoef; iCol++) {
    double d;
    Fs = y_im * static_cast<double>(y[iCol]);
    if (Fs == 0.0) {
      d = -0.0 / static_cast<double>(win_size);
      z_data[iCol].re = d;
      Fs = 0.0;
      z_data[iCol].im = 0.0;
    } else {
      d = 0.0;
      z_data[iCol].re = 0.0;
      Fs /= static_cast<double>(win_size);
      z_data[iCol].im = Fs;
    }
    if (d == 0.0) {
      z_data[iCol].re = std::cos(Fs);
      Fs = std::sin(Fs);
      z_data[iCol].im = Fs;
    } else if (Fs == 0.0) {
      z_data[iCol].re = rtNaN;
      z_data[iCol].im = 0.0;
    } else {
      z_data[iCol].re = rtNaN;
      z_data[iCol].im = rtNaN;
    }
  }
  for (int iCol{0}; iCol < bcoef; iCol++) {
    b_z_data[iCol].re = z_data[iCol].re;
    b_z_data[iCol].im = -z_data[iCol].im;
  }
  bsxfun(Xx, b_z_data, bcoef, xin);
  xin_im = f_data[f_size - 1] - f_data[0];
  br = f_data[0];
  rowIdx.set_size(b_bcoef);
  bcoef = (b_bcoef / 4) << 2;
  nCol = bcoef - 4;
  for (int iCol{0}; iCol <= nCol; iCol += 4) {
    r1 = _mm_loadu_ps(&fcorr[iCol]);
    _mm_storeu_ps(
        &rowIdx[iCol],
        _mm_div_ps(_mm_mul_ps(_mm_sub_ps(r1, _mm_set1_ps(br)),
                              _mm_set1_ps(static_cast<float>(f_size) - 1.0F)),
                   _mm_set1_ps(xin_im)));
  }
  for (int iCol{bcoef}; iCol < b_bcoef; iCol++) {
    rowIdx[iCol] =
        (fcorr[iCol] - br) * (static_cast<float>(f_size) - 1.0F) / xin_im;
  }
  nx = rowIdx.size(0);
  bcoef = (rowIdx.size(0) < 800);
  if (bcoef) {
    for (int b_k{0}; b_k < nx; b_k++) {
      b_x = rowIdx[b_k];
      if (std::abs(rowIdx[b_k]) < 8.388608E+6F) {
        if (rowIdx[b_k] >= 0.5F) {
          b_x = std::floor(rowIdx[b_k] + 0.5F);
        } else if (rowIdx[b_k] > -0.5F) {
          b_x = rowIdx[b_k] * 0.0F;
        } else {
          b_x = std::ceil(rowIdx[b_k] - 0.5F);
        }
      }
      rowIdx[b_k] = b_x;
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(b_x)

    for (int b_k = 0; b_k < nx; b_k++) {
      b_x = rowIdx[b_k];
      if (std::abs(rowIdx[b_k]) < 8.388608E+6F) {
        if (rowIdx[b_k] >= 0.5F) {
          b_x = std::floor(rowIdx[b_k] + 0.5F);
        } else if (rowIdx[b_k] > -0.5F) {
          b_x = rowIdx[b_k] * 0.0F;
        } else {
          b_x = std::ceil(rowIdx[b_k] - 0.5F);
        }
      }
      rowIdx[b_k] = b_x;
    }
  }
  if (bcoef) {
    for (int i3{0}; i3 < nx; i3++) {
      xin_im = rowIdx[i3];
      if (f_size == 0) {
        if (xin_im == 0.0F) {
          xin_im = 0.0F;
        }
      } else if (std::isnan(xin_im) || std::isinf(xin_im)) {
        xin_im = rtNaNF;
      } else {
        xin_im = std::fmod(xin_im, static_cast<float>(f_size));
        if (xin_im == 0.0F) {
          xin_im = 0.0F;
        } else if (xin_im < 0.0F) {
          xin_im += static_cast<float>(f_size);
        }
      }
      rowIdx[i3] = xin_im + 1.0F;
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        b_varargin_1)

    for (int i3 = 0; i3 < nx; i3++) {
      b_varargin_1 = rowIdx[i3];
      if (f_size == 0) {
        if (b_varargin_1 == 0.0F) {
          b_varargin_1 = 0.0F;
        }
      } else if (std::isnan(b_varargin_1) || std::isinf(b_varargin_1)) {
        b_varargin_1 = rtNaNF;
      } else {
        b_varargin_1 = std::fmod(b_varargin_1, static_cast<float>(f_size));
        if (b_varargin_1 == 0.0F) {
          b_varargin_1 = 0.0F;
        } else if (b_varargin_1 < 0.0F) {
          b_varargin_1 += static_cast<float>(f_size);
        }
      }
      rowIdx[i3] = b_varargin_1 + 1.0F;
    }
  }
  xin_im = tout[tout.size(1) - 1] - tout[0];
  br = tout[0];
  bi = static_cast<float>(tout.size(1)) - 1.0F;
  bcoef = tcorr.size(0) * tcorr.size(1);
  colIdx.set_size(bcoef);
  nCol = (bcoef / 4) << 2;
  b_bcoef = nCol - 4;
  for (int iCol{0}; iCol <= b_bcoef; iCol += 4) {
    r1 = _mm_loadu_ps(&tcorr[iCol]);
    _mm_storeu_ps(
        &colIdx[iCol],
        _mm_div_ps(_mm_mul_ps(_mm_sub_ps(r1, _mm_set1_ps(br)), _mm_set1_ps(bi)),
                   _mm_set1_ps(xin_im)));
  }
  for (int iCol{nCol}; iCol < bcoef; iCol++) {
    colIdx[iCol] = (tcorr[iCol] - br) * bi / xin_im;
  }
  b_bcoef = colIdx.size(0);
  if (static_cast<int>(colIdx.size(0) < 800)) {
    for (int c_k{0}; c_k < b_bcoef; c_k++) {
      c_x = colIdx[c_k];
      if (std::abs(colIdx[c_k]) < 8.388608E+6F) {
        if (colIdx[c_k] >= 0.5F) {
          c_x = std::floor(colIdx[c_k] + 0.5F);
        } else if (colIdx[c_k] > -0.5F) {
          c_x = colIdx[c_k] * 0.0F;
        } else {
          c_x = std::ceil(colIdx[c_k] - 0.5F);
        }
      }
      colIdx[c_k] = c_x;
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(c_x)

    for (int c_k = 0; c_k < b_bcoef; c_k++) {
      c_x = colIdx[c_k];
      if (std::abs(colIdx[c_k]) < 8.388608E+6F) {
        if (colIdx[c_k] >= 0.5F) {
          c_x = std::floor(colIdx[c_k] + 0.5F);
        } else if (colIdx[c_k] > -0.5F) {
          c_x = colIdx[c_k] * 0.0F;
        } else {
          c_x = std::ceil(colIdx[c_k] - 0.5F);
        }
      }
      colIdx[c_k] = c_x;
    }
  }
  bcoef = (colIdx.size(0) / 4) << 2;
  nCol = bcoef - 4;
  for (int iCol{0}; iCol <= nCol; iCol += 4) {
    r1 = _mm_loadu_ps(&colIdx[iCol]);
    _mm_storeu_ps(&colIdx[iCol], _mm_add_ps(r1, _mm_set1_ps(1.0F)));
  }
  for (int iCol{bcoef}; iCol < b_bcoef; iCol++) {
    colIdx[iCol] = colIdx[iCol] + 1.0F;
  }
  sst.set_size(f_size, loop_ub);
  bcoef = f_size * tout.size(1);
  for (int iCol{0}; iCol < bcoef; iCol++) {
    sst[iCol].re = 0.0F;
    sst[iCol].im = 0.0F;
  }
  for (int iCol{0}; iCol < nx; iCol++) {
    if ((rowIdx[iCol] >= 1.0F) &&
        (static_cast<double>(rowIdx[iCol]) <= f_size) &&
        (colIdx[iCol] >= 1.0F) &&
        (static_cast<double>(colIdx[iCol]) <= tout.size(1))) {
      bcoef = static_cast<int>(rowIdx[iCol]) - 1;
      nCol = static_cast<int>(colIdx[iCol]) - 1;
      bcoef += sst.size(0) * nCol;
      sst[bcoef].re = sst[bcoef].re + xin[iCol].re;
      sst[bcoef].im = sst[bcoef].im + xin[iCol].im;
    }
  }
  if (std::fmod(static_cast<double>(f_size), 2.0) == 0.0) {
    xin_im = f_data[static_cast<int>(static_cast<double>(f_size) / 2.0) - 1];
    bcoef = (f_size / 4) << 2;
    nCol = bcoef - 4;
    for (int iCol{0}; iCol <= nCol; iCol += 4) {
      r1 = _mm_loadu_ps(&f_data[iCol]);
      _mm_storeu_ps(&f_data[iCol], _mm_sub_ps(r1, _mm_set1_ps(xin_im)));
    }
    for (int iCol{bcoef}; iCol < f_size; iCol++) {
      f_data[iCol] -= xin_im;
    }
  } else {
    xin_im =
        f_data[static_cast<int>((static_cast<double>(f_size) + 1.0) / 2.0) - 1];
    bcoef = (f_size / 4) << 2;
    nCol = bcoef - 4;
    for (int iCol{0}; iCol <= nCol; iCol += 4) {
      r1 = _mm_loadu_ps(&f_data[iCol]);
      _mm_storeu_ps(&f_data[iCol], _mm_sub_ps(r1, _mm_set1_ps(xin_im)));
    }
    for (int iCol{bcoef}; iCol < f_size; iCol++) {
      f_data[iCol] -= xin_im;
    }
  }
  b_signal::internal::spectral::centerest(sst);
  t.set_size(1, loop_ub);
  for (int iCol{0}; iCol < loop_ub; iCol++) {
    t[iCol] = tout[iCol];
  }
  return f_size;
}

} // namespace coder

// End of code generation (fsst.cpp)
