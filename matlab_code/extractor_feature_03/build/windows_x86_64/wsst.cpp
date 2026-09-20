//
// wsst.cpp
//
// Code generation for function 'wsst'
//

// Include files
#include "wsst.h"
#include "abs.h"
#include "atan2.h"
#include "cwtfilterbank.h"
#include "eml_setop.h"
#include "extractAllFeatures_data.h"
#include "fft.h"
#include "find.h"
#include "flip.h"
#include "ifft.h"
#include "log21.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include "omp.h"
#include <cmath>
#include <cstring>
#include <emmintrin.h>

// Function Declarations
static void binary_expand_op_11(coder::array<double, 2U> &in1,
                                const coder::array<creal_T, 2U> &in2,
                                const coder::array<creal_T, 2U> &in3);

static void binary_expand_op_12(coder::array<creal_T, 2U> &in1,
                                const coder::cwtfilterbank &in3,
                                const coder::array<creal_T, 2U> &in4);

static void binary_expand_op_13(coder::array<creal_T, 2U> &in1,
                                const coder::array<creal_T, 2U> &in2,
                                const coder::cwtfilterbank &in3);

// Function Definitions
static void binary_expand_op_11(coder::array<double, 2U> &in1,
                                const coder::array<creal_T, 2U> &in2,
                                const coder::array<creal_T, 2U> &in3)
{
  int b_loop_ub;
  int loop_ub;
  int stride_0_0;
  int stride_1_0;
  if (in3.size(0) == 1) {
    loop_ub = in2.size(0);
  } else {
    loop_ub = in3.size(0);
  }
  in1.set_size(loop_ub, in1.size(1));
  b_loop_ub = in2.size(1);
  in1.set_size(in1.size(0), b_loop_ub);
  stride_0_0 = (in2.size(0) != 1);
  stride_1_0 = (in3.size(0) != 1);
  for (int i{0}; i < b_loop_ub; i++) {
    for (int i1{0}; i1 < loop_ub; i1++) {
      double ai;
      double ar;
      double bi;
      double br;
      double in2_im;
      int ar_tmp;
      ar_tmp = i1 * stride_0_0 + in2.size(0) * i;
      ar = in2[ar_tmp].re;
      ai = in2[ar_tmp].im;
      ar_tmp = i1 * stride_1_0 + in3.size(0) * i;
      br = in3[ar_tmp].re;
      bi = in3[ar_tmp].im;
      if (bi == 0.0) {
        if (ai == 0.0) {
          in2_im = 0.0;
        } else {
          in2_im = ai / br;
        }
      } else if (br == 0.0) {
        if (ar == 0.0) {
          in2_im = 0.0;
        } else {
          in2_im = -(ar / bi);
        }
      } else {
        double brm;
        brm = std::abs(br);
        in2_im = std::abs(bi);
        if (brm > in2_im) {
          in2_im = bi / br;
          in2_im = (ai - in2_im * ar) / (br + in2_im * bi);
        } else if (in2_im == brm) {
          if (br > 0.0) {
            br = 0.5;
          } else {
            br = -0.5;
          }
          if (bi > 0.0) {
            in2_im = 0.5;
          } else {
            in2_im = -0.5;
          }
          in2_im = (ai * br - ar * in2_im) / brm;
        } else {
          in2_im = br / bi;
          in2_im = (in2_im * ai - ar) / (bi + in2_im * br);
        }
      }
      in1[i1 + in1.size(0) * i] = in2_im / 6.2831853071795862;
    }
  }
}

static void binary_expand_op_12(coder::array<creal_T, 2U> &in1,
                                const coder::cwtfilterbank &in3,
                                const coder::array<creal_T, 2U> &in4)
{
  coder::array<creal_T, 2U> in2;
  int aux_0_1;
  int aux_1_1;
  int b_loop_ub;
  int loop_ub;
  int stride_0_1;
  int stride_1_1;
  loop_ub = in3.PsiDFT.size(0);
  b_loop_ub = in4.size(1);
  in2.set_size(loop_ub, b_loop_ub);
  stride_0_1 = (in3.Omega.size(1) != 1);
  stride_1_1 = (in3.PsiDFT.size(1) != 1);
  aux_0_1 = 0;
  aux_1_1 = 0;
  for (int i{0}; i < b_loop_ub; i++) {
    for (int i1{0}; i1 < loop_ub; i1++) {
      double d;
      double d1;
      double in2_im;
      double in2_re;
      int in2_re_tmp;
      in2_re_tmp = i1 + in3.PsiDFT.size(0) * aux_1_1;
      in2_re = in3.PsiDFT[in2_re_tmp] * (in3.Omega[aux_0_1] * 0.0);
      in2_im = in3.PsiDFT[in2_re_tmp] * in3.Omega[aux_0_1];
      d = in4[i].im;
      d1 = in4[i].re;
      in2_re_tmp = i1 + in2.size(0) * i;
      in2[in2_re_tmp].re = in2_re * d1 - in2_im * d;
      in2[in2_re_tmp].im = in2_re * d + in2_im * d1;
    }
    aux_1_1 += stride_1_1;
    aux_0_1 += stride_0_1;
  }
  coder::ifft(in2, in1);
}

static void binary_expand_op_13(coder::array<creal_T, 2U> &in1,
                                const coder::array<creal_T, 2U> &in2,
                                const coder::cwtfilterbank &in3)
{
  coder::array<creal_T, 2U> b_in2;
  int aux_0_1;
  int b_loop_ub;
  int loop_ub;
  int stride_0_1;
  loop_ub = in3.PsiDFT.size(0);
  b_loop_ub = in2.size(1);
  b_in2.set_size(loop_ub, b_loop_ub);
  stride_0_1 = (in3.PsiDFT.size(1) != 1);
  aux_0_1 = 0;
  for (int i{0}; i < b_loop_ub; i++) {
    for (int i1{0}; i1 < loop_ub; i1++) {
      int i2;
      int i3;
      i2 = i1 + in3.PsiDFT.size(0) * aux_0_1;
      i3 = i1 + b_in2.size(0) * i;
      b_in2[i3].re = in3.PsiDFT[i2] * in2[i].re;
      b_in2[i3].im = in3.PsiDFT[i2] * in2[i].im;
    }
    aux_0_1 += stride_0_1;
  }
  coder::ifft(b_in2, in1);
}

namespace coder {
void wsst(const array<double, 1U> &x, double varargin_1,
          array<creal_T, 2U> &sst, array<double, 1U> &f)
{
  __m128d r;
  cwtfilterbank fb;
  array<creal_T, 2U> b_f;
  array<creal_T, 2U> b_xdft;
  array<creal_T, 2U> c_x;
  array<creal_T, 2U> cwtcfs;
  array<creal_T, 2U> padcwtcfs;
  array<creal_T, 2U> paddcwtcfs;
  array<creal_T, 2U> xdft;
  array<creal_T, 1U> z;
  array<double, 2U> b_x;
  array<double, 2U> iRow;
  array<double, 2U> phasetf;
  array<double, 2U> result;
  array<double, 1U> ftmp;
  array<double, 1U> r1;
  array<double, 1U> r2;
  array<int, 2U> counts;
  array<int, 1U> ia;
  array<int, 1U> ib;
  array<boolean_T, 2U> filled;
  double s[2];
  double SF;
  double a;
  double a_tmp;
  double b_a;
  double b_ai;
  double b_ar;
  double b_bi;
  double b_tmp;
  double b_u;
  double b_varargin_1;
  double bi;
  double br;
  double brm;
  double bsum;
  double c_a;
  double c_ai;
  double c_ar;
  double log2Fund;
  double u;
  double x_im;
  int firstBlockLength;
  int hi;
  int i6;
  int lastBlockLength;
  int loop_ub;
  int nblocks;
  int nx;
  int result_tmp;
  int vstride;
  boolean_T guard1;
  SF = 1.0 / (1.0 / varargin_1);
  s[0] = 0.0;
  s[1] = SF / 2.0;
  fb.init(static_cast<double>(x.size(0)), s, SF);
  if (x.size(0) <= 1024) {
    firstBlockLength = x.size(0);
    lastBlockLength = 0;
    nblocks = 1;
  } else {
    firstBlockLength = 1024;
    nblocks = static_cast<unsigned short>(x.size(0)) >> 10;
    lastBlockLength = x.size(0) - (nblocks << 10);
    if (lastBlockLength > 0) {
      nblocks++;
    } else {
      lastBlockLength = 1024;
    }
  }
  SF = x[0];
  for (int k{2}; k <= firstBlockLength; k++) {
    SF += x[k - 1];
  }
  for (int k{2}; k <= nblocks; k++) {
    firstBlockLength = (k - 1) << 10;
    bsum = x[firstBlockLength];
    if (k == nblocks) {
      hi = lastBlockLength;
    } else {
      hi = 1024;
    }
    for (int b_k{2}; b_k <= hi; b_k++) {
      bsum += x[(firstBlockLength + b_k) - 1];
    }
    SF += bsum;
  }
  SF /= static_cast<double>(x.size(0));
  nx = x.size(0);
  b_x.set_size(1, x.size(0));
  firstBlockLength = (x.size(0) / 2) << 1;
  lastBlockLength = firstBlockLength - 2;
  for (int k{0}; k <= lastBlockLength; k += 2) {
    _mm_storeu_pd(&b_x[k], _mm_sub_pd(_mm_loadu_pd(&x[k]), _mm_set1_pd(SF)));
  }
  for (int k{firstBlockLength}; k < nx; k++) {
    b_x[k] = x[k] - SF;
  }
  fft(b_x, xdft);
  if (xdft.size(1) == fb.PsiDFT.size(1)) {
    firstBlockLength = fb.PsiDFT.size(0);
    hi = xdft.size(1);
    b_xdft.set_size(firstBlockLength, xdft.size(1));
    for (int k{0}; k < hi; k++) {
      for (int b_k{0}; b_k < firstBlockLength; b_k++) {
        lastBlockLength = b_k + fb.PsiDFT.size(0) * k;
        nblocks = b_k + b_xdft.size(0) * k;
        b_xdft[nblocks].re = fb.PsiDFT[lastBlockLength] * xdft[k].re;
        b_xdft[nblocks].im = fb.PsiDFT[lastBlockLength] * xdft[k].im;
      }
    }
    ifft(b_xdft, padcwtcfs);
  } else {
    binary_expand_op_13(padcwtcfs, xdft, fb);
  }
  if (fb.Omega.size(1) == 1) {
    firstBlockLength = fb.PsiDFT.size(1);
  } else {
    firstBlockLength = fb.Omega.size(1);
  }
  if ((fb.Omega.size(1) == fb.PsiDFT.size(1)) &&
      (firstBlockLength == xdft.size(1))) {
    hi = fb.PsiDFT.size(0);
    lastBlockLength = fb.Omega.size(1);
    b_xdft.set_size(hi, lastBlockLength);
    for (int k{0}; k < lastBlockLength; k++) {
      for (int b_k{0}; b_k < hi; b_k++) {
        SF = fb.Omega[k] * 0.0;
        bsum = fb.Omega[k];
        firstBlockLength = b_k + fb.PsiDFT.size(0) * k;
        log2Fund = fb.PsiDFT[firstBlockLength] * SF;
        SF = fb.PsiDFT[firstBlockLength] * bsum;
        bsum = xdft[k].im;
        bi = xdft[k].re;
        firstBlockLength = b_k + b_xdft.size(0) * k;
        b_xdft[firstBlockLength].re = log2Fund * bi - SF * bsum;
        b_xdft[firstBlockLength].im = log2Fund * bsum + SF * bi;
      }
    }
    ifft(b_xdft, paddcwtcfs);
  } else {
    binary_expand_op_12(paddcwtcfs, fb, xdft);
  }
  hi = padcwtcfs.size(0);
  cwtcfs.set_size(padcwtcfs.size(0), x.size(0));
  for (int k{0}; k < nx; k++) {
    for (int b_k{0}; b_k < hi; b_k++) {
      firstBlockLength = b_k + padcwtcfs.size(0) * k;
      SF = padcwtcfs[firstBlockLength].re;
      bsum = padcwtcfs[firstBlockLength].im;
      if (bsum == 0.0) {
        firstBlockLength = b_k + cwtcfs.size(0) * k;
        cwtcfs[firstBlockLength].re = SF / 2.0;
        cwtcfs[firstBlockLength].im = 0.0;
      } else if (SF == 0.0) {
        firstBlockLength = b_k + cwtcfs.size(0) * k;
        cwtcfs[firstBlockLength].re = 0.0;
        cwtcfs[firstBlockLength].im = bsum / 2.0;
      } else {
        firstBlockLength = b_k + cwtcfs.size(0) * k;
        cwtcfs[firstBlockLength].re = SF / 2.0;
        cwtcfs[firstBlockLength].im = bsum / 2.0;
      }
    }
  }
  flip(cwtcfs);
  loop_ub = fb.WaveletCenterFrequencies.size(0);
  f.set_size(loop_ub);
  for (int k{0}; k < loop_ub; k++) {
    f[k] = fb.WaveletCenterFrequencies[k];
  }
  lastBlockLength = 2;
  if (f.size(0) != 1) {
    lastBlockLength = 1;
  }
  if (f.size(0) != 0) {
    if (lastBlockLength <= 1) {
      firstBlockLength = f.size(0);
    } else {
      firstBlockLength = 1;
    }
    if (firstBlockLength > 1) {
      vstride = 1;
      for (int k{0}; k <= lastBlockLength - 2; k++) {
        vstride *= f.size(0);
      }
      if (lastBlockLength <= 1) {
        firstBlockLength = f.size(0) - 1;
      } else {
        firstBlockLength = 0;
      }
      hi = (firstBlockLength + 1) >> 1;
      for (int k{0}; k < vstride; k++) {
        for (int b_k{0}; b_k < hi; b_k++) {
          lastBlockLength = k + b_k * vstride;
          SF = f[lastBlockLength];
          nblocks = k + (firstBlockLength - b_k) * vstride;
          f[lastBlockLength] = f[nblocks];
          f[nblocks] = SF;
        }
      }
    }
  }
  ftmp.set_size(loop_ub);
  firstBlockLength = (f.size(0) / 2) << 1;
  lastBlockLength = firstBlockLength - 2;
  for (int k{0}; k <= lastBlockLength; k += 2) {
    r = _mm_loadu_pd(&f[k]);
    r = _mm_div_pd(r, _mm_set1_pd(fb.SamplingFrequency));
    _mm_storeu_pd(&ftmp[k], r);
  }
  for (int k{firstBlockLength}; k < loop_ub; k++) {
    ftmp[k] = f[k] / fb.SamplingFrequency;
  }
  hi = paddcwtcfs.size(0);
  c_x.set_size(paddcwtcfs.size(0), x.size(0));
  for (int k{0}; k < nx; k++) {
    for (int b_k{0}; b_k < hi; b_k++) {
      firstBlockLength = b_k + paddcwtcfs.size(0) * k;
      SF = paddcwtcfs[firstBlockLength].re;
      bsum = paddcwtcfs[firstBlockLength].im;
      if (bsum == 0.0) {
        firstBlockLength = b_k + c_x.size(0) * k;
        c_x[firstBlockLength].re = SF / 2.0;
        c_x[firstBlockLength].im = 0.0;
      } else if (SF == 0.0) {
        firstBlockLength = b_k + c_x.size(0) * k;
        c_x[firstBlockLength].re = 0.0;
        c_x[firstBlockLength].im = bsum / 2.0;
      } else {
        firstBlockLength = b_k + c_x.size(0) * k;
        c_x[firstBlockLength].re = SF / 2.0;
        c_x[firstBlockLength].im = bsum / 2.0;
      }
    }
  }
  flip(c_x);
  if (c_x.size(0) == cwtcfs.size(0)) {
    phasetf.set_size(c_x.size(0), c_x.size(1));
    firstBlockLength = c_x.size(0) * c_x.size(1);
    if (static_cast<int>(firstBlockLength < 800)) {
      for (int i{0}; i < firstBlockLength; i++) {
        double ai;
        double ar;
        ar = c_x[i].re;
        ai = c_x[i].im;
        bsum = cwtcfs[i].re;
        bi = cwtcfs[i].im;
        if (bi == 0.0) {
          if (ai == 0.0) {
            SF = 0.0;
          } else {
            SF = ai / bsum;
          }
        } else if (bsum == 0.0) {
          if (ar == 0.0) {
            SF = 0.0;
          } else {
            SF = -(ar / bi);
          }
        } else {
          log2Fund = std::abs(bsum);
          SF = std::abs(bi);
          if (log2Fund > SF) {
            SF = bi / bsum;
            SF = (ai - SF * ar) / (bsum + SF * bi);
          } else if (SF == log2Fund) {
            if (bsum > 0.0) {
              bsum = 0.5;
            } else {
              bsum = -0.5;
            }
            if (bi > 0.0) {
              SF = 0.5;
            } else {
              SF = -0.5;
            }
            SF = (ai * bsum - ar * SF) / log2Fund;
          } else {
            SF = bsum / bi;
            SF = (SF * ai - ar) / (bi + SF * bsum);
          }
        }
        phasetf[i] = SF / 6.2831853071795862;
      }
    } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        b_ar, b_ai, br, b_bi, x_im, brm)

      for (int i = 0; i < firstBlockLength; i++) {
        b_ar = c_x[i].re;
        b_ai = c_x[i].im;
        br = cwtcfs[i].re;
        b_bi = cwtcfs[i].im;
        if (b_bi == 0.0) {
          if (b_ai == 0.0) {
            x_im = 0.0;
          } else {
            x_im = b_ai / br;
          }
        } else if (br == 0.0) {
          if (b_ar == 0.0) {
            x_im = 0.0;
          } else {
            x_im = -(b_ar / b_bi);
          }
        } else {
          brm = std::abs(br);
          x_im = std::abs(b_bi);
          if (brm > x_im) {
            x_im = b_bi / br;
            x_im = (b_ai - x_im * b_ar) / (br + x_im * b_bi);
          } else if (x_im == brm) {
            if (br > 0.0) {
              br = 0.5;
            } else {
              br = -0.5;
            }
            if (b_bi > 0.0) {
              x_im = 0.5;
            } else {
              x_im = -0.5;
            }
            x_im = (b_ai * br - b_ar * x_im) / brm;
          } else {
            x_im = br / b_bi;
            x_im = (x_im * b_ai - b_ar) / (b_bi + x_im * br);
          }
        }
        phasetf[i] = x_im / 6.2831853071795862;
      }
    }
  } else {
    binary_expand_op_11(phasetf, c_x, cwtcfs);
  }
  nx = phasetf.size(0) * phasetf.size(1);
  nblocks = (nx < 800);
  if (nblocks) {
    for (int b_i{0}; b_i < nx; b_i++) {
      if ((phasetf[b_i] < 0.0) ||
          (std::isinf(phasetf[b_i]) || std::isnan(phasetf[b_i]))) {
        phasetf[b_i] = 0.0;
      }
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

    for (int b_i = 0; b_i < nx; b_i++) {
      if ((phasetf[b_i] < 0.0) ||
          (std::isinf(phasetf[b_i]) || std::isnan(phasetf[b_i]))) {
        phasetf[b_i] = 0.0;
      }
    }
  }
  lastBlockLength = ftmp.size(0);
  if (ftmp.size(0) <= 2) {
    if (ftmp.size(0) == 1) {
      bsum = ftmp[0];
    } else {
      bsum = ftmp[ftmp.size(0) - 1];
      if ((!(ftmp[0] > bsum)) && ((!std::isnan(ftmp[0])) || std::isnan(bsum))) {
        bsum = ftmp[0];
      }
    }
  } else {
    if (!std::isnan(ftmp[0])) {
      firstBlockLength = 1;
    } else {
      boolean_T exitg1;
      firstBlockLength = 0;
      hi = 2;
      exitg1 = false;
      while ((!exitg1) && (hi <= lastBlockLength)) {
        if (!std::isnan(ftmp[hi - 1])) {
          firstBlockLength = hi;
          exitg1 = true;
        } else {
          hi++;
        }
      }
    }
    if (firstBlockLength == 0) {
      bsum = ftmp[0];
    } else {
      bsum = ftmp[firstBlockLength - 1];
      firstBlockLength++;
      for (int k{firstBlockLength}; k <= lastBlockLength; k++) {
        SF = ftmp[k - 1];
        if (bsum > SF) {
          bsum = SF;
        }
      }
    }
  }
  bi = fb.VoicesPerOctave;
  log2Fund = internal::scalar::scalar_real_log2(bsum);
  c_x.set_size(phasetf.size(0), phasetf.size(1));
  for (int k{0}; k < nx; k++) {
    c_x[k].re = phasetf[k];
    c_x[k].im = 0.0;
  }
  vstride = c_x.size(0);
  loop_ub = c_x.size(1);
  b_f.set_size(c_x.size(0), c_x.size(1));
  if (nblocks) {
    for (int c_k{0}; c_k < nx; c_k++) {
      if (c_x[c_k].im == 0.0) {
        if (c_x[c_k].re < 0.0) {
          b_f[c_k].re =
              internal::scalar::scalar_real_log2(std::abs(c_x[c_k].re));
          b_f[c_k].im = 4.5323601418271942;
        } else {
          b_f[c_k].re =
              internal::scalar::scalar_real_log2(std::abs(c_x[c_k].re));
          b_f[c_k].im = 0.0;
        }
      } else {
        bsum = c_x[c_k].re;
        b_a = std::abs(bsum);
        guard1 = false;
        if (b_a > 8.9884656743115785E+307) {
          guard1 = true;
        } else {
          SF = c_x[c_k].im;
          a = std::abs(SF);
          if (a > 8.9884656743115785E+307) {
            guard1 = true;
          } else {
            if (b_a < a) {
              c_a = b_a / a;
              a *= std::sqrt(c_a * c_a + 1.0);
            } else if (b_a > a) {
              a /= b_a;
              a = b_a * std::sqrt(a * a + 1.0);
            } else {
              a = b_a * 1.4142135623730951;
            }
            b_f[c_k].re = internal::scalar::scalar_real_log2(a);
            b_f[c_k].im =
                internal::scalar::b_atan2(SF, bsum) / 0.69314718055994529;
          }
        }
        if (guard1) {
          a = std::abs(bsum / 2.0);
          SF = c_x[c_k].im;
          b_a = std::abs(SF / 2.0);
          if (a < b_a) {
            a /= b_a;
            a = b_a * std::sqrt(a * a + 1.0);
          } else if (a > b_a) {
            b_a /= a;
            a *= std::sqrt(b_a * b_a + 1.0);
          } else {
            a *= 1.4142135623730951;
          }
          b_f[c_k].re = internal::scalar::scalar_real_log2(a) + 1.0;
          b_f[c_k].im =
              internal::scalar::b_atan2(SF, bsum) / 0.69314718055994529;
        }
      }
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        a, b_a, c_a, a_tmp, guard1, b_tmp)

    for (int c_k = 0; c_k < nx; c_k++) {
      if (c_x[c_k].im == 0.0) {
        if (c_x[c_k].re < 0.0) {
          b_f[c_k].re =
              internal::scalar::scalar_real_log2(std::abs(c_x[c_k].re));
          b_f[c_k].im = 4.5323601418271942;
        } else {
          b_f[c_k].re =
              internal::scalar::scalar_real_log2(std::abs(c_x[c_k].re));
          b_f[c_k].im = 0.0;
        }
      } else {
        a_tmp = c_x[c_k].re;
        b_a = std::abs(a_tmp);
        guard1 = false;
        if (b_a > 8.9884656743115785E+307) {
          guard1 = true;
        } else {
          b_tmp = c_x[c_k].im;
          a = std::abs(b_tmp);
          if (a > 8.9884656743115785E+307) {
            guard1 = true;
          } else {
            if (b_a < a) {
              c_a = b_a / a;
              a *= std::sqrt(c_a * c_a + 1.0);
            } else if (b_a > a) {
              a /= b_a;
              a = b_a * std::sqrt(a * a + 1.0);
            } else {
              a = b_a * 1.4142135623730951;
            }
            b_f[c_k].re = internal::scalar::scalar_real_log2(a);
            b_f[c_k].im =
                internal::scalar::b_atan2(b_tmp, a_tmp) / 0.69314718055994529;
          }
        }
        if (guard1) {
          a = std::abs(a_tmp / 2.0);
          c_a = c_x[c_k].im;
          b_a = std::abs(c_a / 2.0);
          if (a < b_a) {
            a /= b_a;
            a = b_a * std::sqrt(a * a + 1.0);
          } else if (a > b_a) {
            b_a /= a;
            a *= std::sqrt(b_a * b_a + 1.0);
          } else {
            a *= 1.4142135623730951;
          }
          b_f[c_k].re = internal::scalar::scalar_real_log2(a) + 1.0;
          b_f[c_k].im =
              internal::scalar::b_atan2(c_a, a_tmp) / 0.69314718055994529;
        }
      }
    }
  }
  lastBlockLength = b_f.size(0) * b_f.size(1);
  hi = (lastBlockLength < 800);
  if (hi) {
    for (int i1{0}; i1 < lastBlockLength; i1++) {
      b_f[i1].re = bi * (b_f[i1].re - log2Fund);
      b_f[i1].im = bi * b_f[i1].im;
    }
    for (int d_k{0}; d_k < lastBlockLength; d_k++) {
      b_u = b_f[d_k].re;
      u = b_f[d_k].im;
      if (std::abs(b_u) < 4.503599627370496E+15) {
        if (b_u >= 0.5) {
          b_f[d_k].re = std::floor(b_u + 0.5);
        } else if (b_u > -0.5) {
          b_f[d_k].re = b_u * 0.0;
        } else {
          b_f[d_k].re = std::ceil(b_u - 0.5);
        }
      } else {
        b_f[d_k].re = b_u;
      }
      if (std::abs(u) < 4.503599627370496E+15) {
        if (u >= 0.5) {
          b_f[d_k].im = std::floor(u + 0.5);
        } else if (u > -0.5) {
          b_f[d_k].im = u * 0.0;
        } else {
          b_f[d_k].im = std::ceil(u - 0.5);
        }
      } else {
        b_f[d_k].im = u;
      }
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

    for (int i1 = 0; i1 < lastBlockLength; i1++) {
      b_f[i1].re = bi * (b_f[i1].re - log2Fund);
      b_f[i1].im = bi * b_f[i1].im;
    }
#pragma omp parallel for num_threads(omp_get_max_threads()) private(u, b_u)

    for (int d_k = 0; d_k < lastBlockLength; d_k++) {
      b_u = b_f[d_k].re;
      if (std::abs(b_u) < 4.503599627370496E+15) {
        if (b_u >= 0.5) {
          b_u = std::floor(b_u + 0.5);
        } else if (b_u > -0.5) {
          b_u *= 0.0;
        } else {
          b_u = std::ceil(b_u - 0.5);
        }
      }
      u = b_f[d_k].im;
      if (std::abs(u) < 4.503599627370496E+15) {
        if (u >= 0.5) {
          u = std::floor(u + 0.5);
        } else if (u > -0.5) {
          u *= 0.0;
        } else {
          u = std::ceil(u - 0.5);
        }
      }
      b_f[d_k].re = b_u;
      b_f[d_k].im = u;
    }
  }
  iRow.set_size(vstride, loop_ub);
  firstBlockLength = cwtcfs.size(0);
  if (hi) {
    for (int i2{0}; i2 < lastBlockLength; i2++) {
      SF = b_f[i2].re + 1.0;
      iRow[i2] =
          std::fmin(std::fmax(SF, 0.0), static_cast<double>(firstBlockLength));
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        b_varargin_1)

    for (int i2 = 0; i2 < lastBlockLength; i2++) {
      b_varargin_1 = b_f[i2].re + 1.0;
      iRow[i2] = std::fmin(std::fmax(b_varargin_1, 0.0),
                           static_cast<double>(firstBlockLength));
    }
  }
  filled.set_size(vstride, loop_ub);
  if (hi) {
    for (int i3{0}; i3 < lastBlockLength; i3++) {
      filled[i3] = ((iRow[i3] > 0.0) && (iRow[i3] <= firstBlockLength) &&
                    ((!std::isinf(iRow[i3])) && (!std::isnan(iRow[i3]))));
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

    for (int i3 = 0; i3 < lastBlockLength; i3++) {
      filled[i3] = ((iRow[i3] > 0.0) && (iRow[i3] <= firstBlockLength) &&
                    ((!std::isinf(iRow[i3])) && (!std::isnan(iRow[i3]))));
    }
  }
  eml_find(filled, ia);
  b_abs(cwtcfs, phasetf);
  filled.set_size(phasetf.size(0), phasetf.size(1));
  firstBlockLength = phasetf.size(0) * phasetf.size(1);
  if (static_cast<int>(firstBlockLength < 800)) {
    for (int i4{0}; i4 < firstBlockLength; i4++) {
      filled[i4] = (phasetf[i4] > 1.4901161193847656E-8);
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

    for (int i4 = 0; i4 < firstBlockLength; i4++) {
      filled[i4] = (phasetf[i4] > 1.4901161193847656E-8);
    }
  }
  eml_find(filled, ib);
  firstBlockLength = ia.size(0);
  r1.set_size(ia.size(0));
  for (int k{0}; k < firstBlockLength; k++) {
    r1[k] = ia[k];
  }
  firstBlockLength = ib.size(0);
  r2.set_size(ib.size(0));
  for (int k{0}; k < firstBlockLength; k++) {
    r2[k] = ib[k];
  }
  do_vectors(r1, r2, ftmp, ia, ib);
  if (cwtcfs.size(1) < 1) {
    fb.Omega.set_size(1, 0);
  } else {
    fb.Omega.set_size(1, cwtcfs.size(1));
    firstBlockLength = cwtcfs.size(1) - 1;
    lastBlockLength = (cwtcfs.size(1) / 2) << 1;
    hi = lastBlockLength - 2;
    for (int k{0}; k <= hi; k += 2) {
      s[0] = k;
      s[1] = k + 1;
      r = _mm_loadu_pd(&s[0]);
      _mm_storeu_pd(&fb.Omega[k], _mm_add_pd(_mm_set1_pd(1.0), r));
    }
    for (int k{lastBlockLength}; k <= firstBlockLength; k++) {
      fb.Omega[k] = static_cast<double>(k) + 1.0;
    }
  }
  firstBlockLength = fb.Omega.size(1);
  nblocks = cwtcfs.size(0);
  phasetf.set_size(cwtcfs.size(0), firstBlockLength);
  for (int k{0}; k < firstBlockLength; k++) {
    lastBlockLength = k * nblocks;
    for (int b_k{0}; b_k < nblocks; b_k++) {
      phasetf[lastBlockLength + b_k] = static_cast<short>(fb.Omega[k]);
    }
  }
  lastBlockLength = ftmp.size(0);
  result.set_size(ftmp.size(0), 2);
  firstBlockLength = ftmp.size(0);
  z.set_size(ftmp.size(0));
  if (static_cast<int>(ftmp.size(0) < 800)) {
    for (int i5{0}; i5 < lastBlockLength; i5++) {
      firstBlockLength = static_cast<int>(ftmp[i5]) - 1;
      result[i5] = iRow[firstBlockLength];
      result[i5 + result.size(0)] = phasetf[firstBlockLength];
      SF = cwtcfs[firstBlockLength].re * 0.69314718055994529;
      bsum = cwtcfs[firstBlockLength].im * 0.69314718055994529;
      if (bsum == 0.0) {
        z[i5].re = SF / bi;
        z[i5].im = 0.0;
      } else if (SF == 0.0) {
        z[i5].re = 0.0;
        z[i5].im = bsum / bi;
      } else {
        z[i5].re = SF / bi;
        z[i5].im = bsum / bi;
      }
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        result_tmp, c_ar, c_ai)

    for (int i5 = 0; i5 < firstBlockLength; i5++) {
      result_tmp = static_cast<int>(ftmp[i5]) - 1;
      result[i5] = iRow[result_tmp];
      result[i5 + result.size(0)] = phasetf[result_tmp];
      c_ar = cwtcfs[result_tmp].re * 0.69314718055994529;
      c_ai = cwtcfs[result_tmp].im * 0.69314718055994529;
      if (c_ai == 0.0) {
        z[i5].re = c_ar / bi;
        z[i5].im = 0.0;
      } else if (c_ar == 0.0) {
        z[i5].re = 0.0;
        z[i5].im = c_ai / bi;
      } else {
        z[i5].re = c_ar / bi;
        z[i5].im = c_ai / bi;
      }
    }
  }
  if (z.size(0) == 1) {
    counts.set_size(nblocks, cwtcfs.size(1));
    hi = nblocks * cwtcfs.size(1);
    if (hi - 1 >= 0) {
      std::memset(&counts[0], 0, static_cast<unsigned int>(hi) * sizeof(int));
    }
    for (int k{0}; k < lastBlockLength; k++) {
      s[0] = result[k];
      s[1] = result[k + result.size(0)];
      firstBlockLength = (static_cast<int>(s[0]) +
                          counts.size(0) * (static_cast<int>(s[1]) - 1)) -
                         1;
      counts[firstBlockLength] = counts[firstBlockLength] + 1;
    }
    sst.set_size(nblocks, cwtcfs.size(1));
    if (static_cast<int>(hi < 800)) {
      for (int e_k{0}; e_k < hi; e_k++) {
        if (counts[e_k] == 0) {
          sst[e_k].re = 0.0;
          sst[e_k].im = 0.0;
        } else {
          firstBlockLength = counts[e_k];
          sst[e_k].re = static_cast<double>(firstBlockLength) * z[0].re;
          sst[e_k].im = static_cast<double>(firstBlockLength) * z[0].im;
        }
      }
    } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(i6)

      for (int e_k = 0; e_k < hi; e_k++) {
        if (counts[e_k] == 0) {
          sst[e_k].re = 0.0;
          sst[e_k].im = 0.0;
        } else {
          i6 = counts[e_k];
          sst[e_k].re = static_cast<double>(i6) * z[0].re;
          sst[e_k].im = static_cast<double>(i6) * z[0].im;
        }
      }
    }
  } else {
    filled.set_size(nblocks, cwtcfs.size(1));
    firstBlockLength = nblocks * cwtcfs.size(1);
    for (int k{0}; k < firstBlockLength; k++) {
      filled[k] = true;
    }
    sst.set_size(nblocks, cwtcfs.size(1));
    for (int k{0}; k < firstBlockLength; k++) {
      sst[k].re = 0.0;
      sst[k].im = 0.0;
    }
    for (int k{0}; k < lastBlockLength; k++) {
      s[0] = result[k];
      s[1] = result[k + result.size(0)];
      firstBlockLength = (static_cast<int>(s[0]) +
                          filled.size(0) * (static_cast<int>(s[1]) - 1)) -
                         1;
      if (filled[firstBlockLength]) {
        filled[firstBlockLength] = false;
        sst[(static_cast<int>(s[0]) +
             sst.size(0) * (static_cast<int>(s[1]) - 1)) -
            1] = z[k];
      } else {
        firstBlockLength = (static_cast<int>(s[0]) +
                            sst.size(0) * (static_cast<int>(s[1]) - 1)) -
                           1;
        sst[firstBlockLength].re = sst[firstBlockLength].re + z[k].re;
        sst[firstBlockLength].im = sst[firstBlockLength].im + z[k].im;
      }
    }
  }
}

} // namespace coder

// End of code generation (wsst.cpp)
