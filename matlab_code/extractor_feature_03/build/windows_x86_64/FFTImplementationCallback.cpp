//
// FFTImplementationCallback.cpp
//
// Code generation for function 'FFTImplementationCallback'
//

// Include files
#include "FFTImplementationCallback.h"
#include "extractAllFeatures_data.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include "omp.h"
#include <algorithm>
#include <cmath>
#include <cstring>

// Function Definitions
namespace coder {
namespace internal {
namespace fft {
void FFTImplementationCallback::b_generate_twiddle_tables(
    int nRows, array<double, 2U> &costab, array<double, 2U> &sintab,
    array<double, 2U> &sintabinv)
{
  array<double, 2U> b_costab;
  array<double, 2U> b_sintab;
  array<double, 2U> b_sintabinv;
  array<double, 2U> costab1q;
  double e;
  int loop_ub;
  int n;
  int n2;
  int nd2;
  e = 6.2831853071795862 / static_cast<double>(nRows);
  n = static_cast<int>((static_cast<unsigned int>(nRows) >> 1) >> 1);
  costab1q.set_size(1, n + 1);
  costab1q[0] = 1.0;
  nd2 = static_cast<int>(static_cast<unsigned int>(n) >> 1);
  loop_ub = static_cast<unsigned short>(nd2);
  if (static_cast<int>(static_cast<unsigned short>(nd2) < 800)) {
    for (int k{0}; k < loop_ub; k++) {
      costab1q[k + 1] = std::cos(e * (static_cast<double>(k) + 1.0));
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

    for (int k = 0; k < loop_ub; k++) {
      costab1q[k + 1] = std::cos(e * (static_cast<double>(k) + 1.0));
    }
  }
  loop_ub = nd2 + 1;
  if (static_cast<int>((n - nd2) - 1 < 800)) {
    for (int b_k{loop_ub}; b_k < n; b_k++) {
      costab1q[b_k] = std::sin(e * static_cast<double>(n - b_k));
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

    for (int b_k = loop_ub; b_k < n; b_k++) {
      costab1q[b_k] = std::sin(e * static_cast<double>(n - b_k));
    }
  }
  costab1q[n] = 0.0;
  n = costab1q.size(1) - 1;
  n2 = (costab1q.size(1) - 1) << 1;
  b_costab.set_size(1, n2 + 1);
  b_sintab.set_size(1, n2 + 1);
  b_costab[0] = 1.0;
  b_sintab[0] = 0.0;
  b_sintabinv.set_size(1, n2 + 1);
  loop_ub = (costab1q.size(1) - 1 < 800);
  if (loop_ub) {
    for (int c_k{0}; c_k < n; c_k++) {
      b_sintabinv[c_k + 1] = costab1q[(n - c_k) - 1];
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

    for (int c_k = 0; c_k < n; c_k++) {
      b_sintabinv[c_k + 1] = costab1q[(n - c_k) - 1];
    }
  }
  nd2 = costab1q.size(1);
  for (int d_k{nd2}; d_k <= n2; d_k++) {
    b_sintabinv[d_k] = costab1q[d_k - n];
  }
  if (loop_ub) {
    for (int e_k{0}; e_k < n; e_k++) {
      b_costab[e_k + 1] = costab1q[e_k + 1];
      b_sintab[e_k + 1] = -costab1q[(n - e_k) - 1];
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

    for (int e_k = 0; e_k < n; e_k++) {
      b_costab[e_k + 1] = costab1q[e_k + 1];
      b_sintab[e_k + 1] = -costab1q[(n - e_k) - 1];
    }
  }
  if (static_cast<int>((n2 - costab1q.size(1)) + 1 < 800)) {
    for (int f_k{nd2}; f_k <= n2; f_k++) {
      b_costab[f_k] = -costab1q[n2 - f_k];
      b_sintab[f_k] = -costab1q[f_k - n];
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

    for (int f_k = nd2; f_k <= n2; f_k++) {
      b_costab[f_k] = -costab1q[n2 - f_k];
      b_sintab[f_k] = -costab1q[f_k - n];
    }
  }
  loop_ub = b_costab.size(1);
  costab.set_size(1, b_costab.size(1));
  for (int d_k{0}; d_k < loop_ub; d_k++) {
    costab[d_k] = b_costab[d_k];
  }
  sintab.set_size(1, b_costab.size(1));
  for (int d_k{0}; d_k < loop_ub; d_k++) {
    sintab[d_k] = b_sintab[d_k];
  }
  sintabinv.set_size(1, b_costab.size(1));
  for (int d_k{0}; d_k < loop_ub; d_k++) {
    sintabinv[d_k] = b_sintabinv[d_k];
  }
}

void FFTImplementationCallback::b_r2br_r2dit_trig_impl(
    const array<creal_T, 1U> &x, int unsigned_nRows,
    const array<double, 2U> &costab, const array<double, 2U> &sintab,
    array<creal_T, 1U> &y)
{
  double im;
  double re;
  double temp_im;
  double temp_re;
  int iDelta;
  int iDelta2;
  int iheight;
  int istart;
  int iy;
  int j;
  int ju;
  int k;
  int nRowsD2;
  y.set_size(unsigned_nRows);
  if (unsigned_nRows > x.size(0)) {
    y.set_size(unsigned_nRows);
    for (int i{0}; i < unsigned_nRows; i++) {
      y[i].re = 0.0;
      y[i].im = 0.0;
    }
  }
  j = x.size(0);
  if (j > unsigned_nRows) {
    j = unsigned_nRows;
  }
  iDelta = unsigned_nRows - 2;
  nRowsD2 = static_cast<int>(static_cast<unsigned int>(unsigned_nRows) >> 1);
  k = static_cast<int>(static_cast<unsigned int>(nRowsD2) >> 1);
  iy = 0;
  ju = 0;
  for (int i{0}; i <= j - 2; i++) {
    boolean_T tst;
    y[iy] = x[i];
    istart = unsigned_nRows;
    tst = true;
    while (tst) {
      istart >>= 1;
      ju ^= istart;
      tst = ((ju & istart) == 0);
    }
    iy = ju;
  }
  if (j - 2 < 0) {
    istart = 0;
  } else {
    istart = j - 1;
  }
  y[iy] = x[istart];
  if (unsigned_nRows > 1) {
    for (int i{0}; i <= iDelta; i += 2) {
      temp_re = y[i + 1].re;
      temp_im = y[i + 1].im;
      re = y[i].re;
      im = y[i].im;
      y[i + 1].re = re - temp_re;
      y[i + 1].im = y[i].im - y[i + 1].im;
      re += temp_re;
      im += temp_im;
      y[i].re = re;
      y[i].im = im;
    }
  }
  iDelta = 2;
  iDelta2 = 4;
  iheight = ((k - 1) << 2) + 1;
  while (k > 0) {
    int b_i;
    for (b_i = 0; b_i < iheight; b_i += iDelta2) {
      istart = b_i + iDelta;
      temp_re = y[istart].re;
      temp_im = y[istart].im;
      y[istart].re = y[b_i].re - temp_re;
      y[istart].im = y[b_i].im - temp_im;
      y[b_i].re = y[b_i].re + temp_re;
      y[b_i].im = y[b_i].im + temp_im;
    }
    istart = 1;
    for (j = k; j < nRowsD2; j += k) {
      double twid_im;
      double twid_re;
      twid_re = costab[j];
      twid_im = sintab[j];
      b_i = istart;
      iy = istart + iheight;
      while (b_i < iy) {
        ju = b_i + iDelta;
        re = y[ju].im;
        im = y[ju].re;
        temp_re = twid_re * im - twid_im * re;
        temp_im = twid_re * re + twid_im * im;
        y[ju].re = y[b_i].re - temp_re;
        y[ju].im = y[b_i].im - temp_im;
        y[b_i].re = y[b_i].re + temp_re;
        y[b_i].im = y[b_i].im + temp_im;
        b_i += iDelta2;
      }
      istart++;
    }
    k = static_cast<int>(static_cast<unsigned int>(k) >> 1);
    iDelta = iDelta2;
    iDelta2 += iDelta2;
    iheight -= iDelta;
  }
}

void FFTImplementationCallback::doHalfLengthBluestein(
    const array<double, 1U> &x, array<creal_T, 1U> &y, int nrowsx, int nRows,
    int nfft, const array<creal_T, 1U> &wwc, const array<double, 2U> &costab,
    const array<double, 2U> &sintab, const array<double, 2U> &costabinv,
    const array<double, 2U> &sintabinv)
{
  array<creal_T, 1U> b_y;
  array<creal_T, 1U> r;
  array<creal_T, 1U> reconVar1;
  array<creal_T, 1U> reconVar2;
  array<creal_T, 1U> ytmp;
  array<double, 2U> a__1;
  array<double, 2U> costable;
  array<double, 2U> hcostab;
  array<double, 2U> hcostabinv;
  array<double, 2U> hsintab;
  array<double, 2U> hsintabinv;
  array<double, 2U> sintable;
  creal_T a;
  creal_T b;
  cuint8_T y_data[16384];
  double b_re_tmp;
  double c_re_tmp;
  double d;
  double d1;
  double d2;
  double d3;
  double d4;
  double d5;
  double d_re_tmp;
  double e_re_tmp;
  double f_re_tmp;
  double g_re_tmp;
  double h_re_tmp;
  double re_tmp;
  int wrapIndex_data[8192];
  int b_b_tmp;
  int b_i;
  int b_tmp;
  int hnRows;
  int i1;
  int i2;
  int i5;
  int minHnrowsNxBy2;
  boolean_T nxeven;
  hnRows = static_cast<int>(static_cast<unsigned int>(nRows) >> 1);
  if (hnRows > nrowsx) {
    if (hnRows - 1 >= 0) {
      std::memset(&y_data[0], 0,
                  static_cast<unsigned int>(hnRows) * sizeof(cuint8_T));
    }
  }
  ytmp.set_size(hnRows);
  for (int i{0}; i < hnRows; i++) {
    ytmp[i].re = 0.0;
    ytmp[i].im = y_data[i].im;
  }
  if ((static_cast<unsigned int>(x.size(0)) & 1U) == 0U) {
    nxeven = true;
    minHnrowsNxBy2 = x.size(0);
  } else if (x.size(0) >= nRows) {
    nxeven = true;
    minHnrowsNxBy2 = nRows;
  } else {
    nxeven = false;
    minHnrowsNxBy2 = x.size(0) - 1;
  }
  if (minHnrowsNxBy2 > nRows) {
    minHnrowsNxBy2 = nRows;
  }
  FFTImplementationCallback::b_generate_twiddle_tables(nRows << 1, costable,
                                                       sintable, a__1);
  FFTImplementationCallback::get_half_twiddle_tables(
      costab, sintab, costabinv, sintabinv, hcostab, hsintab, hcostabinv,
      hsintabinv);
  reconVar1.set_size(hnRows);
  reconVar2.set_size(hnRows);
  b_i = static_cast<unsigned short>(hnRows);
  if (static_cast<int>(static_cast<unsigned short>(hnRows) < 800)) {
    for (int c_i{0}; c_i < b_i; c_i++) {
      b_tmp = c_i << 1;
      d = sintable[b_tmp];
      d1 = costable[b_tmp];
      reconVar1[c_i].re = d + 1.0;
      reconVar1[c_i].im = -d1;
      reconVar2[c_i].re = 1.0 - d;
      reconVar2[c_i].im = d1;
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(d1, d, i2)

    for (int c_i = 0; c_i < b_i; c_i++) {
      i2 = c_i << 1;
      d = sintable[i2];
      d1 = costable[i2];
      reconVar1[c_i].re = d + 1.0;
      reconVar1[c_i].im = -d1;
      reconVar2[c_i].re = 1.0 - d;
      reconVar2[c_i].im = d1;
    }
  }
  for (int i{0}; i < b_i; i++) {
    if (i != 0) {
      wrapIndex_data[i] = (hnRows - i) + 1;
    } else {
      wrapIndex_data[0] = 1;
    }
  }
  minHnrowsNxBy2 =
      static_cast<int>(static_cast<unsigned int>(minHnrowsNxBy2) >> 1);
  i1 = static_cast<unsigned short>(minHnrowsNxBy2);
  if (static_cast<int>(static_cast<unsigned short>(minHnrowsNxBy2) < 800)) {
    for (int k1{0}; k1 < i1; k1++) {
      a = wwc[(hnRows + k1) - 1];
      b_tmp = k1 << 1;
      b.re = x[b_tmp];
      b.im = x[b_tmp + 1];
      ytmp[k1].re = a.re * b.re + a.im * b.im;
      ytmp[k1].im = a.re * b.im - a.im * b.re;
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        b, a, b_b_tmp)

    for (int k1 = 0; k1 < i1; k1++) {
      a = wwc[(hnRows + k1) - 1];
      b_b_tmp = k1 << 1;
      b.re = x[b_b_tmp];
      b.im = x[b_b_tmp + 1];
      ytmp[k1].re = a.re * b.re + a.im * b.im;
      ytmp[k1].im = a.re * b.im - a.im * b.re;
    }
  }
  if (!nxeven) {
    a = wwc[(hnRows + minHnrowsNxBy2) - 1];
    b.re = x[i1 << 1];
    ytmp[minHnrowsNxBy2].re = a.re * b.re + a.im * 0.0;
    ytmp[minHnrowsNxBy2].im = a.re * 0.0 - a.im * b.re;
    if (minHnrowsNxBy2 + 2 <= hnRows) {
      b_tmp = minHnrowsNxBy2 + 2;
      if (b_tmp <= hnRows) {
        std::memset(&ytmp[b_tmp + -1], 0,
                    static_cast<unsigned int>((hnRows - b_tmp) + 1) *
                        sizeof(creal_T));
      }
    }
  } else if (minHnrowsNxBy2 + 1 <= hnRows) {
    b_tmp = minHnrowsNxBy2 + 1;
    if (b_tmp <= hnRows) {
      std::memset(&ytmp[b_tmp + -1], 0,
                  static_cast<unsigned int>((hnRows - b_tmp) + 1) *
                      sizeof(creal_T));
    }
  }
  b_tmp = static_cast<int>(static_cast<unsigned int>(nfft) >> 1);
  FFTImplementationCallback::r2br_r2dit_trig_impl(ytmp, b_tmp, hcostab, hsintab,
                                                  r);
  FFTImplementationCallback::b_r2br_r2dit_trig_impl(wwc, b_tmp, hcostab,
                                                    hsintab, b_y);
  minHnrowsNxBy2 = r.size(0);
  if (static_cast<int>(r.size(0) < 800)) {
    for (int i3{0}; i3 < minHnrowsNxBy2; i3++) {
      re_tmp = r[i3].re;
      b_re_tmp = b_y[i3].im;
      c_re_tmp = r[i3].im;
      e_re_tmp = b_y[i3].re;
      r[i3].re = re_tmp * e_re_tmp - c_re_tmp * b_re_tmp;
      r[i3].im = re_tmp * b_re_tmp + c_re_tmp * e_re_tmp;
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        d_re_tmp, f_re_tmp, g_re_tmp, h_re_tmp)

    for (int i3 = 0; i3 < minHnrowsNxBy2; i3++) {
      d_re_tmp = r[i3].re;
      f_re_tmp = b_y[i3].im;
      g_re_tmp = r[i3].im;
      h_re_tmp = b_y[i3].re;
      r[i3].re = d_re_tmp * h_re_tmp - g_re_tmp * f_re_tmp;
      r[i3].im = d_re_tmp * f_re_tmp + g_re_tmp * h_re_tmp;
    }
  }
  FFTImplementationCallback::b_r2br_r2dit_trig_impl(r, b_tmp, hcostabinv,
                                                    hsintabinv, b_y);
  if (b_y.size(0) > 1) {
    re_tmp = 1.0 / static_cast<double>(b_y.size(0));
    b_tmp = b_y.size(0);
    if (static_cast<int>(b_y.size(0) < 800)) {
      for (int i4{0}; i4 < b_tmp; i4++) {
        b_y[i4].re = re_tmp * b_y[i4].re;
        b_y[i4].im = re_tmp * b_y[i4].im;
      }
    } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

      for (int i4 = 0; i4 < b_tmp; i4++) {
        b_y[i4].re = re_tmp * b_y[i4].re;
        b_y[i4].im = re_tmp * b_y[i4].im;
      }
    }
  }
  minHnrowsNxBy2 = wwc.size(0);
  if (static_cast<int>((wwc.size(0) - hnRows) + 1 < 800)) {
    for (int k{hnRows}; k <= minHnrowsNxBy2; k++) {
      re_tmp = wwc[k - 1].re;
      b_re_tmp = b_y[k - 1].im;
      c_re_tmp = wwc[k - 1].im;
      e_re_tmp = b_y[k - 1].re;
      b_tmp = k - hnRows;
      ytmp[b_tmp].re = re_tmp * e_re_tmp + c_re_tmp * b_re_tmp;
      ytmp[b_tmp].im = re_tmp * b_re_tmp - c_re_tmp * e_re_tmp;
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        d2, d3, d4, d5, i5)

    for (int k = hnRows; k <= minHnrowsNxBy2; k++) {
      d2 = wwc[k - 1].re;
      d3 = b_y[k - 1].im;
      d4 = wwc[k - 1].im;
      d5 = b_y[k - 1].re;
      i5 = k - hnRows;
      ytmp[i5].re = d2 * d5 + d4 * d3;
      ytmp[i5].im = d2 * d3 - d4 * d5;
    }
  }
  for (int i{0}; i < b_i; i++) {
    double b_ytmp_re_tmp;
    double ytmp_im;
    double ytmp_re;
    double ytmp_re_tmp;
    b_tmp = wrapIndex_data[i];
    re_tmp = ytmp[i].re;
    b_re_tmp = reconVar1[i].im;
    c_re_tmp = ytmp[i].im;
    e_re_tmp = reconVar1[i].re;
    ytmp_re = ytmp[b_tmp - 1].re;
    ytmp_im = -ytmp[b_tmp - 1].im;
    ytmp_re_tmp = reconVar2[i].im;
    b_ytmp_re_tmp = reconVar2[i].re;
    y[i].re = 0.5 * ((re_tmp * e_re_tmp - c_re_tmp * b_re_tmp) +
                     (ytmp_re * b_ytmp_re_tmp - ytmp_im * ytmp_re_tmp));
    y[i].im = 0.5 * ((re_tmp * b_re_tmp + c_re_tmp * e_re_tmp) +
                     (ytmp_re * ytmp_re_tmp + ytmp_im * b_ytmp_re_tmp));
    b_tmp = hnRows + i;
    y[b_tmp].re = 0.5 * ((re_tmp * b_ytmp_re_tmp - c_re_tmp * ytmp_re_tmp) +
                         (ytmp_re * e_re_tmp - ytmp_im * b_re_tmp));
    y[b_tmp].im = 0.5 * ((re_tmp * ytmp_re_tmp + c_re_tmp * b_ytmp_re_tmp) +
                         (ytmp_re * b_re_tmp + ytmp_im * e_re_tmp));
  }
}

void FFTImplementationCallback::get_half_twiddle_tables(
    const array<double, 2U> &costab, const array<double, 2U> &sintab,
    const array<double, 2U> &costabinv, const array<double, 2U> &sintabinv,
    array<double, 2U> &hcostab, array<double, 2U> &hsintab,
    array<double, 2U> &hcostabinv, array<double, 2U> &hsintabinv)
{
  int b_i;
  int hszCostab;
  hszCostab = static_cast<unsigned short>(costab.size(1)) >> 1;
  hcostab.set_size(1, hszCostab);
  hsintab.set_size(1, hszCostab);
  hcostabinv.set_size(1, hszCostab);
  hsintabinv.set_size(1, hszCostab);
  if (static_cast<int>(hszCostab < 800)) {
    for (int i{0}; i < hszCostab; i++) {
      b_i = ((i + 1) << 1) - 2;
      hcostab[i] = costab[b_i];
      hsintab[i] = sintab[b_i];
      hcostabinv[i] = costabinv[b_i];
      hsintabinv[i] = sintabinv[b_i];
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(b_i)

    for (int i = 0; i < hszCostab; i++) {
      b_i = ((i + 1) << 1) - 2;
      hcostab[i] = costab[b_i];
      hsintab[i] = sintab[b_i];
      hcostabinv[i] = costabinv[b_i];
      hsintabinv[i] = sintabinv[b_i];
    }
  }
}

int FFTImplementationCallback::r2br_r2dit_trig(const creal32_T x_data[],
                                               int x_size, int n1_unsigned,
                                               const float costab_data[],
                                               const float sintab_data[],
                                               creal32_T y_data[])
{
  float re;
  float temp_im;
  float temp_re;
  float temp_re_tmp;
  float twid_re;
  int iDelta;
  int iDelta2;
  int iheight;
  int ihi;
  int istart;
  int iy;
  int ju;
  int k;
  int nRowsD2;
  int y_size;
  y_size = n1_unsigned;
  if (n1_unsigned > x_size) {
    std::memset(&y_data[0], 0,
                static_cast<unsigned int>(n1_unsigned) * sizeof(creal32_T));
  }
  iDelta = n1_unsigned - 2;
  nRowsD2 = static_cast<int>(static_cast<unsigned int>(n1_unsigned) >> 1);
  k = static_cast<int>(static_cast<unsigned int>(nRowsD2) >> 1);
  iy = 0;
  ju = 0;
  if (x_size <= n1_unsigned) {
    istart = x_size;
  } else {
    istart = n1_unsigned;
  }
  ihi = static_cast<unsigned short>(istart - 1);
  for (int i{0}; i < ihi; i++) {
    boolean_T tst;
    y_data[iy] = x_data[i];
    istart = n1_unsigned;
    tst = true;
    while (tst) {
      istart >>= 1;
      ju ^= istart;
      tst = ((ju & istart) == 0);
    }
    iy = ju;
  }
  if (ihi - 1 < 0) {
    ihi = 0;
  }
  y_data[iy] = x_data[ihi];
  if (n1_unsigned > 1) {
    for (int i{0}; i <= iDelta; i += 2) {
      temp_re = y_data[i + 1].re;
      temp_re_tmp = y_data[i + 1].im;
      temp_im = temp_re_tmp;
      re = y_data[i].re;
      twid_re = y_data[i].im;
      y_data[i + 1].re = re - temp_re;
      temp_re_tmp = twid_re - temp_re_tmp;
      y_data[i + 1].im = temp_re_tmp;
      re += temp_re;
      y_data[i].re = re;
      y_data[i].im = twid_re + temp_im;
    }
  }
  iDelta = 2;
  iDelta2 = 4;
  iheight = ((k - 1) << 2) + 1;
  while (k > 0) {
    int b_i;
    for (b_i = 0; b_i < iheight; b_i += iDelta2) {
      istart = b_i + iDelta;
      temp_re = y_data[istart].re;
      temp_im = y_data[istart].im;
      y_data[istart].re = y_data[b_i].re - temp_re;
      y_data[istart].im = y_data[b_i].im - temp_im;
      y_data[b_i].re += temp_re;
      y_data[b_i].im += temp_im;
    }
    istart = 1;
    for (iy = k; iy < nRowsD2; iy += k) {
      float twid_im;
      twid_re = costab_data[iy];
      twid_im = sintab_data[iy];
      b_i = istart;
      ihi = istart + iheight;
      while (b_i < ihi) {
        ju = b_i + iDelta;
        temp_re_tmp = y_data[ju].im;
        re = y_data[ju].re;
        temp_re = twid_re * re - twid_im * temp_re_tmp;
        temp_im = twid_re * temp_re_tmp + twid_im * re;
        y_data[ju].re = y_data[b_i].re - temp_re;
        y_data[ju].im = y_data[b_i].im - temp_im;
        y_data[b_i].re += temp_re;
        y_data[b_i].im += temp_im;
        b_i += iDelta2;
      }
      istart++;
    }
    k = static_cast<int>(static_cast<unsigned int>(k) >> 1);
    iDelta = iDelta2;
    iDelta2 += iDelta2;
    iheight -= iDelta;
  }
  if (n1_unsigned > 1) {
    temp_re_tmp = 1.0F / static_cast<float>(n1_unsigned);
    if (static_cast<int>(n1_unsigned < 800)) {
      for (int c_i{0}; c_i < n1_unsigned; c_i++) {
        y_data[c_i].re *= temp_re_tmp;
        y_data[c_i].im *= temp_re_tmp;
      }
    } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

      for (int c_i = 0; c_i < n1_unsigned; c_i++) {
        y_data[c_i].re *= temp_re_tmp;
        y_data[c_i].im *= temp_re_tmp;
      }
    }
  }
  return y_size;
}

void FFTImplementationCallback::r2br_r2dit_trig_impl(
    const array<creal_T, 1U> &x, int unsigned_nRows,
    const array<double, 2U> &costab, const array<double, 2U> &sintab,
    array<creal_T, 1U> &y)
{
  array<creal_T, 1U> b_y;
  double im;
  double re;
  double temp_im;
  double temp_re;
  int iDelta;
  int iDelta2;
  int iheight;
  int istart;
  int iy;
  int j;
  int ju;
  int k;
  int loop_ub;
  int nRowsD2;
  b_y.set_size(unsigned_nRows);
  if (unsigned_nRows > x.size(0)) {
    b_y.set_size(unsigned_nRows);
    std::memset(&b_y[0], 0,
                static_cast<unsigned int>(unsigned_nRows) * sizeof(creal_T));
  }
  loop_ub = b_y.size(0);
  y.set_size(b_y.size(0));
  for (int i{0}; i < loop_ub; i++) {
    y[i] = b_y[i];
  }
  j = x.size(0);
  if (j > unsigned_nRows) {
    j = unsigned_nRows;
  }
  iDelta = unsigned_nRows - 2;
  nRowsD2 = static_cast<int>(static_cast<unsigned int>(unsigned_nRows) >> 1);
  k = static_cast<int>(static_cast<unsigned int>(nRowsD2) >> 1);
  iy = 0;
  ju = 0;
  for (int i{0}; i <= j - 2; i++) {
    boolean_T tst;
    y[iy] = x[i];
    istart = unsigned_nRows;
    tst = true;
    while (tst) {
      istart >>= 1;
      ju ^= istart;
      tst = ((ju & istart) == 0);
    }
    iy = ju;
  }
  if (j - 2 < 0) {
    istart = 0;
  } else {
    istart = j - 1;
  }
  y[iy] = x[istart];
  b_y.set_size(b_y.size(0));
  for (int i{0}; i < loop_ub; i++) {
    b_y[i] = y[i];
  }
  if (unsigned_nRows > 1) {
    for (int i{0}; i <= iDelta; i += 2) {
      temp_re = b_y[i + 1].re;
      temp_im = b_y[i + 1].im;
      re = b_y[i].re;
      im = b_y[i].im;
      b_y[i + 1].re = re - temp_re;
      b_y[i + 1].im = b_y[i].im - b_y[i + 1].im;
      re += temp_re;
      im += temp_im;
      b_y[i].re = re;
      b_y[i].im = im;
    }
  }
  iDelta = 2;
  iDelta2 = 4;
  iheight = ((k - 1) << 2) + 1;
  while (k > 0) {
    int b_i;
    for (b_i = 0; b_i < iheight; b_i += iDelta2) {
      istart = b_i + iDelta;
      temp_re = b_y[istart].re;
      temp_im = b_y[istart].im;
      b_y[istart].re = b_y[b_i].re - temp_re;
      b_y[istart].im = b_y[b_i].im - temp_im;
      b_y[b_i].re = b_y[b_i].re + temp_re;
      b_y[b_i].im = b_y[b_i].im + temp_im;
    }
    istart = 1;
    for (j = k; j < nRowsD2; j += k) {
      double twid_im;
      double twid_re;
      twid_re = costab[j];
      twid_im = sintab[j];
      b_i = istart;
      iy = istart + iheight;
      while (b_i < iy) {
        ju = b_i + iDelta;
        re = b_y[ju].im;
        im = b_y[ju].re;
        temp_re = twid_re * im - twid_im * re;
        temp_im = twid_re * re + twid_im * im;
        b_y[ju].re = b_y[b_i].re - temp_re;
        b_y[ju].im = b_y[b_i].im - temp_im;
        b_y[b_i].re = b_y[b_i].re + temp_re;
        b_y[b_i].im = b_y[b_i].im + temp_im;
        b_i += iDelta2;
      }
      istart++;
    }
    k = static_cast<int>(static_cast<unsigned int>(k) >> 1);
    iDelta = iDelta2;
    iDelta2 += iDelta2;
    iheight -= iDelta;
  }
  y.set_size(loop_ub);
  for (int i{0}; i < loop_ub; i++) {
    y[i] = b_y[i];
  }
}

void FFTImplementationCallback::doHalfLengthRadix2(
    const array<double, 1U> &x, array<creal_T, 1U> &y, int unsigned_nRows,
    const array<double, 2U> &costab, const array<double, 2U> &sintab)
{
  array<creal_T, 1U> b_y;
  array<creal_T, 1U> reconVar1;
  array<creal_T, 1U> reconVar2;
  array<double, 2U> hcostab;
  array<double, 2U> hsintab;
  double d;
  double d1;
  double im;
  double re;
  double temp2_im;
  double temp2_re;
  double temp_im;
  double temp_re;
  double y_im_tmp;
  double y_re_tmp;
  int bitrevIndex_data[8192];
  int wrapIndex_data[8192];
  int b_i;
  int hszCostab;
  int iDelta;
  int iDelta2;
  int iheight;
  int j;
  int ju;
  int k;
  int loop_ub;
  int nRows;
  int nRowsD2;
  int temp_re_tmp;
  boolean_T tst;
  nRows = static_cast<int>(static_cast<unsigned int>(unsigned_nRows) >> 1);
  j = y.size(0);
  if (j > nRows) {
    j = nRows;
  }
  iDelta = nRows - 2;
  nRowsD2 = static_cast<int>(static_cast<unsigned int>(nRows) >> 1);
  k = static_cast<int>(static_cast<unsigned int>(nRowsD2) >> 1);
  hszCostab = static_cast<unsigned short>(costab.size(1)) >> 1;
  hcostab.set_size(1, hszCostab);
  hsintab.set_size(1, hszCostab);
  if (static_cast<int>(hszCostab < 800)) {
    for (int i{0}; i < hszCostab; i++) {
      b_i = ((i + 1) << 1) - 2;
      hcostab[i] = costab[b_i];
      hsintab[i] = sintab[b_i];
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(b_i)

    for (int i = 0; i < hszCostab; i++) {
      b_i = ((i + 1) << 1) - 2;
      hcostab[i] = costab[b_i];
      hsintab[i] = sintab[b_i];
    }
  }
  reconVar1.set_size(nRows);
  reconVar2.set_size(nRows);
  ju = static_cast<unsigned short>(nRows);
  if (static_cast<int>(static_cast<unsigned short>(nRows) < 800)) {
    for (int c_i{0}; c_i < ju; c_i++) {
      d = sintab[c_i];
      d1 = costab[c_i];
      reconVar1[c_i].re = d + 1.0;
      reconVar1[c_i].im = -d1;
      reconVar2[c_i].re = 1.0 - d;
      reconVar2[c_i].im = d1;
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(d1, d)

    for (int c_i = 0; c_i < ju; c_i++) {
      d = sintab[c_i];
      d1 = costab[c_i];
      reconVar1[c_i].re = d + 1.0;
      reconVar1[c_i].im = -d1;
      reconVar2[c_i].re = 1.0 - d;
      reconVar2[c_i].im = d1;
    }
  }
  for (int d_i{0}; d_i < ju; d_i++) {
    if (d_i != 0) {
      wrapIndex_data[d_i] = (nRows - d_i) + 1;
    } else {
      wrapIndex_data[0] = 1;
    }
  }
  ju = 0;
  hszCostab = 1;
  if (nRows - 1 >= 0) {
    std::memset(&bitrevIndex_data[0], 0,
                static_cast<unsigned int>(nRows) * sizeof(int));
  }
  temp_re_tmp = static_cast<unsigned short>(j - 1);
  for (int d_i{0}; d_i < temp_re_tmp; d_i++) {
    bitrevIndex_data[d_i] = hszCostab;
    hszCostab = nRows;
    tst = true;
    while (tst) {
      hszCostab >>= 1;
      ju = static_cast<int>(static_cast<unsigned int>(ju) ^
                            static_cast<unsigned int>(hszCostab));
      tst = ((static_cast<unsigned int>(ju) &
              static_cast<unsigned int>(hszCostab)) == 0U);
    }
    hszCostab = ju + 1;
  }
  bitrevIndex_data[j - 1] = hszCostab;
  if ((static_cast<unsigned int>(x.size(0)) & 1U) == 0U) {
    tst = true;
    hszCostab = x.size(0);
  } else if (x.size(0) >= unsigned_nRows) {
    tst = true;
    hszCostab = unsigned_nRows;
  } else {
    tst = false;
    hszCostab = x.size(0) - 1;
  }
  if (hszCostab > unsigned_nRows) {
    hszCostab = unsigned_nRows;
  }
  hszCostab = static_cast<int>(static_cast<unsigned int>(hszCostab) >> 1);
  ju = static_cast<unsigned short>(hszCostab);
  for (int d_i{0}; d_i < ju; d_i++) {
    temp_re_tmp = d_i << 1;
    j = bitrevIndex_data[d_i];
    y[j - 1].re = x[temp_re_tmp];
    y[j - 1].im = x[temp_re_tmp + 1];
  }
  if (!tst) {
    y[bitrevIndex_data[hszCostab] - 1].re =
        x[static_cast<unsigned short>(hszCostab) << 1];
    y[bitrevIndex_data[hszCostab] - 1].im = 0.0;
  }
  loop_ub = y.size(0);
  b_y.set_size(loop_ub);
  for (int d_i{0}; d_i < loop_ub; d_i++) {
    b_y[d_i] = y[d_i];
  }
  if (nRows > 1) {
    for (int d_i{0}; d_i <= iDelta; d_i += 2) {
      temp_re = b_y[d_i + 1].re;
      temp_im = b_y[d_i + 1].im;
      re = b_y[d_i].re;
      im = b_y[d_i].im;
      b_y[d_i + 1].re = re - temp_re;
      b_y[d_i + 1].im = b_y[d_i].im - b_y[d_i + 1].im;
      re += temp_re;
      im += temp_im;
      b_y[d_i].re = re;
      b_y[d_i].im = im;
    }
  }
  iDelta = 2;
  iDelta2 = 4;
  iheight = ((k - 1) << 2) + 1;
  while (k > 0) {
    int e_i;
    for (e_i = 0; e_i < iheight; e_i += iDelta2) {
      hszCostab = e_i + iDelta;
      temp_re = b_y[hszCostab].re;
      temp_im = b_y[hszCostab].im;
      b_y[hszCostab].re = b_y[e_i].re - temp_re;
      b_y[hszCostab].im = b_y[e_i].im - temp_im;
      b_y[e_i].re = b_y[e_i].re + temp_re;
      b_y[e_i].im = b_y[e_i].im + temp_im;
    }
    hszCostab = 1;
    for (j = k; j < nRowsD2; j += k) {
      temp2_re = hcostab[j];
      temp2_im = hsintab[j];
      e_i = hszCostab;
      ju = hszCostab + iheight;
      while (e_i < ju) {
        temp_re_tmp = e_i + iDelta;
        re = b_y[temp_re_tmp].im;
        im = b_y[temp_re_tmp].re;
        temp_re = temp2_re * im - temp2_im * re;
        temp_im = temp2_re * re + temp2_im * im;
        b_y[temp_re_tmp].re = b_y[e_i].re - temp_re;
        b_y[temp_re_tmp].im = b_y[e_i].im - temp_im;
        b_y[e_i].re = b_y[e_i].re + temp_re;
        b_y[e_i].im = b_y[e_i].im + temp_im;
        e_i += iDelta2;
      }
      hszCostab++;
    }
    k = static_cast<int>(static_cast<unsigned int>(k) >> 1);
    iDelta = iDelta2;
    iDelta2 += iDelta2;
    iheight -= iDelta;
  }
  for (int d_i{0}; d_i < loop_ub; d_i++) {
    y[d_i] = b_y[d_i];
  }
  re = b_y[0].re * reconVar1[0].re;
  im = b_y[0].re * reconVar1[0].im;
  y_re_tmp = b_y[0].re * reconVar2[0].re;
  y_im_tmp = b_y[0].re * reconVar2[0].im;
  y[0].re = 0.5 * ((re - b_y[0].im * reconVar1[0].im) +
                   (y_re_tmp - -b_y[0].im * reconVar2[0].im));
  y[0].im = 0.5 * ((im + b_y[0].im * reconVar1[0].re) +
                   (y_im_tmp + -b_y[0].im * reconVar2[0].re));
  y[nRows].re = 0.5 * ((y_re_tmp - b_y[0].im * reconVar2[0].im) +
                       (re - -b_y[0].im * reconVar1[0].im));
  y[nRows].im = 0.5 * ((y_im_tmp + b_y[0].im * reconVar2[0].re) +
                       (im + -b_y[0].im * reconVar1[0].re));
  for (int d_i{2}; d_i <= nRowsD2; d_i++) {
    temp_re = y[d_i - 1].re;
    temp_im = y[d_i - 1].im;
    hszCostab = wrapIndex_data[d_i - 1];
    temp2_re = y[hszCostab - 1].re;
    temp2_im = y[hszCostab - 1].im;
    re = reconVar1[d_i - 1].im;
    im = reconVar1[d_i - 1].re;
    y_re_tmp = reconVar2[d_i - 1].im;
    y_im_tmp = reconVar2[d_i - 1].re;
    y[d_i - 1].re = 0.5 * ((temp_re * im - temp_im * re) +
                           (temp2_re * y_im_tmp - -temp2_im * y_re_tmp));
    y[d_i - 1].im = 0.5 * ((temp_re * re + temp_im * im) +
                           (temp2_re * y_re_tmp + -temp2_im * y_im_tmp));
    ju = (nRows + d_i) - 1;
    y[ju].re = 0.5 * ((temp_re * y_im_tmp - temp_im * y_re_tmp) +
                      (temp2_re * im - -temp2_im * re));
    y[ju].im = 0.5 * ((temp_re * y_re_tmp + temp_im * y_im_tmp) +
                      (temp2_re * re + -temp2_im * im));
    re = reconVar1[hszCostab - 1].im;
    im = reconVar1[hszCostab - 1].re;
    y_re_tmp = reconVar2[hszCostab - 1].im;
    y_im_tmp = reconVar2[hszCostab - 1].re;
    y[hszCostab - 1].re = 0.5 * ((temp2_re * im - temp2_im * re) +
                                 (temp_re * y_im_tmp - -temp_im * y_re_tmp));
    y[hszCostab - 1].im = 0.5 * ((temp2_re * re + temp2_im * im) +
                                 (temp_re * y_re_tmp + -temp_im * y_im_tmp));
    hszCostab = (hszCostab + nRows) - 1;
    y[hszCostab].re = 0.5 * ((temp2_re * y_im_tmp - temp2_im * y_re_tmp) +
                             (temp_re * im - -temp_im * re));
    y[hszCostab].im = 0.5 * ((temp2_re * y_re_tmp + temp2_im * y_im_tmp) +
                             (temp_re * re + -temp_im * im));
  }
  double b_y_re_tmp;
  double c_y_re_tmp;
  temp_re = y[nRowsD2].re;
  temp_im = y[nRowsD2].im;
  im = reconVar1[nRowsD2].im;
  y_re_tmp = reconVar1[nRowsD2].re;
  y_im_tmp = temp_re * y_re_tmp;
  temp2_re = temp_re * im;
  temp2_im = reconVar2[nRowsD2].im;
  b_y_re_tmp = reconVar2[nRowsD2].re;
  c_y_re_tmp = temp_re * b_y_re_tmp;
  re = temp_re * temp2_im;
  y[nRowsD2].re =
      0.5 * ((y_im_tmp - temp_im * im) + (c_y_re_tmp - -temp_im * temp2_im));
  y[nRowsD2].im =
      0.5 * ((temp2_re + temp_im * y_re_tmp) + (re + -temp_im * b_y_re_tmp));
  hszCostab = nRows + nRowsD2;
  y[hszCostab].re =
      0.5 * ((c_y_re_tmp - temp_im * temp2_im) + (y_im_tmp - -temp_im * im));
  y[hszCostab].im =
      0.5 * ((re + temp_im * b_y_re_tmp) + (temp2_re + -temp_im * y_re_tmp));
}

void FFTImplementationCallback::dobluesteinfft(const array<creal32_T, 2U> &x,
                                               int n2blue, int nfft,
                                               const float costab_data[],
                                               const float sintab_data[],
                                               const float sintabinv_data[],
                                               array<creal32_T, 2U> &y)
{
  creal32_T fv_data[1032];
  creal32_T fy_data[1032];
  creal32_T wwc_data[515];
  creal32_T tmp_data[258];
  float im;
  float re;
  float temp_im;
  float temp_re;
  float temp_re_tmp;
  float twid_im;
  float twid_re;
  int b_i;
  int b_k;
  int b_y;
  int c_k;
  int d_k;
  int i;
  int iDelta;
  int iDelta2;
  int iheight;
  int iheight_tmp;
  int ihi;
  int istart;
  int ju;
  int minNrowsNx;
  int nInt2;
  int nInt2m1;
  int nRowsD2;
  int rt;
  int xoff;
  boolean_T tst;
  nInt2m1 = (nfft + nfft) - 1;
  rt = 0;
  wwc_data[nfft - 1].re = 1.0F;
  wwc_data[nfft - 1].im = 0.0F;
  nInt2 = nfft << 1;
  i = static_cast<unsigned short>(nfft - 1);
  for (int k{0}; k < i; k++) {
    float nt_im;
    float nt_re;
    b_y = ((k + 1) << 1) - 1;
    if (nInt2 - rt <= b_y) {
      rt += b_y - nInt2;
    } else {
      rt += b_y;
    }
    nt_im = -3.14159274F * static_cast<float>(rt) / static_cast<float>(nfft);
    nt_re = std::cos(nt_im);
    nt_im = std::sin(nt_im);
    b_y = (nfft - k) - 2;
    wwc_data[b_y].re = nt_re;
    wwc_data[b_y].im = -nt_im;
  }
  b_y = nInt2m1 - 1;
  for (int k{b_y}; k >= nfft; k--) {
    wwc_data[k] = wwc_data[(nInt2m1 - k) - 1];
  }
  rt = x.size(0);
  y.set_size(nfft, x.size(1));
  if (nfft > x.size(0)) {
    y.set_size(nfft, x.size(1));
    b_y = nfft * x.size(1);
    for (int k{0}; k < b_y; k++) {
      y[k].re = 0.0F;
      y[k].im = 0.0F;
    }
  }
  b_y = x.size(1);
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        xoff, minNrowsNx, ju, b_k, istart, temp_re, temp_im, ihi, temp_re_tmp, \
            re, nRowsD2, c_k, tst, im, iDelta, iDelta2, d_k, iheight_tmp,      \
            iheight, b_i, twid_re, twid_im, fv_data)                           \
    firstprivate(fy_data, tmp_data)

  for (int chan = 0; chan < b_y; chan++) {
    xoff = chan * rt;
    if (nfft > x.size(0)) {
      std::memset(&fy_data[0], 0,
                  static_cast<unsigned int>(nfft) * sizeof(creal32_T));
    }
    if (nfft - 1 >= 0) {
      std::copy(&fy_data[0], &fy_data[nfft], &tmp_data[0]);
    }
    minNrowsNx = x.size(0);
    if (nfft <= minNrowsNx) {
      minNrowsNx = nfft;
    }
    ju = static_cast<unsigned short>(minNrowsNx);
    for (b_k = 0; b_k < ju; b_k++) {
      istart = (nfft + b_k) - 1;
      temp_re = wwc_data[istart].re;
      temp_im = wwc_data[istart].im;
      ihi = xoff + b_k;
      temp_re_tmp = x[ihi].im;
      re = x[ihi].re;
      tmp_data[b_k].re = temp_re * re + temp_im * temp_re_tmp;
      tmp_data[b_k].im = temp_re * temp_re_tmp - temp_im * re;
    }
    istart = minNrowsNx + 1;
    if (istart <= nfft) {
      std::memset(&tmp_data[istart + -1], 0,
                  static_cast<unsigned int>((nfft - istart) + 1) *
                      sizeof(creal32_T));
    }
    if (n2blue > nfft) {
      std::memset(&fy_data[0], 0,
                  static_cast<unsigned int>(n2blue) * sizeof(creal32_T));
    }
    ihi = n2blue;
    if (nfft <= n2blue) {
      ihi = nfft;
    }
    ju = n2blue - 2;
    nRowsD2 = static_cast<int>(static_cast<unsigned int>(n2blue) >> 1);
    c_k = static_cast<int>(static_cast<unsigned int>(nRowsD2) >> 1);
    xoff = 0;
    minNrowsNx = 0;
    for (b_k = 0; b_k <= ihi - 2; b_k++) {
      fy_data[xoff] = tmp_data[b_k];
      istart = n2blue;
      tst = true;
      while (tst) {
        istart >>= 1;
        minNrowsNx ^= istart;
        tst = ((minNrowsNx & istart) == 0);
      }
      xoff = minNrowsNx;
    }
    if (ihi - 2 < 0) {
      istart = 0;
    } else {
      istart = ihi - 1;
    }
    fy_data[xoff] = tmp_data[istart];
    if (n2blue > 1) {
      for (b_k = 0; b_k <= ju; b_k += 2) {
        temp_re = fy_data[b_k + 1].re;
        temp_re_tmp = fy_data[b_k + 1].im;
        temp_im = temp_re_tmp;
        re = fy_data[b_k].re;
        im = fy_data[b_k].im;
        fy_data[b_k + 1].re = re - temp_re;
        temp_re_tmp = im - temp_re_tmp;
        fy_data[b_k + 1].im = temp_re_tmp;
        re += temp_re;
        fy_data[b_k].re = re;
        fy_data[b_k].im = im + temp_im;
      }
    }
    iDelta = 2;
    iDelta2 = 4;
    d_k = c_k;
    iheight_tmp = (c_k - 1) << 2;
    iheight = iheight_tmp + 1;
    while (d_k > 0) {
      for (b_i = 0; b_i < iheight; b_i += iDelta2) {
        istart = b_i + iDelta;
        temp_re = fy_data[istart].re;
        temp_im = fy_data[istart].im;
        fy_data[istart].re = fy_data[b_i].re - temp_re;
        fy_data[istart].im = fy_data[b_i].im - temp_im;
        fy_data[b_i].re += temp_re;
        fy_data[b_i].im += temp_im;
      }
      xoff = 1;
      for (minNrowsNx = d_k; minNrowsNx < nRowsD2; minNrowsNx += d_k) {
        twid_re = costab_data[minNrowsNx];
        twid_im = sintab_data[minNrowsNx];
        b_i = xoff;
        ihi = xoff + iheight;
        while (b_i < ihi) {
          ju = b_i + iDelta;
          temp_re_tmp = fy_data[ju].im;
          temp_im = fy_data[ju].re;
          temp_re = twid_re * temp_im - twid_im * temp_re_tmp;
          temp_im = twid_re * temp_re_tmp + twid_im * temp_im;
          fy_data[ju].re = fy_data[b_i].re - temp_re;
          fy_data[ju].im = fy_data[b_i].im - temp_im;
          fy_data[b_i].re += temp_re;
          fy_data[b_i].im += temp_im;
          b_i += iDelta2;
        }
        xoff++;
      }
      d_k = static_cast<int>(static_cast<unsigned int>(d_k) >> 1);
      iDelta = iDelta2;
      iDelta2 += iDelta2;
      iheight -= iDelta;
    }
    if (n2blue > nInt2m1) {
      if (n2blue - 1 >= 0) {
        std::memset(&fv_data[0], 0,
                    static_cast<unsigned int>(n2blue) * sizeof(creal32_T));
      }
    }
    xoff = n2blue;
    if (nInt2m1 <= n2blue) {
      xoff = nInt2m1;
    }
    ihi = n2blue - 2;
    minNrowsNx = 0;
    ju = 0;
    for (b_k = 0; b_k <= xoff - 2; b_k++) {
      fv_data[minNrowsNx] = wwc_data[b_k];
      istart = n2blue;
      tst = true;
      while (tst) {
        istart >>= 1;
        ju ^= istart;
        tst = ((ju & istart) == 0);
      }
      minNrowsNx = ju;
    }
    if (xoff - 2 < 0) {
      istart = 0;
    } else {
      istart = xoff - 1;
    }
    fv_data[minNrowsNx] = wwc_data[istart];
    if (n2blue > 1) {
      for (b_k = 0; b_k <= ihi; b_k += 2) {
        temp_re = fv_data[b_k + 1].re;
        re = fv_data[b_k + 1].im;
        temp_im = re;
        im = fv_data[b_k].re;
        temp_re_tmp = fv_data[b_k].im;
        fv_data[b_k + 1].re = im - temp_re;
        re = temp_re_tmp - re;
        fv_data[b_k + 1].im = re;
        im += temp_re;
        fv_data[b_k].re = im;
        fv_data[b_k].im = temp_re_tmp + temp_im;
      }
    }
    iDelta2 = 2;
    d_k = 4;
    ju = iheight_tmp + 1;
    while (c_k > 0) {
      for (iDelta = 0; iDelta < ju; iDelta += d_k) {
        istart = iDelta + iDelta2;
        temp_re = fv_data[istart].re;
        temp_im = fv_data[istart].im;
        fv_data[istart].re = fv_data[iDelta].re - temp_re;
        fv_data[istart].im = fv_data[iDelta].im - temp_im;
        fv_data[iDelta].re += temp_re;
        fv_data[iDelta].im += temp_im;
      }
      istart = 1;
      for (xoff = c_k; xoff < nRowsD2; xoff += c_k) {
        twid_re = costab_data[xoff];
        twid_im = sintab_data[xoff];
        iDelta = istart;
        minNrowsNx = istart + ju;
        while (iDelta < minNrowsNx) {
          ihi = iDelta + iDelta2;
          re = fv_data[ihi].im;
          im = fv_data[ihi].re;
          temp_re = twid_re * im - twid_im * re;
          temp_im = twid_re * re + twid_im * im;
          fv_data[ihi].re = fv_data[iDelta].re - temp_re;
          fv_data[ihi].im = fv_data[iDelta].im - temp_im;
          fv_data[iDelta].re += temp_re;
          fv_data[iDelta].im += temp_im;
          iDelta += d_k;
        }
        istart++;
      }
      c_k = static_cast<int>(static_cast<unsigned int>(c_k) >> 1);
      iDelta2 = d_k;
      d_k += d_k;
      ju -= iDelta2;
    }
    for (b_k = 0; b_k < n2blue; b_k++) {
      temp_re_tmp = fy_data[b_k].re;
      re = fv_data[b_k].im;
      im = fy_data[b_k].im;
      temp_im = fv_data[b_k].re;
      fy_data[b_k].re = temp_re_tmp * temp_im - im * re;
      fy_data[b_k].im = temp_re_tmp * re + im * temp_im;
    }
    FFTImplementationCallback::r2br_r2dit_trig(
        fy_data, n2blue, n2blue, costab_data, sintabinv_data, fv_data);
    istart = static_cast<int>(static_cast<float>(nfft));
    for (b_k = istart; b_k <= nInt2m1; b_k++) {
      temp_re_tmp = wwc_data[b_k - 1].re;
      re = fv_data[b_k - 1].im;
      im = wwc_data[b_k - 1].im;
      temp_im = fv_data[b_k - 1].re;
      ihi = b_k - static_cast<int>(static_cast<float>(nfft));
      tmp_data[ihi].re = temp_re_tmp * temp_im + im * re;
      tmp_data[ihi].im = temp_re_tmp * re - im * temp_im;
    }
    for (b_k = 0; b_k < nfft; b_k++) {
      y[b_k + y.size(0) * chan] = tmp_data[b_k];
    }
  }
}

void FFTImplementationCallback::dobluesteinfft(
    const array<double, 1U> &x, int n2blue, int nfft,
    const array<double, 2U> &costab, const array<double, 2U> &sintab,
    const array<double, 2U> &sintabinv, array<creal_T, 1U> &y)
{
  array<creal_T, 1U> b_y;
  array<creal_T, 1U> c_y;
  array<creal_T, 1U> r;
  array<creal_T, 1U> wwc;
  creal_T nt;
  double d10;
  double d3;
  double d4;
  double d5;
  double d6;
  double d7;
  double d8;
  double d9;
  double nt_tmp;
  int i3;
  int minNrowsNx;
  int rt;
  unsigned int u;
  u = static_cast<unsigned int>(nfft) & 1U;
  if (u == 0U) {
    int b_nInt2m1;
    int i;
    int nInt2;
    int nInt2m1;
    nInt2m1 = static_cast<int>(static_cast<unsigned int>(nfft) >> 1);
    b_nInt2m1 = (nInt2m1 + nInt2m1) - 1;
    wwc.set_size(b_nInt2m1);
    rt = 0;
    wwc[nInt2m1 - 1].re = 1.0;
    wwc[nInt2m1 - 1].im = 0.0;
    nInt2 = nInt2m1 << 1;
    i = static_cast<unsigned short>(nInt2m1 - 1);
    for (int k{0}; k < i; k++) {
      minNrowsNx = ((k + 1) << 1) - 1;
      if (nInt2 - rt <= minNrowsNx) {
        rt += minNrowsNx - nInt2;
      } else {
        rt += minNrowsNx;
      }
      nt_tmp = -3.1415926535897931 * static_cast<double>(rt) /
               static_cast<double>(nInt2m1);
      nt.re = std::cos(nt_tmp);
      nt.im = std::sin(nt_tmp);
      minNrowsNx = (nInt2m1 - k) - 2;
      wwc[minNrowsNx].re = nt.re;
      wwc[minNrowsNx].im = -nt.im;
    }
    minNrowsNx = b_nInt2m1 - 1;
    for (int k{minNrowsNx}; k >= nInt2m1; k--) {
      wwc[k] = wwc[(b_nInt2m1 - k) - 1];
    }
  } else {
    int i;
    int nInt2;
    int nInt2m1;
    nInt2m1 = (nfft + nfft) - 1;
    wwc.set_size(nInt2m1);
    rt = 0;
    wwc[nfft - 1].re = 1.0;
    wwc[nfft - 1].im = 0.0;
    nInt2 = nfft << 1;
    i = static_cast<unsigned short>(nfft - 1);
    for (int k{0}; k < i; k++) {
      minNrowsNx = ((k + 1) << 1) - 1;
      if (nInt2 - rt <= minNrowsNx) {
        rt += minNrowsNx - nInt2;
      } else {
        rt += minNrowsNx;
      }
      nt_tmp = -3.1415926535897931 * static_cast<double>(rt) /
               static_cast<double>(nfft);
      nt.re = std::cos(nt_tmp);
      nt.im = std::sin(nt_tmp);
      minNrowsNx = (nfft - k) - 2;
      wwc[minNrowsNx].re = nt.re;
      wwc[minNrowsNx].im = -nt.im;
    }
    minNrowsNx = nInt2m1 - 1;
    for (int k{minNrowsNx}; k >= nfft; k--) {
      wwc[k] = wwc[(nInt2m1 - k) - 1];
    }
  }
  y.set_size(nfft);
  if (nfft > x.size(0)) {
    y.set_size(nfft);
    for (int k{0}; k < nfft; k++) {
      y[k].re = 0.0;
      y[k].im = 0.0;
    }
  }
  if ((n2blue != 1) && (u == 0U)) {
    FFTImplementationCallback::doHalfLengthBluestein(
        x, y, x.size(0), nfft, n2blue, wwc, costab, sintab, costab, sintabinv);
  } else {
    double d;
    double d1;
    double d2;
    minNrowsNx = x.size(0);
    if (nfft <= minNrowsNx) {
      minNrowsNx = nfft;
    }
    rt = static_cast<unsigned short>(minNrowsNx);
    if (static_cast<int>(static_cast<unsigned short>(minNrowsNx) < 800)) {
      for (int b_k{0}; b_k < rt; b_k++) {
        nt = wwc[(nfft + b_k) - 1];
        y[b_k].re = nt.re * x[b_k];
        y[b_k].im = nt.im * -x[b_k];
      }
    } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(nt)

      for (int b_k = 0; b_k < rt; b_k++) {
        nt = wwc[(nfft + b_k) - 1];
        y[b_k].re = nt.re * x[b_k];
        y[b_k].im = nt.im * -x[b_k];
      }
    }
    minNrowsNx++;
    for (int k{minNrowsNx}; k <= nfft; k++) {
      y[k - 1].re = 0.0;
      y[k - 1].im = 0.0;
    }
    FFTImplementationCallback::b_r2br_r2dit_trig_impl(y, n2blue, costab, sintab,
                                                      b_y);
    FFTImplementationCallback::b_r2br_r2dit_trig_impl(wwc, n2blue, costab,
                                                      sintab, r);
    minNrowsNx = b_y.size(0);
    c_y.set_size(b_y.size(0));
    if (static_cast<int>(b_y.size(0) < 800)) {
      for (int i1{0}; i1 < minNrowsNx; i1++) {
        nt_tmp = b_y[i1].re;
        d = r[i1].im;
        d1 = b_y[i1].im;
        d2 = r[i1].re;
        c_y[i1].re = nt_tmp * d2 - d1 * d;
        c_y[i1].im = nt_tmp * d + d1 * d2;
      }
    } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        d3, d4, d5, d6)

      for (int i1 = 0; i1 < minNrowsNx; i1++) {
        d3 = b_y[i1].re;
        d4 = r[i1].im;
        d5 = b_y[i1].im;
        d6 = r[i1].re;
        c_y[i1].re = d3 * d6 - d5 * d4;
        c_y[i1].im = d3 * d4 + d5 * d6;
      }
    }
    FFTImplementationCallback::b_r2br_r2dit_trig_impl(c_y, n2blue, costab,
                                                      sintabinv, b_y);
    if (b_y.size(0) > 1) {
      nt_tmp = 1.0 / static_cast<double>(b_y.size(0));
      minNrowsNx = b_y.size(0);
      if (static_cast<int>(b_y.size(0) < 800)) {
        for (int i2{0}; i2 < minNrowsNx; i2++) {
          b_y[i2].re = nt_tmp * b_y[i2].re;
          b_y[i2].im = nt_tmp * b_y[i2].im;
        }
      } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

        for (int i2 = 0; i2 < minNrowsNx; i2++) {
          b_y[i2].re = nt_tmp * b_y[i2].re;
          b_y[i2].im = nt_tmp * b_y[i2].im;
        }
      }
    }
    rt = wwc.size(0);
    if (static_cast<int>((wwc.size(0) - nfft) + 1 < 800)) {
      for (int c_k{nfft}; c_k <= rt; c_k++) {
        nt_tmp = wwc[c_k - 1].re;
        d = b_y[c_k - 1].im;
        d1 = wwc[c_k - 1].im;
        d2 = b_y[c_k - 1].re;
        minNrowsNx = c_k - nfft;
        y[minNrowsNx].re = nt_tmp * d2 + d1 * d;
        y[minNrowsNx].im = nt_tmp * d - d1 * d2;
      }
    } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        d7, d8, d9, d10, i3)

      for (int c_k = nfft; c_k <= rt; c_k++) {
        d7 = wwc[c_k - 1].re;
        d8 = b_y[c_k - 1].im;
        d9 = wwc[c_k - 1].im;
        d10 = b_y[c_k - 1].re;
        i3 = c_k - nfft;
        y[i3].re = d7 * d10 + d9 * d8;
        y[i3].im = d7 * d8 - d9 * d10;
      }
    }
  }
}

void FFTImplementationCallback::dobluesteinfft(
    const array<creal_T, 2U> &x, int n2blue, int nfft,
    const array<double, 2U> &costab, const array<double, 2U> &sintab,
    const array<double, 2U> &sintabinv, array<creal_T, 2U> &y)
{
  array<creal_T, 1U> b_fv;
  array<creal_T, 1U> fv;
  array<creal_T, 1U> r;
  array<creal_T, 1U> wwc;
  double a_im;
  double a_re;
  double ar;
  double b_re_tmp;
  double re_tmp;
  int a_re_tmp;
  int b_k;
  int b_y;
  int minNrowsNx;
  int nInt2;
  int nInt2m1;
  int rt;
  int xoff;
  nInt2m1 = (nfft + nfft) - 1;
  wwc.set_size(nInt2m1);
  rt = 0;
  wwc[nfft - 1].re = 1.0;
  wwc[nfft - 1].im = 0.0;
  nInt2 = nfft << 1;
  for (int k{0}; k <= nfft - 2; k++) {
    double nt_im;
    double nt_re;
    b_y = ((k + 1) << 1) - 1;
    if (nInt2 - rt <= b_y) {
      rt += b_y - nInt2;
    } else {
      rt += b_y;
    }
    nt_im = 3.1415926535897931 * static_cast<double>(rt) /
            static_cast<double>(nfft);
    nt_re = std::cos(nt_im);
    nt_im = std::sin(nt_im);
    b_y = (nfft - k) - 2;
    wwc[b_y].re = nt_re;
    wwc[b_y].im = -nt_im;
  }
  b_y = nInt2m1 - 1;
  for (int k{b_y}; k >= nfft; k--) {
    wwc[k] = wwc[(nInt2m1 - k) - 1];
  }
  rt = x.size(0);
  y.set_size(nfft, x.size(1));
  if (nfft > x.size(0)) {
    y.set_size(nfft, x.size(1));
    b_y = nfft * x.size(1);
    for (int k{0}; k < b_y; k++) {
      y[k].re = 0.0;
      y[k].im = 0.0;
    }
  }
  b_y = x.size(1);
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        fv, b_fv, r, xoff, minNrowsNx, b_k, a_re_tmp, a_re, a_im, re_tmp,      \
            b_re_tmp, ar)

  for (int chan = 0; chan < b_y; chan++) {
    xoff = chan * rt;
    r.set_size(nfft);
    if (nfft > x.size(0)) {
      r.set_size(nfft);
      std::memset(&r[0], 0, static_cast<unsigned int>(nfft) * sizeof(creal_T));
    }
    minNrowsNx = x.size(0);
    if (nfft <= minNrowsNx) {
      minNrowsNx = nfft;
    }
    for (b_k = 0; b_k < minNrowsNx; b_k++) {
      a_re_tmp = (nfft + b_k) - 1;
      a_re = wwc[a_re_tmp].re;
      a_im = wwc[a_re_tmp].im;
      a_re_tmp = xoff + b_k;
      re_tmp = x[a_re_tmp].im;
      b_re_tmp = x[a_re_tmp].re;
      r[b_k].re = a_re * b_re_tmp + a_im * re_tmp;
      r[b_k].im = a_re * re_tmp - a_im * b_re_tmp;
    }
    a_re_tmp = minNrowsNx + 1;
    if (a_re_tmp <= nfft) {
      std::memset(&r[a_re_tmp + -1], 0,
                  static_cast<unsigned int>((nfft - a_re_tmp) + 1) *
                      sizeof(creal_T));
    }
    FFTImplementationCallback::b_r2br_r2dit_trig_impl(r, n2blue, costab, sintab,
                                                      b_fv);
    FFTImplementationCallback::b_r2br_r2dit_trig_impl(wwc, n2blue, costab,
                                                      sintab, fv);
    a_re_tmp = b_fv.size(0);
    fv.set_size(b_fv.size(0));
    for (b_k = 0; b_k < a_re_tmp; b_k++) {
      a_re = b_fv[b_k].re;
      a_im = fv[b_k].im;
      re_tmp = b_fv[b_k].im;
      b_re_tmp = fv[b_k].re;
      fv[b_k].re = a_re * b_re_tmp - re_tmp * a_im;
      fv[b_k].im = a_re * a_im + re_tmp * b_re_tmp;
    }
    FFTImplementationCallback::b_r2br_r2dit_trig_impl(fv, n2blue, costab,
                                                      sintabinv, b_fv);
    if (b_fv.size(0) > 1) {
      a_re = 1.0 / static_cast<double>(b_fv.size(0));
      a_re_tmp = b_fv.size(0);
      for (b_k = 0; b_k < a_re_tmp; b_k++) {
        b_fv[b_k].re = a_re * b_fv[b_k].re;
        b_fv[b_k].im = a_re * b_fv[b_k].im;
      }
    }
    xoff = wwc.size(0);
    for (b_k = nfft; b_k <= xoff; b_k++) {
      a_re = wwc[b_k - 1].re;
      a_im = b_fv[b_k - 1].im;
      re_tmp = wwc[b_k - 1].im;
      b_re_tmp = b_fv[b_k - 1].re;
      ar = a_re * b_re_tmp + re_tmp * a_im;
      a_re = a_re * a_im - re_tmp * b_re_tmp;
      if (a_re == 0.0) {
        a_re_tmp = b_k - nfft;
        r[a_re_tmp].re = ar / static_cast<double>(nfft);
        r[a_re_tmp].im = 0.0;
      } else if (ar == 0.0) {
        a_re_tmp = b_k - nfft;
        r[a_re_tmp].re = 0.0;
        r[a_re_tmp].im = a_re / static_cast<double>(nfft);
      } else {
        a_re_tmp = b_k - nfft;
        r[a_re_tmp].re = ar / static_cast<double>(nfft);
        r[a_re_tmp].im = a_re / static_cast<double>(nfft);
      }
    }
    a_re_tmp = y.size(0);
    for (b_k = 0; b_k < a_re_tmp; b_k++) {
      y[b_k + y.size(0) * chan] = r[b_k];
    }
  }
}

void FFTImplementationCallback::generate_twiddle_tables(
    int nRows, boolean_T useRadix2, array<double, 2U> &costab,
    array<double, 2U> &sintab, array<double, 2U> &sintabinv)
{
  array<double, 2U> b_costab;
  array<double, 2U> b_sintab;
  array<double, 2U> b_sintabinv;
  array<double, 2U> costab1q;
  double e;
  int loop_ub;
  int n;
  int nd2;
  e = 6.2831853071795862 / static_cast<double>(nRows);
  n = static_cast<int>((static_cast<unsigned int>(nRows) >> 1) >> 1);
  costab1q.set_size(1, n + 1);
  costab1q[0] = 1.0;
  nd2 = static_cast<int>(static_cast<unsigned int>(n) >> 1);
  loop_ub = static_cast<unsigned short>(nd2);
  if (static_cast<int>(static_cast<unsigned short>(nd2) < 800)) {
    for (int k{0}; k < loop_ub; k++) {
      costab1q[k + 1] = std::cos(e * (static_cast<double>(k) + 1.0));
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

    for (int k = 0; k < loop_ub; k++) {
      costab1q[k + 1] = std::cos(e * (static_cast<double>(k) + 1.0));
    }
  }
  loop_ub = nd2 + 1;
  if (static_cast<int>((n - nd2) - 1 < 800)) {
    for (int b_k{loop_ub}; b_k < n; b_k++) {
      costab1q[b_k] = std::sin(e * static_cast<double>(n - b_k));
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

    for (int b_k = loop_ub; b_k < n; b_k++) {
      costab1q[b_k] = std::sin(e * static_cast<double>(n - b_k));
    }
  }
  costab1q[n] = 0.0;
  if (!useRadix2) {
    int n2;
    n = costab1q.size(1) - 1;
    n2 = (costab1q.size(1) - 1) << 1;
    b_costab.set_size(1, n2 + 1);
    b_sintab.set_size(1, n2 + 1);
    b_costab[0] = 1.0;
    b_sintab[0] = 0.0;
    b_sintabinv.set_size(1, n2 + 1);
    loop_ub = (costab1q.size(1) - 1 < 800);
    if (loop_ub) {
      for (int e_k{0}; e_k < n; e_k++) {
        b_sintabinv[e_k + 1] = costab1q[(n - e_k) - 1];
      }
    } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

      for (int e_k = 0; e_k < n; e_k++) {
        b_sintabinv[e_k + 1] = costab1q[(n - e_k) - 1];
      }
    }
    nd2 = costab1q.size(1);
    for (int f_k{nd2}; f_k <= n2; f_k++) {
      b_sintabinv[f_k] = costab1q[f_k - n];
    }
    if (loop_ub) {
      for (int g_k{0}; g_k < n; g_k++) {
        b_costab[g_k + 1] = costab1q[g_k + 1];
        b_sintab[g_k + 1] = -costab1q[(n - g_k) - 1];
      }
    } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

      for (int g_k = 0; g_k < n; g_k++) {
        b_costab[g_k + 1] = costab1q[g_k + 1];
        b_sintab[g_k + 1] = -costab1q[(n - g_k) - 1];
      }
    }
    if (static_cast<int>((n2 - costab1q.size(1)) + 1 < 800)) {
      for (int h_k{nd2}; h_k <= n2; h_k++) {
        b_costab[h_k] = -costab1q[n2 - h_k];
        b_sintab[h_k] = -costab1q[h_k - n];
      }
    } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

      for (int h_k = nd2; h_k <= n2; h_k++) {
        b_costab[h_k] = -costab1q[n2 - h_k];
        b_sintab[h_k] = -costab1q[h_k - n];
      }
    }
    loop_ub = b_costab.size(1);
    costab.set_size(1, b_costab.size(1));
    for (int f_k{0}; f_k < loop_ub; f_k++) {
      costab[f_k] = b_costab[f_k];
    }
    sintab.set_size(1, b_costab.size(1));
    for (int f_k{0}; f_k < loop_ub; f_k++) {
      sintab[f_k] = b_sintab[f_k];
    }
    sintabinv.set_size(1, b_costab.size(1));
    for (int f_k{0}; f_k < loop_ub; f_k++) {
      sintabinv[f_k] = b_sintabinv[f_k];
    }
  } else {
    nd2 = costab1q.size(1) - 1;
    n = (costab1q.size(1) - 1) << 1;
    costab.set_size(1, n + 1);
    sintab.set_size(1, n + 1);
    costab[0] = 1.0;
    sintab[0] = 0.0;
    if (static_cast<int>(costab1q.size(1) - 1 < 800)) {
      for (int c_k{0}; c_k < nd2; c_k++) {
        costab[c_k + 1] = costab1q[c_k + 1];
        sintab[c_k + 1] = -costab1q[(nd2 - c_k) - 1];
      }
    } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

      for (int c_k = 0; c_k < nd2; c_k++) {
        costab[c_k + 1] = costab1q[c_k + 1];
        sintab[c_k + 1] = -costab1q[(nd2 - c_k) - 1];
      }
    }
    loop_ub = costab1q.size(1);
    if (static_cast<int>((n - costab1q.size(1)) + 1 < 800)) {
      for (int d_k{loop_ub}; d_k <= n; d_k++) {
        costab[d_k] = -costab1q[n - d_k];
        sintab[d_k] = -costab1q[d_k - nd2];
      }
    } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

      for (int d_k = loop_ub; d_k <= n; d_k++) {
        costab[d_k] = -costab1q[n - d_k];
        sintab[d_k] = -costab1q[d_k - nd2];
      }
    }
    sintabinv.set_size(1, 0);
  }
}

int FFTImplementationCallback::get_algo_sizes(int nfft, boolean_T useRadix2,
                                              int &nRows)
{
  int n2blue;
  n2blue = 1;
  if (useRadix2) {
    nRows = nfft;
  } else {
    if (nfft > 0) {
      int pmax;
      n2blue = (nfft + nfft) - 1;
      pmax = 31;
      if (n2blue <= 1) {
        pmax = 0;
      } else {
        int pmin;
        boolean_T exitg1;
        pmin = 0;
        exitg1 = false;
        while ((!exitg1) && (pmax - pmin > 1)) {
          int k;
          int pow2p;
          k = (pmin + pmax) >> 1;
          pow2p = 1 << k;
          if (pow2p == n2blue) {
            pmax = k;
            exitg1 = true;
          } else if (pow2p > n2blue) {
            pmax = k;
          } else {
            pmin = k;
          }
        }
      }
      n2blue = 1 << pmax;
    }
    nRows = n2blue;
  }
  return n2blue;
}

void FFTImplementationCallback::r2br_r2dit_trig(const array<creal32_T, 2U> &x,
                                                array<creal32_T, 2U> &y)
{
  static const float fv[257]{
      1.0F,           0.999924719F,   0.999698818F,   0.999322414F,
      0.99879545F,    0.998118103F,   0.997290432F,   0.996312618F,
      0.99518472F,    0.993907F,      0.992479563F,   0.990902662F,
      0.989176512F,   0.987301409F,   0.985277653F,   0.983105481F,
      0.980785251F,   0.97831738F,    0.975702107F,   0.972939968F,
      0.970031261F,   0.966976464F,   0.963776052F,   0.960430503F,
      0.956940353F,   0.953306F,      0.949528158F,   0.945607305F,
      0.941544056F,   0.937339F,      0.932992816F,   0.928506076F,
      0.923879504F,   0.919113874F,   0.914209723F,   0.909167945F,
      0.903989315F,   0.898674488F,   0.893224299F,   0.887639642F,
      0.881921232F,   0.876070082F,   0.870086968F,   0.863972843F,
      0.857728601F,   0.851355195F,   0.84485358F,    0.838224709F,
      0.831469595F,   0.824589252F,   0.817584813F,   0.81045717F,
      0.803207517F,   0.795836926F,   0.78834641F,    0.780737221F,
      0.773010433F,   0.765167236F,   0.757208824F,   0.749136388F,
      0.740951121F,   0.732654274F,   0.724247098F,   0.715730786F,
      0.707106769F,   0.698376298F,   0.689540565F,   0.680601F,
      0.671559F,      0.662415802F,   0.653172851F,   0.643831551F,
      0.634393334F,   0.624859512F,   0.615231633F,   0.605511F,
      0.59569931F,    0.585797906F,   0.575808227F,   0.565731823F,
      0.555570245F,   0.545325041F,   0.534997642F,   0.524589717F,
      0.514102757F,   0.50353837F,    0.492898226F,   0.482183754F,
      0.471396744F,   0.460538715F,   0.449611336F,   0.438616246F,
      0.427555084F,   0.416429579F,   0.40524134F,    0.393992066F,
      0.382683456F,   0.371317208F,   0.359895051F,   0.348418683F,
      0.336889863F,   0.32531032F,    0.313681751F,   0.302005947F,
      0.290284663F,   0.27851969F,    0.266712785F,   0.254865676F,
      0.242980197F,   0.231058121F,   0.219101235F,   0.207111388F,
      0.195090324F,   0.183039889F,   0.170961902F,   0.15885815F,
      0.146730468F,   0.134580716F,   0.122410677F,   0.110222206F,
      0.0980171412F,  0.0857973173F,  0.0735645667F,  0.0613207407F,
      0.0490676761F,  0.0368072242F,  0.024541229F,   0.0122715384F,
      0.0F,           -0.0122715384F, -0.024541229F,  -0.0368072242F,
      -0.0490676761F, -0.0613207407F, -0.0735645667F, -0.0857973173F,
      -0.0980171412F, -0.110222206F,  -0.122410677F,  -0.134580716F,
      -0.146730468F,  -0.15885815F,   -0.170961902F,  -0.183039889F,
      -0.195090324F,  -0.207111388F,  -0.219101235F,  -0.231058121F,
      -0.242980197F,  -0.254865676F,  -0.266712785F,  -0.27851969F,
      -0.290284663F,  -0.302005947F,  -0.313681751F,  -0.32531032F,
      -0.336889863F,  -0.348418683F,  -0.359895051F,  -0.371317208F,
      -0.382683456F,  -0.393992066F,  -0.40524134F,   -0.416429579F,
      -0.427555084F,  -0.438616246F,  -0.449611336F,  -0.460538715F,
      -0.471396744F,  -0.482183754F,  -0.492898226F,  -0.50353837F,
      -0.514102757F,  -0.524589717F,  -0.534997642F,  -0.545325041F,
      -0.555570245F,  -0.565731823F,  -0.575808227F,  -0.585797906F,
      -0.59569931F,   -0.605511F,     -0.615231633F,  -0.624859512F,
      -0.634393334F,  -0.643831551F,  -0.653172851F,  -0.662415802F,
      -0.671559F,     -0.680601F,     -0.689540565F,  -0.698376298F,
      -0.707106769F,  -0.715730786F,  -0.724247098F,  -0.732654274F,
      -0.740951121F,  -0.749136388F,  -0.757208824F,  -0.765167236F,
      -0.773010433F,  -0.780737221F,  -0.78834641F,   -0.795836926F,
      -0.803207517F,  -0.81045717F,   -0.817584813F,  -0.824589252F,
      -0.831469595F,  -0.838224709F,  -0.84485358F,   -0.851355195F,
      -0.857728601F,  -0.863972843F,  -0.870086968F,  -0.876070082F,
      -0.881921232F,  -0.887639642F,  -0.893224299F,  -0.898674488F,
      -0.903989315F,  -0.909167945F,  -0.914209723F,  -0.919113874F,
      -0.923879504F,  -0.928506076F,  -0.932992816F,  -0.937339F,
      -0.941544056F,  -0.945607305F,  -0.949528158F,  -0.953306F,
      -0.956940353F,  -0.960430503F,  -0.963776052F,  -0.966976464F,
      -0.970031261F,  -0.972939968F,  -0.975702107F,  -0.97831738F,
      -0.980785251F,  -0.983105481F,  -0.985277653F,  -0.987301409F,
      -0.989176512F,  -0.990902662F,  -0.992479563F,  -0.993907F,
      -0.99518472F,   -0.996312618F,  -0.997290432F,  -0.998118103F,
      -0.99879545F,   -0.999322414F,  -0.999698818F,  -0.999924719F,
      -1.0F};
  static const float fv1[257]{
      0.0F,           -0.0122715384F, -0.024541229F,  -0.0368072242F,
      -0.0490676761F, -0.0613207407F, -0.0735645667F, -0.0857973173F,
      -0.0980171412F, -0.110222206F,  -0.122410677F,  -0.134580716F,
      -0.146730468F,  -0.15885815F,   -0.170961902F,  -0.183039889F,
      -0.195090324F,  -0.207111388F,  -0.219101235F,  -0.231058121F,
      -0.242980197F,  -0.254865676F,  -0.266712785F,  -0.27851969F,
      -0.290284663F,  -0.302005947F,  -0.313681751F,  -0.32531032F,
      -0.336889863F,  -0.348418683F,  -0.359895051F,  -0.371317208F,
      -0.382683456F,  -0.393992066F,  -0.40524134F,   -0.416429579F,
      -0.427555084F,  -0.438616246F,  -0.449611336F,  -0.460538715F,
      -0.471396744F,  -0.482183754F,  -0.492898226F,  -0.50353837F,
      -0.514102757F,  -0.524589717F,  -0.534997642F,  -0.545325041F,
      -0.555570245F,  -0.565731823F,  -0.575808227F,  -0.585797906F,
      -0.59569931F,   -0.605511F,     -0.615231633F,  -0.624859512F,
      -0.634393334F,  -0.643831551F,  -0.653172851F,  -0.662415802F,
      -0.671559F,     -0.680601F,     -0.689540565F,  -0.698376298F,
      -0.707106769F,  -0.715730786F,  -0.724247098F,  -0.732654274F,
      -0.740951121F,  -0.749136388F,  -0.757208824F,  -0.765167236F,
      -0.773010433F,  -0.780737221F,  -0.78834641F,   -0.795836926F,
      -0.803207517F,  -0.81045717F,   -0.817584813F,  -0.824589252F,
      -0.831469595F,  -0.838224709F,  -0.84485358F,   -0.851355195F,
      -0.857728601F,  -0.863972843F,  -0.870086968F,  -0.876070082F,
      -0.881921232F,  -0.887639642F,  -0.893224299F,  -0.898674488F,
      -0.903989315F,  -0.909167945F,  -0.914209723F,  -0.919113874F,
      -0.923879504F,  -0.928506076F,  -0.932992816F,  -0.937339F,
      -0.941544056F,  -0.945607305F,  -0.949528158F,  -0.953306F,
      -0.956940353F,  -0.960430503F,  -0.963776052F,  -0.966976464F,
      -0.970031261F,  -0.972939968F,  -0.975702107F,  -0.97831738F,
      -0.980785251F,  -0.983105481F,  -0.985277653F,  -0.987301409F,
      -0.989176512F,  -0.990902662F,  -0.992479563F,  -0.993907F,
      -0.99518472F,   -0.996312618F,  -0.997290432F,  -0.998118103F,
      -0.99879545F,   -0.999322414F,  -0.999698818F,  -0.999924719F,
      -1.0F,          -0.999924719F,  -0.999698818F,  -0.999322414F,
      -0.99879545F,   -0.998118103F,  -0.997290432F,  -0.996312618F,
      -0.99518472F,   -0.993907F,     -0.992479563F,  -0.990902662F,
      -0.989176512F,  -0.987301409F,  -0.985277653F,  -0.983105481F,
      -0.980785251F,  -0.97831738F,   -0.975702107F,  -0.972939968F,
      -0.970031261F,  -0.966976464F,  -0.963776052F,  -0.960430503F,
      -0.956940353F,  -0.953306F,     -0.949528158F,  -0.945607305F,
      -0.941544056F,  -0.937339F,     -0.932992816F,  -0.928506076F,
      -0.923879504F,  -0.919113874F,  -0.914209723F,  -0.909167945F,
      -0.903989315F,  -0.898674488F,  -0.893224299F,  -0.887639642F,
      -0.881921232F,  -0.876070082F,  -0.870086968F,  -0.863972843F,
      -0.857728601F,  -0.851355195F,  -0.84485358F,   -0.838224709F,
      -0.831469595F,  -0.824589252F,  -0.817584813F,  -0.81045717F,
      -0.803207517F,  -0.795836926F,  -0.78834641F,   -0.780737221F,
      -0.773010433F,  -0.765167236F,  -0.757208824F,  -0.749136388F,
      -0.740951121F,  -0.732654274F,  -0.724247098F,  -0.715730786F,
      -0.707106769F,  -0.698376298F,  -0.689540565F,  -0.680601F,
      -0.671559F,     -0.662415802F,  -0.653172851F,  -0.643831551F,
      -0.634393334F,  -0.624859512F,  -0.615231633F,  -0.605511F,
      -0.59569931F,   -0.585797906F,  -0.575808227F,  -0.565731823F,
      -0.555570245F,  -0.545325041F,  -0.534997642F,  -0.524589717F,
      -0.514102757F,  -0.50353837F,   -0.492898226F,  -0.482183754F,
      -0.471396744F,  -0.460538715F,  -0.449611336F,  -0.438616246F,
      -0.427555084F,  -0.416429579F,  -0.40524134F,   -0.393992066F,
      -0.382683456F,  -0.371317208F,  -0.359895051F,  -0.348418683F,
      -0.336889863F,  -0.32531032F,   -0.313681751F,  -0.302005947F,
      -0.290284663F,  -0.27851969F,   -0.266712785F,  -0.254865676F,
      -0.242980197F,  -0.231058121F,  -0.219101235F,  -0.207111388F,
      -0.195090324F,  -0.183039889F,  -0.170961902F,  -0.15885815F,
      -0.146730468F,  -0.134580716F,  -0.122410677F,  -0.110222206F,
      -0.0980171412F, -0.0857973173F, -0.0735645667F, -0.0613207407F,
      -0.0490676761F, -0.0368072242F, -0.024541229F,  -0.0122715384F,
      -0.0F};
  float im;
  float re;
  float temp_im;
  float temp_re;
  float twid_im;
  float twid_re;
  int b_i;
  int c_i;
  int i1;
  int iDelta;
  int iDelta2;
  int iheight;
  int ihi;
  int iy;
  int ju;
  int k;
  int loop_ub;
  int nrows;
  int xoff;
  boolean_T tst;
  nrows = x.size(0);
  y.set_size(512, x.size(1));
  loop_ub = 512 * x.size(1);
  for (int i{0}; i < loop_ub; i++) {
    y[i].re = 0.0F;
    y[i].im = 0.0F;
  }
  loop_ub = x.size(1);
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        xoff, b_i, iy, ju, ihi, tst, iDelta, temp_re, temp_im, re, im,         \
            iDelta2, k, iheight, c_i, twid_re, twid_im, i1)

  for (int chan = 0; chan < loop_ub; chan++) {
    xoff = chan * nrows;
    for (b_i = 0; b_i < 512; b_i++) {
      iy = b_i + 512 * chan;
      y[iy].re = 0.0F;
      y[iy].im = 0.0F;
    }
    iy = 0;
    ju = 0;
    ihi = static_cast<unsigned short>(x.size(0) - 1);
    for (b_i = 0; b_i < ihi; b_i++) {
      y[iy + 512 * chan] = x[xoff + b_i];
      iy = 512;
      tst = true;
      while (tst) {
        iy >>= 1;
        ju ^= iy;
        tst = ((ju & iy) == 0);
      }
      iy = ju;
    }
    y[iy + 512 * chan] = x[xoff + static_cast<unsigned short>(x.size(0) - 1)];
    for (b_i = 0; b_i <= 510; b_i += 2) {
      iy = b_i + 512 * chan;
      temp_re = y[iy + 1].re;
      temp_im = y[iy + 1].im;
      re = y[iy].re;
      im = y[iy].im;
      y[iy + 1].re = re - temp_re;
      y[iy + 1].im = y[iy].im - y[iy + 1].im;
      re += temp_re;
      im += temp_im;
      y[iy].re = re;
      y[iy].im = im;
    }
    iDelta = 2;
    iDelta2 = 4;
    k = 128;
    iheight = 509;
    while (k > 0) {
      for (c_i = 0; c_i < iheight; c_i += iDelta2) {
        iy = (c_i + iDelta) + 512 * chan;
        temp_re = y[iy].re;
        temp_im = y[iy].im;
        ju = c_i + 512 * chan;
        y[iy].re = y[ju].re - temp_re;
        y[iy].im = y[ju].im - temp_im;
        y[ju].re = y[ju].re + temp_re;
        y[ju].im = y[ju].im + temp_im;
      }
      iy = 1;
      for (ju = k; ju < 256; ju += k) {
        twid_re = fv[ju];
        twid_im = fv1[ju];
        c_i = iy;
        ihi = iy + iheight;
        while (c_i < ihi) {
          xoff = (c_i + iDelta) + 512 * chan;
          re = y[xoff].im;
          im = y[xoff].re;
          temp_re = twid_re * im - twid_im * re;
          temp_im = twid_re * re + twid_im * im;
          i1 = c_i + 512 * chan;
          y[xoff].re = y[i1].re - temp_re;
          y[xoff].im = y[i1].im - temp_im;
          y[i1].re = y[i1].re + temp_re;
          y[i1].im = y[i1].im + temp_im;
          c_i += iDelta2;
        }
        iy++;
      }
      k >>= 1;
      iDelta = iDelta2;
      iDelta2 += iDelta2;
      iheight -= iDelta;
    }
  }
}

void FFTImplementationCallback::r2br_r2dit_trig(const array<creal32_T, 2U> &x,
                                                int n1_unsigned,
                                                const float costab_data[],
                                                const float sintab_data[],
                                                array<creal32_T, 2U> &y)
{
  creal32_T y_data[1032];
  creal32_T tmp_data[258];
  float re;
  float temp_im;
  float temp_re;
  float temp_re_tmp;
  float twid_im;
  float twid_re;
  int b_i;
  int c_i;
  int iDelta;
  int iDelta2;
  int iheight;
  int iy;
  int ju;
  int k;
  int loop_ub;
  int nRowsD2;
  int nrows;
  int u1;
  int xoff;
  boolean_T tst;
  nrows = x.size(0);
  y.set_size(n1_unsigned, x.size(1));
  if (n1_unsigned > x.size(0)) {
    y.set_size(n1_unsigned, x.size(1));
    loop_ub = n1_unsigned * x.size(1);
    for (int i{0}; i < loop_ub; i++) {
      y[i].re = 0.0F;
      y[i].im = 0.0F;
    }
  }
  loop_ub = x.size(1);
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        xoff, iy, u1, iDelta, nRowsD2, k, ju, b_i, tst, temp_re, temp_re_tmp,  \
            temp_im, re, twid_re, iDelta2, iheight, c_i, twid_im)              \
    firstprivate(y_data, tmp_data)

  for (int chan = 0; chan < loop_ub; chan++) {
    xoff = chan * nrows;
    if (n1_unsigned > x.size(0)) {
      std::memset(&y_data[0], 0,
                  static_cast<unsigned int>(n1_unsigned) * sizeof(creal32_T));
    }
    if (n1_unsigned - 1 >= 0) {
      std::copy(&y_data[0], &y_data[n1_unsigned], &tmp_data[0]);
    }
    iy = x.size(0);
    u1 = n1_unsigned;
    if (iy <= n1_unsigned) {
      u1 = iy;
    }
    iDelta = n1_unsigned - 2;
    nRowsD2 = static_cast<int>(static_cast<unsigned int>(n1_unsigned) >> 1);
    k = static_cast<int>(static_cast<unsigned int>(nRowsD2) >> 1);
    iy = 0;
    ju = 0;
    for (b_i = 0; b_i <= u1 - 2; b_i++) {
      tmp_data[iy] = x[xoff + b_i];
      iy = n1_unsigned;
      tst = true;
      while (tst) {
        iy >>= 1;
        ju ^= iy;
        tst = ((ju & iy) == 0);
      }
      iy = ju;
    }
    if (u1 - 2 >= 0) {
      xoff = (xoff + u1) - 1;
    }
    tmp_data[iy] = x[xoff];
    if (n1_unsigned - 1 >= 0) {
      std::copy(&tmp_data[0], &tmp_data[n1_unsigned], &y_data[0]);
    }
    if (n1_unsigned > 1) {
      for (b_i = 0; b_i <= iDelta; b_i += 2) {
        temp_re = y_data[b_i + 1].re;
        temp_re_tmp = y_data[b_i + 1].im;
        temp_im = temp_re_tmp;
        re = y_data[b_i].re;
        twid_re = y_data[b_i].im;
        y_data[b_i + 1].re = re - temp_re;
        temp_re_tmp = twid_re - temp_re_tmp;
        y_data[b_i + 1].im = temp_re_tmp;
        re += temp_re;
        y_data[b_i].re = re;
        y_data[b_i].im = twid_re + temp_im;
      }
    }
    iDelta = 2;
    iDelta2 = 4;
    iheight = ((k - 1) << 2) + 1;
    while (k > 0) {
      for (c_i = 0; c_i < iheight; c_i += iDelta2) {
        iy = c_i + iDelta;
        temp_re = y_data[iy].re;
        temp_im = y_data[iy].im;
        y_data[iy].re = y_data[c_i].re - temp_re;
        y_data[iy].im = y_data[c_i].im - temp_im;
        y_data[c_i].re += temp_re;
        y_data[c_i].im += temp_im;
      }
      iy = 1;
      for (ju = k; ju < nRowsD2; ju += k) {
        twid_re = costab_data[ju];
        twid_im = sintab_data[ju];
        c_i = iy;
        xoff = iy + iheight;
        while (c_i < xoff) {
          u1 = c_i + iDelta;
          temp_re_tmp = y_data[u1].im;
          re = y_data[u1].re;
          temp_re = twid_re * re - twid_im * temp_re_tmp;
          temp_im = twid_re * temp_re_tmp + twid_im * re;
          y_data[u1].re = y_data[c_i].re - temp_re;
          y_data[u1].im = y_data[c_i].im - temp_im;
          y_data[c_i].re += temp_re;
          y_data[c_i].im += temp_im;
          c_i += iDelta2;
        }
        iy++;
      }
      k = static_cast<int>(static_cast<unsigned int>(k) >> 1);
      iDelta = iDelta2;
      iDelta2 += iDelta2;
      iheight -= iDelta;
    }
    for (b_i = 0; b_i < n1_unsigned; b_i++) {
      y[b_i + y.size(0) * chan] = y_data[b_i];
    }
  }
}

void FFTImplementationCallback::r2br_r2dit_trig(const array<creal_T, 2U> &x,
                                                int n1_unsigned,
                                                const array<double, 2U> &costab,
                                                const array<double, 2U> &sintab,
                                                array<creal_T, 2U> &y)
{
  array<creal_T, 1U> r;
  double im;
  double re;
  double temp_im;
  double temp_re;
  double twid_im;
  double twid_re;
  int b_i;
  int c_i;
  int iDelta;
  int iDelta2;
  int iheight;
  int iy;
  int ju;
  int k;
  int loop_ub;
  int nRowsD2;
  int nrows;
  int u1;
  int xoff;
  boolean_T tst;
  nrows = x.size(0);
  y.set_size(n1_unsigned, x.size(1));
  if (n1_unsigned > x.size(0)) {
    y.set_size(n1_unsigned, x.size(1));
    loop_ub = n1_unsigned * x.size(1);
    for (int i{0}; i < loop_ub; i++) {
      y[i].re = 0.0;
      y[i].im = 0.0;
    }
  }
  loop_ub = x.size(1);
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        r, xoff, iy, u1, iDelta, nRowsD2, k, ju, b_i, tst, temp_re, temp_im,   \
            re, im, iDelta2, iheight, c_i, twid_re, twid_im)

  for (int chan = 0; chan < loop_ub; chan++) {
    xoff = chan * nrows;
    r.set_size(n1_unsigned);
    if (n1_unsigned > x.size(0)) {
      r.set_size(n1_unsigned);
      std::memset(&r[0], 0,
                  static_cast<unsigned int>(n1_unsigned) * sizeof(creal_T));
    }
    iy = x.size(0);
    u1 = n1_unsigned;
    if (iy <= n1_unsigned) {
      u1 = iy;
    }
    iDelta = n1_unsigned - 2;
    nRowsD2 = static_cast<int>(static_cast<unsigned int>(n1_unsigned) >> 1);
    k = static_cast<int>(static_cast<unsigned int>(nRowsD2) >> 1);
    iy = 0;
    ju = 0;
    for (b_i = 0; b_i <= u1 - 2; b_i++) {
      r[iy] = x[xoff + b_i];
      iy = n1_unsigned;
      tst = true;
      while (tst) {
        iy >>= 1;
        ju ^= iy;
        tst = ((ju & iy) == 0);
      }
      iy = ju;
    }
    if (u1 - 2 >= 0) {
      xoff = (xoff + u1) - 1;
    }
    r[iy] = x[xoff];
    if (n1_unsigned > 1) {
      for (b_i = 0; b_i <= iDelta; b_i += 2) {
        temp_re = r[b_i + 1].re;
        temp_im = r[b_i + 1].im;
        re = r[b_i].re;
        im = r[b_i].im;
        r[b_i + 1].re = re - temp_re;
        r[b_i + 1].im = r[b_i].im - r[b_i + 1].im;
        re += temp_re;
        im += temp_im;
        r[b_i].re = re;
        r[b_i].im = im;
      }
    }
    iDelta = 2;
    iDelta2 = 4;
    iheight = ((k - 1) << 2) + 1;
    while (k > 0) {
      for (c_i = 0; c_i < iheight; c_i += iDelta2) {
        iy = c_i + iDelta;
        temp_re = r[iy].re;
        temp_im = r[iy].im;
        r[iy].re = r[c_i].re - temp_re;
        r[iy].im = r[c_i].im - temp_im;
        r[c_i].re = r[c_i].re + temp_re;
        r[c_i].im = r[c_i].im + temp_im;
      }
      iy = 1;
      for (ju = k; ju < nRowsD2; ju += k) {
        twid_re = costab[ju];
        twid_im = sintab[ju];
        c_i = iy;
        xoff = iy + iheight;
        while (c_i < xoff) {
          u1 = c_i + iDelta;
          re = r[u1].im;
          im = r[u1].re;
          temp_re = twid_re * im - twid_im * re;
          temp_im = twid_re * re + twid_im * im;
          r[u1].re = r[c_i].re - temp_re;
          r[u1].im = r[c_i].im - temp_im;
          r[c_i].re = r[c_i].re + temp_re;
          r[c_i].im = r[c_i].im + temp_im;
          c_i += iDelta2;
        }
        iy++;
      }
      k = static_cast<int>(static_cast<unsigned int>(k) >> 1);
      iDelta = iDelta2;
      iDelta2 += iDelta2;
      iheight -= iDelta;
    }
    iy = y.size(0);
    for (b_i = 0; b_i < iy; b_i++) {
      y[b_i + y.size(0) * chan] = r[b_i];
    }
  }
  if (y.size(0) > 1) {
    double b;
    b = 1.0 / static_cast<double>(y.size(0));
    loop_ub = y.size(0) * y.size(1);
    if (static_cast<int>(loop_ub < 800)) {
      for (int i1{0}; i1 < loop_ub; i1++) {
        y[i1].re = b * y[i1].re;
        y[i1].im = b * y[i1].im;
      }
    } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

      for (int i1 = 0; i1 < loop_ub; i1++) {
        y[i1].re = b * y[i1].re;
        y[i1].im = b * y[i1].im;
      }
    }
  }
}

} // namespace fft
} // namespace internal
} // namespace coder

// End of code generation (FFTImplementationCallback.cpp)
