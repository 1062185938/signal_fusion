//
// bsxfun.cpp
//
// Code generation for function 'bsxfun'
//

// Include files
#include "bsxfun.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include <cstring>

// Function Definitions
namespace coder {
void bsxfun(const float a_data[], int a_size, const array<creal32_T, 2U> &b,
            array<creal32_T, 2U> &c)
{
  int i;
  int u0;
  u0 = b.size(0);
  if (u0 > a_size) {
    u0 = a_size;
  }
  if (b.size(0) == 1) {
    u0 = a_size;
  } else if (a_size == 1) {
    u0 = b.size(0);
  } else if (a_size == b.size(0)) {
    u0 = a_size;
  }
  i = b.size(1);
  c.set_size(u0, b.size(1));
  if ((u0 != 0) && (b.size(1) != 0)) {
    int acoef;
    int b_bcoef;
    int bcoef;
    bcoef = (b.size(1) != 1);
    acoef = (a_size != 1);
    b_bcoef = (b.size(0) != 1);
    for (int k{0}; k < i; k++) {
      int varargin_3;
      varargin_3 = bcoef * k;
      for (int b_k{0}; b_k < u0; b_k++) {
        float f;
        int i1;
        int i2;
        f = a_data[acoef * b_k];
        i1 = b_bcoef * b_k + b.size(0) * varargin_3;
        i2 = b_k + c.size(0) * k;
        c[i2].re = f * b[i1].re;
        c[i2].im = f * b[i1].im;
      }
    }
  }
}

void bsxfun(const array<float, 2U> &a, const array<double, 2U> &b,
            array<float, 2U> &c)
{
  int csz_idx_1;
  int u0;
  u0 = b.size(1);
  csz_idx_1 = a.size(1);
  if (u0 <= csz_idx_1) {
    csz_idx_1 = u0;
  }
  if (b.size(1) == 1) {
    csz_idx_1 = a.size(1);
  } else if (a.size(1) == 1) {
    csz_idx_1 = b.size(1);
  } else if (a.size(1) == b.size(1)) {
    csz_idx_1 = a.size(1);
  }
  u0 = b.size(0);
  c.set_size(b.size(0), csz_idx_1);
  if ((b.size(0) != 0) && (csz_idx_1 != 0)) {
    int acoef;
    acoef = (a.size(1) != 1);
    for (int k{0}; k < csz_idx_1; k++) {
      int varargin_2;
      varargin_2 = acoef * k;
      for (int b_k{0}; b_k < u0; b_k++) {
        c[b_k + c.size(0) * k] = a[varargin_2];
      }
    }
  }
}

void bsxfun(const array<creal32_T, 2U> &a, const creal_T b_data[], int b_size,
            array<creal32_T, 2U> &c)
{
  int i;
  int u1;
  u1 = a.size(0);
  if (b_size <= u1) {
    u1 = b_size;
  }
  if (b_size == 1) {
    u1 = a.size(0);
  } else if (a.size(0) == 1) {
    u1 = b_size;
  } else if (a.size(0) == b_size) {
    u1 = a.size(0);
  }
  i = a.size(1);
  c.set_size(u1, a.size(1));
  if ((u1 != 0) && (a.size(1) != 0)) {
    int acoef;
    int b_acoef;
    int bcoef;
    acoef = (a.size(1) != 1);
    b_acoef = (a.size(0) != 1);
    bcoef = (b_size != 1);
    for (int k{0}; k < i; k++) {
      int varargin_2;
      varargin_2 = acoef * k;
      for (int b_k{0}; b_k < u1; b_k++) {
        float b_im;
        float b_re;
        float f;
        float f1;
        int b_re_tmp;
        b_re_tmp = bcoef * b_k;
        b_re = static_cast<float>(b_data[b_re_tmp].re);
        b_im = static_cast<float>(b_data[b_re_tmp].im);
        b_re_tmp = b_acoef * b_k + a.size(0) * varargin_2;
        f = a[b_re_tmp].re;
        f1 = a[b_re_tmp].im;
        b_re_tmp = b_k + c.size(0) * k;
        c[b_re_tmp].re = f * b_re - f1 * b_im;
        c[b_re_tmp].im = f * b_im + f1 * b_re;
      }
    }
  }
}

} // namespace coder

// End of code generation (bsxfun.cpp)
