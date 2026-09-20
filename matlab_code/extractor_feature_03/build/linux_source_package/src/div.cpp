//
// div.cpp
//
// Code generation for function 'div'
//

// Include files
#include "div.h"
#include "combineVectorElements.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include "omp.h"
#include <cmath>
#include <cstring>

// Function Definitions
float binary_expand_op_1(const coder::array<creal32_T, 1U> &in1, int in2,
                         int in3, int in4, float in5)
{
  coder::array<creal32_T, 1U> b_in1;
  creal32_T fc;
  float b_in1_im;
  float f2;
  float f3;
  float in1_im;
  float in1_re;
  float out1;
  int b_in1_re_tmp;
  int in1_re_tmp;
  int loop_ub;
  int stride_0_0;
  int stride_1_0;
  in1_re_tmp = (in4 - in3) + 1;
  if (in1_re_tmp == 1) {
    loop_ub = in2 + 1;
  } else {
    loop_ub = in1_re_tmp;
  }
  b_in1.set_size(loop_ub);
  stride_0_0 = (in2 + 1 != 1);
  stride_1_0 = (in1_re_tmp != 1);
  if (static_cast<int>(loop_ub < 800)) {
    for (int i{0}; i < loop_ub; i++) {
      float f;
      float f1;
      in1_re_tmp = i * stride_0_0;
      out1 = in1[in1_re_tmp].re;
      in1_im = -in1[in1_re_tmp].im;
      in1_re_tmp = in3 + i * stride_1_0;
      f = in1[in1_re_tmp].im;
      f1 = in1[in1_re_tmp].re;
      b_in1[i].re = out1 * f1 - in1_im * f;
      b_in1[i].im = out1 * f + in1_im * f1;
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        b_in1_re_tmp, in1_re, b_in1_im, f2, f3)

    for (int i = 0; i < loop_ub; i++) {
      b_in1_re_tmp = i * stride_0_0;
      in1_re = in1[b_in1_re_tmp].re;
      b_in1_im = -in1[b_in1_re_tmp].im;
      b_in1_re_tmp = in3 + i * stride_1_0;
      f2 = in1[b_in1_re_tmp].im;
      f3 = in1[b_in1_re_tmp].re;
      b_in1[i].re = in1_re * f3 - b_in1_im * f2;
      b_in1[i].im = in1_re * f2 + b_in1_im * f3;
    }
  }
  fc = coder::combineVectorElements(b_in1);
  out1 = std::abs(fc.re);
  in1_im = std::abs(fc.im);
  if (out1 < in1_im) {
    out1 /= in1_im;
    out1 = in1_im * std::sqrt(out1 * out1 + 1.0F);
  } else if (out1 > in1_im) {
    in1_im /= out1;
    out1 *= std::sqrt(in1_im * in1_im + 1.0F);
  } else if (std::isnan(in1_im)) {
    out1 = rtNaNF;
  } else {
    out1 *= 1.41421354F;
  }
  out1 /= in5 + 1.1920929E-7F;
  return out1;
}

// End of code generation (div.cpp)
