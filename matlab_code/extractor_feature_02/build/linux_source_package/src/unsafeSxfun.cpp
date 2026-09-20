//
// unsafeSxfun.cpp
//
// Code generation for function 'unsafeSxfun'
//

// Include files
#include "unsafeSxfun.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include "omp.h"
#include <cstring>

// Function Definitions
void binary_expand_op_5(coder::array<float, 1U> &in1,
                        const coder::array<float, 1U> &in3, int in4, int in5,
                        const coder::array<float, 1U> &in6, int in7, int in8,
                        int in9)
{
  coder::array<float, 1U> b_in3;
  float b_varargin_1;
  int i;
  int loop_ub;
  int stride_0_0;
  int stride_1_0;
  int stride_2_0;
  stride_2_0 = (in8 - in7) + 1;
  stride_1_0 = (in5 - in4) + 1;
  if (in9 + 1 == 1) {
    if (stride_2_0 == 1) {
      loop_ub = stride_1_0;
    } else {
      loop_ub = stride_2_0;
    }
  } else {
    loop_ub = in9 + 1;
  }
  b_in3.set_size(loop_ub);
  stride_0_0 = (stride_1_0 != 1);
  stride_1_0 = (stride_2_0 != 1);
  stride_2_0 = (in9 + 1 != 1);
  i = (loop_ub < 800);
  if (i) {
    for (int i1{0}; i1 < loop_ub; i1++) {
      b_in3[i1] =
          (in3[in4 + i1 * stride_0_0] - 2.0F * in6[in7 + i1 * stride_1_0]) +
          in6[i1 * stride_2_0];
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

    for (int i1 = 0; i1 < loop_ub; i1++) {
      b_in3[i1] =
          (in3[in4 + i1 * stride_0_0] - 2.0F * in6[in7 + i1 * stride_1_0]) +
          in6[i1 * stride_2_0];
    }
  }
  in1.set_size(loop_ub);
  if (i) {
    for (int i2{0}; i2 < loop_ub; i2++) {
      float varargin_1;
      varargin_1 = b_in3[i2];
      in1[i2] = varargin_1 * varargin_1;
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        b_varargin_1)

    for (int i2 = 0; i2 < loop_ub; i2++) {
      b_varargin_1 = b_in3[i2];
      in1[i2] = b_varargin_1 * b_varargin_1;
    }
  }
}

// End of code generation (unsafeSxfun.cpp)
