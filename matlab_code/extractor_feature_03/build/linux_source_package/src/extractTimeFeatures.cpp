//
// extractTimeFeatures.cpp
//
// Code generation for function 'extractTimeFeatures'
//

// Include files
#include "extractTimeFeatures.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include "omp.h"
#include <cstring>

// Function Definitions
void binary_expand_op(coder::array<float, 1U> &in1,
                      const coder::array<float, 1U> &in2, int in3, int in4,
                      int in5)
{
  int loop_ub;
  int stride_0_0;
  int stride_1_0;
  stride_1_0 = (in5 - in4) + 1;
  if (stride_1_0 == 1) {
    loop_ub = in3 + 1;
  } else {
    loop_ub = stride_1_0;
  }
  in1.set_size(loop_ub);
  stride_0_0 = (in3 + 1 != 1);
  stride_1_0 = (stride_1_0 != 1);
  if (static_cast<int>(loop_ub < 800)) {
    for (int i{0}; i < loop_ub; i++) {
      in1[i] = in2[i * stride_0_0] * in2[in4 + i * stride_1_0];
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

    for (int i = 0; i < loop_ub; i++) {
      in1[i] = in2[i * stride_0_0] * in2[in4 + i * stride_1_0];
    }
  }
}

// End of code generation (extractTimeFeatures.cpp)
