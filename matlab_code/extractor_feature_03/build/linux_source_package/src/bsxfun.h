//
// bsxfun.h
//
// Code generation for function 'bsxfun'
//

#ifndef BSXFUN_H
#define BSXFUN_H

// Include files
#include "rtwtypes.h"
#include "coder_array.h"
#include <cstddef>
#include <cstdlib>

// Function Declarations
namespace coder {
void bsxfun(const float a_data[], int a_size, const array<creal32_T, 2U> &b,
            array<creal32_T, 2U> &c);

void bsxfun(const array<float, 2U> &a, const array<double, 2U> &b,
            array<float, 2U> &c);

void bsxfun(const array<creal32_T, 2U> &a, const creal_T b_data[], int b_size,
            array<creal32_T, 2U> &c);

} // namespace coder

#endif
// End of code generation (bsxfun.h)
