//
// abs.h
//
// Code generation for function 'abs'
//

#ifndef ABS_H
#define ABS_H

// Include files
#include "rtwtypes.h"
#include "coder_array.h"
#include <cstddef>
#include <cstdlib>

// Function Declarations
namespace coder {
float b_abs(const creal32_T x);

void b_abs(const array<creal32_T, 1U> &x, array<float, 1U> &y);

void b_abs(const array<creal32_T, 2U> &x, array<float, 2U> &y);

void b_abs(const array<creal_T, 2U> &x, array<double, 2U> &y);

void c_abs(const array<creal32_T, 2U> &x, array<float, 2U> &y);

} // namespace coder

#endif
// End of code generation (abs.h)
