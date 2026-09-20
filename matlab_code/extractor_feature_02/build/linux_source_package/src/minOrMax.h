//
// minOrMax.h
//
// Code generation for function 'minOrMax'
//

#ifndef MINORMAX_H
#define MINORMAX_H

// Include files
#include "rtwtypes.h"
#include "coder_array.h"
#include <cstddef>
#include <cstdlib>

// Function Declarations
namespace coder {
namespace internal {
float b_maximum(const float x[512]);

float b_maximum(const float x[512], int &idx);

float maximum(const array<float, 1U> &x);

float maximum(const float x[4096], int &idx);

float maximum(const float x[4096]);

float maximum(const array<float, 1U> &x, int &idx);

} // namespace internal
} // namespace coder

#endif
// End of code generation (minOrMax.h)
