//
// std.h
//
// Code generation for function 'std'
//

#ifndef STD_H
#define STD_H

// Include files
#include "rtwtypes.h"
#include "coder_array.h"
#include <cstddef>
#include <cstdlib>

// Function Declarations
namespace coder {
float b_std(const array<creal32_T, 1U> &x);

float b_std(const float x_data[], int x_size);

float b_std(const float x[4096]);

} // namespace coder

#endif
// End of code generation (std.h)
