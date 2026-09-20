//
// fft.h
//
// Code generation for function 'fft'
//

#ifndef FFT_H
#define FFT_H

// Include files
#include "rtwtypes.h"
#include "coder_array.h"
#include <cstddef>
#include <cstdlib>

// Function Declarations
namespace coder {
void fft(const creal32_T x[256], creal32_T y[256]);

void fft(const array<double, 2U> &x, array<creal_T, 2U> &y);

} // namespace coder

#endif
// End of code generation (fft.h)
