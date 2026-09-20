//
// computeDFT.h
//
// Code generation for function 'computeDFT'
//

#ifndef COMPUTEDFT_H
#define COMPUTEDFT_H

// Include files
#include "rtwtypes.h"
#include "coder_array.h"
#include <cstddef>
#include <cstdlib>

// Function Declarations
namespace coder {
int computeDFTviaFFT(const array<creal32_T, 2U> &xin, double nx, double nfft,
                     double Fs, array<creal32_T, 2U> &Xx, double f_data[]);

}

#endif
// End of code generation (computeDFT.h)
