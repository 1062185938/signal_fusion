//
// spectrogram.h
//
// Code generation for function 'spectrogram'
//

#ifndef SPECTROGRAM_H
#define SPECTROGRAM_H

// Include files
#include "rtwtypes.h"
#include "coder_array.h"
#include <cstddef>
#include <cstdlib>

// Function Declarations
namespace coder {
void spectrogram(const array<creal32_T, 1U> &x, const float varargin_1_data[],
                 int varargin_1_size, double varargin_2, double varargin_4,
                 array<creal32_T, 2U> &varargout_1, float varargout_2[512],
                 array<float, 2U> &varargout_3);

}

#endif
// End of code generation (spectrogram.h)
