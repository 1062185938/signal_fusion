//
// computeperiodogram.h
//
// Code generation for function 'computeperiodogram'
//

#ifndef COMPUTEPERIODOGRAM_H
#define COMPUTEPERIODOGRAM_H

// Include files
#include "rtwtypes.h"
#include <cstddef>
#include <cstdlib>

// Function Declarations
namespace coder {
void computeperiodogram(const creal32_T x_data[], int x_size,
                        const float win_data[], int win_size, float sampleFreq,
                        float Pxx[4096], float F[4096]);

}

#endif
// End of code generation (computeperiodogram.h)
