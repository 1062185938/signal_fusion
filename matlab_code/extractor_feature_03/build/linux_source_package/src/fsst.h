//
// fsst.h
//
// Code generation for function 'fsst'
//

#ifndef FSST_H
#define FSST_H

// Include files
#include "rtwtypes.h"
#include "coder_array.h"
#include <cstddef>
#include <cstdlib>

// Function Declarations
namespace coder {
int fsst(const array<creal32_T, 1U> &x, double varargin_1,
         const double varargin_2_data[], int varargin_2_size,
         array<creal32_T, 2U> &sst, float f_data[], array<float, 2U> &t);

}

#endif
// End of code generation (fsst.h)
