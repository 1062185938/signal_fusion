//
// ppval.h
//
// Code generation for function 'ppval'
//

#ifndef PPVAL_H
#define PPVAL_H

// Include files
#include "rtwtypes.h"
#include <cstddef>
#include <cstdlib>

// Type Declarations
struct d_struct_T;

// Function Declarations
namespace coder {
int ppval(const d_struct_T &pp, const double x_data[], int x_size,
          float v_data[]);

}

#endif
// End of code generation (ppval.h)
