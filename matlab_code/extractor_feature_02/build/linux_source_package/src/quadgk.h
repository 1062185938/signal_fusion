//
// quadgk.h
//
// Code generation for function 'quadgk'
//

#ifndef QUADGK_H
#define QUADGK_H

// Include files
#include "rtwtypes.h"
#include <cstddef>
#include <cstdlib>

// Type Declarations
namespace coder {
class anonymous_function;

}

// Function Declarations
namespace coder {
double quadgk(const anonymous_function fun);

int split(double x[650], int nx, double &pathlen);

} // namespace coder

#endif
// End of code generation (quadgk.h)
