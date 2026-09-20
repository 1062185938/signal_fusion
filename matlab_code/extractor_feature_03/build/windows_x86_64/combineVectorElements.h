//
// combineVectorElements.h
//
// Code generation for function 'combineVectorElements'
//

#ifndef COMBINEVECTORELEMENTS_H
#define COMBINEVECTORELEMENTS_H

// Include files
#include "rtwtypes.h"
#include "coder_array.h"
#include <cstddef>
#include <cstdlib>

// Function Declarations
namespace coder {
int b_combineVectorElements(const boolean_T x_data[], int x_size);

float combineVectorElements(const float x[4096]);

creal32_T combineVectorElements(const array<creal32_T, 1U> &x);

} // namespace coder

#endif
// End of code generation (combineVectorElements.h)
