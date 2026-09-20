//
// extractAllFeatures_internal_types.h
//
// Code generation for function 'extractAllFeatures'
//

#ifndef EXTRACTALLFEATURES_INTERNAL_TYPES_H
#define EXTRACTALLFEATURES_INTERNAL_TYPES_H

// Include files
#include "extractAllFeatures_types.h"
#include "rtwtypes.h"
#include "coder_bounded_array.h"

// Type Definitions
struct struct_T {
  double be;
  double ga;
};

struct b_struct_T {
  double alpha;
  double anorm;
};

struct c_struct_T {
  double alpha;
};

struct d_struct_T {
  coder::bounded_array<double, 258U, 2U> breaks;
  coder::bounded_array<float, 1028U, 2U> coefs;
};

#endif
// End of code generation (extractAllFeatures_internal_types.h)
