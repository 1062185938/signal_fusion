//
// extractAllFeatures.h
//
// Code generation for function 'extractAllFeatures'
//

#ifndef EXTRACTALLFEATURES_H
#define EXTRACTALLFEATURES_H

// Include files
#include "extractAllFeatures_spec.h"
#include "rtwtypes.h"
#include <cstddef>
#include <cstdlib>

// Function Declarations
EXTRACTALLFEATURES_DLL_EXPORT extern void
extractAllFeatures(const float iData[16384], const float qData[16384],
                   int signalLength, double sampleRate, float features[64],
                   int *status);

#endif
// End of code generation (extractAllFeatures.h)
