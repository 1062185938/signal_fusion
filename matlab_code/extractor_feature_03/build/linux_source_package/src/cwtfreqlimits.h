//
// cwtfreqlimits.h
//
// Code generation for function 'cwtfreqlimits'
//

#ifndef CWTFREQLIMITS_H
#define CWTFREQLIMITS_H

// Include files
#include "rtwtypes.h"
#include <cstddef>
#include <cstdlib>

// Function Declarations
namespace coder {
namespace wavelet {
namespace internal {
namespace cwt {
double getFreqFromCutoffAmor(double cutoff, double cf);

double getFreqFromCutoffBump(double cutoff, double cf);

double getFreqFromCutoffMorse(double cutoff, double cf, double ga, double be);

} // namespace cwt
} // namespace internal
} // namespace wavelet
} // namespace coder

#endif
// End of code generation (cwtfreqlimits.h)
