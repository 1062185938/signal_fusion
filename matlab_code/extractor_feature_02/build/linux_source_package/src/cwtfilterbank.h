//
// cwtfilterbank.h
//
// Code generation for function 'cwtfilterbank'
//

#ifndef CWTFILTERBANK_H
#define CWTFILTERBANK_H

// Include files
#include "rtwtypes.h"
#include "coder_array.h"
#include <cstddef>
#include <cstdlib>

// Type Definitions
namespace coder {
class cwtfilterbank {
public:
  cwtfilterbank *init(double varargin_2, const double varargin_8[2],
                      double varargin_12);

private:
  cwtfilterbank *setProperties(double varargin_2, const double varargin_8[2],
                               double varargin_12);

public:
  double VoicesPerOctave;
  char Wavelet[4];
  double SamplingFrequency;
  double SignalLength;
  double FrequencyLimits[2];
  double TimeBandwidth;
  double WaveletParameters[2];
  char Boundary[8];
  array<double, 2U> Scales;
  array<double, 2U> PsiDFT;
  array<double, 1U> WaveletCenterFrequencies;
  array<double, 2U> Omega;

private:
  double Beta;
  double Gamma;
  double SignalPad;
  double CutOff;
};

} // namespace coder

#endif
// End of code generation (cwtfilterbank.h)
