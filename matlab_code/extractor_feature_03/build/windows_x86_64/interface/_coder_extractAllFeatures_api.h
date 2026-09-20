//
// _coder_extractAllFeatures_api.h
//
// Code generation for function 'extractAllFeatures'
//

#ifndef _CODER_EXTRACTALLFEATURES_API_H
#define _CODER_EXTRACTALLFEATURES_API_H

// Include files
#include "extractAllFeatures_spec.h"
#include "emlrt.h"
#include "mex.h"
#include "tmwtypes.h"
#include <algorithm>
#include <cstring>

// Variable Declarations
extern emlrtCTX emlrtRootTLSGlobal;
extern emlrtContext emlrtContextGlobal;

// Function Declarations
void extractAllFeatures(real32_T iData[16384], real32_T qData[16384],
                        int32_T signalLength, real_T sampleRate,
                        real32_T features[64], int32_T *status);

void extractAllFeatures_api(const mxArray *const prhs[4], int32_T nlhs,
                            const mxArray *plhs[2]);

void extractAllFeatures_atexit();

void extractAllFeatures_initialize();

void extractAllFeatures_terminate();

void extractAllFeatures_xil_shutdown();

void extractAllFeatures_xil_terminate();

#endif
// End of code generation (_coder_extractAllFeatures_api.h)
