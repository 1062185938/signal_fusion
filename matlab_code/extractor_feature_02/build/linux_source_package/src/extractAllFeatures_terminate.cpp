//
// extractAllFeatures_terminate.cpp
//
// Code generation for function 'extractAllFeatures_terminate'
//

// Include files
#include "extractAllFeatures_terminate.h"
#include "extractAllFeatures_data.h"
#include "rt_nonfinite.h"
#include "omp.h"
#include <cstring>

// Function Definitions
void extractAllFeatures_terminate()
{
  omp_destroy_nest_lock(&extractAllFeatures_nestLockGlobal);
  isInitialized_extractAllFeatures = false;
}

// End of code generation (extractAllFeatures_terminate.cpp)
