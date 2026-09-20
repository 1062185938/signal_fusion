//
// extractAllFeatures_initialize.cpp
//
// Code generation for function 'extractAllFeatures_initialize'
//

// Include files
#include "extractAllFeatures_initialize.h"
#include "extractAllFeatures_data.h"
#include "rt_nonfinite.h"
#include "omp.h"
#include <cstring>

// Function Definitions
void extractAllFeatures_initialize()
{
  omp_init_nest_lock(&extractAllFeatures_nestLockGlobal);
  isInitialized_extractAllFeatures = true;
}

// End of code generation (extractAllFeatures_initialize.cpp)
