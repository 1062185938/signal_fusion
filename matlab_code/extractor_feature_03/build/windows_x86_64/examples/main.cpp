//
// main.cpp
//
// Code generation for function 'main'
//

/*************************************************************************/
/* This automatically generated example C++ main file shows how to call  */
/* entry-point functions that MATLAB Coder generated. You must customize */
/* this file for your application. Do not modify this file directly.     */
/* Instead, make a copy of this file, modify it, and integrate it into   */
/* your development environment.                                         */
/*                                                                       */
/* This file initializes entry-point function arguments to a default     */
/* size and value before calling the entry-point functions. It does      */
/* not store or use any values returned from the entry-point functions.  */
/* If necessary, it does pre-allocate memory for returned values.        */
/* You can use this file as a starting point for a main function that    */
/* you can deploy in your application.                                   */
/*                                                                       */
/* After you copy the file, and before you deploy it, you must make the  */
/* following changes:                                                    */
/* * For variable-size function arguments, change the example sizes to   */
/* the sizes that your application requires.                             */
/* * Change the example values of function arguments to the values that  */
/* your application requires.                                            */
/* * If the entry-point functions return values, store these values or   */
/* otherwise use them as required by your application.                   */
/*                                                                       */
/*************************************************************************/

// Include files
#include "main.h"
#include "extractAllFeatures.h"
#include "extractAllFeatures_initialize.h"
#include "extractAllFeatures_terminate.h"
#include "rt_nonfinite.h"
#include <cstring>

// Function Declarations
static void argInit_16384x1_real32_T(float result[16384]);

static int argInit_int32_T();

static float argInit_real32_T();

static double argInit_real_T();

// Function Definitions
static void argInit_16384x1_real32_T(float result[16384])
{
  // Loop over the array to initialize each element.
  for (int idx0{0}; idx0 < 16384; idx0++) {
    // Set the value of the array element.
    // Change this value to the value that the application requires.
    result[idx0] = argInit_real32_T();
  }
}

static int argInit_int32_T()
{
  return 0;
}

static float argInit_real32_T()
{
  return 0.0F;
}

static double argInit_real_T()
{
  return 0.0;
}

int main(int, char **)
{
  // Initialize the application.
  // You do not need to do this more than one time.
  extractAllFeatures_initialize();
  // Invoke the entry-point functions.
  // You can call entry-point functions multiple times.
  main_extractAllFeatures();
  // Terminate the application.
  // You do not need to do this more than one time.
  extractAllFeatures_terminate();
  return 0;
}

void main_extractAllFeatures()
{
  static float iData_tmp[16384];
  float features[64];
  int status;
  // Initialize function 'extractAllFeatures' input arguments.
  // Initialize function input argument 'iData'.
  argInit_16384x1_real32_T(iData_tmp);
  // Initialize function input argument 'qData'.
  // Call the entry-point 'extractAllFeatures'.
  extractAllFeatures(iData_tmp, iData_tmp, argInit_int32_T(), argInit_real_T(),
                     features, &status);
}

// End of code generation (main.cpp)
