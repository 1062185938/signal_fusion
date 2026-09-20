//
// linspace.cpp
//
// Code generation for function 'linspace'
//

// Include files
#include "linspace.h"
#include "extractAllFeatures_data.h"
#include "rt_nonfinite.h"
#include <cstring>
#include <emmintrin.h>
#include <xmmintrin.h>

// Function Definitions
namespace coder {
void linspace(float d2, float y[11])
{
  __m128 r;
  float delta1;
  y[10] = d2;
  y[0] = 0.0F;
  delta1 = d2 / 10.0F;
  r = _mm_set1_ps(delta1);
  _mm_storeu_ps(&y[1],
                _mm_mul_ps(_mm_cvtepi32_ps(_mm_add_epi32(
                               _mm_set1_epi32(1),
                               _mm_loadu_si128((const __m128i *)&iv[0]))),
                           r));
  _mm_storeu_ps(&y[5],
                _mm_mul_ps(_mm_cvtepi32_ps(_mm_add_epi32(
                               _mm_set1_epi32(5),
                               _mm_loadu_si128((const __m128i *)&iv[0]))),
                           r));
  y[9] = 9.0F * delta1;
}

} // namespace coder

// End of code generation (linspace.cpp)
