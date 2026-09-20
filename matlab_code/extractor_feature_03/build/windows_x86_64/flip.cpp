//
// flip.cpp
//
// Code generation for function 'flip'
//

// Include files
#include "flip.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include <cstring>

// Function Definitions
namespace coder {
void flip(array<creal_T, 2U> &x)
{
  int dim;
  dim = (x.size(0) == 1);
  if (x.size(0) != 0) {
    int i;
    i = x.size(dim);
    if (i > 1) {
      int npages;
      int pagelen;
      int vstride;
      vstride = 1;
      for (int k{0}; k < dim; k++) {
        vstride *= x.size(0);
      }
      pagelen = vstride * i;
      npages = 1;
      dim += 2;
      for (int k{dim}; k < 3; k++) {
        npages *= x.size(1);
      }
      dim = i >> 1;
      for (int k{0}; k < npages; k++) {
        for (int b_i{0}; b_i < vstride; b_i++) {
          int offset;
          offset = k * pagelen + b_i;
          for (int b_k{0}; b_k < dim; b_k++) {
            double tmp_im;
            double tmp_re;
            int i1;
            int tmp_re_tmp;
            tmp_re_tmp = offset + b_k * vstride;
            tmp_re = x[tmp_re_tmp].re;
            tmp_im = x[tmp_re_tmp].im;
            i1 = offset + ((i - b_k) - 1) * vstride;
            x[tmp_re_tmp] = x[i1];
            x[i1].re = tmp_re;
            x[i1].im = tmp_im;
          }
        }
      }
    }
  }
}

} // namespace coder

// End of code generation (flip.cpp)
