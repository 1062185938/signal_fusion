//
// centerest.cpp
//
// Code generation for function 'centerest'
//

// Include files
#include "centerest.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include <cmath>
#include <cstring>

// Function Definitions
namespace coder {
namespace b_signal {
namespace internal {
namespace spectral {
void centerest(array<creal32_T, 2U> &y)
{
  array<creal32_T, 2U> buffer;
  if (std::fmod(static_cast<double>(y.size(0)), 2.0) == 0.0) {
    double p;
    int dim;
    p = static_cast<double>(y.size(0)) / 2.0 - 1.0;
    dim = 1;
    if (y.size(0) != 1) {
      dim = 0;
    }
    if ((y.size(0) != 0) && ((y.size(0) != 1) || (y.size(1) != 1))) {
      int b_i1;
      int ib;
      int lowerDim;
      int npages;
      int stride;
      int u1;
      short unnamed_idx_1;
      boolean_T shiftright;
      if (p < 0.0) {
        b_i1 = -static_cast<int>(p);
        shiftright = false;
      } else {
        b_i1 = static_cast<int>(p);
        shiftright = true;
      }
      ib = y.size(dim);
      if (ib <= 1) {
        b_i1 = 0;
      } else {
        if (b_i1 > ib) {
          b_i1 -= ib * static_cast<int>(static_cast<unsigned int>(b_i1) /
                                        static_cast<unsigned int>(ib));
        }
        if (b_i1 > (ib >> 1)) {
          b_i1 = ib - b_i1;
          shiftright = !shiftright;
        }
      }
      lowerDim = y.size(0);
      u1 = y.size(1);
      if (lowerDim >= u1) {
        u1 = lowerDim;
      }
      if (y.size(0) == 0) {
        u1 = 0;
      }
      unnamed_idx_1 = static_cast<short>(static_cast<unsigned short>(u1) >> 1);
      buffer.set_size(1, static_cast<int>(unnamed_idx_1));
      lowerDim = unnamed_idx_1;
      if (lowerDim - 1 >= 0) {
        std::memset(&buffer[0], 0,
                    static_cast<unsigned int>(lowerDim) * sizeof(creal32_T));
      }
      stride = 1;
      for (int k{0}; k < dim; k++) {
        stride *= y.size(0);
      }
      npages = 1;
      lowerDim = dim + 2;
      for (int k{lowerDim}; k < 3; k++) {
        npages *= y.size(1);
      }
      lowerDim = stride * ib;
      if ((ib > 1) && (b_i1 > 0)) {
        for (int i{0}; i < npages; i++) {
          dim = i * lowerDim;
          for (int j{0}; j < stride; j++) {
            int i1;
            i1 = dim + j;
            if (shiftright) {
              for (int k{0}; k < b_i1; k++) {
                buffer[k] = y[i1 + ((k + ib) - b_i1) * stride];
              }
              u1 = b_i1 + 1;
              for (int k{ib}; k >= u1; k--) {
                y[i1 + (k - 1) * stride] = y[i1 + ((k - b_i1) - 1) * stride];
              }
              for (int k{0}; k < b_i1; k++) {
                y[i1 + k * stride] = buffer[k];
              }
            } else {
              for (int k{0}; k < b_i1; k++) {
                buffer[k] = y[i1 + k * stride];
              }
              u1 = ib - b_i1;
              for (int k{0}; k < u1; k++) {
                y[i1 + k * stride] = y[i1 + (k + b_i1) * stride];
              }
              for (int k{0}; k < b_i1; k++) {
                y[i1 + ((k + ib) - b_i1) * stride] = buffer[k];
              }
            }
          }
        }
      }
    }
  } else if (y.size(0) > 1) {
    int dim;
    int i1;
    int npages;
    dim = (static_cast<unsigned short>(y.size(0)) >> 1) - 1;
    i1 = y.size(1);
    npages = y.size(0);
    if ((dim + 1) << 1 == y.size(0)) {
      int stride;
      stride = 1;
      for (int k{0}; k < i1; k++) {
        int b_i1;
        int ib;
        b_i1 = stride;
        stride += npages;
        ib = b_i1 + dim;
        for (int i{0}; i <= dim; i++) {
          float xtmp_im;
          float xtmp_re;
          int lowerDim;
          int u1;
          u1 = (b_i1 + i) - 1;
          xtmp_re = y[u1].re;
          xtmp_im = y[u1].im;
          lowerDim = ib + i;
          y[u1] = y[lowerDim];
          y[lowerDim].re = xtmp_re;
          y[lowerDim].im = xtmp_im;
        }
      }
    } else {
      int stride;
      stride = 1;
      for (int k{0}; k < i1; k++) {
        float xtmp_im;
        float xtmp_re;
        int b_i1;
        int ib;
        b_i1 = stride;
        stride += npages;
        ib = b_i1 + dim;
        xtmp_re = y[ib].re;
        xtmp_im = y[ib].im;
        for (int i{0}; i <= dim; i++) {
          int lowerDim;
          int u1;
          u1 = ib + i;
          lowerDim = (b_i1 + i) - 1;
          y[u1] = y[lowerDim];
          y[lowerDim] = y[u1 + 1];
        }
        ib = (ib + dim) + 1;
        y[ib].re = xtmp_re;
        y[ib].im = xtmp_im;
      }
    }
  }
}

} // namespace spectral
} // namespace internal
} // namespace b_signal
} // namespace coder

// End of code generation (centerest.cpp)
