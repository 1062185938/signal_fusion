//
// FFTImplementationCallback.h
//
// Code generation for function 'FFTImplementationCallback'
//

#ifndef FFTIMPLEMENTATIONCALLBACK_H
#define FFTIMPLEMENTATIONCALLBACK_H

// Include files
#include "rtwtypes.h"
#include "coder_array.h"
#include <cstddef>
#include <cstdlib>

// Type Definitions
namespace coder {
namespace internal {
namespace fft {
class FFTImplementationCallback {
public:
  static int get_algo_sizes(int nfft, boolean_T useRadix2, int &nRows);
  static void r2br_r2dit_trig(const array<creal32_T, 2U> &x,
                              array<creal32_T, 2U> &y);
  static void r2br_r2dit_trig(const array<creal32_T, 2U> &x, int n1_unsigned,
                              const float costab_data[],
                              const float sintab_data[],
                              array<creal32_T, 2U> &y);
  static void dobluesteinfft(const array<creal32_T, 2U> &x, int n2blue,
                             int nfft, const float costab_data[],
                             const float sintab_data[],
                             const float sintabinv_data[],
                             array<creal32_T, 2U> &y);
  static void generate_twiddle_tables(int nRows, boolean_T useRadix2,
                                      array<double, 2U> &costab,
                                      array<double, 2U> &sintab,
                                      array<double, 2U> &sintabinv);
  static void dobluesteinfft(const array<double, 1U> &x, int n2blue, int nfft,
                             const array<double, 2U> &costab,
                             const array<double, 2U> &sintab,
                             const array<double, 2U> &sintabinv,
                             array<creal_T, 1U> &y);
  static void r2br_r2dit_trig(const array<creal_T, 2U> &x, int n1_unsigned,
                              const array<double, 2U> &costab,
                              const array<double, 2U> &sintab,
                              array<creal_T, 2U> &y);
  static void dobluesteinfft(const array<creal_T, 2U> &x, int n2blue, int nfft,
                             const array<double, 2U> &costab,
                             const array<double, 2U> &sintab,
                             const array<double, 2U> &sintabinv,
                             array<creal_T, 2U> &y);
  static void doHalfLengthRadix2(const array<double, 1U> &x,
                                 array<creal_T, 1U> &y, int unsigned_nRows,
                                 const array<double, 2U> &costab,
                                 const array<double, 2U> &sintab);

protected:
  static int r2br_r2dit_trig(const creal32_T x_data[], int x_size,
                             int n1_unsigned, const float costab_data[],
                             const float sintab_data[], creal32_T y_data[]);
  static void b_generate_twiddle_tables(int nRows, array<double, 2U> &costab,
                                        array<double, 2U> &sintab,
                                        array<double, 2U> &sintabinv);
  static void get_half_twiddle_tables(
      const array<double, 2U> &costab, const array<double, 2U> &sintab,
      const array<double, 2U> &costabinv, const array<double, 2U> &sintabinv,
      array<double, 2U> &hcostab, array<double, 2U> &hsintab,
      array<double, 2U> &hcostabinv, array<double, 2U> &hsintabinv);
  static void r2br_r2dit_trig_impl(const array<creal_T, 1U> &x,
                                   int unsigned_nRows,
                                   const array<double, 2U> &costab,
                                   const array<double, 2U> &sintab,
                                   array<creal_T, 1U> &y);
  static void b_r2br_r2dit_trig_impl(const array<creal_T, 1U> &x,
                                     int unsigned_nRows,
                                     const array<double, 2U> &costab,
                                     const array<double, 2U> &sintab,
                                     array<creal_T, 1U> &y);
  static void doHalfLengthBluestein(
      const array<double, 1U> &x, array<creal_T, 1U> &y, int nrowsx, int nRows,
      int nfft, const array<creal_T, 1U> &wwc, const array<double, 2U> &costab,
      const array<double, 2U> &sintab, const array<double, 2U> &costabinv,
      const array<double, 2U> &sintabinv);
};

} // namespace fft
} // namespace internal
} // namespace coder

#endif
// End of code generation (FFTImplementationCallback.h)
