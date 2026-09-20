//
// fft.cpp
//
// Code generation for function 'fft'
//

// Include files
#include "fft.h"
#include "FFTImplementationCallback.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include <cstring>

// Function Definitions
namespace coder {
void fft(const creal32_T x[256], creal32_T y[256])
{
  static const float fv[129]{
      1.0F,           0.999698818F,  0.99879545F,    0.997290432F,
      0.99518472F,    0.992479563F,  0.989176512F,   0.985277653F,
      0.980785251F,   0.975702107F,  0.970031261F,   0.963776052F,
      0.956940353F,   0.949528158F,  0.941544056F,   0.932992816F,
      0.923879504F,   0.914209723F,  0.903989315F,   0.893224299F,
      0.881921232F,   0.870086968F,  0.857728601F,   0.84485358F,
      0.831469595F,   0.817584813F,  0.803207517F,   0.78834641F,
      0.773010433F,   0.757208824F,  0.740951121F,   0.724247098F,
      0.707106769F,   0.689540565F,  0.671559F,      0.653172851F,
      0.634393334F,   0.615231633F,  0.59569931F,    0.575808227F,
      0.555570245F,   0.534997642F,  0.514102757F,   0.492898226F,
      0.471396744F,   0.449611336F,  0.427555084F,   0.40524134F,
      0.382683456F,   0.359895051F,  0.336889863F,   0.313681751F,
      0.290284663F,   0.266712785F,  0.242980197F,   0.219101235F,
      0.195090324F,   0.170961902F,  0.146730468F,   0.122410677F,
      0.0980171412F,  0.0735645667F, 0.0490676761F,  0.024541229F,
      0.0F,           -0.024541229F, -0.0490676761F, -0.0735645667F,
      -0.0980171412F, -0.122410677F, -0.146730468F,  -0.170961902F,
      -0.195090324F,  -0.219101235F, -0.242980197F,  -0.266712785F,
      -0.290284663F,  -0.313681751F, -0.336889863F,  -0.359895051F,
      -0.382683456F,  -0.40524134F,  -0.427555084F,  -0.449611336F,
      -0.471396744F,  -0.492898226F, -0.514102757F,  -0.534997642F,
      -0.555570245F,  -0.575808227F, -0.59569931F,   -0.615231633F,
      -0.634393334F,  -0.653172851F, -0.671559F,     -0.689540565F,
      -0.707106769F,  -0.724247098F, -0.740951121F,  -0.757208824F,
      -0.773010433F,  -0.78834641F,  -0.803207517F,  -0.817584813F,
      -0.831469595F,  -0.84485358F,  -0.857728601F,  -0.870086968F,
      -0.881921232F,  -0.893224299F, -0.903989315F,  -0.914209723F,
      -0.923879504F,  -0.932992816F, -0.941544056F,  -0.949528158F,
      -0.956940353F,  -0.963776052F, -0.970031261F,  -0.975702107F,
      -0.980785251F,  -0.985277653F, -0.989176512F,  -0.992479563F,
      -0.99518472F,   -0.997290432F, -0.99879545F,   -0.999698818F,
      -1.0F};
  static const float fv1[129]{
      0.0F,           -0.024541229F,  -0.0490676761F, -0.0735645667F,
      -0.0980171412F, -0.122410677F,  -0.146730468F,  -0.170961902F,
      -0.195090324F,  -0.219101235F,  -0.242980197F,  -0.266712785F,
      -0.290284663F,  -0.313681751F,  -0.336889863F,  -0.359895051F,
      -0.382683456F,  -0.40524134F,   -0.427555084F,  -0.449611336F,
      -0.471396744F,  -0.492898226F,  -0.514102757F,  -0.534997642F,
      -0.555570245F,  -0.575808227F,  -0.59569931F,   -0.615231633F,
      -0.634393334F,  -0.653172851F,  -0.671559F,     -0.689540565F,
      -0.707106769F,  -0.724247098F,  -0.740951121F,  -0.757208824F,
      -0.773010433F,  -0.78834641F,   -0.803207517F,  -0.817584813F,
      -0.831469595F,  -0.84485358F,   -0.857728601F,  -0.870086968F,
      -0.881921232F,  -0.893224299F,  -0.903989315F,  -0.914209723F,
      -0.923879504F,  -0.932992816F,  -0.941544056F,  -0.949528158F,
      -0.956940353F,  -0.963776052F,  -0.970031261F,  -0.975702107F,
      -0.980785251F,  -0.985277653F,  -0.989176512F,  -0.992479563F,
      -0.99518472F,   -0.997290432F,  -0.99879545F,   -0.999698818F,
      -1.0F,          -0.999698818F,  -0.99879545F,   -0.997290432F,
      -0.99518472F,   -0.992479563F,  -0.989176512F,  -0.985277653F,
      -0.980785251F,  -0.975702107F,  -0.970031261F,  -0.963776052F,
      -0.956940353F,  -0.949528158F,  -0.941544056F,  -0.932992816F,
      -0.923879504F,  -0.914209723F,  -0.903989315F,  -0.893224299F,
      -0.881921232F,  -0.870086968F,  -0.857728601F,  -0.84485358F,
      -0.831469595F,  -0.817584813F,  -0.803207517F,  -0.78834641F,
      -0.773010433F,  -0.757208824F,  -0.740951121F,  -0.724247098F,
      -0.707106769F,  -0.689540565F,  -0.671559F,     -0.653172851F,
      -0.634393334F,  -0.615231633F,  -0.59569931F,   -0.575808227F,
      -0.555570245F,  -0.534997642F,  -0.514102757F,  -0.492898226F,
      -0.471396744F,  -0.449611336F,  -0.427555084F,  -0.40524134F,
      -0.382683456F,  -0.359895051F,  -0.336889863F,  -0.313681751F,
      -0.290284663F,  -0.266712785F,  -0.242980197F,  -0.219101235F,
      -0.195090324F,  -0.170961902F,  -0.146730468F,  -0.122410677F,
      -0.0980171412F, -0.0735645667F, -0.0490676761F, -0.024541229F,
      -0.0F};
  float re;
  float temp_im;
  float temp_re;
  float temp_re_tmp;
  float twid_re;
  int iDelta;
  int iDelta2;
  int iheight;
  int iy;
  int ju;
  int k;
  iy = 0;
  ju = 0;
  for (int i{0}; i < 255; i++) {
    boolean_T tst;
    y[iy] = x[i];
    iy = 256;
    tst = true;
    while (tst) {
      iy >>= 1;
      ju ^= iy;
      tst = ((ju & iy) == 0);
    }
    iy = ju;
  }
  y[iy] = x[255];
  for (int i{0}; i <= 254; i += 2) {
    temp_re = y[i + 1].re;
    temp_re_tmp = y[i + 1].im;
    temp_im = temp_re_tmp;
    re = y[i].re;
    twid_re = y[i].im;
    y[i + 1].re = re - temp_re;
    temp_re_tmp = twid_re - temp_re_tmp;
    y[i + 1].im = temp_re_tmp;
    re += temp_re;
    y[i].re = re;
    y[i].im = twid_re + temp_im;
  }
  iDelta = 2;
  iDelta2 = 4;
  k = 64;
  iheight = 253;
  while (k > 0) {
    int b_i;
    for (b_i = 0; b_i < iheight; b_i += iDelta2) {
      iy = b_i + iDelta;
      temp_re = y[iy].re;
      temp_im = y[iy].im;
      y[iy].re = y[b_i].re - temp_re;
      y[iy].im = y[b_i].im - temp_im;
      y[b_i].re += temp_re;
      y[b_i].im += temp_im;
    }
    iy = 1;
    for (ju = k; ju < 128; ju += k) {
      float twid_im;
      int ihi;
      twid_re = fv[ju];
      twid_im = fv1[ju];
      b_i = iy;
      ihi = iy + iheight;
      while (b_i < ihi) {
        int b_temp_re_tmp;
        b_temp_re_tmp = b_i + iDelta;
        temp_re_tmp = y[b_temp_re_tmp].im;
        re = y[b_temp_re_tmp].re;
        temp_re = twid_re * re - twid_im * temp_re_tmp;
        temp_im = twid_re * temp_re_tmp + twid_im * re;
        y[b_temp_re_tmp].re = y[b_i].re - temp_re;
        y[b_temp_re_tmp].im = y[b_i].im - temp_im;
        y[b_i].re += temp_re;
        y[b_i].im += temp_im;
        b_i += iDelta2;
      }
      iy++;
    }
    k >>= 1;
    iDelta = iDelta2;
    iDelta2 += iDelta2;
    iheight -= iDelta;
  }
}

void fft(const array<double, 2U> &x, array<creal_T, 2U> &y)
{
  array<creal_T, 1U> yCol;
  array<double, 2U> costab;
  array<double, 2U> sintab;
  array<double, 2U> sintabinv;
  array<double, 1U> b_x;
  int nRows;
  if (x.size(1) == 0) {
    y.set_size(1, 0);
  } else {
    int N2blue;
    boolean_T useRadix2;
    useRadix2 = ((static_cast<unsigned int>(x.size(1)) &
                  static_cast<unsigned int>(x.size(1) - 1)) == 0U);
    N2blue = internal::fft::FFTImplementationCallback::get_algo_sizes(
        x.size(1), useRadix2, nRows);
    internal::fft::FFTImplementationCallback::generate_twiddle_tables(
        nRows, useRadix2, costab, sintab, sintabinv);
    if (useRadix2) {
      b_x = x.reshape(x.size(1));
      nRows = x.size(1);
      yCol.set_size(nRows);
      if (nRows > b_x.size(0)) {
        yCol.set_size(nRows);
        std::memset(&yCol[0], 0,
                    static_cast<unsigned int>(nRows) * sizeof(creal_T));
      }
      internal::fft::FFTImplementationCallback::doHalfLengthRadix2(
          b_x, yCol, nRows, costab, sintab);
    } else {
      b_x = x.reshape(x.size(1));
      internal::fft::FFTImplementationCallback::dobluesteinfft(
          b_x, N2blue, x.size(1), costab, sintab, sintabinv, yCol);
    }
    nRows = x.size(1);
    y.set_size(1, x.size(1));
    for (int i{0}; i < nRows; i++) {
      y[i] = yCol[i];
    }
  }
}

} // namespace coder

// End of code generation (fft.cpp)
