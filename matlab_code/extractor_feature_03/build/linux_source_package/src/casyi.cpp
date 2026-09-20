//
// casyi.cpp
//
// Code generation for function 'casyi'
//

// Include files
#include "casyi.h"
#include "extractAllFeatures_data.h"
#include "rt_nonfinite.h"
#include <cmath>
#include <cstring>

// Function Definitions
namespace coder {
int casyi(const creal_T z, creal_T &y)
{
  double ak1_im;
  double ak1_re;
  double az;
  double r;
  double sqk;
  double yi;
  double yr;
  int nz;
  nz = 0;
  sqk = std::abs(z.re);
  yi = std::abs(z.im);
  if (sqk < yi) {
    r = sqk / yi;
    az = yi * std::sqrt(r * r + 1.0);
  } else if (sqk > yi) {
    r = yi / sqk;
    az = sqk * std::sqrt(r * r + 1.0);
  } else if (std::isnan(yi)) {
    az = rtNaN;
  } else {
    az = sqk * 1.4142135623730951;
  }
  if (z.im == 0.0) {
    ak1_re = 0.15915494309189535 / z.re;
    ak1_im = 0.0;
  } else if (z.re == 0.0) {
    ak1_re = 0.0;
    ak1_im = -(0.15915494309189535 / z.im);
  } else if (sqk > yi) {
    yi = z.im / z.re;
    r = z.re + yi * z.im;
    ak1_re = (yi * 0.0 + 0.15915494309189535) / r;
    ak1_im = (0.0 - yi * 0.15915494309189535) / r;
  } else if (yi == sqk) {
    if (z.re > 0.0) {
      yi = 0.5;
    } else {
      yi = -0.5;
    }
    ak1_re = 0.15915494309189535 * yi / sqk;
    if (z.im > 0.0) {
      r = 0.5;
    } else {
      r = -0.5;
    }
    ak1_im = (0.0 * yi - 0.15915494309189535 * r) / sqk;
  } else {
    yi = z.re / z.im;
    r = z.im + yi * z.re;
    ak1_re = yi * 0.15915494309189535 / r;
    ak1_im = (yi * 0.0 - 0.15915494309189535) / r;
  }
  if (ak1_im == 0.0) {
    if (ak1_re < 0.0) {
      yr = 0.0;
      yi = std::sqrt(-ak1_re);
    } else {
      yr = std::sqrt(ak1_re);
      yi = 0.0;
    }
  } else if (ak1_re == 0.0) {
    if (ak1_im < 0.0) {
      yr = std::sqrt(-ak1_im / 2.0);
      yi = -yr;
    } else {
      yr = std::sqrt(ak1_im / 2.0);
      yi = yr;
    }
  } else if (std::isnan(ak1_re)) {
    yr = rtNaN;
    yi = rtNaN;
  } else if (std::isnan(ak1_im)) {
    yr = rtNaN;
    yi = rtNaN;
  } else if (std::isinf(ak1_im)) {
    yr = std::abs(ak1_im);
    yi = ak1_im;
  } else if (std::isinf(ak1_re)) {
    if (ak1_re < 0.0) {
      yr = 0.0;
      yi = ak1_im * -ak1_re;
    } else {
      yr = ak1_re;
      yi = 0.0;
    }
  } else {
    yr = std::abs(ak1_re);
    r = std::abs(ak1_im);
    if ((yr > 4.4942328371557893E+307) || (r > 4.4942328371557893E+307)) {
      yr *= 0.5;
      r *= 0.5;
      if (yr < r) {
        yi = yr / r;
        yi = r * std::sqrt(yi * yi + 1.0);
      } else if (yr > r) {
        r /= yr;
        yi = yr * std::sqrt(r * r + 1.0);
      } else {
        yi = yr * 1.4142135623730951;
      }
      if (yi > yr) {
        yr = std::sqrt(yi) * std::sqrt(yr / yi + 1.0);
      } else {
        yr = std::sqrt(yi) * 1.4142135623730951;
      }
    } else {
      if (yr < r) {
        yi = yr / r;
        r *= std::sqrt(yi * yi + 1.0);
      } else if (yr > r) {
        r /= yr;
        r = yr * std::sqrt(r * r + 1.0);
      } else {
        r = yr * 1.4142135623730951;
      }
      yr = std::sqrt((r + yr) * 0.5);
    }
    if (ak1_re > 0.0) {
      yi = 0.5 * (ak1_im / yr);
    } else {
      if (ak1_im < 0.0) {
        yi = -yr;
      } else {
        yi = yr;
      }
      yr = 0.5 * (ak1_im / yi);
    }
  }
  if (sqk > 700.92179369444591) {
    nz = -1;
    y.re = rtNaN;
    y.im = 0.0;
  } else {
    double aa;
    double ak;
    double bb;
    double cs1_im;
    double cs1_re;
    double cs2_im;
    double cs2_re;
    double dk_im;
    double dk_re;
    double ez_im;
    double ez_re;
    double im;
    double re;
    double sgn;
    double tmp_im;
    double tmp_re;
    int bk;
    signed char p1_im;
    boolean_T errflag;
    boolean_T exitg1;
    if (z.re == 0.0) {
      tmp_re = std::cos(z.im);
      tmp_im = std::sin(z.im);
    } else if (z.im == 0.0) {
      tmp_re = std::exp(z.re);
      tmp_im = 0.0;
    } else if (std::isinf(z.im) && std::isinf(z.re) && (z.re < 0.0)) {
      tmp_re = 0.0;
      tmp_im = 0.0;
    } else {
      r = std::exp(z.re / 2.0);
      tmp_re = r * (r * std::cos(z.im));
      tmp_im = r * (r * std::sin(z.im));
    }
    re = yr * tmp_re - yi * tmp_im;
    im = yr * tmp_im + yi * tmp_re;
    ez_re = 8.0 * z.re;
    ez_im = 8.0 * z.im;
    ak1_re = 8.0 * az;
    if (z.im != 0.0) {
      bk = 1;
      if (z.im < 0.0) {
        bk = -1;
      }
      p1_im = static_cast<signed char>(bk);
    } else {
      p1_im = 0;
    }
    sqk = -1.0;
    az = 2.2204460492503131E-16 / ak1_re;
    sgn = 1.0;
    cs1_re = 1.0;
    cs1_im = 0.0;
    cs2_re = 1.0;
    cs2_im = 0.0;
    tmp_re = 1.0;
    tmp_im = 0.0;
    ak = 0.0;
    aa = 1.0;
    bb = ak1_re;
    dk_re = ez_re;
    dk_im = ez_im;
    errflag = true;
    bk = 0;
    exitg1 = false;
    while ((!exitg1) && (bk < 45)) {
      tmp_re *= sqk;
      tmp_im *= sqk;
      if (dk_im == 0.0) {
        if (tmp_im == 0.0) {
          ak1_im = tmp_re / dk_re;
          tmp_im = 0.0;
        } else if (tmp_re == 0.0) {
          ak1_im = 0.0;
          tmp_im /= dk_re;
        } else {
          ak1_im = tmp_re / dk_re;
          tmp_im /= dk_re;
        }
      } else if (dk_re == 0.0) {
        if (tmp_re == 0.0) {
          ak1_im = tmp_im / dk_im;
          tmp_im = 0.0;
        } else if (tmp_im == 0.0) {
          ak1_im = 0.0;
          tmp_im = -(tmp_re / dk_im);
        } else {
          ak1_im = tmp_im / dk_im;
          tmp_im = -(tmp_re / dk_im);
        }
      } else {
        yr = std::abs(dk_re);
        yi = std::abs(dk_im);
        if (yr > yi) {
          yi = dk_im / dk_re;
          r = dk_re + yi * dk_im;
          ak1_im = (tmp_re + yi * tmp_im) / r;
          tmp_im = (tmp_im - yi * tmp_re) / r;
        } else if (yi == yr) {
          if (dk_re > 0.0) {
            yi = 0.5;
          } else {
            yi = -0.5;
          }
          if (dk_im > 0.0) {
            r = 0.5;
          } else {
            r = -0.5;
          }
          ak1_im = (tmp_re * yi + tmp_im * r) / yr;
          tmp_im = (tmp_im * yi - tmp_re * r) / yr;
        } else {
          yi = dk_re / dk_im;
          r = dk_im + yi * dk_re;
          ak1_im = (yi * tmp_re + tmp_im) / r;
          tmp_im = (yi * tmp_im - tmp_re) / r;
        }
      }
      tmp_re = ak1_im;
      cs2_re += ak1_im;
      cs2_im += tmp_im;
      sgn = -sgn;
      cs1_re += ak1_im * sgn;
      cs1_im += tmp_im * sgn;
      dk_re += ez_re;
      dk_im += ez_im;
      aa = aa * std::abs(sqk) / bb;
      bb += ak1_re;
      ak += 8.0;
      sqk -= ak;
      if (aa <= az) {
        errflag = false;
        exitg1 = true;
      } else {
        bk++;
      }
    }
    if (errflag) {
      nz = -2;
    } else {
      if (z.re + z.re < 700.92179369444591) {
        tmp_re = -2.0 * z.re;
        tmp_im = -2.0 * z.im;
        if (tmp_re == 0.0) {
          tmp_re = std::cos(tmp_im);
          tmp_im = std::sin(tmp_im);
        } else if (tmp_im == 0.0) {
          tmp_re = std::exp(tmp_re);
          tmp_im = 0.0;
        } else if (std::isinf(tmp_im) && std::isinf(tmp_re) && (tmp_re < 0.0)) {
          tmp_re = 0.0;
          tmp_im = 0.0;
        } else {
          r = std::exp(tmp_re / 2.0);
          tmp_re = r * (r * std::cos(tmp_im));
          tmp_im = r * (r * std::sin(tmp_im));
        }
        r = tmp_re * cs2_re - tmp_im * cs2_im;
        yi = tmp_re * cs2_im + tmp_im * cs2_re;
        cs1_re += r * 0.0 - yi * static_cast<double>(p1_im);
        cs1_im += r * static_cast<double>(p1_im) + yi * 0.0;
      }
      y.re = cs1_re * re - cs1_im * im;
      y.im = cs1_re * im + cs1_im * re;
    }
  }
  return nz;
}

} // namespace coder

// End of code generation (casyi.cpp)
