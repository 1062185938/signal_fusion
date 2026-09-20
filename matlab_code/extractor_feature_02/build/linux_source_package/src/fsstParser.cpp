//
// fsstParser.cpp
//
// Code generation for function 'fsstParser'
//

// Include files
#include "fsstParser.h"
#include "casyi.h"
#include "cmlri.h"
#include "gammaln.h"
#include "log.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include "omp.h"
#include <cmath>
#include <cstring>

// Function Definitions
namespace coder {
namespace b_signal {
namespace internal {
namespace fsst {
double fsstParser(const array<creal32_T, 1U> &x, double varargin_1,
                  double varargin_2_data[], int &varargin_2_size)
{
  creal_T cz;
  creal_T tmp;
  creal_T zd;
  double Fs;
  double aa;
  double acz;
  double ak;
  double az;
  double b_atol;
  double cz_tmp;
  double im;
  double r;
  double rs;
  double s;
  int b_nw;
  int c_nw;
  int inw;
  int iseven;
  int mid;
  int nw;
  boolean_T guard1;
  nw = static_cast<int>(std::fmin(256.0, static_cast<double>(x.size(0))));
  iseven = 1 - static_cast<int>(static_cast<unsigned int>(nw) & 1U);
  mid = (nw >> 1) + 1;
  if (static_cast<int>(((nw - mid) + 1) * 10 < 800)) {
    for (int k{mid}; k <= nw; k++) {
      r = static_cast<double>(iseven + ((k - mid) << 1)) /
          (static_cast<double>(nw) - 1.0);
      Fs = 10.0 * std::sqrt((1.0 - r) * (r + 1.0));
      zd.re = Fs;
      zd.im = 0.0;
      if (!std::isnan(Fs)) {
        if (Fs > 0.0) {
          az = Fs;
        } else {
          az = 0.0;
        }
        guard1 = false;
        if (az <= 2.0) {
          c_nw = 0;
          if ((Fs > 0.0) && (!(Fs < 2.2250738585072014E-305))) {
            tmp.re = 0.5 * Fs;
            tmp.im = 0.0;
            if (Fs > 4.7170688552396617E-153) {
              cz_tmp = tmp.re * tmp.re;
              cz.re = cz_tmp;
              Fs = tmp.re * 0.0;
              Fs += Fs;
              cz.im = Fs;
              if (cz_tmp > Fs) {
                acz = cz_tmp;
              } else if (std::isnan(Fs)) {
                acz = rtNaN;
              } else {
                acz = cz_tmp * 1.4142135623730951;
              }
            } else {
              cz.re = 0.0;
              cz.im = 0.0;
              acz = 0.0;
            }
            b_log(tmp);
            r = 1.0;
            gammaln(r);
            tmp.re = tmp.re * 0.0 - r;
            tmp.im *= 0.0;
            if (tmp.re > -700.92179369444591) {
              b_atol = 2.2204460492503131E-16 * acz;
              if (!(acz < 2.2204460492503131E-16)) {
                tmp.re = 1.0;
                tmp.im = 0.0;
                ak = 3.0;
                s = 1.0;
                aa = 2.0;
                do {
                  rs = 1.0 / s;
                  Fs = tmp.re * cz.re - tmp.im * cz.im;
                  cz_tmp = tmp.re * cz.im + tmp.im * cz.re;
                  tmp.re = rs * Fs;
                  tmp.im = rs * cz_tmp;
                  s += ak;
                  ak += 2.0;
                  aa = aa * acz * rs;
                } while (!!(aa > b_atol));
              }
            } else {
              c_nw = 1;
              if (acz > 0.0) {
                c_nw = -1;
              }
            }
          }
          if (c_nw < 0) {
            b_nw = 1;
          } else {
            b_nw = c_nw;
          }
          if ((1 - b_nw != 0) && (c_nw < 0)) {
            guard1 = true;
          }
        } else {
          guard1 = true;
        }
        if (guard1) {
          if (az < 21.784271729432426) {
            cmlri(zd, tmp);
          } else {
            casyi(zd, tmp);
          }
        }
      }
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        rs, inw, aa, s, ak, b_atol, r, acz, cz, tmp, c_nw, az, zd, guard1, im)

    for (int k = mid; k <= nw; k++) {
      r = static_cast<double>(iseven + ((k - mid) << 1)) /
          (static_cast<double>(nw) - 1.0);
      r = 10.0 * std::sqrt((1.0 - r) * (r + 1.0));
      zd.re = r;
      zd.im = 0.0;
      if (!std::isnan(r)) {
        if (r > 0.0) {
          az = r;
        } else {
          az = 0.0;
        }
        guard1 = false;
        if (az <= 2.0) {
          c_nw = 0;
          if ((r > 0.0) && (!(r < 2.2250738585072014E-305))) {
            tmp.re = 0.5 * r;
            tmp.im = 0.0;
            if (r > 4.7170688552396617E-153) {
              acz = tmp.re * tmp.re;
              cz.re = acz;
              r = tmp.re * 0.0;
              r += r;
              cz.im = r;
              if (!(acz > r)) {
                if (std::isnan(r)) {
                  acz = rtNaN;
                } else {
                  acz *= 1.4142135623730951;
                }
              }
            } else {
              cz.re = 0.0;
              cz.im = 0.0;
              acz = 0.0;
            }
            b_log(tmp);
            r = 1.0;
            gammaln(r);
            r = tmp.re * 0.0 - r;
            tmp.re = r;
            tmp.im *= 0.0;
            if (r > -700.92179369444591) {
              b_atol = 2.2204460492503131E-16 * acz;
              if (!(acz < 2.2204460492503131E-16)) {
                tmp.re = 1.0;
                tmp.im = 0.0;
                ak = 3.0;
                s = 1.0;
                aa = 2.0;
                do {
                  rs = 1.0 / s;
                  r = tmp.re * cz.re - tmp.im * cz.im;
                  im = tmp.re * cz.im + tmp.im * cz.re;
                  tmp.re = rs * r;
                  tmp.im = rs * im;
                  s += ak;
                  ak += 2.0;
                  aa = aa * acz * rs;
                } while (!!(aa > b_atol));
              }
            } else {
              c_nw = 1;
              if (acz > 0.0) {
                c_nw = -1;
              }
            }
          }
          if (c_nw < 0) {
            inw = 1;
          } else {
            inw = c_nw;
          }
          if ((1 - inw != 0) && (c_nw < 0)) {
            guard1 = true;
          }
        } else {
          guard1 = true;
        }
        if (guard1) {
          if (az < 21.784271729432426) {
            cmlri(zd, tmp);
          } else {
            casyi(zd, tmp);
          }
        }
      }
    }
  }
  Fs = varargin_1;
  if (varargin_2_size == 1) {
    if (varargin_2_data[0] == std::floor(varargin_2_data[0])) {
      b_nw = static_cast<int>(varargin_2_data[0]);
    } else {
      cz_tmp = varargin_2_data[0];
      if (varargin_2_data[0] < 4.503599627370496E+15) {
        if (varargin_2_data[0] >= 0.5) {
          cz_tmp = std::floor(varargin_2_data[0] + 0.5);
        } else {
          cz_tmp = varargin_2_data[0] * 0.0;
        }
      }
      b_nw = static_cast<int>(cz_tmp);
    }
    varargin_2_size = b_nw;
    for (int i{0}; i < b_nw; i++) {
      varargin_2_data[i] = 1.0;
    }
  }
  return Fs;
}

} // namespace fsst
} // namespace internal
} // namespace b_signal
} // namespace coder

// End of code generation (fsstParser.cpp)
