//
// cmlri.cpp
//
// Code generation for function 'cmlri'
//

// Include files
#include "cmlri.h"
#include "extractAllFeatures_data.h"
#include "gammaln.h"
#include "log.h"
#include "rt_nonfinite.h"
#include <cmath>
#include <cstring>

// Function Definitions
namespace coder {
int cmlri(const creal_T z, creal_T &y)
{
  creal_T rz;
  double ack;
  double ak;
  double az;
  double b_tmp;
  double bk;
  double ck_im;
  double ck_re;
  double flooraz;
  double p1_im;
  double p1_re;
  double p2_im;
  double p2_re;
  double pt_im;
  double pt_re;
  double rho;
  double rho2;
  double s_re;
  double tst;
  int icounter;
  int itime;
  int nz;
  boolean_T earlyExit;
  boolean_T exitg1;
  nz = 0;
  s_re = std::abs(z.re);
  b_tmp = std::abs(z.im);
  if (s_re < b_tmp) {
    rho2 = s_re / b_tmp;
    az = b_tmp * std::sqrt(rho2 * rho2 + 1.0);
  } else if (s_re > b_tmp) {
    rho2 = b_tmp / s_re;
    az = s_re * std::sqrt(rho2 * rho2 + 1.0);
  } else if (std::isnan(b_tmp)) {
    az = rtNaN;
  } else {
    az = s_re * 1.4142135623730951;
  }
  flooraz = std::floor(az);
  if (z.im == 0.0) {
    ck_re = (flooraz + 1.0) / z.re;
    ck_im = 0.0;
    rz.re = 2.0 / z.re;
    rz.im = 0.0;
  } else if (z.re == 0.0) {
    ck_re = 0.0;
    ck_im = -((flooraz + 1.0) / z.im);
    rz.re = 0.0;
    rz.im = -(2.0 / z.im);
  } else if (s_re > b_tmp) {
    rho = z.im / z.re;
    bk = z.re + rho * z.im;
    rho2 = rho * 0.0;
    ck_re = ((flooraz + 1.0) + rho2) / bk;
    ck_im = (0.0 - rho * (flooraz + 1.0)) / bk;
    rz.re = (rho2 + 2.0) / bk;
    rz.im = (0.0 - rho * 2.0) / bk;
  } else if (b_tmp == s_re) {
    if (z.re > 0.0) {
      rho2 = 0.5;
    } else {
      rho2 = -0.5;
    }
    ck_re = (flooraz + 1.0) * rho2 / s_re;
    rho = 0.0 * rho2;
    if (z.im > 0.0) {
      bk = 0.5;
    } else {
      bk = -0.5;
    }
    ck_im = (rho - (flooraz + 1.0) * bk) / s_re;
    rz.re = 2.0 * rho2 / s_re;
    rz.im = (rho - 2.0 * bk) / s_re;
  } else {
    rho = z.re / z.im;
    bk = z.im + rho * z.re;
    ck_re = rho * (flooraz + 1.0) / bk;
    rho2 = rho * 0.0;
    ck_im = (rho2 - (flooraz + 1.0)) / bk;
    rz.re = rho * 2.0 / bk;
    rz.im = (rho2 - 2.0) / bk;
  }
  p1_re = 0.0;
  p1_im = 0.0;
  p2_re = 1.0;
  p2_im = 0.0;
  ack = ((flooraz + 1.0) + 1.0) / az;
  rho = ack + std::sqrt(ack * ack - 1.0);
  rho2 = rho * rho;
  tst = (rho2 + rho2) / ((rho2 - 1.0) * (rho - 1.0)) / 2.2204460492503131E-16;
  ak = flooraz + 1.0;
  earlyExit = true;
  icounter = 1;
  itime = 0;
  exitg1 = false;
  while ((!exitg1) && (itime < 80)) {
    icounter++;
    pt_re = p2_re;
    pt_im = p2_im;
    rho2 = ck_re * p2_im + ck_im * p2_re;
    p2_re = p1_re - (ck_re * p2_re - ck_im * p2_im);
    p2_im = p1_im - rho2;
    p1_re = pt_re;
    p1_im = pt_im;
    ck_re += rz.re;
    ck_im += rz.im;
    rho2 = std::abs(p2_re);
    bk = std::abs(p2_im);
    if (rho2 < bk) {
      rho2 /= bk;
      rho = bk * std::sqrt(rho2 * rho2 + 1.0);
    } else if (rho2 > bk) {
      bk /= rho2;
      rho = rho2 * std::sqrt(bk * bk + 1.0);
    } else if (std::isnan(bk)) {
      rho = rtNaN;
    } else {
      rho = rho2 * 1.4142135623730951;
    }
    if (rho > tst * ak * ak) {
      earlyExit = false;
      exitg1 = true;
    } else {
      ak++;
      itime++;
    }
  }
  if (earlyExit) {
    nz = -2;
  } else {
    int kcounter;
    boolean_T guard1;
    kcounter = 1;
    guard1 = false;
    if (static_cast<int>(flooraz) <= 0) {
      int i;
      p1_re = 0.0;
      p1_im = 0.0;
      p2_re = 1.0;
      p2_im = 0.0;
      if (z.im == 0.0) {
        ck_re = 1.0 / z.re;
        ck_im = 0.0;
      } else if (z.re == 0.0) {
        ck_re = 0.0;
        ck_im = -(1.0 / z.im);
      } else if (s_re > b_tmp) {
        rho2 = z.im / z.re;
        rho = z.re + rho2 * z.im;
        ck_re = (rho2 * 0.0 + 1.0) / rho;
        ck_im = (0.0 - rho2) / rho;
      } else if (b_tmp == s_re) {
        if (z.re > 0.0) {
          rho2 = 0.5;
        } else {
          rho2 = -0.5;
        }
        ck_re = rho2 / s_re;
        if (z.im > 0.0) {
          rho = 0.5;
        } else {
          rho = -0.5;
        }
        ck_im = (0.0 * rho2 - rho) / s_re;
      } else {
        rho2 = z.re / z.im;
        rho = z.im + rho2 * z.re;
        ck_re = rho2 / rho;
        ck_im = (rho2 * 0.0 - 1.0) / rho;
      }
      tst = std::sqrt(1.0 / az / 2.2204460492503131E-16);
      itime = 1;
      earlyExit = true;
      i = 0;
      exitg1 = false;
      while ((!exitg1) && (i < 80)) {
        kcounter++;
        pt_re = p2_re;
        pt_im = p2_im;
        rho = ck_re * p2_im + ck_im * p2_re;
        p2_re = p1_re - (ck_re * p2_re - ck_im * p2_im);
        p2_im = p1_im - rho;
        p1_re = pt_re;
        p1_im = pt_im;
        ck_re += rz.re;
        ck_im += rz.im;
        b_tmp = std::abs(p2_re);
        rho2 = std::abs(p2_im);
        if (b_tmp < rho2) {
          b_tmp /= rho2;
          rho = rho2 * std::sqrt(b_tmp * b_tmp + 1.0);
        } else if (b_tmp > rho2) {
          rho2 /= b_tmp;
          rho = b_tmp * std::sqrt(rho2 * rho2 + 1.0);
        } else if (std::isnan(rho2)) {
          rho = rtNaN;
        } else {
          rho = b_tmp * 1.4142135623730951;
        }
        if (rho >= tst * ak * ak) {
          if (itime == 2) {
            earlyExit = false;
            exitg1 = true;
          } else {
            rho2 = std::abs(ck_re);
            b_tmp = std::abs(ck_im);
            if (rho2 < b_tmp) {
              rho2 /= b_tmp;
              ack = b_tmp * std::sqrt(rho2 * rho2 + 1.0);
            } else if (rho2 > b_tmp) {
              b_tmp /= rho2;
              ack = rho2 * std::sqrt(b_tmp * b_tmp + 1.0);
            } else if (std::isnan(b_tmp)) {
              ack = rtNaN;
            } else {
              ack = rho2 * 1.4142135623730951;
            }
            b_tmp = std::abs(pt_re);
            rho2 = std::abs(pt_im);
            if (b_tmp < rho2) {
              b_tmp /= rho2;
              b_tmp = rho2 * std::sqrt(b_tmp * b_tmp + 1.0);
            } else if (b_tmp > rho2) {
              rho2 /= b_tmp;
              b_tmp *= std::sqrt(rho2 * rho2 + 1.0);
            } else if (std::isnan(rho2)) {
              b_tmp = rtNaN;
            } else {
              b_tmp *= 1.4142135623730951;
            }
            rho = std::fmin(ack + std::sqrt(ack * ack - 1.0), rho / b_tmp);
            tst *= std::sqrt(rho / (rho * rho - 1.0));
            itime = 2;
            i++;
          }
        } else {
          i++;
        }
      }
      if (earlyExit) {
        nz = -2;
      } else {
        guard1 = true;
      }
    } else {
      guard1 = true;
    }
    if (guard1) {
      itime = icounter + static_cast<int>(flooraz);
      if (itime >= kcounter) {
        kcounter = itime;
      }
      tst = kcounter;
      p1_re = 0.0;
      p1_im = 0.0;
      p2_re = 1.0020841800044864E-289;
      p2_im = 0.0;
      rho = static_cast<double>(kcounter) + 1.0;
      gammaln(rho);
      az = 1.0;
      gammaln(az);
      bk = std::exp((rho - rho) - az);
      s_re = 0.0;
      ak = 0.0;
      for (int b_i{0}; b_i < kcounter; b_i++) {
        pt_re = p2_re;
        pt_im = p2_im;
        rho2 = tst * rz.re;
        b_tmp = tst * rz.im;
        rho = rho2 * p2_im + b_tmp * p2_re;
        p2_re = p1_re + (rho2 * p2_re - b_tmp * p2_im);
        p2_im = p1_im + rho;
        p1_re = pt_re;
        p1_im = pt_im;
        ack = bk * (1.0 - 0.0 / tst);
        bk += ack;
        s_re += bk * pt_re;
        ak += bk * pt_im;
        bk = ack;
        tst--;
      }
      y.re = p2_re;
      y.im = p2_im;
      b_log(rz);
      rho2 = 0.0 * rz.im;
      rho = 0.0 * rz.re;
      ck_re = ((rho - rho2) + z.re) - az;
      ck_im = (rho2 + rho) + z.im;
      p2_re += s_re;
      p2_im += ak;
      rho2 = std::abs(p2_re);
      rho = std::abs(p2_im);
      if (rho2 < rho) {
        rho2 /= rho;
        rho *= std::sqrt(rho2 * rho2 + 1.0);
      } else if (rho2 > rho) {
        rho /= rho2;
        rho = rho2 * std::sqrt(rho * rho + 1.0);
      } else if (std::isnan(rho)) {
        rho = rtNaN;
      } else {
        rho = rho2 * 1.4142135623730951;
      }
      p1_re = 1.0 / rho;
      if (ck_re == 0.0) {
        ck_re = std::cos(ck_im);
        ck_im = std::sin(ck_im);
      } else if (ck_im == 0.0) {
        ck_re = std::exp(ck_re);
        ck_im = 0.0;
      } else if (std::isinf(ck_im) && std::isinf(ck_re) && (ck_re < 0.0)) {
        ck_re = 0.0;
        ck_im = 0.0;
      } else {
        rho2 = std::exp(ck_re / 2.0);
        ck_re = rho2 * (rho2 * std::cos(ck_im));
        ck_im = rho2 * (rho2 * std::sin(ck_im));
      }
      b_tmp = ck_re * p1_re - ck_im * 0.0;
      rho = ck_re * 0.0 + ck_im * p1_re;
      bk = p2_re * p1_re + p2_im * 0.0;
      rho2 = p2_re * 0.0 - p2_im * p1_re;
      ck_re = b_tmp * bk - rho * rho2;
      ck_im = b_tmp * rho2 + rho * bk;
      rho2 = y.re * ck_im + y.im * ck_re;
      y.re = y.re * ck_re - y.im * ck_im;
      y.im = rho2;
    }
  }
  return nz;
}

} // namespace coder

// End of code generation (cmlri.cpp)
