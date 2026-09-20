//
// cwtfreqlimits.cpp
//
// Code generation for function 'cwtfreqlimits'
//

// Include files
#include "cwtfreqlimits.h"
#include "anonymous_function.h"
#include "extractAllFeatures_internal_types.h"
#include "extractAllFeatures_rtwutil.h"
#include "rt_nonfinite.h"
#include <cmath>
#include <cstring>

// Function Definitions
namespace coder {
namespace wavelet {
namespace internal {
namespace cwt {
double getFreqFromCutoffAmor(double cutoff, double cf)
{
  c_anonymous_function psihat;
  double fa;
  double omegac;
  double p;
  psihat.workspace.alpha = 2.0 * cutoff;
  p = cf - cf;
  fa = psihat.workspace.alpha - 2.0 * std::exp(-(p * p) / 2.0);
  if (fa > 0.0) {
    omegac = cf + 38.729833462074168;
  } else {
    double a;
    double fb;
    a = cf;
    omegac = cf + 38.729833462074168;
    p = (cf + 38.729833462074168) - cf;
    fb = psihat.workspace.alpha - 2.0 * std::exp(-(p * p) / 2.0);
    if (fa == 0.0) {
      omegac = cf;
    } else if (!(fb == 0.0)) {
      double c;
      double d;
      double e;
      double fc;
      boolean_T exitg1;
      fc = fb;
      c = cf + 38.729833462074168;
      e = 0.0;
      d = 0.0;
      exitg1 = false;
      while ((!exitg1) && ((fb != 0.0) && (a != omegac))) {
        double m;
        double toler;
        if ((fb > 0.0) == (fc > 0.0)) {
          c = a;
          fc = fa;
          d = omegac - a;
          e = d;
        }
        if (std::abs(fc) < std::abs(fb)) {
          a = omegac;
          omegac = c;
          c = a;
          fa = fb;
          fb = fc;
          fc = fa;
        }
        m = 0.5 * (c - omegac);
        toler = 4.4408920985006262E-16 * std::fmax(std::abs(omegac), 1.0);
        if ((std::abs(m) <= toler) || (fb == 0.0)) {
          exitg1 = true;
        } else {
          if ((std::abs(e) < toler) || (std::abs(fa) <= std::abs(fb))) {
            d = m;
            e = m;
          } else {
            double s;
            s = fb / fa;
            if (a == c) {
              p = 2.0 * m * s;
              fa = 1.0 - s;
            } else {
              double r;
              fa /= fc;
              r = fb / fc;
              p = s * (2.0 * m * fa * (fa - r) - (omegac - a) * (r - 1.0));
              fa = (fa - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if (p > 0.0) {
              fa = -fa;
            } else {
              p = -p;
            }
            if ((2.0 * p < 3.0 * m * fa - std::abs(toler * fa)) &&
                (p < std::abs(0.5 * e * fa))) {
              e = d;
              d = p / fa;
            } else {
              d = m;
              e = m;
            }
          }
          a = omegac;
          fa = fb;
          if (std::abs(d) > toler) {
            omegac += d;
          } else if (omegac > c) {
            omegac -= toler;
          } else {
            omegac += toler;
          }
          p = omegac - cf;
          fb = psihat.workspace.alpha - 2.0 * std::exp(-(p * p) / 2.0);
        }
      }
    }
  }
  return omegac;
}

double getFreqFromCutoffBump(double cutoff, double cf)
{
  double omegac;
  int exponent;
  if (cutoff < 4.94065645841247E-323) {
    omegac = std::abs(cf + 0.6);
    if (std::isinf(omegac) || std::isnan(omegac)) {
      omegac = rtNaN;
    } else if (omegac < 4.4501477170144028E-308) {
      omegac = 4.94065645841247E-324;
    } else {
      std::frexp(omegac, &exponent);
      omegac = std::ldexp(1.0, exponent - 53);
    }
    omegac = (cf + 0.6) - 10.0 * omegac;
  } else {
    double a;
    double epsilon;
    double fa_tmp;
    double fb;
    a = 4.94065645841247E-324;
    epsilon = 0.99999999999999978;
    fa_tmp = std::log(2.0 * cutoff);
    omegac = ((fa_tmp + 1.0) - 0.69314718055994529) - 1.0;
    fb = ((fa_tmp + 2.251799813685248E+15) - 0.69314718055994529) - 1.0;
    if (((fa_tmp + 1.0) - 0.69314718055994529) - 1.0 == 0.0) {
      epsilon = 4.94065645841247E-324;
    } else {
      double c;
      double d;
      double e;
      double fc;
      boolean_T exitg1;
      fc = ((fa_tmp + 2.251799813685248E+15) - 0.69314718055994529) - 1.0;
      c = 0.99999999999999978;
      e = 0.0;
      d = 0.0;
      exitg1 = false;
      while ((!exitg1) && ((fb != 0.0) && (a != epsilon))) {
        double m;
        double toler;
        if ((fb > 0.0) == (fc > 0.0)) {
          c = a;
          fc = omegac;
          d = epsilon - a;
          e = d;
        }
        if (std::abs(fc) < std::abs(fb)) {
          a = epsilon;
          epsilon = c;
          c = a;
          omegac = fb;
          fb = fc;
          fc = omegac;
        }
        m = 0.5 * (c - epsilon);
        toler = 4.4408920985006262E-16 * std::fmax(std::abs(epsilon), 1.0);
        if ((std::abs(m) <= toler) || (fb == 0.0)) {
          exitg1 = true;
        } else {
          if ((std::abs(e) < toler) || (std::abs(omegac) <= std::abs(fb))) {
            d = m;
            e = m;
          } else {
            double q;
            double s;
            s = fb / omegac;
            if (a == c) {
              omegac = 2.0 * m * s;
              q = 1.0 - s;
            } else {
              double r;
              q = omegac / fc;
              r = fb / fc;
              omegac = s * (2.0 * m * q * (q - r) - (epsilon - a) * (r - 1.0));
              q = (q - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if (omegac > 0.0) {
              q = -q;
            } else {
              omegac = -omegac;
            }
            if ((2.0 * omegac < 3.0 * m * q - std::abs(toler * q)) &&
                (omegac < std::abs(0.5 * e * q))) {
              e = d;
              d = omegac / q;
            } else {
              d = m;
              e = m;
            }
          }
          a = epsilon;
          omegac = fb;
          if (std::abs(d) > toler) {
            epsilon += d;
          } else if (epsilon > c) {
            epsilon -= toler;
          } else {
            epsilon += toler;
          }
          fb = ((1.0 / (1.0 - epsilon * epsilon) + fa_tmp) -
                0.69314718055994529) -
               1.0;
        }
      }
    }
    omegac = 0.6 * epsilon + cf;
  }
  return omegac;
}

double getFreqFromCutoffMorse(double cutoff, double cf, double ga, double be)
{
  b_anonymous_function psihat;
  double fa;
  double omax;
  double omegac;
  psihat.workspace.anorm =
      2.0 * std::exp(be / ga * ((std::log(ga) - std::log(be)) + 1.0));
  psihat.workspace.alpha = 2.0 * cutoff;
  omax = rt_powd_snf(750.0, 1.0 / ga);
  fa = psihat.workspace.alpha - psihat.workspace.anorm * rt_powd_snf(cf, be) *
                                    std::exp(-rt_powd_snf(cf, ga));
  if (fa >= 0.0) {
    if (psihat.workspace.alpha - psihat.workspace.anorm *
                                     rt_powd_snf(omax, be) *
                                     std::exp(-rt_powd_snf(omax, ga)) ==
        fa) {
      omegac = omax;
    } else {
      omegac = cf;
    }
  } else {
    double a;
    double fb;
    a = cf;
    omegac = omax;
    fb = psihat.workspace.alpha - psihat.workspace.anorm *
                                      rt_powd_snf(omax, be) *
                                      std::exp(-rt_powd_snf(omax, ga));
    if (!(fb == 0.0)) {
      double d;
      double e;
      double fc;
      boolean_T exitg1;
      fc = fb;
      e = 0.0;
      d = 0.0;
      exitg1 = false;
      while ((!exitg1) && ((fb != 0.0) && (a != omegac))) {
        double m;
        double toler;
        if ((fb > 0.0) == (fc > 0.0)) {
          omax = a;
          fc = fa;
          d = omegac - a;
          e = d;
        }
        if (std::abs(fc) < std::abs(fb)) {
          a = omegac;
          omegac = omax;
          omax = a;
          fa = fb;
          fb = fc;
          fc = fa;
        }
        m = 0.5 * (omax - omegac);
        toler = 4.4408920985006262E-16 * std::fmax(std::abs(omegac), 1.0);
        if ((std::abs(m) <= toler) || (fb == 0.0)) {
          exitg1 = true;
        } else {
          if ((std::abs(e) < toler) || (std::abs(fa) <= std::abs(fb))) {
            d = m;
            e = m;
          } else {
            double q;
            double s;
            s = fb / fa;
            if (a == omax) {
              fa = 2.0 * m * s;
              q = 1.0 - s;
            } else {
              double r;
              q = fa / fc;
              r = fb / fc;
              fa = s * (2.0 * m * q * (q - r) - (omegac - a) * (r - 1.0));
              q = (q - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if (fa > 0.0) {
              q = -q;
            } else {
              fa = -fa;
            }
            if ((2.0 * fa < 3.0 * m * q - std::abs(toler * q)) &&
                (fa < std::abs(0.5 * e * q))) {
              e = d;
              d = fa / q;
            } else {
              d = m;
              e = m;
            }
          }
          a = omegac;
          fa = fb;
          if (std::abs(d) > toler) {
            omegac += d;
          } else if (omegac > omax) {
            omegac -= toler;
          } else {
            omegac += toler;
          }
          fb = psihat.workspace.alpha - psihat.workspace.anorm *
                                            rt_powd_snf(omegac, be) *
                                            std::exp(-rt_powd_snf(omegac, ga));
        }
      }
    }
  }
  return omegac;
}

} // namespace cwt
} // namespace internal
} // namespace wavelet
} // namespace coder

// End of code generation (cwtfreqlimits.cpp)
