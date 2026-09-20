//
// wavCFandSD.cpp
//
// Code generation for function 'wavCFandSD'
//

// Include files
#include "wavCFandSD.h"
#include "anonymous_function.h"
#include "extractAllFeatures_data.h"
#include "extractAllFeatures_internal_types.h"
#include "extractAllFeatures_rtwutil.h"
#include "gammaln.h"
#include "quadgk.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include <cmath>
#include <cstring>
#include <emmintrin.h>

// Function Definitions
namespace coder {
namespace wavelet {
namespace internal {
namespace cwt {
double wavCFandSD(char wname[4], double varargin_1, double varargin_2,
                  double &sigmaT, double &cf)
{
  anonymous_function derivSq;
  array<double, 2U> b_x;
  array<double, 2U> fx;
  array<double, 2U> x;
  array<double, 2U> xt;
  double interval[650];
  double FourierFactor;
  double be;
  double d;
  double d1;
  double err_ok;
  double halfh;
  wname[0] = cv[static_cast<int>(static_cast<unsigned char>(wname[0]) & 127U)];
  if (wname[0] == 'm') {
    double abserrsubk;
    double b_sigt_tmp;
    double intFsq;
    double midpt;
    double q_ok;
    double sigt_tmp;
    double tkd1mtk;
    double tol;
    double x_tmp_tmp;
    tkd1mtk = std::log(varargin_1);
    midpt = std::log(varargin_2);
    cf = std::exp(1.0 / varargin_1 * (midpt - tkd1mtk));
    x_tmp_tmp = 2.0 * varargin_2;
    halfh = (x_tmp_tmp + 1.0) / varargin_1;
    gammaln(halfh);
    abserrsubk = 2.0 * (varargin_2 - 1.0);
    tol = (varargin_2 - 1.0) + varargin_1;
    intFsq = 2.0 * tol;
    be = (varargin_2 - 1.0) + varargin_1 / 2.0;
    q_ok = 2.0 * be;
    FourierFactor = varargin_2 / varargin_1;
    err_ok = (abserrsubk + 1.0) / varargin_1;
    gammaln(err_ok);
    d = (intFsq + 1.0) / varargin_1;
    gammaln(d);
    d1 = (q_ok + 1.0) / varargin_1;
    gammaln(d1);
    sigt_tmp = 2.0 * (FourierFactor * ((tkd1mtk + 1.0) - midpt));
    b_sigt_tmp =
        x_tmp_tmp / varargin_1 * ((tkd1mtk + 1.0) - std::log(x_tmp_tmp));
    FourierFactor = 2.0 / varargin_1 * std::log(FourierFactor);
    FourierFactor = std::sqrt(
        (std::exp(((((((sigt_tmp - 2.0 * ((varargin_2 - 1.0) / varargin_1 *
                                          ((tkd1mtk + 1.0) -
                                           std::log(varargin_2 - 1.0)))) +
                       abserrsubk / varargin_1 *
                           ((tkd1mtk + 1.0) - std::log(abserrsubk))) -
                      b_sigt_tmp) +
                     FourierFactor) +
                    2.0 * midpt) +
                   err_ok) -
                  halfh) +
         std::exp(
             ((((((sigt_tmp - 2.0 * (tol / varargin_1 *
                                     ((tkd1mtk + 1.0) - std::log(tol)))) +
                  intFsq / varargin_1 * ((tkd1mtk + 1.0) - std::log(intFsq))) -
                 b_sigt_tmp) +
                FourierFactor) +
               2.0 * tkd1mtk) +
              d) -
             halfh)) -
        std::exp(
            ((((((((sigt_tmp - 2.0 * (be / varargin_1 *
                                      ((tkd1mtk + 1.0) - std::log(be)))) +
                   q_ok / varargin_1 * ((tkd1mtk + 1.0) - std::log(q_ok))) -
                  b_sigt_tmp) +
                 FourierFactor) +
                0.69314718055994529) +
               midpt) +
              tkd1mtk) +
             d1) -
            halfh));
    sigmaT = FourierFactor;
    if (std::isinf(FourierFactor) || std::isnan(FourierFactor)) {
      double subs[1298];
      double errsub[649];
      double qsub[649];
      int nt;
      derivSq.workspace.be = varargin_2;
      derivSq.workspace.ga = varargin_1;
      interval[0] = 0.0;
      interval[1] = 1.0;
      std::memset(&interval[2], 0, 648U * sizeof(double));
      intFsq = 0.0;
      nt = split(interval, 2, be);
      if (!(be > 0.0)) {
        intFsq = rtInf * (rt_powd_snf(rtInf, x_tmp_tmp) *
                          std::exp(-2.0 * rt_powd_snf(rtInf, varargin_1)));
      } else {
        int ix;
        int nsubs;
        boolean_T first_iteration;
        nsubs = nt - 1;
        nt = static_cast<unsigned short>(nt - 1);
        for (int k{0}; k < nt; k++) {
          ix = k << 1;
          subs[ix] = interval[k];
          subs[ix + 1] = interval[k + 1];
        }
        q_ok = 0.0;
        err_ok = 0.0;
        first_iteration = true;
        int exitg1;
        do {
          int loop_ub;
          boolean_T guard1;
          exitg1 = 0;
          x.set_size(1, 15 * nsubs);
          ix = -1;
          nsubs = static_cast<unsigned short>(nsubs);
          for (int b_k{0}; b_k < nsubs; b_k++) {
            nt = b_k << 1;
            FourierFactor = subs[nt];
            tkd1mtk = subs[nt + 1];
            midpt = (FourierFactor + tkd1mtk) / 2.0;
            halfh = (tkd1mtk - FourierFactor) / 2.0;
            for (int k{0}; k <= 12; k += 2) {
              _mm_storeu_pd(&x[(ix + k) + 1],
                            _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(&dv[k]),
                                                  _mm_set1_pd(halfh)),
                                       _mm_set1_pd(midpt)));
            }
            x[ix + 15] = dv[14] * halfh + midpt;
            ix += 15;
          }
          loop_ub = x.size(1);
          b_x.set_size(1, x.size(1));
          xt.set_size(1, x.size(1));
          nt = (x.size(1) / 2) << 1;
          ix = nt - 2;
          for (int k{0}; k <= ix; k += 2) {
            __m128d r;
            __m128d r1;
            r = _mm_loadu_pd(&x[k]);
            r1 = _mm_sub_pd(_mm_set1_pd(1.0), r);
            r = _mm_div_pd(r, r1);
            _mm_storeu_pd(&b_x[k], _mm_mul_pd(r, r));
            _mm_storeu_pd(&xt[k], _mm_div_pd(_mm_mul_pd(_mm_set1_pd(2.0), r),
                                             _mm_mul_pd(r1, r1)));
          }
          for (int k{nt}; k < loop_ub; k++) {
            FourierFactor = x[k];
            tkd1mtk = FourierFactor / (1.0 - FourierFactor);
            b_x[k] = tkd1mtk * tkd1mtk;
            xt[k] =
                2.0 * tkd1mtk / ((1.0 - FourierFactor) * (1.0 - FourierFactor));
          }
          guard1 = false;
          if (!first_iteration) {
            boolean_T exitg2;
            FourierFactor = b_x[0];
            first_iteration = false;
            nt = 0;
            exitg2 = false;
            while ((!exitg2) && (nt <= b_x.size(1) - 2)) {
              tkd1mtk = FourierFactor;
              FourierFactor = b_x[nt + 1];
              if (std::abs(FourierFactor - b_x[nt]) <=
                  2.2204460492503131E-14 * std::fmax(tkd1mtk, FourierFactor)) {
                first_iteration = true;
                exitg2 = true;
              } else {
                nt++;
              }
            }
            if (first_iteration) {
              first_iteration = true;
              fx.set_size(1, x.size(1));
              if (loop_ub - 1 >= 0) {
                std::memset(&fx[0], 0,
                            static_cast<unsigned int>(loop_ub) *
                                sizeof(double));
              }
            } else {
              guard1 = true;
            }
          } else {
            guard1 = true;
          }
          if (guard1) {
            first_iteration = false;
            nt = b_x.size(1);
            fx.set_size(1, b_x.size(1));
            for (int k{0}; k < nt; k++) {
              FourierFactor = b_x[k];
              fx[k] = std::exp(-2.0 * rt_powd_snf(FourierFactor, varargin_1));
            }
            fx.set_size(1, b_x.size(1));
            nt = b_x.size(1) - 1;
            for (int k{0}; k <= nt; k++) {
              FourierFactor = b_x[k];
              fx[k] = rt_powd_snf(FourierFactor, x_tmp_tmp) * fx[k] * xt[k];
            }
          }
          if (first_iteration) {
            exitg1 = 1;
          } else {
            int nrefine;
            midpt = 0.0;
            ix = -1;
            for (int b_k{0}; b_k < nsubs; b_k++) {
              tkd1mtk = 0.0;
              abserrsubk = 0.0;
              for (int k{0}; k < 15; k++) {
                FourierFactor = fx[(ix + k) + 1];
                tkd1mtk += dv1[k] * FourierFactor;
                abserrsubk += dv2[k] * FourierFactor;
              }
              ix += 15;
              nt = b_k << 1;
              halfh = (subs[nt + 1] - subs[nt]) / 2.0;
              FourierFactor = tkd1mtk * halfh;
              qsub[b_k] = FourierFactor;
              midpt += FourierFactor;
              errsub[b_k] = abserrsubk * halfh;
            }
            intFsq = midpt + q_ok;
            tol = std::fmax(1.0E-10, 1.0E-6 * std::abs(intFsq));
            FourierFactor = 2.0 * tol / be;
            tkd1mtk = 0.0;
            nrefine = 0;
            for (int k{0}; k < nsubs; k++) {
              midpt = errsub[k];
              abserrsubk = std::abs(midpt);
              nt = k << 1;
              halfh = subs[nt];
              if (abserrsubk <=
                  FourierFactor * ((subs[nt + 1] - halfh) / 2.0)) {
                err_ok += midpt;
                q_ok += qsub[k];
              } else {
                tkd1mtk += abserrsubk;
                nrefine++;
                ix = (nrefine - 1) << 1;
                subs[ix] = halfh;
                subs[ix + 1] = subs[nt + 1];
              }
            }
            FourierFactor = std::abs(err_ok) + tkd1mtk;
            if ((!std::isinf(intFsq)) && (!std::isnan(intFsq)) &&
                ((!std::isinf(FourierFactor)) &&
                 (!std::isnan(FourierFactor))) &&
                (nrefine != 0) && (!(FourierFactor <= tol))) {
              nsubs = nrefine << 1;
              if (nsubs > 650) {
                exitg1 = 1;
              } else {
                for (int k{nrefine}; k >= 1; k--) {
                  nt = (k << 1) - 1;
                  ix = (k - 1) << 1;
                  loop_ub = nt << 1;
                  subs[loop_ub + 1] = subs[ix + 1];
                  subs[loop_ub] = (subs[ix] + subs[ix + 1]) / 2.0;
                  nt = (nt - 1) << 1;
                  subs[nt + 1] = subs[loop_ub];
                  subs[nt] = subs[ix];
                }
                first_iteration = false;
              }
            } else {
              exitg1 = 1;
            }
          }
        } while (exitg1 == 0);
      }
      sigmaT = std::sqrt(cf * cf * (quadgk(derivSq) / intFsq));
    }
  } else if (wname[0] == 'a') {
    cf = 6.0;
    sigmaT = 1.4142135623730951;
  } else {
    cf = 5.0;
    sigmaT = 5.847705;
  }
  return 6.2831853071795862 / cf;
}

} // namespace cwt
} // namespace internal
} // namespace wavelet
} // namespace coder

// End of code generation (wavCFandSD.cpp)
