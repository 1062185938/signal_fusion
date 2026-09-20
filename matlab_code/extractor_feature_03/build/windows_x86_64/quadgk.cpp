//
// quadgk.cpp
//
// Code generation for function 'quadgk'
//

// Include files
#include "quadgk.h"
#include "anonymous_function.h"
#include "extractAllFeatures_data.h"
#include "extractAllFeatures_internal_types.h"
#include "extractAllFeatures_rtwutil.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include <cmath>
#include <cstring>
#include <emmintrin.h>

// Function Definitions
namespace coder {
double quadgk(const anonymous_function fun)
{
  array<double, 2U> b_x;
  array<double, 2U> fx;
  array<double, 2U> x;
  array<double, 2U> xt;
  double subs[1298];
  double interval[650];
  double errsub[649];
  double qsub[649];
  double pathlen;
  double q;
  int nt;
  interval[0] = 0.0;
  interval[1] = 1.0;
  std::memset(&interval[2], 0, 648U * sizeof(double));
  q = 0.0;
  nt = split(interval, 2, pathlen);
  if (!(pathlen > 0.0)) {
    double absxk;
    absxk = fun.workspace.be * rt_powd_snf(rtInf, fun.workspace.be - 1.0) -
            fun.workspace.ga *
                rt_powd_snf(rtInf, (fun.workspace.be + fun.workspace.ga) - 1.0);
    q = rtInf *
        (absxk * absxk * std::exp(-2.0 * rt_powd_snf(rtInf, fun.workspace.ga)));
  } else {
    double err_ok;
    double q_ok;
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
      double absxk;
      double halfh;
      double midpt;
      double tkd1mtk;
      int loop_ub;
      boolean_T guard1;
      exitg1 = 0;
      x.set_size(1, 15 * nsubs);
      ix = -1;
      nsubs = static_cast<unsigned short>(nsubs);
      for (int b_k{0}; b_k < nsubs; b_k++) {
        nt = b_k << 1;
        tkd1mtk = subs[nt];
        absxk = subs[nt + 1];
        midpt = (tkd1mtk + absxk) / 2.0;
        halfh = (absxk - tkd1mtk) / 2.0;
        for (int k{0}; k <= 12; k += 2) {
          _mm_storeu_pd(
              &x[(ix + k) + 1],
              _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(&dv[k]), _mm_set1_pd(halfh)),
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
        absxk = x[k];
        tkd1mtk = absxk / (1.0 - absxk);
        b_x[k] = tkd1mtk * tkd1mtk;
        xt[k] = 2.0 * tkd1mtk / ((1.0 - absxk) * (1.0 - absxk));
      }
      guard1 = false;
      if (!first_iteration) {
        boolean_T exitg2;
        absxk = b_x[0];
        first_iteration = false;
        nt = 0;
        exitg2 = false;
        while ((!exitg2) && (nt <= b_x.size(1) - 2)) {
          tkd1mtk = absxk;
          absxk = b_x[nt + 1];
          if (std::abs(absxk - b_x[nt]) <=
              2.2204460492503131E-14 * std::fmax(tkd1mtk, absxk)) {
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
                        static_cast<unsigned int>(loop_ub) * sizeof(double));
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
          absxk = b_x[k];
          fx[k] = std::exp(-2.0 * rt_powd_snf(absxk, fun.workspace.ga));
        }
        absxk = (fun.workspace.be + fun.workspace.ga) - 1.0;
        fx.set_size(1, b_x.size(1));
        nt = b_x.size(1) - 1;
        for (int k{0}; k <= nt; k++) {
          tkd1mtk = b_x[k];
          tkd1mtk =
              fun.workspace.be * rt_powd_snf(tkd1mtk, fun.workspace.be - 1.0) -
              fun.workspace.ga * rt_powd_snf(tkd1mtk, absxk);
          fx[k] = tkd1mtk * tkd1mtk * fx[k] * xt[k];
        }
      }
      if (first_iteration) {
        exitg1 = 1;
      } else {
        double abserrsubk;
        double tol;
        int nrefine;
        midpt = 0.0;
        ix = -1;
        for (int b_k{0}; b_k < nsubs; b_k++) {
          absxk = 0.0;
          abserrsubk = 0.0;
          for (int k{0}; k < 15; k++) {
            tkd1mtk = fx[(ix + k) + 1];
            absxk += dv1[k] * tkd1mtk;
            abserrsubk += dv2[k] * tkd1mtk;
          }
          ix += 15;
          nt = b_k << 1;
          halfh = (subs[nt + 1] - subs[nt]) / 2.0;
          tkd1mtk = absxk * halfh;
          qsub[b_k] = tkd1mtk;
          midpt += tkd1mtk;
          errsub[b_k] = abserrsubk * halfh;
        }
        q = midpt + q_ok;
        tol = std::fmax(1.0E-10, 1.0E-6 * std::abs(q));
        tkd1mtk = 2.0 * tol / pathlen;
        absxk = 0.0;
        nrefine = 0;
        for (int k{0}; k < nsubs; k++) {
          midpt = errsub[k];
          abserrsubk = std::abs(midpt);
          nt = k << 1;
          halfh = subs[nt];
          if (abserrsubk <= tkd1mtk * ((subs[nt + 1] - halfh) / 2.0)) {
            err_ok += midpt;
            q_ok += qsub[k];
          } else {
            absxk += abserrsubk;
            nrefine++;
            ix = (nrefine - 1) << 1;
            subs[ix] = halfh;
            subs[ix + 1] = subs[nt + 1];
          }
        }
        tkd1mtk = std::abs(err_ok) + absxk;
        if ((!std::isinf(q)) && (!std::isnan(q)) &&
            ((!std::isinf(tkd1mtk)) && (!std::isnan(tkd1mtk))) &&
            (nrefine != 0) && (!(tkd1mtk <= tol))) {
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
  return q;
}

int split(double x[650], int nx, double &pathlen)
{
  double y;
  int lidx;
  int nxnew;
  pathlen = x[1] - x[0];
  if (pathlen > 0.0) {
    y = std::ceil(pathlen * (10.0 / pathlen));
    nxnew = static_cast<int>(y - 1.0) + 2;
    if (static_cast<int>(y - 1.0) + 2 > 2) {
      double delta;
      x[static_cast<int>(y - 1.0) + 1] = x[1];
      delta =
          (x[1] - x[0]) / static_cast<double>(static_cast<int>(y - 1.0) + 1);
      nx = static_cast<int>(y - 1.0);
      for (int j{nx}; j >= 1; j--) {
        x[j] = x[0] + static_cast<double>(j) * delta;
      }
    }
    nx = static_cast<int>(y - 1.0) + 2;
  } else {
    nxnew = 2;
  }
  lidx = 0;
  for (int j{2}; j <= nx; j++) {
    y = x[j - 1];
    if (std::abs(y - x[lidx]) > 0.0) {
      lidx++;
      x[lidx] = y;
    } else {
      nxnew--;
    }
  }
  if (nxnew < 2) {
    x[1] = x[nx - 1];
    nxnew = 2;
  }
  return nxnew;
}

} // namespace coder

// End of code generation (quadgk.cpp)
