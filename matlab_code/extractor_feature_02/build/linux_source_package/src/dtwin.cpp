//
// dtwin.cpp
//
// Code generation for function 'dtwin'
//

// Include files
#include "dtwin.h"
#include "extractAllFeatures_internal_types.h"
#include "ppval.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include "coder_bounded_array.h"
#include "omp.h"
#include <algorithm>
#include <cstring>
#include <emmintrin.h>
#include <xmmintrin.h>

// Function Definitions
namespace coder {
namespace b_signal {
namespace internal {
namespace spectral {
int dtwin(const float w_data[], int w_size, double Fs, float Wdt_data[])
{
  __m128i r;
  array<short, 2U> y;
  d_struct_T expl_temp;
  double md_data[258];
  float pp_coefs_data[1028];
  float xv_data[4];
  float endslopes_idx_1;
  int Wdt_size;
  int dvdf_tmp;
  int loop_ub;
  int outsize_idx_0;
  int outsize_idx_1;
  int pp_coefs;
  short t0_breaks_data[258];
  short b_iv[8];
  short i;
  boolean_T has_endslopes;
  if (w_size < 1) {
    y.set_size(1, 0);
  } else {
    y.set_size(1, w_size);
    Wdt_size = (w_size / 8) << 3;
    pp_coefs = Wdt_size - 8;
    for (int k{0}; k <= pp_coefs; k += 8) {
      b_iv[0] = static_cast<short>(k);
      b_iv[1] = static_cast<short>(k + 1);
      b_iv[2] = static_cast<short>(k + 2);
      b_iv[3] = static_cast<short>(k + 3);
      b_iv[4] = static_cast<short>(k + 4);
      b_iv[5] = static_cast<short>(k + 5);
      b_iv[6] = static_cast<short>(k + 6);
      b_iv[7] = static_cast<short>(k + 7);
      r = _mm_loadu_si128((const __m128i *)&b_iv[0]);
      _mm_storeu_si128((__m128i *)&y[k], _mm_add_epi16(_mm_set1_epi16(1), r));
    }
    for (int k{Wdt_size}; k < w_size; k++) {
      y[k] = static_cast<short>(k + 1);
    }
  }
  has_endslopes = (w_size == y.size(1) + 2);
  if ((y.size(1) <= 2) || ((y.size(1) <= 3) && (!has_endslopes))) {
    has_endslopes = (w_size == y.size(1) + 2);
    if (y.size(1) <= 2) {
      if (has_endslopes) {
        pp_coefs = 4;
      } else {
        pp_coefs = 2;
      }
    } else {
      pp_coefs = 3;
    }
    outsize_idx_1 = 1;
    if (y.size(1) <= 2) {
      if (has_endslopes) {
        float endslopes_idx_0;
        endslopes_idx_0 = w_data[0];
        endslopes_idx_1 = w_data[w_size - 1];
        Wdt_size = y.size(1);
        if (Wdt_size - 2 >= 0) {
          float divdifij;
          float dzzdx;
          float f;
          Wdt_size = y[1] - y[0];
          f = w_data[1];
          divdifij = (w_data[2] - f) / static_cast<float>(Wdt_size);
          dzzdx = (divdifij - endslopes_idx_0) / static_cast<float>(Wdt_size);
          endslopes_idx_1 =
              (endslopes_idx_1 - divdifij) / static_cast<float>(Wdt_size);
          xv_data[0] = (endslopes_idx_1 - dzzdx) / static_cast<float>(Wdt_size);
          xv_data[1] = 2.0F * dzzdx - endslopes_idx_1;
          xv_data[2] = endslopes_idx_0;
          xv_data[3] = f;
        }
        std::copy(&xv_data[0], &xv_data[pp_coefs], &pp_coefs_data[0]);
      } else {
        pp_coefs_data[0] =
            (w_data[1] - w_data[0]) / static_cast<float>(y[1] - y[0]);
        pp_coefs_data[1] = w_data[0];
      }
      Wdt_size = y.size(1);
      loop_ub = y.size(1);
      if (Wdt_size - 1 >= 0) {
        std::copy(&y[0], &y[Wdt_size], &t0_breaks_data[0]);
      }
    } else {
      Wdt_size = y[1] - y[0];
      endslopes_idx_1 = (w_data[1] - w_data[0]) / static_cast<float>(Wdt_size);
      pp_coefs_data[0] =
          ((w_data[2] - w_data[1]) / static_cast<float>(y[2] - y[1]) -
           endslopes_idx_1) /
          static_cast<float>(y[2] - y[0]);
      pp_coefs_data[1] =
          endslopes_idx_1 - pp_coefs_data[0] * static_cast<float>(Wdt_size);
      pp_coefs_data[2] = w_data[0];
      loop_ub = 2;
      t0_breaks_data[0] = y[0];
      t0_breaks_data[1] = y[2];
    }
  } else {
    double b_r;
    float s_data[258];
    float dvdf_data[257];
    int d31;
    int dnnm2;
    int nxm1;
    int s_size_idx_1;
    int yoffset;
    short dx_data[257];
    short szs_idx_1;
    nxm1 = y.size(1) - 1;
    if (has_endslopes) {
      szs_idx_1 = static_cast<short>(w_size - 2);
      yoffset = 1;
    } else {
      szs_idx_1 = static_cast<short>(w_size);
      yoffset = 0;
    }
    s_size_idx_1 = szs_idx_1;
    if (static_cast<int>(y.size(1) - 1 < 800)) {
      for (int b_k{0}; b_k < nxm1; b_k++) {
        i = static_cast<short>(y[b_k + 1] - y[b_k]);
        dx_data[b_k] = i;
        Wdt_size = yoffset + b_k;
        dvdf_data[b_k] =
            (w_data[Wdt_size + 1] - w_data[Wdt_size]) / static_cast<float>(i);
      }
    } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(i, dvdf_tmp)

      for (int b_k = 0; b_k < nxm1; b_k++) {
        i = static_cast<short>(y[b_k + 1] - y[b_k]);
        dx_data[b_k] = i;
        dvdf_tmp = yoffset + b_k;
        dvdf_data[b_k] =
            (w_data[dvdf_tmp + 1] - w_data[dvdf_tmp]) / static_cast<float>(i);
      }
    }
    outsize_idx_1 = (nxm1 - 1 < 800);
    if (outsize_idx_1) {
      for (int c_k{2}; c_k <= nxm1; c_k++) {
        s_data[c_k - 1] =
            3.0F * (static_cast<float>(dx_data[c_k - 1]) * dvdf_data[c_k - 2] +
                    static_cast<float>(dx_data[c_k - 2]) * dvdf_data[c_k - 1]);
      }
    } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

      for (int c_k = 2; c_k <= nxm1; c_k++) {
        s_data[c_k - 1] =
            3.0F * (static_cast<float>(dx_data[c_k - 1]) * dvdf_data[c_k - 2] +
                    static_cast<float>(dx_data[c_k - 2]) * dvdf_data[c_k - 1]);
      }
    }
    if (has_endslopes) {
      d31 = 0;
      dnnm2 = 0;
      s_data[0] = static_cast<float>(dx_data[1]) * w_data[0];
      s_data[nxm1] =
          static_cast<float>(dx_data[y.size(1) - 3]) * w_data[y.size(1) + 1];
    } else {
      d31 = y[2] - y[0];
      dnnm2 = y[nxm1] - y[y.size(1) - 3];
      s_data[0] = (static_cast<float>((dx_data[0] + (d31 << 1)) * dx_data[1]) *
                       dvdf_data[0] +
                   static_cast<float>(dx_data[0] * dx_data[0]) * dvdf_data[1]) /
                  static_cast<float>(d31);
      Wdt_size = dx_data[y.size(1) - 2];
      s_data[nxm1] =
          (static_cast<float>((Wdt_size + (dnnm2 << 1)) *
                              dx_data[y.size(1) - 3]) *
               dvdf_data[y.size(1) - 2] +
           static_cast<float>(Wdt_size * Wdt_size) * dvdf_data[y.size(1) - 3]) /
          static_cast<float>(dnnm2);
    }
    loop_ub = y.size(1);
    Wdt_size = dx_data[1];
    md_data[0] = dx_data[1];
    pp_coefs = dx_data[y.size(1) - 3];
    md_data[nxm1] = pp_coefs;
    if (outsize_idx_1) {
      for (int d_k{2}; d_k <= nxm1; d_k++) {
        md_data[d_k - 1] =
            2.0 * static_cast<double>(dx_data[d_k - 1] + dx_data[d_k - 2]);
      }
    } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

      for (int d_k = 2; d_k <= nxm1; d_k++) {
        md_data[d_k - 1] =
            2.0 * static_cast<double>(dx_data[d_k - 1] + dx_data[d_k - 2]);
      }
    }
    b_r = static_cast<double>(Wdt_size) /
          static_cast<double>(static_cast<short>(md_data[0]));
    md_data[1] = static_cast<double>(static_cast<short>(md_data[1])) -
                 b_r * static_cast<double>(d31);
    s_data[1] -= static_cast<float>(b_r) * s_data[0];
    for (int k{3}; k <= nxm1; k++) {
      b_r = static_cast<double>(dx_data[k - 1]) / md_data[k - 2];
      md_data[k - 1] -= b_r * static_cast<double>(dx_data[k - 3]);
      s_data[k - 1] -= static_cast<float>(b_r) * s_data[k - 2];
    }
    b_r = static_cast<double>(dnnm2) / md_data[y.size(1) - 2];
    md_data[nxm1] -= b_r * static_cast<double>(pp_coefs);
    s_data[nxm1] -= static_cast<float>(b_r) * s_data[y.size(1) - 2];
    s_data[nxm1] /= static_cast<float>(md_data[nxm1]);
    for (int k{nxm1}; k >= 2; k--) {
      s_data[k - 1] =
          (s_data[k - 1] - static_cast<float>(dx_data[k - 2]) * s_data[k]) /
          static_cast<float>(md_data[k - 1]);
    }
    s_data[0] = (s_data[0] - static_cast<float>(d31) * s_data[1]) /
                static_cast<float>(md_data[0]);
    outsize_idx_1 = s_size_idx_1 - 1;
    pp_coefs = 4;
    for (int k{0}; k <= loop_ub - 2; k++) {
      float divdifij;
      float endslopes_idx_0;
      endslopes_idx_1 = dvdf_data[k];
      divdifij = s_data[k];
      szs_idx_1 = dx_data[k];
      endslopes_idx_0 =
          (endslopes_idx_1 - divdifij) / static_cast<float>(szs_idx_1);
      endslopes_idx_1 =
          (s_data[k + 1] - endslopes_idx_1) / static_cast<float>(szs_idx_1);
      pp_coefs_data[k] =
          (endslopes_idx_1 - endslopes_idx_0) / static_cast<float>(szs_idx_1);
      pp_coefs_data[(s_size_idx_1 + k) - 1] =
          2.0F * endslopes_idx_0 - endslopes_idx_1;
      pp_coefs_data[((s_size_idx_1 - 1) << 1) + k] = divdifij;
      pp_coefs_data[3 * (s_size_idx_1 - 1) + k] = w_data[yoffset + k];
    }
    std::copy(&y[0], &y[loop_ub], &t0_breaks_data[0]);
  }
  expl_temp.coefs.size[0] = static_cast<short>(outsize_idx_1);
  expl_temp.coefs.size[1] = pp_coefs - 1;
  Wdt_size = static_cast<short>(outsize_idx_1) * (pp_coefs - 1);
  if (Wdt_size - 1 >= 0) {
    std::memset(&expl_temp.coefs.data[0], 0,
                static_cast<unsigned int>(Wdt_size) * sizeof(float));
  }
  if (outsize_idx_1 - 1 >= 0) {
    outsize_idx_0 = pp_coefs;
  }
  for (int j{0}; j < outsize_idx_1; j++) {
    for (int k{0}; k < outsize_idx_0; k++) {
      xv_data[k] = pp_coefs_data[j + k * outsize_idx_1];
    }
    for (int k{0}; k <= outsize_idx_0 - 2; k++) {
      expl_temp.coefs.data[j + k * outsize_idx_1] =
          xv_data[k] * (static_cast<float>(outsize_idx_0 - k) - 1.0F);
    }
  }
  if (w_size < 1) {
    outsize_idx_1 = 0;
    y.set_size(1, 0);
  } else {
    outsize_idx_1 = w_size;
    y.set_size(1, w_size);
    Wdt_size = (w_size / 8) << 3;
    pp_coefs = Wdt_size - 8;
    for (int k{0}; k <= pp_coefs; k += 8) {
      b_iv[0] = static_cast<short>(k);
      b_iv[1] = static_cast<short>(k + 1);
      b_iv[2] = static_cast<short>(k + 2);
      b_iv[3] = static_cast<short>(k + 3);
      b_iv[4] = static_cast<short>(k + 4);
      b_iv[5] = static_cast<short>(k + 5);
      b_iv[6] = static_cast<short>(k + 6);
      b_iv[7] = static_cast<short>(k + 7);
      r = _mm_loadu_si128((const __m128i *)&b_iv[0]);
      _mm_storeu_si128((__m128i *)&y[k], _mm_add_epi16(_mm_set1_epi16(1), r));
    }
    for (int k{Wdt_size}; k < w_size; k++) {
      y[k] = static_cast<short>(k + 1);
    }
  }
  expl_temp.breaks.size[0] = 1;
  expl_temp.breaks.size[1] = loop_ub;
  for (int k{0}; k < loop_ub; k++) {
    expl_temp.breaks.data[k] = t0_breaks_data[k];
  }
  for (int k{0}; k < outsize_idx_1; k++) {
    md_data[k] = y[k];
  }
  Wdt_size = ppval(expl_temp, md_data, outsize_idx_1, Wdt_data);
  endslopes_idx_1 = static_cast<float>(Fs / 6.2831853071795862);
  pp_coefs = (Wdt_size / 4) << 2;
  outsize_idx_1 = pp_coefs - 4;
  for (int k{0}; k <= outsize_idx_1; k += 4) {
    __m128 r1;
    r1 = _mm_loadu_ps(&Wdt_data[k]);
    _mm_storeu_ps(&Wdt_data[k], _mm_mul_ps(r1, _mm_set1_ps(endslopes_idx_1)));
  }
  for (int k{pp_coefs}; k < Wdt_size; k++) {
    Wdt_data[k] *= endslopes_idx_1;
  }
  return Wdt_size;
}

} // namespace spectral
} // namespace internal
} // namespace b_signal
} // namespace coder

// End of code generation (dtwin.cpp)
