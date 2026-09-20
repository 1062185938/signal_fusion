//
// extractAllFeatures.cpp
//
// Code generation for function 'extractAllFeatures'
//

// Include files
#include "extractAllFeatures.h"
#include "abs.h"
#include "angle.h"
#include "any1.h"
#include "blockedSummation.h"
#include "combineVectorElements.h"
#include "extractAllFeatures_data.h"
#include "extractAllFeatures_initialize.h"
#include "extractAllFeatures_rtwutil.h"
#include "extractFrequencyFeatures.h"
#include "extractTimeFrequencyFeatures.h"
#include "linspace.h"
#include "log.h"
#include "log2.h"
#include "minOrMax.h"
#include "rms.h"
#include "rt_nonfinite.h"
#include "std.h"
#include "coder_array.h"
#include "omp.h"
#include <cmath>
#include <cstring>
#include <xmmintrin.h>

// Function Definitions
void extractAllFeatures(const float iData[16384], const float qData[16384],
                        int signalLength, double sampleRate, float features[62],
                        int *status)
{
  static float tmp_data[16383];
  coder::array<creal32_T, 1U> b_x;
  coder::array<creal32_T, 1U> b_x0;
  coder::array<creal32_T, 1U> x;
  coder::array<creal32_T, 1U> x0;
  coder::array<float, 1U> absX0;
  coder::array<float, 1U> centeredAmplitude;
  coder::array<float, 1U> y;
  creal32_T b_varargout_1;
  creal32_T e_varargin_1;
  creal32_T fc;
  creal32_T varargin_1;
  creal32_T varargout_1;
  float b_varargin_1;
  float b_varargout_1_tmp;
  float b_x0_im;
  float b_x0_re;
  float c_r;
  float c_varargin_1;
  float c_x0_im;
  float d_r;
  float d_varargin_1;
  float f;
  float f1;
  float f_varargin_1;
  float g_varargin_1;
  float re;
  float varargout_1_tmp;
  float x0_re;
  if (!isInitialized_extractAllFeatures) {
    extractAllFeatures_initialize();
  }
  //  EXTRACTALLFEATURES
  //  IQ 特征提取动态库统一入口。
  //
  //  Input:
  //    iData         16384×1 single，I 路输入缓冲区
  //    qData         16384×1 single，Q 路输入缓冲区
  //    signalLength  int32，实际有效 IQ 采样点数量
  //    sampleRate    double，采样率 Hz
  //
  //  Output:
  //    features      1×62 single 特征向量
  //    status        int32 状态码
  //
  //  Feature order:
  //
  //     1 - 12   Time-domain features
  //    13 - 37   Frequency-domain features
  //    38 - 62   Time-frequency features
  //
  //  Status:
  //     0   正常
  //    -1   输入长度非法
  //    -2   采样率非法
  //    -3   输入包含 NaN 或 Inf
  //  1. 固定输出
  std::memset(&features[0], 0, 62U * sizeof(float));
  *status = 0;
  //  2. 输入长度检查
  if ((signalLength < 32) || (signalLength > 16384)) {
    *status = -1;

    //  3. 采样率检查
  } else if (std::isnan(sampleRate) || std::isinf(sampleRate) ||
             (sampleRate <= 0.0)) {
    *status = -2;
  } else {
    int i;
    boolean_T absX0_data[16384];
    //  4. 提取实际有效 IQ 数据
    //  5. 输入数据检查
    i = (signalLength < 800);
    if (i) {
      for (int i1{0}; i1 < signalLength; i1++) {
        absX0_data[i1] = std::isnan(iData[i1]);
      }
    } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

      for (int i1 = 0; i1 < signalLength; i1++) {
        absX0_data[i1] = std::isnan(iData[i1]);
      }
    }
    if (coder::any(absX0_data, signalLength)) {
      *status = -3;
    } else {
      if (i) {
        for (int i2{0}; i2 < signalLength; i2++) {
          absX0_data[i2] = std::isinf(iData[i2]);
        }
      } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

        for (int i2 = 0; i2 < signalLength; i2++) {
          absX0_data[i2] = std::isinf(iData[i2]);
        }
      }
      if (coder::any(absX0_data, signalLength)) {
        *status = -3;
      } else {
        if (i) {
          for (int i3{0}; i3 < signalLength; i3++) {
            absX0_data[i3] = std::isnan(qData[i3]);
          }
        } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

          for (int i3 = 0; i3 < signalLength; i3++) {
            absX0_data[i3] = std::isnan(qData[i3]);
          }
        }
        if (coder::any(absX0_data, signalLength)) {
          *status = -3;
        } else {
          if (i) {
            for (int i4{0}; i4 < signalLength; i4++) {
              absX0_data[i4] = std::isinf(qData[i4]);
            }
          } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

            for (int i4 = 0; i4 < signalLength; i4++) {
              absX0_data[i4] = std::isinf(qData[i4]);
            }
          }
          if (coder::any(absX0_data, signalLength)) {
            *status = -3;
          } else {
            creal32_T M20;
            float M20_im_tmp;
            float br;
            float fAmplitudeEntropy;
            float fC20_tmp;
            float fStandardDeviation;
            float maximumAmplitude;
            float meanAmplitude;
            float meanPower;
            float r;
            float x0_im;
            int b_loop_ub;
            int c_loop_ub;
            int i6;
            int loop_ub;
            int vectorUB;
            //  6. 构造复数 IQ
            x.set_size(signalLength);
            for (int k{0}; k < signalLength; k++) {
              x[k].re = iData[k];
              x[k].im = qData[k];
            }
            //  7. 时域特征
            //
            //  输出：
            //    1×12 single
            //  EXTRACTTIMEFEATURES
            //  从单个复数 IQ 段中提取 12 个时域和统计特征。
            //
            //  Input:
            //    x        复数 IQ 向量。输入被视为一个完整的段。
            //
            //  Output:
            //    features 1×12 单精度特征向量。
            //
            //  Feature order:
            //     1  RMS                     均方根
            //     2  Standard deviation      标准差
            //     3  PAPR_dB                 峰均功率比
            //     4  Amplitude skewness      幅度偏度
            //     5  Amplitude kurtosis      幅度峰度
            //     6  Envelope CV             包络变异系数
            //     7  Normalized Amplitude  entropy      归一化幅度熵
            //     8  Normalized C20          归一化 C20
            //     9  Normalized C40          归一化 C40
            //    10  Normalized C41          归一化 C41
            //    11  Normalized C42          归一化 C42
            //    12  Differential phase standard deviation  差分相位标准差
            //
            // 注意事项
            //    1   删去Clearance factor裕度因子，Crest
            //    factor峰值因子，同样是描述峰值突出程度，与PARP重合度太高 2
            //    删去Impulse factor，脉冲因子，与当前业务契合度不够
            //  1. 输入
            //  2. 去除直流分量
            varargin_1 = coder::blockedSummation(x, x.size(0));
            br = static_cast<float>(x.size(0));
            if (varargin_1.im == 0.0F) {
              M20.re = varargin_1.re / static_cast<float>(x.size(0));
              M20.im = 0.0F;
            } else if (varargin_1.re == 0.0F) {
              M20.re = 0.0F;
              M20.im = varargin_1.im / static_cast<float>(x.size(0));
            } else {
              M20.re = varargin_1.re / static_cast<float>(x.size(0));
              M20.im = varargin_1.im / static_cast<float>(x.size(0));
            }
            loop_ub = x.size(0);
            x0.set_size(x.size(0));
            b_loop_ub = x.size(0);
            if (i) {
              for (int i5{0}; i5 < loop_ub; i5++) {
                x0[i5].re = x[i5].re - M20.re;
                x0[i5].im = x[i5].im - M20.im;
              }
            } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

              for (int i5 = 0; i5 < b_loop_ub; i5++) {
                x0[i5].re = x[i5].re - M20.re;
                x0[i5].im = x[i5].im - M20.im;
              }
            }
            coder::b_abs(x0, absX0);
            //  3. 基本幅度和峰值相关特征
            // IQ信号的均方根
            features[0] = coder::rms(x0);
            //  标准差，可以用于描述信号幅度的整体波动程度。
            fStandardDeviation = coder::b_std(x0);
            maximumAmplitude = coder::internal::maximum(absX0);
            meanAmplitude = coder::blockedSummation(absX0, absX0.size(0)) /
                            static_cast<float>(absX0.size(0));
            c_loop_ub = absX0.size(0);
            y.set_size(absX0.size(0));
            b_loop_ub = absX0.size(0);
            i6 = (absX0.size(0) < 800);
            if (i6) {
              for (int i7{0}; i7 < c_loop_ub; i7++) {
                r = absX0[i7];
                y[i7] = r * r;
              }
            } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        b_varargin_1)

              for (int i7 = 0; i7 < b_loop_ub; i7++) {
                b_varargin_1 = absX0[i7];
                y[i7] = b_varargin_1 * b_varargin_1;
              }
            }
            meanPower = coder::blockedSummation(y, y.size(0)) /
                        static_cast<float>(y.size(0));
            //  峰均功率比：
            //
            //    PAPR = max(|x|^2) / mean(|x|^2)
            //
            //  使用 dB 表示，更符合通信信号分析中的常见定义。
            if (meanPower <= 1.1920929E-7F) {
              features[2] = 0.0F;
            } else {
              features[2] =
                  10.0F * std::log10(maximumAmplitude * maximumAmplitude /
                                         (meanPower + 1.1920929E-7F) +
                                     1.1920929E-7F);
            }
            //  包络变异系数：
            //
            //    Envelope CV = std(|x|) / mean(|x|)
            //
            //  该特征不反映绝对幅度大小，而是描述包络相对波动程度。
            if (meanAmplitude <= 1.1920929E-7F) {
              features[5] = 0.0F;
            } else {
              features[5] =
                  fStandardDeviation / (meanAmplitude + 1.1920929E-7F);
            }
            //  4. 幅度偏度和峰度
            //
            //
            //
            //  skewness:
            //    E[(x - mu)^3] / (E[(x - mu)^2])^(3/2)
            //
            //  kurtosis:
            //    E[(x - mu)^4] / (E[(x - mu)^2])^2
            //
            centeredAmplitude.set_size(c_loop_ub);
            b_loop_ub = (absX0.size(0) / 4) << 2;
            vectorUB = b_loop_ub - 4;
            for (int k{0}; k <= vectorUB; k += 4) {
              __m128 b_r;
              b_r = _mm_loadu_ps(&absX0[k]);
              _mm_storeu_ps(&centeredAmplitude[k],
                            _mm_sub_ps(b_r, _mm_set1_ps(meanAmplitude)));
            }
            for (int k{b_loop_ub}; k < c_loop_ub; k++) {
              centeredAmplitude[k] = absX0[k] - meanAmplitude;
            }
            b_loop_ub = centeredAmplitude.size(0);
            y.set_size(centeredAmplitude.size(0));
            vectorUB = (centeredAmplitude.size(0) < 800);
            if (vectorUB) {
              for (int i8{0}; i8 < b_loop_ub; i8++) {
                r = centeredAmplitude[i8];
                y[i8] = r * r;
              }
            } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        c_varargin_1)

              for (int i8 = 0; i8 < b_loop_ub; i8++) {
                c_varargin_1 = centeredAmplitude[i8];
                y[i8] = c_varargin_1 * c_varargin_1;
              }
            }
            meanAmplitude = coder::blockedSummation(y, y.size(0)) /
                            static_cast<float>(y.size(0));
            if (meanAmplitude <= 1.1920929E-7F) {
              features[3] = 0.0F;
              features[4] = 0.0F;
            } else {
              y.set_size(b_loop_ub);
              if (vectorUB) {
                for (int i9{0}; i9 < b_loop_ub; i9++) {
                  r = centeredAmplitude[i9];
                  y[i9] = rt_powf_snf(r, 3.0F);
                }
              } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        d_varargin_1)

                for (int i9 = 0; i9 < b_loop_ub; i9++) {
                  d_varargin_1 = centeredAmplitude[i9];
                  y[i9] = rt_powf_snf(d_varargin_1, 3.0F);
                }
              }
              features[3] =
                  coder::blockedSummation(y, y.size(0)) /
                  static_cast<float>(y.size(0)) /
                  (meanAmplitude * std::sqrt(meanAmplitude) + 1.1920929E-7F);
              y.set_size(b_loop_ub);
              if (vectorUB) {
                for (int i11{0}; i11 < b_loop_ub; i11++) {
                  r = centeredAmplitude[i11];
                  y[i11] = rt_powf_snf(r, 4.0F);
                }
              } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        f_varargin_1)

                for (int i11 = 0; i11 < b_loop_ub; i11++) {
                  f_varargin_1 = centeredAmplitude[i11];
                  y[i11] = rt_powf_snf(f_varargin_1, 4.0F);
                }
              }
              features[4] = coder::blockedSummation(y, y.size(0)) /
                            static_cast<float>(y.size(0)) /
                            (meanAmplitude * meanAmplitude + 1.1920929E-7F);
            }
            //  5. 高阶矩和累积量
            // 对高阶累积量进行能量归一化，是为了减小接收幅度或增益变化的影响；取模则可以减小未知载波相位带来的影响。
            b_x.set_size(loop_ub);
            if (i) {
              for (int i10{0}; i10 < loop_ub; i10++) {
                varargin_1 = x0[i10];
                M20.re = varargin_1.re * varargin_1.re -
                         varargin_1.im * varargin_1.im;
                r = varargin_1.re * varargin_1.im;
                M20.im = r + r;
                b_x[i10] = M20;
              }
            } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        e_varargin_1, varargout_1, varargout_1_tmp)

              for (int i10 = 0; i10 < loop_ub; i10++) {
                e_varargin_1 = x0[i10];
                varargout_1.re = e_varargin_1.re * e_varargin_1.re -
                                 e_varargin_1.im * e_varargin_1.im;
                varargout_1_tmp = e_varargin_1.re * e_varargin_1.im;
                varargout_1.im = varargout_1_tmp + varargout_1_tmp;
                b_x[i10] = varargout_1;
              }
            }
            varargin_1 = coder::blockedSummation(b_x, b_x.size(0));
            if (varargin_1.im == 0.0F) {
              M20.re = varargin_1.re / br;
              M20.im = 0.0F;
            } else if (varargin_1.re == 0.0F) {
              M20.re = 0.0F;
              M20.im = varargin_1.im / br;
            } else {
              M20.re = varargin_1.re / br;
              M20.im = varargin_1.im / br;
            }
            fAmplitudeEntropy = meanPower * meanPower;
            fC20_tmp = coder::b_abs(M20);
            features[7] = fC20_tmp / (meanPower + 1.1920929E-7F);
            b_x.set_size(loop_ub);
            if (i) {
              for (int i12{0}; i12 < loop_ub; i12++) {
                varargin_1 = x0[i12];
                if ((varargin_1.im == 0.0F) && (varargin_1.re >= 0.0F)) {
                  varargin_1.re = rt_powf_snf(varargin_1.re, 4.0F);
                  varargin_1.im = 0.0F;
                } else if (varargin_1.re == 0.0F) {
                  varargin_1.re = rt_powf_snf(varargin_1.im, 4.0F);
                  varargin_1.im = 0.0F;
                } else {
                  coder::b_log(varargin_1);
                  r = 4.0F * varargin_1.re;
                  meanAmplitude = 4.0F * varargin_1.im;
                  if (r == 0.0F) {
                    varargin_1.re = std::cos(meanAmplitude);
                    varargin_1.im = std::sin(meanAmplitude);
                  } else if (meanAmplitude == 0.0F) {
                    varargin_1.re = std::exp(r);
                    varargin_1.im = 0.0F;
                  } else if (std::isinf(meanAmplitude) && std::isinf(r) &&
                             (r < 0.0F)) {
                    varargin_1.re = 0.0F;
                    varargin_1.im = 0.0F;
                  } else {
                    r = std::exp(r / 2.0F);
                    varargin_1.re = r * (r * std::cos(meanAmplitude));
                    varargin_1.im = r * (r * std::sin(meanAmplitude));
                  }
                }
                b_x[i12] = varargin_1;
              }
            } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        b_varargout_1, c_r, b_varargout_1_tmp)

              for (int i12 = 0; i12 < loop_ub; i12++) {
                b_varargout_1 = x0[i12];
                if ((b_varargout_1.im == 0.0F) && (b_varargout_1.re >= 0.0F)) {
                  b_varargout_1.re = rt_powf_snf(b_varargout_1.re, 4.0F);
                  b_varargout_1.im = 0.0F;
                } else if (b_varargout_1.re == 0.0F) {
                  b_varargout_1.re = rt_powf_snf(b_varargout_1.im, 4.0F);
                  b_varargout_1.im = 0.0F;
                } else {
                  coder::b_log(b_varargout_1);
                  c_r = 4.0F * b_varargout_1.re;
                  b_varargout_1_tmp = 4.0F * b_varargout_1.im;
                  if (c_r == 0.0F) {
                    b_varargout_1.re = std::cos(b_varargout_1_tmp);
                    b_varargout_1.im = std::sin(b_varargout_1_tmp);
                  } else if (b_varargout_1_tmp == 0.0F) {
                    b_varargout_1.re = std::exp(c_r);
                    b_varargout_1.im = 0.0F;
                  } else if (std::isinf(b_varargout_1_tmp) && std::isinf(c_r) &&
                             (c_r < 0.0F)) {
                    b_varargout_1.re = 0.0F;
                    b_varargout_1.im = 0.0F;
                  } else {
                    c_r = std::exp(c_r / 2.0F);
                    b_varargout_1.re =
                        c_r * (c_r * std::cos(b_varargout_1_tmp));
                    b_varargout_1.im =
                        c_r * (c_r * std::sin(b_varargout_1_tmp));
                  }
                }
                b_x[i12] = b_varargout_1;
              }
            }
            varargin_1 = coder::blockedSummation(b_x, b_x.size(0));
            if (varargin_1.im == 0.0F) {
              meanAmplitude = varargin_1.re / br;
              r = 0.0F;
            } else if (varargin_1.re == 0.0F) {
              meanAmplitude = 0.0F;
              r = varargin_1.im / br;
            } else {
              meanAmplitude = varargin_1.re / br;
              r = varargin_1.im / br;
            }
            M20_im_tmp = M20.re * M20.im;
            varargin_1.re =
                meanAmplitude - 3.0F * (M20.re * M20.re - M20.im * M20.im);
            varargin_1.im = r - 3.0F * (M20_im_tmp + M20_im_tmp);
            features[8] =
                coder::b_abs(varargin_1) / (fAmplitudeEntropy + 1.1920929E-7F);
            b_x.set_size(loop_ub);
            if (i) {
              for (int i13{0}; i13 < loop_ub; i13++) {
                varargin_1 = x0[i13];
                if ((varargin_1.im == 0.0F) && (varargin_1.re >= 0.0F)) {
                  M20_im_tmp = rt_powf_snf(varargin_1.re, 3.0F);
                  meanAmplitude = 0.0F;
                } else if (varargin_1.re == 0.0F) {
                  M20_im_tmp = 0.0F;
                  meanAmplitude = -rt_powf_snf(varargin_1.im, 3.0F);
                } else {
                  coder::b_log(varargin_1);
                  M20_im_tmp = 3.0F * varargin_1.re;
                  r = 3.0F * varargin_1.im;
                  if (M20_im_tmp == 0.0F) {
                    M20_im_tmp = std::cos(r);
                    meanAmplitude = std::sin(r);
                  } else if (r == 0.0F) {
                    M20_im_tmp = std::exp(M20_im_tmp);
                    meanAmplitude = 0.0F;
                  } else if (std::isinf(r) && std::isinf(M20_im_tmp) &&
                             (M20_im_tmp < 0.0F)) {
                    M20_im_tmp = 0.0F;
                    meanAmplitude = 0.0F;
                  } else {
                    meanAmplitude = std::exp(M20_im_tmp / 2.0F);
                    M20_im_tmp = meanAmplitude * (meanAmplitude * std::cos(r));
                    meanAmplitude *= meanAmplitude * std::sin(r);
                  }
                }
                r = x0[i13].re;
                x0_im = -x0[i13].im;
                b_x[i13].re = M20_im_tmp * r - meanAmplitude * x0_im;
                b_x[i13].im = M20_im_tmp * x0_im + meanAmplitude * r;
              }
            } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        fc, re, d_r, x0_re, b_x0_im)

              for (int i13 = 0; i13 < loop_ub; i13++) {
                fc = x0[i13];
                if ((fc.im == 0.0F) && (fc.re >= 0.0F)) {
                  re = rt_powf_snf(fc.re, 3.0F);
                  d_r = 0.0F;
                } else if (fc.re == 0.0F) {
                  re = 0.0F;
                  d_r = -rt_powf_snf(fc.im, 3.0F);
                } else {
                  coder::b_log(fc);
                  d_r = 3.0F * fc.re;
                  x0_re = 3.0F * fc.im;
                  if (d_r == 0.0F) {
                    re = std::cos(x0_re);
                    d_r = std::sin(x0_re);
                  } else if (x0_re == 0.0F) {
                    re = std::exp(d_r);
                    d_r = 0.0F;
                  } else if (std::isinf(x0_re) && std::isinf(d_r) &&
                             (d_r < 0.0F)) {
                    re = 0.0F;
                    d_r = 0.0F;
                  } else {
                    d_r = std::exp(d_r / 2.0F);
                    re = d_r * (d_r * std::cos(x0_re));
                    d_r *= d_r * std::sin(x0_re);
                  }
                }
                x0_re = x0[i13].re;
                b_x0_im = -x0[i13].im;
                b_x[i13].re = re * x0_re - d_r * b_x0_im;
                b_x[i13].im = re * b_x0_im + d_r * x0_re;
              }
            }
            varargin_1 = coder::blockedSummation(b_x, b_x.size(0));
            if (varargin_1.im == 0.0F) {
              x0_im = varargin_1.re / br;
              r = 0.0F;
            } else if (varargin_1.re == 0.0F) {
              x0_im = 0.0F;
              r = varargin_1.im / br;
            } else {
              x0_im = varargin_1.re / br;
              r = varargin_1.im / br;
            }
            varargin_1.re = x0_im - 3.0F * M20.re * meanPower;
            varargin_1.im = r - 3.0F * M20.im * meanPower;
            features[9] =
                coder::b_abs(varargin_1) / (fAmplitudeEntropy + 1.1920929E-7F);
            y.set_size(c_loop_ub);
            if (i6) {
              for (int i14{0}; i14 < c_loop_ub; i14++) {
                r = absX0[i14];
                y[i14] = rt_powf_snf(r, 4.0F);
              }
            } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        g_varargin_1)

              for (int i14 = 0; i14 < c_loop_ub; i14++) {
                g_varargin_1 = absX0[i14];
                y[i14] = rt_powf_snf(g_varargin_1, 4.0F);
              }
            }
            features[10] = std::abs((coder::blockedSummation(y, y.size(0)) /
                                         static_cast<float>(y.size(0)) -
                                     fC20_tmp * fC20_tmp) -
                                    2.0F * fAmplitudeEntropy) /
                           (fAmplitudeEntropy + 1.1920929E-7F);
            //  6. 归一化幅度熵
            //
            //  将幅度范围划分为 10 个等宽区间。
            //
            //  原始香农熵的最大值为：
            //
            //    log2(10)
            //
            //  因此，将计算结果除以 log2(10)，使输出大致位于 [0,1]。
            fAmplitudeEntropy = 0.0F;
            if (maximumAmplitude > 1.1920929E-7F) {
              float edges[11];
              float counts[10];
              coder::linspace(maximumAmplitude, edges);
              for (int binIndex{0}; binIndex < 10; binIndex++) {
                if (binIndex + 1 == 10) {
                  for (int k{0}; k < c_loop_ub; k++) {
                    absX0_data[k] =
                        ((absX0[k] >= edges[9]) && (absX0[k] <= edges[10]));
                  }
                  counts[9] = static_cast<float>(
                      coder::combineVectorElements(absX0_data, c_loop_ub));
                } else {
                  for (int k{0}; k < c_loop_ub; k++) {
                    absX0_data[k] = ((absX0[k] >= edges[binIndex]) &&
                                     (absX0[k] < edges[binIndex + 1]));
                  }
                  counts[binIndex] = static_cast<float>(
                      coder::combineVectorElements(absX0_data, c_loop_ub));
                }
              }
              r = counts[0];
              for (int k{0}; k < 9; k++) {
                r += counts[k + 1];
              }
              if (r > 0.0F) {
                for (int k{0}; k < 10; k++) {
                  meanAmplitude = counts[k] / r;
                  if (meanAmplitude > 0.0F) {
                    fAmplitudeEntropy -=
                        meanAmplitude * coder::b_log2(meanAmplitude);
                  }
                }
                fAmplitudeEntropy /= 3.32192802F;
              }
            }
            //  7. 差分相位标准差
            //
            //  Differential phase:
            //
            //    deltaPhi[n] = angle(x[n] * conj(x[n-1]))
            //
            //  这种计算方式可以消除恒定初始相位的影响。
            //
            //  这里不对整段相位进行 unwrap，因为差分相位本身已经通过 angle()
            //  限制在 [-pi, pi] 内。该特征仍可能受到载波频偏以及低幅度噪声
            //  样本的影响。
            b_x0.set_size(x0.size(0) - 1);
            if (static_cast<int>(x0.size(0) - 1 < 800)) {
              for (int i15{0}; i15 <= loop_ub - 2; i15++) {
                r = x0[i15].re;
                meanAmplitude = -x0[i15].im;
                M20_im_tmp = x0[i15 + 1].re;
                x0_im = x0[i15 + 1].im;
                b_x0[i15].re = M20_im_tmp * r - x0_im * meanAmplitude;
                b_x0[i15].im = M20_im_tmp * meanAmplitude + x0_im * r;
              }
            } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        b_x0_re, c_x0_im, f, f1)

              for (int i15 = 0; i15 <= loop_ub - 2; i15++) {
                b_x0_re = x0[i15].re;
                c_x0_im = -x0[i15].im;
                f = x0[i15 + 1].re;
                f1 = x0[i15 + 1].im;
                b_x0[i15].re = f * b_x0_re - f1 * c_x0_im;
                b_x0[i15].im = f * c_x0_im + f1 * b_x0_re;
              }
            }
            b_loop_ub = coder::angle(b_x0, tmp_data);
            features[11] = coder::b_std(tmp_data, b_loop_ub);
            //  8. 固定尺寸输出
            features[1] = fStandardDeviation;
            features[6] = fAmplitudeEntropy;
            //  9. 数值保护
            for (int k{0}; k < 12; k++) {
              r = features[k];
              if (std::isnan(r) || std::isinf(r)) {
                features[k] = 0.0F;
              }
            }
            //  8. 频域特征
            //
            //  输出：
            //    1×25 single
            extractFrequencyFeatures(x, sampleRate, &features[12]);
            //  9. 时频特征
            //
            //  输出：
            //    1×25 single
            extractTimeFrequencyFeatures(x, sampleRate, &features[37]);
            //  10. 拼接固定 62 维输出
          }
        }
      }
    }
  }
}

// End of code generation (extractAllFeatures.cpp)
