//
// extractTimeFrequencyFeatures.cpp
//
// Code generation for function 'extractTimeFrequencyFeatures'
//

// Include files
#include "extractTimeFrequencyFeatures.h"
#include "abs.h"
#include "blockedSummation.h"
#include "extractAllFeatures_data.h"
#include "extractAllFeatures_rtwutil.h"
#include "fsst.h"
#include "hamming.h"
#include "log.h"
#include "mean.h"
#include "minOrMax.h"
#include "rt_nonfinite.h"
#include "spectrogram.h"
#include "unsafeSxfun.h"
#include "wsst.h"
#include "coder_array.h"
#include "omp.h"
#include <cmath>
#include <cstring>
#include <xmmintrin.h>

// Function Declarations
static void binary_expand_op_1(coder::array<float, 1U> &in1,
                               const coder::array<float, 1U> &in3, float in4,
                               const coder::array<float, 2U> &in5, int in6);

static void binary_expand_op_2(coder::array<float, 1U> &in1,
                               const coder::array<float, 1U> &in2,
                               const coder::array<float, 2U> &in3, int in4);

static void binary_expand_op_3(coder::array<float, 1U> &in1,
                               const coder::array<float, 1U> &in2, float in3);

static void binary_expand_op_4(coder::array<float, 2U> &in1,
                               const coder::array<double, 2U> &in3,
                               const coder::array<double, 2U> &in4);

static void binary_expand_op_6(coder::array<float, 1U> &in1,
                               const coder::array<float, 1U> &in2,
                               const coder::array<float, 1U> &in3, float in4);

// Function Definitions
static void binary_expand_op_1(coder::array<float, 1U> &in1,
                               const coder::array<float, 1U> &in3, float in4,
                               const coder::array<float, 2U> &in5, int in6)
{
  float b_varargin_1;
  int loop_ub;
  int stride_0_0;
  int stride_1_0;
  if (in5.size(0) == 1) {
    loop_ub = in3.size(0);
  } else {
    loop_ub = in5.size(0);
  }
  in1.set_size(loop_ub);
  stride_0_0 = (in3.size(0) != 1);
  stride_1_0 = (in5.size(0) != 1);
  if (static_cast<int>(loop_ub < 800)) {
    for (int i{0}; i < loop_ub; i++) {
      float varargin_1;
      varargin_1 = in3[i * stride_0_0] - in4;
      in1[i] =
          varargin_1 * varargin_1 * in5[i * stride_1_0 + in5.size(0) * in6];
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        b_varargin_1)

    for (int i = 0; i < loop_ub; i++) {
      b_varargin_1 = in3[i * stride_0_0] - in4;
      in1[i] =
          b_varargin_1 * b_varargin_1 * in5[i * stride_1_0 + in5.size(0) * in6];
    }
  }
}

static void binary_expand_op_2(coder::array<float, 1U> &in1,
                               const coder::array<float, 1U> &in2,
                               const coder::array<float, 2U> &in3, int in4)
{
  int loop_ub;
  int stride_0_0;
  int stride_1_0;
  if (in3.size(0) == 1) {
    loop_ub = in2.size(0);
  } else {
    loop_ub = in3.size(0);
  }
  in1.set_size(loop_ub);
  stride_0_0 = (in2.size(0) != 1);
  stride_1_0 = (in3.size(0) != 1);
  if (static_cast<int>(loop_ub < 800)) {
    for (int i{0}; i < loop_ub; i++) {
      in1[i] = in2[i * stride_0_0] * in3[i * stride_1_0 + in3.size(0) * in4];
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

    for (int i = 0; i < loop_ub; i++) {
      in1[i] = in2[i * stride_0_0] * in3[i * stride_1_0 + in3.size(0) * in4];
    }
  }
}

static void binary_expand_op_3(coder::array<float, 1U> &in1,
                               const coder::array<float, 1U> &in2, float in3)
{
  coder::array<float, 1U> b_in2;
  int loop_ub;
  int stride_0_0;
  int stride_1_0;
  if (in1.size(0) == 1) {
    loop_ub = in2.size(0);
  } else {
    loop_ub = in1.size(0);
  }
  b_in2.set_size(loop_ub);
  stride_0_0 = (in2.size(0) != 1);
  stride_1_0 = (in1.size(0) != 1);
  if (static_cast<int>(loop_ub < 800)) {
    for (int i{0}; i < loop_ub; i++) {
      b_in2[i] = in2[i * stride_0_0] * (in1[i * stride_1_0] - in3);
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

    for (int i = 0; i < loop_ub; i++) {
      b_in2[i] = in2[i * stride_0_0] * (in1[i * stride_1_0] - in3);
    }
  }
  in1.set_size(loop_ub);
  for (int i1{0}; i1 < loop_ub; i1++) {
    in1[i1] = b_in2[i1];
  }
}

static void binary_expand_op_4(coder::array<float, 2U> &in1,
                               const coder::array<double, 2U> &in3,
                               const coder::array<double, 2U> &in4)
{
  int b_loop_ub;
  int loop_ub;
  int stride_0_0;
  int stride_1_0;
  if (in4.size(0) == 1) {
    loop_ub = in3.size(0);
  } else {
    loop_ub = in4.size(0);
  }
  in1.set_size(loop_ub, in1.size(1));
  b_loop_ub = in3.size(1);
  in1.set_size(in1.size(0), b_loop_ub);
  stride_0_0 = (in3.size(0) != 1);
  stride_1_0 = (in4.size(0) != 1);
  for (int i{0}; i < b_loop_ub; i++) {
    for (int i1{0}; i1 < loop_ub; i1++) {
      double b_varargin_1;
      double varargin_1;
      varargin_1 = in3[i1 * stride_0_0 + in3.size(0) * i];
      b_varargin_1 = in4[i1 * stride_1_0 + in4.size(0) * i];
      in1[i1 + in1.size(0) * i] = static_cast<float>(
          rt_powd_snf(varargin_1, 2.0) + rt_powd_snf(b_varargin_1, 2.0));
    }
  }
}

static void binary_expand_op_6(coder::array<float, 1U> &in1,
                               const coder::array<float, 1U> &in2,
                               const coder::array<float, 1U> &in3, float in4)
{
  int loop_ub;
  int stride_0_0;
  int stride_1_0;
  if (in3.size(0) == 1) {
    loop_ub = in2.size(0);
  } else {
    loop_ub = in3.size(0);
  }
  in1.set_size(loop_ub);
  stride_0_0 = (in2.size(0) != 1);
  stride_1_0 = (in3.size(0) != 1);
  if (static_cast<int>(loop_ub < 800)) {
    for (int i{0}; i < loop_ub; i++) {
      in1[i] = in2[i * stride_0_0] * (in3[i * stride_1_0] - in4);
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

    for (int i = 0; i < loop_ub; i++) {
      in1[i] = in2[i * stride_0_0] * (in3[i * stride_1_0] - in4);
    }
  }
}

void extractTimeFrequencyFeatures(const coder::array<creal32_T, 1U> &x,
                                  double sampleRate, float features[25])
{
  coder::array<creal_T, 2U> wsstMatrixI;
  coder::array<creal_T, 2U> wsstMatrixQ;
  coder::array<creal32_T, 2U> fsstMatrix;
  coder::array<creal32_T, 2U> stftMatrix;
  coder::array<creal32_T, 1U> x0;
  coder::array<creal32_T, 1U> xIn;
  coder::array<double, 2U> r1;
  coder::array<double, 2U> r2;
  coder::array<double, 1U> a__8;
  coder::array<double, 1U> b_x0;
  coder::array<double, 1U> wsstFrequencyAxis;
  coder::array<float, 2U> a__1;
  coder::array<float, 2U> a__5;
  coder::array<float, 2U> fsstPowerMatrix;
  coder::array<float, 2U> fsstProbabilityMatrix;
  coder::array<float, 2U> powerMatrix;
  coder::array<float, 2U> wsstPowerMatrix;
  coder::array<float, 1U> b_fsstPowerMatrix_data;
  coder::array<float, 1U> centeredFSSTTime;
  coder::array<float, 1U> d_fsstPowerMatrix;
  coder::array<float, 1U> fsstRidgeFrequencySequence;
  coder::array<float, 1U> r;
  coder::array<float, 1U> ridgeFrequencySequence;
  coder::array<float, 1U> spectralCentroidSequence;
  coder::array<float, 1U> spectralCrestSequence;
  coder::array<float, 1U> spectralEntropySequence;
  coder::array<float, 1U> spectralFlatnessSequence;
  coder::array<float, 1U> spectralKurtosisSequence;
  coder::array<float, 1U> spectralSkewnessSequence;
  coder::array<float, 1U> spectralSpreadSequence;
  creal32_T y;
  double tmp_data[1024];
  double q_varargin_1;
  double r_varargin_1;
  float b_y[512];
  float frequencyAxis[512];
  float frequencyDifference[512];
  float probabilitySpectrum[512];
  float fsstPowerMatrix_data[258];
  float fv[4];
  float b_varargin_1;
  float c_varargin_1;
  float d_varargin_1;
  float e_varargin_1;
  float f_varargin_1;
  float g_varargin_1;
  float h_varargin_1;
  float i_varargin_1;
  float j_varargin_1;
  float k_varargin_1;
  float l_varargin_1;
  float localCentroid;
  float m_varargin_1;
  float n_varargin_1;
  float s_varargin_1;
  float t_varargin_1;
  float u_varargin_1;
  float varargin_1;
  int i;
  int loop_ub;
  int upperRidgeIndex;
  //  EXTRACTTIMEFREQUENCYFEATURES
  //  从一段 IQ 信号中提取 25 个基于语谱图的时频特征
  //  Input:
  //    x           复数 IQ 向量。输入被视为一个完整的段。
  //
  //    sampleRate  采样率
  //
  //  Output:
  //    features    1-by-25 的单精度特征向量
  //
  //  Feature order:
  //     1  SpectralKurtosisMean 谱峰度均值
  //     2  SpectralKurtosisStd 谱峰度标准差
  //     3  SpectralSkewnessMean 谱偏度均值
  //     4  SpectralSkewnessStd 谱偏度标准差
  //     5  SpectralCrestMean 谱峰因子均值
  //     6  SpectralCrestStd 谱峰因子标准差
  //     7  SpectralFlatnessMean 谱平坦度均值
  //     8  SpectralFlatnessStd 谱平坦度标准差
  //     9  SpectralEntropyMean 谱熵均值
  //    10  SpectralEntropyStd 谱熵标准差
  //    11  NormalizedSTFTRidgeFrequencyStd STFT归一化脊频率标准差
  //    12  NormalizedSTFTRidgeSlope STFT归一化脊频率斜率
  //    13  NormalizedSpectralCentroidStd  归一化谱质心标准差
  //    14  NormalizedSpectralSpreadMean  归一化频谱扩展均值
  //    15  NormalizedSpectralSpreadStd  归一化频谱扩展标准差
  //
  //    16  NormalizedFSSTRidgeFrequencyStd
  //                                       FSST归一化脊频率标准差
  //    17  NormalizedFSSTRidgeSlope
  //                                       FSST归一化脊频率斜率
  //    18  NormalizedFSSTRidgeCurvatureRMS
  //                                       FSST归一化脊线曲率RMS
  //    19  FSSTRidgeEnergyRatio
  //                                       FSST脊线能量占比
  //    20  NormalizedFSSTTimeFrequencyEntropy
  //                                       FSST归一化全局时频熵
  //
  //    21  NormalizedWSSTRidgeFrequencyStd
  //                                       WSST归一化脊频率标准差
  //    22  NormalizedWSSTRidgeSlope
  //                                       WSST归一化脊频率斜率
  //    23  NormalizedWSSTBandwidthMean
  //                                       WSST归一化局部带宽均值
  //    24  NormalizedWSSTBandwidthStd
  //                                       WSST归一化局部带宽标准差
  //    25  WSSTRidgeEnergyRatio
  //                                       WSST脊线能量占比
  //  1. 输入特征
  loop_ub = x.size(0);
  xIn.set_size(x.size(0));
  upperRidgeIndex = x.size(0);
  i = (x.size(0) < 800);
  if (i) {
    for (int i1{0}; i1 < loop_ub; i1++) {
      xIn[i1] = x[i1];
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

    for (int i1 = 0; i1 < upperRidgeIndex; i1++) {
      xIn[i1] = x[i1];
    }
  }
  // 检验输入长度和采样率检查
  std::memset(&features[0], 0, 25U * sizeof(float));
  //  2. 去除直流偏量
  y = coder::blockedSummation(xIn, xIn.size(0));
  if (y.im == 0.0F) {
    y.re /= static_cast<float>(xIn.size(0));
    y.im = 0.0F;
  } else if (y.re == 0.0F) {
    y.re = 0.0F;
    y.im /= static_cast<float>(xIn.size(0));
  } else {
    y.re /= static_cast<float>(xIn.size(0));
    y.im /= static_cast<float>(xIn.size(0));
  }
  x0.set_size(loop_ub);
  if (i) {
    for (int i2{0}; i2 < loop_ub; i2++) {
      x0[i2].re = xIn[i2].re - y.re;
      x0[i2].im = xIn[i2].im - y.im;
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

    for (int i2 = 0; i2 < loop_ub; i2++) {
      x0[i2].re = xIn[i2].re - y.re;
      x0[i2].im = xIn[i2].im - y.im;
    }
  }
  //  对于零信号或极弱信号，返回全零的有限特征向量。
  coder::b_abs(x0, fsstRidgeFrequencySequence);
  upperRidgeIndex = fsstRidgeFrequencySequence.size(0);
  spectralKurtosisSequence.set_size(fsstRidgeFrequencySequence.size(0));
  if (static_cast<int>(fsstRidgeFrequencySequence.size(0) < 800)) {
    for (int i3{0}; i3 < upperRidgeIndex; i3++) {
      localCentroid = fsstRidgeFrequencySequence[i3];
      spectralKurtosisSequence[i3] = localCentroid * localCentroid;
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(varargin_1)

    for (int i3 = 0; i3 < upperRidgeIndex; i3++) {
      varargin_1 = fsstRidgeFrequencySequence[i3];
      spectralKurtosisSequence[i3] = varargin_1 * varargin_1;
    }
  }
  if (!(coder::blockedSummation(spectralKurtosisSequence,
                                spectralKurtosisSequence.size(0)) <=
        1.0E-20F)) {
    float analysisWindow_data[258];
    int lowerRidgeIndex;
    int windowLength;
    //  3. 配置语谱图
    //
    //  窗长约为输入长度的四分之一：
    //
    //    128 个采样点  -> 窗长 32 点
    //    512 个采样点  -> 窗长 128 点
    //    4096 个采样点 -> 窗长 256 点
    //
    //  最大窗长限制为 256 点。
    //  使用 50% 重叠和固定的 512 点 FFT
    windowLength =
        static_cast<int>(std::floor(static_cast<double>(xIn.size(0)) / 4.0));
    if (windowLength < 32) {
      windowLength = 32;
    } else if (windowLength > 256) {
      windowLength = 256;
    }
    //  确保窗长不超过信号长度。
    if (windowLength > xIn.size(0)) {
      windowLength = loop_ub;
    }
    windowLength =
        static_cast<int>(std::floor(static_cast<double>(windowLength) / 2.0))
        << 1;
    upperRidgeIndex =
        coder::hamming(static_cast<double>(windowLength), tmp_data);
    for (int k{0}; k < upperRidgeIndex; k++) {
      analysisWindow_data[k] = static_cast<float>(tmp_data[k]);
    }
    //  4. 计算中心化的复数 IQ 语谱图
    coder::spectrogram(x0, analysisWindow_data, upperRidgeIndex,
                       std::floor(static_cast<double>(windowLength) / 2.0),
                       sampleRate, stftMatrix, frequencyAxis, a__1);
    //  将 STFT 系数转换为非负的时频功率分布。
    coder::b_abs(stftMatrix, powerMatrix);
    upperRidgeIndex = powerMatrix.size(1) << 9;
    powerMatrix.set_size(512, powerMatrix.size(1));
    lowerRidgeIndex = powerMatrix.size(1);
    if (static_cast<int>(upperRidgeIndex < 800)) {
      for (int i4{0}; i4 < upperRidgeIndex; i4++) {
        localCentroid = powerMatrix[i4];
        powerMatrix[i4] = localCentroid * localCentroid;
      }
    } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        b_varargin_1)

      for (int i4 = 0; i4 < upperRidgeIndex; i4++) {
        b_varargin_1 = powerMatrix[i4];
        powerMatrix[i4] = b_varargin_1 * b_varargin_1;
      }
    }
    if (powerMatrix.size(1) != 0) {
      __m128 r3;
      __m128 r4;
      float localSecondMoment;
      float localSpread;
      float localTotalPower;
      float wsstRidgeEnergy;
      int b_fsstPowerMatrix;
      int b_loop_ub;
      int c_fsstPowerMatrix;
      int d_loop_ub;
      int scalarLB;
      //  5. 为局部频谱特征分配序列存储空间
      // 因为要转换成C代码，所以数组的大小不能在运行时动态增长，所以在使用这个数组之前，需要先定下长度
      spectralKurtosisSequence.set_size(lowerRidgeIndex);
      spectralSkewnessSequence.set_size(lowerRidgeIndex);
      spectralCrestSequence.set_size(lowerRidgeIndex);
      spectralFlatnessSequence.set_size(lowerRidgeIndex);
      spectralEntropySequence.set_size(lowerRidgeIndex);
      ridgeFrequencySequence.set_size(lowerRidgeIndex);
      spectralCentroidSequence.set_size(lowerRidgeIndex);
      spectralSpreadSequence.set_size(lowerRidgeIndex);
      //  6. 在每个时间帧内计算五种频谱描述子
      //
      //  此处显式实现描述子，不移动负频率轴。
      //  直接使用中心化的物理频率轴进行谱偏度和谱峰度的计算。
      for (int frameIndex{0}; frameIndex < lowerRidgeIndex; frameIndex++) {
        spectralKurtosisSequence[frameIndex] = 0.0F;
        spectralSkewnessSequence[frameIndex] = 0.0F;
        spectralCrestSequence[frameIndex] = 0.0F;
        spectralFlatnessSequence[frameIndex] = 0.0F;
        spectralEntropySequence[frameIndex] = 0.0F;
        ridgeFrequencySequence[frameIndex] = 0.0F;
        spectralCentroidSequence[frameIndex] = 0.0F;
        spectralSpreadSequence[frameIndex] = 0.0F;
        localTotalPower = powerMatrix[512 * frameIndex];
        for (int k{0}; k < 511; k++) {
          localTotalPower += powerMatrix[(k + 512 * frameIndex) + 1];
        }
        if (!(localTotalPower <= 1.0E-20F)) {
          //      %% 6.1 局部频谱质心
          for (int k{0}; k <= 508; k += 4) {
            r3 = _mm_loadu_ps(&frequencyAxis[k]);
            r4 = _mm_loadu_ps(&powerMatrix[k + 512 * frameIndex]);
            _mm_storeu_ps(&frequencyDifference[k], _mm_mul_ps(r3, r4));
          }
          localCentroid = frequencyDifference[0];
          for (int k{0}; k < 511; k++) {
            localCentroid += frequencyDifference[k + 1];
          }
          localCentroid /= localTotalPower;
          spectralCentroidSequence[frameIndex] = localCentroid;
          //     %% 6.2 局部频谱扩展
          for (int k{0}; k <= 508; k += 4) {
            r3 = _mm_loadu_ps(&frequencyAxis[k]);
            r3 = _mm_sub_ps(r3, _mm_set1_ps(localCentroid));
            _mm_storeu_ps(&frequencyDifference[k], r3);
            r4 = _mm_loadu_ps(&powerMatrix[k + 512 * frameIndex]);
            _mm_storeu_ps(&probabilitySpectrum[k],
                          _mm_mul_ps(_mm_mul_ps(r3, r3), r4));
          }
          localCentroid = probabilitySpectrum[0];
          for (int k{0}; k < 511; k++) {
            localCentroid += probabilitySpectrum[k + 1];
          }
          localSecondMoment = std::fmax(localCentroid / localTotalPower, 0.0F);
          localSpread = std::sqrt(localSecondMoment);
          spectralSpreadSequence[frameIndex] = localSpread;
          //     %% 6.3 谱偏度和谱峰度
          if (localSecondMoment > 1.0E-20F) {
            for (int k{0}; k < 512; k++) {
              localCentroid = frequencyDifference[k];
              b_y[k] = rt_powf_snf(localCentroid, 4.0F);
              probabilitySpectrum[k] = rt_powf_snf(localCentroid, 3.0F) *
                                       powerMatrix[k + 512 * frameIndex];
            }
            localCentroid = probabilitySpectrum[0];
            for (int k{0}; k < 511; k++) {
              localCentroid += probabilitySpectrum[k + 1];
            }
            spectralSkewnessSequence[frameIndex] =
                localCentroid / localTotalPower /
                (rt_powf_snf(localSpread, 3.0F) + 1.0E-20F);
            for (int k{0}; k <= 508; k += 4) {
              r3 = _mm_loadu_ps(&b_y[k]);
              r4 = _mm_loadu_ps(&powerMatrix[k + 512 * frameIndex]);
              _mm_storeu_ps(&b_y[k], _mm_mul_ps(r3, r4));
            }
            localCentroid = b_y[0];
            for (int k{0}; k < 511; k++) {
              localCentroid += b_y[k + 1];
            }
            spectralKurtosisSequence[frameIndex] =
                localCentroid / localTotalPower /
                (localSecondMoment * localSecondMoment + 1.0E-20F);
          }
          //     %% 6.4 谱峰因子
          localCentroid = localTotalPower / 512.0F;
          spectralCrestSequence[frameIndex] =
              coder::internal::b_maximum(&powerMatrix[512 * frameIndex]) /
              (localCentroid + 1.0E-20F);
          //     %% 6.5 谱平坦度
          //
          localSpread = std::fmax(localCentroid * 1.0E-12F, 1.0E-20F);
          for (int k{0}; k < 512; k++) {
            frequencyDifference[k] = std::log(
                std::fmax(powerMatrix[k + 512 * frameIndex], localSpread));
          }
          spectralFlatnessSequence[frameIndex] =
              std::exp(coder::mean(frequencyDifference)) /
              (localCentroid + localSpread);
          //     %% 6.6 归一化谱熵
          for (int k{0}; k < 512; k++) {
            localCentroid = powerMatrix[k + 512 * frameIndex] / localTotalPower;
            localCentroid *= std::log(localCentroid + 1.0E-20F);
            probabilitySpectrum[k] = localCentroid;
          }
          localCentroid = probabilitySpectrum[0];
          for (int k{0}; k < 511; k++) {
            localCentroid += probabilitySpectrum[k + 1];
          }
          spectralEntropySequence[frameIndex] = -localCentroid / 6.23832464F;
          coder::internal::b_maximum(&powerMatrix[512 * frameIndex],
                                     upperRidgeIndex);
          ridgeFrequencySequence[frameIndex] =
              frequencyAxis[upperRidgeIndex - 1];
        }
      }
      //  7. 局部频谱描述子的时间统计量
      //  =========================================================================
      //  局部函数：均值和总体标准差
      //  =========================================================================
      localSpread = coder::blockedSummation(spectralKurtosisSequence,
                                            spectralKurtosisSequence.size(0)) /
                    static_cast<float>(spectralKurtosisSequence.size(0));
      upperRidgeIndex = (spectralKurtosisSequence.size(0) < 800);
      if (upperRidgeIndex) {
        for (int i5{0}; i5 < lowerRidgeIndex; i5++) {
          localCentroid = spectralKurtosisSequence[i5] - localSpread;
          spectralKurtosisSequence[i5] = localCentroid * localCentroid;
        }
      } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        c_varargin_1)

        for (int i5 = 0; i5 < lowerRidgeIndex; i5++) {
          c_varargin_1 = spectralKurtosisSequence[i5] - localSpread;
          spectralKurtosisSequence[i5] = c_varargin_1 * c_varargin_1;
        }
      }
      features[1] = std::sqrt(
          std::fmax(coder::blockedSummation(spectralKurtosisSequence,
                                            spectralKurtosisSequence.size(0)) /
                        static_cast<float>(spectralKurtosisSequence.size(0)),
                    0.0F));
      features[0] = localSpread;
      //  =========================================================================
      //  局部函数：均值和总体标准差
      //  =========================================================================
      localSpread = coder::blockedSummation(spectralSkewnessSequence,
                                            spectralSkewnessSequence.size(0)) /
                    static_cast<float>(spectralSkewnessSequence.size(0));
      spectralKurtosisSequence.set_size(lowerRidgeIndex);
      if (upperRidgeIndex) {
        for (int i6{0}; i6 < lowerRidgeIndex; i6++) {
          localCentroid = spectralSkewnessSequence[i6] - localSpread;
          spectralKurtosisSequence[i6] = localCentroid * localCentroid;
        }
      } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        d_varargin_1)

        for (int i6 = 0; i6 < lowerRidgeIndex; i6++) {
          d_varargin_1 = spectralSkewnessSequence[i6] - localSpread;
          spectralKurtosisSequence[i6] = d_varargin_1 * d_varargin_1;
        }
      }
      features[3] = std::sqrt(
          std::fmax(coder::blockedSummation(spectralKurtosisSequence,
                                            spectralKurtosisSequence.size(0)) /
                        static_cast<float>(spectralKurtosisSequence.size(0)),
                    0.0F));
      features[2] = localSpread;
      //  =========================================================================
      //  局部函数：均值和总体标准差
      //  =========================================================================
      localSpread = coder::blockedSummation(spectralCrestSequence,
                                            spectralCrestSequence.size(0)) /
                    static_cast<float>(spectralCrestSequence.size(0));
      spectralKurtosisSequence.set_size(lowerRidgeIndex);
      if (upperRidgeIndex) {
        for (int i7{0}; i7 < lowerRidgeIndex; i7++) {
          localCentroid = spectralCrestSequence[i7] - localSpread;
          spectralKurtosisSequence[i7] = localCentroid * localCentroid;
        }
      } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        e_varargin_1)

        for (int i7 = 0; i7 < lowerRidgeIndex; i7++) {
          e_varargin_1 = spectralCrestSequence[i7] - localSpread;
          spectralKurtosisSequence[i7] = e_varargin_1 * e_varargin_1;
        }
      }
      features[5] = std::sqrt(
          std::fmax(coder::blockedSummation(spectralKurtosisSequence,
                                            spectralKurtosisSequence.size(0)) /
                        static_cast<float>(spectralKurtosisSequence.size(0)),
                    0.0F));
      features[4] = localSpread;
      //  =========================================================================
      //  局部函数：均值和总体标准差
      //  =========================================================================
      localSpread = coder::blockedSummation(spectralFlatnessSequence,
                                            spectralFlatnessSequence.size(0)) /
                    static_cast<float>(spectralFlatnessSequence.size(0));
      spectralKurtosisSequence.set_size(lowerRidgeIndex);
      if (upperRidgeIndex) {
        for (int i8{0}; i8 < lowerRidgeIndex; i8++) {
          localCentroid = spectralFlatnessSequence[i8] - localSpread;
          spectralKurtosisSequence[i8] = localCentroid * localCentroid;
        }
      } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        f_varargin_1)

        for (int i8 = 0; i8 < lowerRidgeIndex; i8++) {
          f_varargin_1 = spectralFlatnessSequence[i8] - localSpread;
          spectralKurtosisSequence[i8] = f_varargin_1 * f_varargin_1;
        }
      }
      features[7] = std::sqrt(
          std::fmax(coder::blockedSummation(spectralKurtosisSequence,
                                            spectralKurtosisSequence.size(0)) /
                        static_cast<float>(spectralKurtosisSequence.size(0)),
                    0.0F));
      features[6] = localSpread;
      //  =========================================================================
      //  局部函数：均值和总体标准差
      //  =========================================================================
      localSpread = coder::blockedSummation(spectralEntropySequence,
                                            spectralEntropySequence.size(0)) /
                    static_cast<float>(spectralEntropySequence.size(0));
      spectralKurtosisSequence.set_size(lowerRidgeIndex);
      features[8] = localSpread;
      //  8. 提取主导时频脊线
      //  Use the peak-power frequency in each STFT frame as a deterministic
      //  ridge.
      if (upperRidgeIndex) {
        for (int i9{0}; i9 < lowerRidgeIndex; i9++) {
          localCentroid = spectralEntropySequence[i9] - localSpread;
          spectralKurtosisSequence[i9] = localCentroid * localCentroid;
          ridgeFrequencySequence[i9] =
              ridgeFrequencySequence[i9] / static_cast<float>(sampleRate);
        }
      } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        g_varargin_1)

        for (int i9 = 0; i9 < lowerRidgeIndex; i9++) {
          g_varargin_1 = spectralEntropySequence[i9] - localSpread;
          spectralKurtosisSequence[i9] = g_varargin_1 * g_varargin_1;
          ridgeFrequencySequence[i9] =
              ridgeFrequencySequence[i9] / static_cast<float>(sampleRate);
        }
      }
      features[9] = std::sqrt(
          std::fmax(coder::blockedSummation(spectralKurtosisSequence,
                                            spectralKurtosisSequence.size(0)) /
                        static_cast<float>(spectralKurtosisSequence.size(0)),
                    0.0F));
      //  =========================================================================
      //  局部函数：均值和总体标准差
      //  =========================================================================
      localSpread = coder::blockedSummation(ridgeFrequencySequence,
                                            ridgeFrequencySequence.size(0)) /
                    static_cast<float>(ridgeFrequencySequence.size(0));
      spectralKurtosisSequence.set_size(lowerRidgeIndex);
      //  9. 计算归一化脊频率斜率
      //
      //  时间轴归一化到 [0,1]，因此斜率表示
      //  在整个 IQ 片段持续时间内归一化频率的变化量。
      //  10. 瞬时频率
      //
      //  instfreq 计算时频功率分布的第一条件谱矩。
      //  Use the power-weighted frequency centroid of each STFT frame.
      if (upperRidgeIndex) {
        for (int i10{0}; i10 < lowerRidgeIndex; i10++) {
          localCentroid = ridgeFrequencySequence[i10] - localSpread;
          spectralKurtosisSequence[i10] = localCentroid * localCentroid;
          spectralCentroidSequence[i10] =
              spectralCentroidSequence[i10] / static_cast<float>(sampleRate);
        }
      } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        h_varargin_1)

        for (int i10 = 0; i10 < lowerRidgeIndex; i10++) {
          h_varargin_1 = ridgeFrequencySequence[i10] - localSpread;
          spectralKurtosisSequence[i10] = h_varargin_1 * h_varargin_1;
          spectralCentroidSequence[i10] =
              spectralCentroidSequence[i10] / static_cast<float>(sampleRate);
        }
      }
      features[10] = std::sqrt(
          std::fmax(coder::blockedSummation(spectralKurtosisSequence,
                                            spectralKurtosisSequence.size(0)) /
                        static_cast<float>(spectralKurtosisSequence.size(0)),
                    0.0F));
      //  =========================================================================
      //  局部函数：均值和总体标准差
      //  =========================================================================
      localSpread = coder::blockedSummation(spectralCentroidSequence,
                                            spectralCentroidSequence.size(0)) /
                    static_cast<float>(spectralCentroidSequence.size(0));
      spectralKurtosisSequence.set_size(lowerRidgeIndex);
      //  11. 瞬时带宽
      //
      //  ScaleFactor = 1 使结果等于频谱标准差，
      //  而不使用 MATLAB 默认的额外比例因子。
      //  Use the power-weighted spectral spread of each STFT frame.
      if (upperRidgeIndex) {
        for (int i11{0}; i11 < lowerRidgeIndex; i11++) {
          localCentroid = spectralCentroidSequence[i11] - localSpread;
          spectralKurtosisSequence[i11] = localCentroid * localCentroid;
          spectralSpreadSequence[i11] =
              spectralSpreadSequence[i11] / static_cast<float>(sampleRate);
        }
      } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        i_varargin_1)

        for (int i11 = 0; i11 < lowerRidgeIndex; i11++) {
          i_varargin_1 = spectralCentroidSequence[i11] - localSpread;
          spectralKurtosisSequence[i11] = i_varargin_1 * i_varargin_1;
          spectralSpreadSequence[i11] =
              spectralSpreadSequence[i11] / static_cast<float>(sampleRate);
        }
      }
      features[12] = std::sqrt(
          std::fmax(coder::blockedSummation(spectralKurtosisSequence,
                                            spectralKurtosisSequence.size(0)) /
                        static_cast<float>(spectralKurtosisSequence.size(0)),
                    0.0F));
      //  =========================================================================
      //  局部函数：均值和总体标准差
      //  =========================================================================
      localSpread = coder::blockedSummation(spectralSpreadSequence,
                                            spectralSpreadSequence.size(0)) /
                    static_cast<float>(spectralSpreadSequence.size(0));
      spectralKurtosisSequence.set_size(lowerRidgeIndex);
      if (upperRidgeIndex) {
        for (int i12{0}; i12 < lowerRidgeIndex; i12++) {
          localCentroid = spectralSpreadSequence[i12] - localSpread;
          spectralKurtosisSequence[i12] = localCentroid * localCentroid;
        }
      } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        j_varargin_1)

        for (int i12 = 0; i12 < lowerRidgeIndex; i12++) {
          j_varargin_1 = spectralSpreadSequence[i12] - localSpread;
          spectralKurtosisSequence[i12] = j_varargin_1 * j_varargin_1;
        }
      }
      features[14] = std::sqrt(
          std::fmax(coder::blockedSummation(spectralKurtosisSequence,
                                            spectralKurtosisSequence.size(0)) /
                        static_cast<float>(spectralKurtosisSequence.size(0)),
                    0.0F));
      features[13] = localSpread;
      //  12. FSST 时频特征
      //
      //  Fourier Synchrosqueezed Transform
      //
      //  FSST 可以直接处理复数 IQ 信号。
      //  使用与 Spectrogram 相同的分析窗长度，使两种时频表示
      //  的基本时间尺度保持一致。
      //
      //  MATLAB Coder 要求 FSST 的分析窗为 double 类型。
      features[16] = 0.0F;
      features[17] = 0.0F;
      features[18] = 0.0F;
      features[19] = 0.0F;
      //  12.1 计算 FSST
      upperRidgeIndex =
          coder::hamming(static_cast<double>(windowLength), tmp_data);
      coder::fsst(x0, sampleRate, tmp_data, upperRidgeIndex, fsstMatrix,
                  analysisWindow_data, a__5);
      coder::c_abs(fsstMatrix, fsstPowerMatrix);
      b_loop_ub = fsstPowerMatrix.size(0) * fsstPowerMatrix.size(1);
      b_fsstPowerMatrix = fsstPowerMatrix.size(0);
      c_fsstPowerMatrix = fsstPowerMatrix.size(1);
      if (static_cast<int>(b_loop_ub < 800)) {
        for (int i13{0}; i13 < b_loop_ub; i13++) {
          localCentroid = fsstPowerMatrix[i13];
          fsstPowerMatrix[i13] = localCentroid * localCentroid;
        }
      } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        k_varargin_1)

        for (int i13 = 0; i13 < b_loop_ub; i13++) {
          k_varargin_1 = fsstPowerMatrix[i13];
          fsstPowerMatrix[i13] = k_varargin_1 * k_varargin_1;
        }
      }
      d_fsstPowerMatrix = fsstPowerMatrix.reshape(b_loop_ub);
      localSecondMoment = coder::blockedSummation(d_fsstPowerMatrix, b_loop_ub);
      //  12.2 提取 FSST 主脊线
      //
      //  每个时间位置选取功率最大的频率作为主脊线。
      //
      //  同时统计主脊线所在频率点及其上下各一个频率 bin 的能量，
      //  用于计算 Ridge Energy Ratio。
      fsstRidgeFrequencySequence.set_size(c_fsstPowerMatrix);
      if (c_fsstPowerMatrix - 1 >= 0) {
        std::memset(&fsstRidgeFrequencySequence[0], 0,
                    static_cast<unsigned int>(c_fsstPowerMatrix) *
                        sizeof(float));
      }
      localTotalPower = 0.0F;
      if (localSecondMoment > 1.0E-20F) {
        for (int frameIndex{0}; frameIndex < c_fsstPowerMatrix; frameIndex++) {
          spectralKurtosisSequence.set_size(b_fsstPowerMatrix);
          for (int k{0}; k < b_fsstPowerMatrix; k++) {
            spectralKurtosisSequence[k] =
                fsstPowerMatrix[k + fsstPowerMatrix.size(0) * frameIndex];
          }
          if (!(coder::blockedSummation(spectralKurtosisSequence,
                                        b_fsstPowerMatrix) <= 1.0E-20F)) {
            coder::internal::maximum(spectralKurtosisSequence, upperRidgeIndex);
            fsstRidgeFrequencySequence[frameIndex] =
                analysisWindow_data[upperRidgeIndex - 1];
            //  主脊线上下各取一个相邻频率 bin。
            lowerRidgeIndex = static_cast<int>(
                std::fmax(static_cast<double>(upperRidgeIndex) - 1.0, 1.0));
            upperRidgeIndex = static_cast<int>(
                std::fmin(static_cast<double>(upperRidgeIndex) + 1.0,
                          static_cast<double>(b_fsstPowerMatrix)));
            if (lowerRidgeIndex > upperRidgeIndex) {
              lowerRidgeIndex = 0;
              upperRidgeIndex = 0;
            } else {
              lowerRidgeIndex--;
            }
            upperRidgeIndex -= lowerRidgeIndex;
            for (int k{0}; k < upperRidgeIndex; k++) {
              fsstPowerMatrix_data[k] =
                  fsstPowerMatrix[(lowerRidgeIndex + k) +
                                  fsstPowerMatrix.size(0) * frameIndex];
            }
            b_fsstPowerMatrix_data.set(&fsstPowerMatrix_data[0],
                                       upperRidgeIndex);
            localTotalPower += coder::blockedSummation(b_fsstPowerMatrix_data,
                                                       upperRidgeIndex);
          }
        }
      }
      //  12.3 FSST 归一化脊频率标准差
      spectralSkewnessSequence.set_size(c_fsstPowerMatrix);
      scalarLB = (fsstRidgeFrequencySequence.size(0) / 4) << 2;
      upperRidgeIndex = scalarLB - 4;
      for (int k{0}; k <= upperRidgeIndex; k += 4) {
        r3 = _mm_loadu_ps(&fsstRidgeFrequencySequence[k]);
        r3 = _mm_div_ps(r3, _mm_set1_ps(static_cast<float>(sampleRate)));
        _mm_storeu_ps(&spectralSkewnessSequence[k], r3);
        _mm_storeu_ps(&fsstRidgeFrequencySequence[k], r3);
      }
      for (int k{scalarLB}; k < c_fsstPowerMatrix; k++) {
        localCentroid =
            fsstRidgeFrequencySequence[k] / static_cast<float>(sampleRate);
        spectralSkewnessSequence[k] = localCentroid;
        fsstRidgeFrequencySequence[k] = localCentroid;
      }
      //  =========================================================================
      //  局部函数：均值和总体标准差
      //  =========================================================================
      if (spectralSkewnessSequence.size(0) == 0) {
        features[15] = 0.0F;
      } else {
        localSpread =
            coder::blockedSummation(spectralSkewnessSequence,
                                    spectralSkewnessSequence.size(0)) /
            static_cast<float>(spectralSkewnessSequence.size(0));
        upperRidgeIndex = spectralSkewnessSequence.size(0);
        spectralKurtosisSequence.set_size(spectralSkewnessSequence.size(0));
        if (static_cast<int>(spectralSkewnessSequence.size(0) < 800)) {
          for (int i14{0}; i14 < upperRidgeIndex; i14++) {
            localCentroid = spectralSkewnessSequence[i14] - localSpread;
            spectralKurtosisSequence[i14] = localCentroid * localCentroid;
          }
        } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        l_varargin_1)

          for (int i14 = 0; i14 < upperRidgeIndex; i14++) {
            l_varargin_1 = spectralSkewnessSequence[i14] - localSpread;
            spectralKurtosisSequence[i14] = l_varargin_1 * l_varargin_1;
          }
        }
        features[15] = std::sqrt(std::fmax(
            coder::blockedSummation(spectralKurtosisSequence,
                                    spectralKurtosisSequence.size(0)) /
                static_cast<float>(spectralKurtosisSequence.size(0)),
            0.0F));
      }
      //  12.4 FSST 归一化脊频率斜率
      //
      //  时间轴归一化到 [0,1]。
      //
      //  正斜率：
      //    主频总体向高频方向移动
      //
      //  负斜率：
      //    主频总体向低频方向移动
      if (fsstPowerMatrix.size(1) >= 2) {
        localCentroid = static_cast<float>(fsstPowerMatrix.size(1)) - 1.0F;
        spectralKurtosisSequence.set_size(c_fsstPowerMatrix);
        upperRidgeIndex = fsstPowerMatrix.size(1) - 1;
        lowerRidgeIndex = scalarLB - 4;
        for (int k{0}; k <= lowerRidgeIndex; k += 4) {
          fv[0] = static_cast<float>(k);
          fv[1] = static_cast<float>(k + 1);
          fv[2] = static_cast<float>(k + 2);
          fv[3] = static_cast<float>(k + 3);
          r3 = _mm_loadu_ps(&fv[0]);
          _mm_storeu_ps(&spectralKurtosisSequence[k],
                        _mm_div_ps(r3, _mm_set1_ps(localCentroid)));
        }
        for (int k{scalarLB}; k <= upperRidgeIndex; k++) {
          spectralKurtosisSequence[k] = static_cast<float>(k) / localCentroid;
        }
        localCentroid =
            coder::blockedSummation(spectralKurtosisSequence,
                                    spectralKurtosisSequence.size(0)) /
            static_cast<float>(spectralKurtosisSequence.size(0));
        localSpread =
            coder::blockedSummation(spectralSkewnessSequence,
                                    spectralSkewnessSequence.size(0)) /
            static_cast<float>(spectralSkewnessSequence.size(0));
        upperRidgeIndex = spectralKurtosisSequence.size(0);
        centeredFSSTTime.set_size(spectralKurtosisSequence.size(0));
        scalarLB = (spectralKurtosisSequence.size(0) / 4) << 2;
        lowerRidgeIndex = scalarLB - 4;
        for (int k{0}; k <= lowerRidgeIndex; k += 4) {
          r3 = _mm_loadu_ps(&spectralKurtosisSequence[k]);
          _mm_storeu_ps(&centeredFSSTTime[k],
                        _mm_sub_ps(r3, _mm_set1_ps(localCentroid)));
        }
        for (int k{scalarLB}; k < upperRidgeIndex; k++) {
          centeredFSSTTime[k] = spectralKurtosisSequence[k] - localCentroid;
        }
        scalarLB = centeredFSSTTime.size(0);
        spectralKurtosisSequence.set_size(centeredFSSTTime.size(0));
        upperRidgeIndex = centeredFSSTTime.size(0);
        if (static_cast<int>(centeredFSSTTime.size(0) < 800)) {
          for (int i17{0}; i17 < scalarLB; i17++) {
            localCentroid = centeredFSSTTime[i17];
            spectralKurtosisSequence[i17] = localCentroid * localCentroid;
          }
        } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        m_varargin_1)

          for (int i17 = 0; i17 < upperRidgeIndex; i17++) {
            m_varargin_1 = centeredFSSTTime[i17];
            spectralKurtosisSequence[i17] = m_varargin_1 * m_varargin_1;
          }
        }
        localCentroid = coder::blockedSummation(
            spectralKurtosisSequence, spectralKurtosisSequence.size(0));
        if (localCentroid > 1.0E-20F) {
          if (centeredFSSTTime.size(0) == spectralSkewnessSequence.size(0)) {
            spectralKurtosisSequence.set_size(scalarLB);
            upperRidgeIndex = (centeredFSSTTime.size(0) / 4) << 2;
            lowerRidgeIndex = upperRidgeIndex - 4;
            for (int k{0}; k <= lowerRidgeIndex; k += 4) {
              r3 = _mm_loadu_ps(&spectralSkewnessSequence[k]);
              r4 = _mm_loadu_ps(&centeredFSSTTime[k]);
              _mm_storeu_ps(
                  &spectralKurtosisSequence[k],
                  _mm_mul_ps(r4, _mm_sub_ps(r3, _mm_set1_ps(localSpread))));
            }
            for (int k{upperRidgeIndex}; k < scalarLB; k++) {
              spectralKurtosisSequence[k] =
                  centeredFSSTTime[k] *
                  (spectralSkewnessSequence[k] - localSpread);
            }
          } else {
            binary_expand_op_6(spectralKurtosisSequence, centeredFSSTTime,
                               spectralSkewnessSequence, localSpread);
          }
          features[16] =
              coder::blockedSummation(spectralKurtosisSequence,
                                      spectralKurtosisSequence.size(0)) /
              localCentroid;
        }
      }
      //  12.5 FSST 脊线曲率 RMS
      //
      //  使用归一化频率脊线的离散二阶差分：
      //
      //    f[n+1] - 2*f[n] + f[n-1]
      //
      //  表示脊线偏离恒定斜率变化的程度。
      //
      //  数值较小：
      //    脊线更接近直线或平稳频率
      //
      //  数值较大：
      //    脊线存在弯曲、跳变或扫频速率变化
      if (fsstPowerMatrix.size(1) >= 3) {
        int c_loop_ub;
        int i15;
        int i16;
        if (spectralSkewnessSequence.size(0) < 3) {
          i15 = -1;
          scalarLB = -1;
          i16 = -1;
          lowerRidgeIndex = -1;
        } else {
          i15 = 1;
          scalarLB = fsstRidgeFrequencySequence.size(0) - 1;
          i16 = 0;
          lowerRidgeIndex = fsstRidgeFrequencySequence.size(0) - 2;
        }
        if (spectralSkewnessSequence.size(0) - 2 < 1) {
          upperRidgeIndex = 0;
        } else {
          upperRidgeIndex = fsstRidgeFrequencySequence.size(0) - 2;
        }
        c_loop_ub = scalarLB - i15;
        windowLength = lowerRidgeIndex - i16;
        if (c_loop_ub == 1) {
          d_loop_ub = windowLength;
        } else {
          d_loop_ub = c_loop_ub;
        }
        if ((c_loop_ub == windowLength) && (d_loop_ub == upperRidgeIndex)) {
          spectralKurtosisSequence.set_size(c_loop_ub);
          if (static_cast<int>(c_loop_ub < 800)) {
            for (int i18{0}; i18 < c_loop_ub; i18++) {
              localCentroid =
                  (fsstRidgeFrequencySequence[(i15 + i18) + 1] -
                   2.0F * spectralSkewnessSequence[(i16 + i18) + 1]) +
                  spectralSkewnessSequence[i18];
              spectralKurtosisSequence[i18] = localCentroid * localCentroid;
            }
          } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        n_varargin_1)

            for (int i18 = 0; i18 < c_loop_ub; i18++) {
              n_varargin_1 =
                  (fsstRidgeFrequencySequence[(i15 + i18) + 1] -
                   2.0F * spectralSkewnessSequence[(i16 + i18) + 1]) +
                  spectralSkewnessSequence[i18];
              spectralKurtosisSequence[i18] = n_varargin_1 * n_varargin_1;
            }
          }
        } else {
          binary_expand_op_5(spectralKurtosisSequence,
                             fsstRidgeFrequencySequence, i15 + 1, scalarLB,
                             spectralSkewnessSequence, i16 + 1, lowerRidgeIndex,
                             upperRidgeIndex - 1);
        }
        features[17] = std::sqrt(
            coder::blockedSummation(spectralKurtosisSequence,
                                    spectralKurtosisSequence.size(0)) /
            static_cast<float>(spectralKurtosisSequence.size(0)));
      }
      //  12.6 FSST 主脊线能量占比
      //
      //  描述 FSST 能量是否集中在主要时频轨迹附近。
      //  12.7 FSST 全局归一化时频熵
      //
      //  将整张 FSST 时频功率图归一化为概率分布：
      //
      //    p(i,j) = P(i,j) / sum(P)
      //
      //  再计算整张时频图的归一化 Shannon entropy。
      //
      //  较小：
      //    能量集中在较少时频位置
      //
      //  较大：
      //    时频能量分布更分散、更复杂
      if (localSecondMoment > 1.0E-20F) {
        features[18] = localTotalPower / (localSecondMoment + 1.0E-20F);
        fsstProbabilityMatrix.set_size(b_fsstPowerMatrix, c_fsstPowerMatrix);
        upperRidgeIndex = (b_loop_ub / 4) << 2;
        lowerRidgeIndex = upperRidgeIndex - 4;
        for (int k{0}; k <= lowerRidgeIndex; k += 4) {
          r3 = _mm_loadu_ps(&fsstPowerMatrix[k]);
          _mm_storeu_ps(&fsstProbabilityMatrix[k],
                        _mm_div_ps(r3, _mm_set1_ps(localSecondMoment)));
        }
        for (int k{upperRidgeIndex}; k < b_loop_ub; k++) {
          fsstProbabilityMatrix[k] = fsstPowerMatrix[k] / localSecondMoment;
        }
        localCentroid = std::log(static_cast<float>(b_loop_ub));
        if (localCentroid > 0.0F) {
          lowerRidgeIndex =
              fsstProbabilityMatrix.size(0) * fsstProbabilityMatrix.size(1);
          r.set_size(lowerRidgeIndex);
          scalarLB = (lowerRidgeIndex / 4) << 2;
          upperRidgeIndex = scalarLB - 4;
          for (int k{0}; k <= upperRidgeIndex; k += 4) {
            r3 = _mm_loadu_ps(&fsstProbabilityMatrix[k]);
            _mm_storeu_ps(&r[k], _mm_add_ps(r3, _mm_set1_ps(1.0E-20F)));
          }
          for (int k{scalarLB}; k < lowerRidgeIndex; k++) {
            r[k] = fsstProbabilityMatrix[k] + 1.0E-20F;
          }
          coder::b_log(r);
          spectralKurtosisSequence.set_size(lowerRidgeIndex);
          upperRidgeIndex = scalarLB - 4;
          for (int k{0}; k <= upperRidgeIndex; k += 4) {
            r3 = _mm_loadu_ps(&fsstProbabilityMatrix[k]);
            r4 = _mm_loadu_ps(&r[k]);
            _mm_storeu_ps(&spectralKurtosisSequence[k], _mm_mul_ps(r3, r4));
          }
          for (int k{scalarLB}; k < lowerRidgeIndex; k++) {
            spectralKurtosisSequence[k] = fsstProbabilityMatrix[k] * r[k];
          }
          features[19] =
              -coder::blockedSummation(spectralKurtosisSequence,
                                       spectralKurtosisSequence.size(0)) /
              localCentroid;
        }
      }
      //  13. WSST 时频特征
      //
      //  Wavelet Synchrosqueezed Transform
      //
      //  MATLAB 的 wsst() 只支持实数输入。
      //
      //  对复数 IQ 信号分别计算：
      //
      //    WSST(I)
      //    WSST(Q)
      //
      //  然后构造联合时频功率：
      //
      //    Pwsst = |WSST(I)|^2 + |WSST(Q)|^2
      //
      //  WSST 内部使用 double，以提高 MATLAB 与生成 C/C++
      //  结果之间的一致性。
      features[21] = 0.0F;
      features[24] = 0.0F;
      //  13.1 分别计算 I/Q 两路 WSST
      //  'amor' 为固定的解析 Morlet 小波。
      //  字面量在代码生成时作为编译期常量。
      b_x0.set_size(loop_ub);
      if (i) {
        for (int i19{0}; i19 < loop_ub; i19++) {
          b_x0[i19] = x0[i19].re;
        }
      } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

        for (int i19 = 0; i19 < loop_ub; i19++) {
          b_x0[i19] = x0[i19].re;
        }
      }
      coder::wsst(b_x0, sampleRate, wsstMatrixI, wsstFrequencyAxis);
      b_x0.set_size(loop_ub);
      if (i) {
        for (int i20{0}; i20 < loop_ub; i20++) {
          b_x0[i20] = x0[i20].im;
        }
      } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

        for (int i20 = 0; i20 < loop_ub; i20++) {
          b_x0[i20] = x0[i20].im;
        }
      }
      coder::wsst(b_x0, sampleRate, wsstMatrixQ, a__8);
      //  13.2 构造 I/Q 联合 WSST 功率图
      coder::b_abs(wsstMatrixI, r1);
      coder::b_abs(wsstMatrixQ, r2);
      if (r1.size(0) == r2.size(0)) {
        wsstPowerMatrix.set_size(r1.size(0), r1.size(1));
        upperRidgeIndex = r1.size(0) * r1.size(1);
        if (static_cast<int>(upperRidgeIndex < 800)) {
          for (int i21{0}; i21 < upperRidgeIndex; i21++) {
            double o_varargin_1;
            double p_varargin_1;
            o_varargin_1 = r1[i21];
            p_varargin_1 = r2[i21];
            wsstPowerMatrix[i21] =
                static_cast<float>(rt_powd_snf(o_varargin_1, 2.0) +
                                   rt_powd_snf(p_varargin_1, 2.0));
          }
        } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        q_varargin_1, r_varargin_1)

          for (int i21 = 0; i21 < upperRidgeIndex; i21++) {
            q_varargin_1 = r1[i21];
            r_varargin_1 = r2[i21];
            wsstPowerMatrix[i21] =
                static_cast<float>(rt_powd_snf(q_varargin_1, 2.0) +
                                   rt_powd_snf(r_varargin_1, 2.0));
          }
        }
      } else {
        binary_expand_op_4(wsstPowerMatrix, r1, r2);
      }
      upperRidgeIndex = wsstFrequencyAxis.size(0);
      spectralSkewnessSequence.set_size(wsstFrequencyAxis.size(0));
      for (int k{0}; k < upperRidgeIndex; k++) {
        spectralSkewnessSequence[k] = static_cast<float>(wsstFrequencyAxis[k]);
      }
      upperRidgeIndex = wsstPowerMatrix.size(0) * wsstPowerMatrix.size(1);
      d_fsstPowerMatrix = wsstPowerMatrix.reshape(upperRidgeIndex);
      localSecondMoment =
          coder::blockedSummation(d_fsstPowerMatrix, upperRidgeIndex);
      //  13.3 分配 WSST 局部特征序列
      d_loop_ub = wsstPowerMatrix.size(1);
      fsstRidgeFrequencySequence.set_size(wsstPowerMatrix.size(1));
      centeredFSSTTime.set_size(wsstPowerMatrix.size(1));
      if (d_loop_ub - 1 >= 0) {
        std::memset(&fsstRidgeFrequencySequence[0], 0,
                    static_cast<unsigned int>(d_loop_ub) * sizeof(float));
        std::memset(&centeredFSSTTime[0], 0,
                    static_cast<unsigned int>(d_loop_ub) * sizeof(float));
      }
      wsstRidgeEnergy = 0.0F;
      //  13.4 每个时间位置提取脊线和局部带宽
      if (localSecondMoment > 1.0E-20F) {
        for (int frameIndex{0}; frameIndex < d_loop_ub; frameIndex++) {
          upperRidgeIndex = wsstPowerMatrix.size(0);
          spectralKurtosisSequence.set_size(wsstPowerMatrix.size(0));
          for (int k{0}; k < upperRidgeIndex; k++) {
            spectralKurtosisSequence[k] =
                wsstPowerMatrix[k + wsstPowerMatrix.size(0) * frameIndex];
          }
          localTotalPower = coder::blockedSummation(spectralKurtosisSequence,
                                                    wsstPowerMatrix.size(0));
          if (!(localTotalPower <= 1.0E-20F)) {
            //         %% WSST 主脊频率
            coder::internal::maximum(spectralKurtosisSequence, upperRidgeIndex);
            fsstRidgeFrequencySequence[frameIndex] =
                spectralSkewnessSequence[upperRidgeIndex - 1];
            //         %% WSST 主脊线邻域能量
            lowerRidgeIndex = static_cast<int>(
                std::fmax(static_cast<double>(upperRidgeIndex) - 1.0, 1.0));
            upperRidgeIndex = static_cast<int>(
                std::fmin(static_cast<double>(upperRidgeIndex) + 1.0,
                          static_cast<double>(wsstPowerMatrix.size(0))));
            if (lowerRidgeIndex > upperRidgeIndex) {
              scalarLB = 0;
              upperRidgeIndex = 0;
            } else {
              scalarLB = lowerRidgeIndex - 1;
            }
            upperRidgeIndex -= scalarLB;
            spectralCrestSequence.set_size(upperRidgeIndex);
            for (int k{0}; k < upperRidgeIndex; k++) {
              spectralCrestSequence[k] =
                  wsstPowerMatrix[(scalarLB + k) +
                                  wsstPowerMatrix.size(0) * frameIndex];
            }
            wsstRidgeEnergy +=
                coder::blockedSummation(spectralCrestSequence, upperRidgeIndex);
            //         %% WSST 局部频谱质心
            if (spectralSkewnessSequence.size(0) == wsstPowerMatrix.size(0)) {
              upperRidgeIndex = spectralSkewnessSequence.size(0);
              spectralKurtosisSequence.set_size(
                  spectralSkewnessSequence.size(0));
              scalarLB = (spectralSkewnessSequence.size(0) / 4) << 2;
              lowerRidgeIndex = scalarLB - 4;
              for (int k{0}; k <= lowerRidgeIndex; k += 4) {
                r3 = _mm_loadu_ps(&spectralSkewnessSequence[k]);
                r4 = _mm_loadu_ps(
                    &wsstPowerMatrix[k + wsstPowerMatrix.size(0) * frameIndex]);
                _mm_storeu_ps(&spectralKurtosisSequence[k], _mm_mul_ps(r3, r4));
              }
              for (int k{scalarLB}; k < upperRidgeIndex; k++) {
                spectralKurtosisSequence[k] =
                    spectralSkewnessSequence[k] *
                    wsstPowerMatrix[k + wsstPowerMatrix.size(0) * frameIndex];
              }
            } else {
              binary_expand_op_2(spectralKurtosisSequence,
                                 spectralSkewnessSequence, wsstPowerMatrix,
                                 frameIndex);
            }
            localCentroid =
                coder::blockedSummation(spectralKurtosisSequence,
                                        spectralKurtosisSequence.size(0)) /
                localTotalPower;
            //         %% WSST 局部频率扩展 / 带宽
            if (spectralSkewnessSequence.size(0) == wsstPowerMatrix.size(0)) {
              upperRidgeIndex = spectralSkewnessSequence.size(0);
              spectralKurtosisSequence.set_size(
                  spectralSkewnessSequence.size(0));
              for (int k{0}; k < upperRidgeIndex; k++) {
                localSpread = spectralSkewnessSequence[k] - localCentroid;
                spectralKurtosisSequence[k] =
                    localSpread * localSpread *
                    wsstPowerMatrix[k + wsstPowerMatrix.size(0) * frameIndex];
              }
            } else {
              binary_expand_op_1(spectralKurtosisSequence,
                                 spectralSkewnessSequence, localCentroid,
                                 wsstPowerMatrix, frameIndex);
            }
            centeredFSSTTime[frameIndex] =
                std::sqrt(std::fmax(
                    coder::blockedSummation(spectralKurtosisSequence,
                                            spectralKurtosisSequence.size(0)) /
                        localTotalPower,
                    0.0F)) /
                static_cast<float>(sampleRate);
          }
        }
      }
      //  13.5 WSST 归一化脊频率标准差
      spectralSkewnessSequence.set_size(wsstPowerMatrix.size(1));
      upperRidgeIndex = (fsstRidgeFrequencySequence.size(0) / 4) << 2;
      lowerRidgeIndex = upperRidgeIndex - 4;
      for (int k{0}; k <= lowerRidgeIndex; k += 4) {
        r3 = _mm_loadu_ps(&fsstRidgeFrequencySequence[k]);
        _mm_storeu_ps(
            &spectralSkewnessSequence[k],
            _mm_div_ps(r3, _mm_set1_ps(static_cast<float>(sampleRate))));
      }
      for (int k{upperRidgeIndex}; k < d_loop_ub; k++) {
        spectralSkewnessSequence[k] =
            fsstRidgeFrequencySequence[k] / static_cast<float>(sampleRate);
      }
      //  =========================================================================
      //  局部函数：均值和总体标准差
      //  =========================================================================
      if (spectralSkewnessSequence.size(0) == 0) {
        features[20] = 0.0F;
      } else {
        localSpread =
            coder::blockedSummation(spectralSkewnessSequence,
                                    spectralSkewnessSequence.size(0)) /
            static_cast<float>(spectralSkewnessSequence.size(0));
        upperRidgeIndex = spectralSkewnessSequence.size(0);
        spectralKurtosisSequence.set_size(spectralSkewnessSequence.size(0));
        if (static_cast<int>(spectralSkewnessSequence.size(0) < 800)) {
          for (int i22{0}; i22 < upperRidgeIndex; i22++) {
            localCentroid = spectralSkewnessSequence[i22] - localSpread;
            spectralKurtosisSequence[i22] = localCentroid * localCentroid;
          }
        } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        s_varargin_1)

          for (int i22 = 0; i22 < upperRidgeIndex; i22++) {
            s_varargin_1 = spectralSkewnessSequence[i22] - localSpread;
            spectralKurtosisSequence[i22] = s_varargin_1 * s_varargin_1;
          }
        }
        features[20] = std::sqrt(std::fmax(
            coder::blockedSummation(spectralKurtosisSequence,
                                    spectralKurtosisSequence.size(0)) /
                static_cast<float>(spectralKurtosisSequence.size(0)),
            0.0F));
      }
      //  13.6 WSST 归一化脊频率斜率
      if (wsstPowerMatrix.size(1) >= 2) {
        localCentroid = static_cast<float>(wsstPowerMatrix.size(1)) - 1.0F;
        spectralKurtosisSequence.set_size(wsstPowerMatrix.size(1));
        upperRidgeIndex = wsstPowerMatrix.size(1) - 1;
        lowerRidgeIndex = (wsstPowerMatrix.size(1) / 4) << 2;
        scalarLB = lowerRidgeIndex - 4;
        for (int k{0}; k <= scalarLB; k += 4) {
          fv[0] = static_cast<float>(k);
          fv[1] = static_cast<float>(k + 1);
          fv[2] = static_cast<float>(k + 2);
          fv[3] = static_cast<float>(k + 3);
          r3 = _mm_loadu_ps(&fv[0]);
          _mm_storeu_ps(&spectralKurtosisSequence[k],
                        _mm_div_ps(r3, _mm_set1_ps(localCentroid)));
        }
        for (int k{lowerRidgeIndex}; k <= upperRidgeIndex; k++) {
          spectralKurtosisSequence[k] = static_cast<float>(k) / localCentroid;
        }
        localCentroid =
            coder::blockedSummation(spectralKurtosisSequence,
                                    spectralKurtosisSequence.size(0)) /
            static_cast<float>(spectralKurtosisSequence.size(0));
        localSpread =
            coder::blockedSummation(spectralSkewnessSequence,
                                    spectralSkewnessSequence.size(0)) /
            static_cast<float>(spectralSkewnessSequence.size(0));
        upperRidgeIndex = spectralKurtosisSequence.size(0);
        fsstRidgeFrequencySequence.set_size(spectralKurtosisSequence.size(0));
        lowerRidgeIndex = (spectralKurtosisSequence.size(0) / 4) << 2;
        scalarLB = lowerRidgeIndex - 4;
        for (int k{0}; k <= scalarLB; k += 4) {
          r3 = _mm_loadu_ps(&spectralKurtosisSequence[k]);
          _mm_storeu_ps(&fsstRidgeFrequencySequence[k],
                        _mm_sub_ps(r3, _mm_set1_ps(localCentroid)));
        }
        for (int k{lowerRidgeIndex}; k < upperRidgeIndex; k++) {
          fsstRidgeFrequencySequence[k] =
              spectralKurtosisSequence[k] - localCentroid;
        }
        scalarLB = fsstRidgeFrequencySequence.size(0);
        spectralKurtosisSequence.set_size(fsstRidgeFrequencySequence.size(0));
        upperRidgeIndex = fsstRidgeFrequencySequence.size(0);
        if (static_cast<int>(fsstRidgeFrequencySequence.size(0) < 800)) {
          for (int i24{0}; i24 < scalarLB; i24++) {
            localCentroid = fsstRidgeFrequencySequence[i24];
            spectralKurtosisSequence[i24] = localCentroid * localCentroid;
          }
        } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        u_varargin_1)

          for (int i24 = 0; i24 < upperRidgeIndex; i24++) {
            u_varargin_1 = fsstRidgeFrequencySequence[i24];
            spectralKurtosisSequence[i24] = u_varargin_1 * u_varargin_1;
          }
        }
        localCentroid = coder::blockedSummation(
            spectralKurtosisSequence, spectralKurtosisSequence.size(0));
        if (localCentroid > 1.0E-20F) {
          if (fsstRidgeFrequencySequence.size(0) ==
              spectralSkewnessSequence.size(0)) {
            spectralSkewnessSequence.set_size(scalarLB);
            upperRidgeIndex = (fsstRidgeFrequencySequence.size(0) / 4) << 2;
            lowerRidgeIndex = upperRidgeIndex - 4;
            for (int k{0}; k <= lowerRidgeIndex; k += 4) {
              r3 = _mm_loadu_ps(&spectralSkewnessSequence[k]);
              r4 = _mm_loadu_ps(&fsstRidgeFrequencySequence[k]);
              _mm_storeu_ps(
                  &spectralSkewnessSequence[k],
                  _mm_mul_ps(r4, _mm_sub_ps(r3, _mm_set1_ps(localSpread))));
            }
            for (int k{upperRidgeIndex}; k < scalarLB; k++) {
              spectralSkewnessSequence[k] =
                  fsstRidgeFrequencySequence[k] *
                  (spectralSkewnessSequence[k] - localSpread);
            }
          } else {
            binary_expand_op_3(spectralSkewnessSequence,
                               fsstRidgeFrequencySequence, localSpread);
          }
          features[21] =
              coder::blockedSummation(spectralSkewnessSequence,
                                      spectralSkewnessSequence.size(0)) /
              localCentroid;
        }
      }
      //  13.7 WSST 归一化局部带宽均值和标准差
      //  =========================================================================
      //  局部函数：均值和总体标准差
      //  =========================================================================
      if (centeredFSSTTime.size(0) == 0) {
        localSpread = 0.0F;
        features[23] = 0.0F;
      } else {
        localSpread = coder::blockedSummation(centeredFSSTTime,
                                              centeredFSSTTime.size(0)) /
                      static_cast<float>(centeredFSSTTime.size(0));
        spectralKurtosisSequence.set_size(d_loop_ub);
        if (static_cast<int>(centeredFSSTTime.size(0) < 800)) {
          for (int i23{0}; i23 < d_loop_ub; i23++) {
            localCentroid = centeredFSSTTime[i23] - localSpread;
            spectralKurtosisSequence[i23] = localCentroid * localCentroid;
          }
        } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        t_varargin_1)

          for (int i23 = 0; i23 < d_loop_ub; i23++) {
            t_varargin_1 = centeredFSSTTime[i23] - localSpread;
            spectralKurtosisSequence[i23] = t_varargin_1 * t_varargin_1;
          }
        }
        features[23] = std::sqrt(std::fmax(
            coder::blockedSummation(spectralKurtosisSequence,
                                    spectralKurtosisSequence.size(0)) /
                static_cast<float>(spectralKurtosisSequence.size(0)),
            0.0F));
      }
      features[22] = localSpread;
      //  13.8 WSST 主脊线能量占比
      if (localSecondMoment > 1.0E-20F) {
        features[24] = wsstRidgeEnergy / (localSecondMoment + 1.0E-20F);
      }
      //  14. 组装25维度特征
      features[11] = 0.0F;
      //  FSST
      //  WSST
      //  =========================================================================
      //  局部函数：将 NaN 和 Inf 替换为零
      //  =========================================================================
      for (int k{0}; k < 25; k++) {
        localCentroid = features[k];
        if (std::isnan(localCentroid) || std::isinf(localCentroid)) {
          features[k] = 0.0F;
        }
      }
    }
  }
}

// End of code generation (extractTimeFrequencyFeatures.cpp)
