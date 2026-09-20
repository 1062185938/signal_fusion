//
// extractFrequencyFeatures.cpp
//
// Code generation for function 'extractFrequencyFeatures'
//

// Include files
#include "extractFrequencyFeatures.h"
#include "abs.h"
#include "blockedSummation.h"
#include "combineVectorElements.h"
#include "extractAllFeatures_data.h"
#include "extractAllFeatures_rtwutil.h"
#include "fft.h"
#include "hamming.h"
#include "log2.h"
#include "minOrMax.h"
#include "pwelch.h"
#include "rt_nonfinite.h"
#include "std.h"
#include "coder_array.h"
#include "omp.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <xmmintrin.h>

// Function Definitions
void extractFrequencyFeatures(const coder::array<creal32_T, 1U> &x,
                              double sampleRate, float features[25])
{
  static creal32_T bispectrumMatrix[65536];
  static creal32_T segmentSpectra[32512];
  static const float bispectrumWindow[256]{
      0.08F,         0.0801385418F, 0.0805540904F, 0.0812463909F, 0.082215026F,
      0.0834594145F, 0.084978804F,  0.0867722854F, 0.088838771F,  0.0911770165F,
      0.0937856212F, 0.0966630131F, 0.0998074487F, 0.103217036F,  0.106889732F,
      0.110823311F,  0.115015417F,  0.119463511F,  0.124164924F,  0.129116818F,
      0.134316221F,  0.139759988F,  0.14544484F,   0.151367366F,  0.157523975F,
      0.163910985F,  0.170524538F,  0.177360639F,  0.184415191F,  0.191683933F,
      0.199162483F,  0.206846341F,  0.214730874F,  0.222811356F,  0.231082886F,
      0.239540488F,  0.248179093F,  0.256993473F,  0.265978307F,  0.275128245F,
      0.284437686F,  0.293901086F,  0.303512752F,  0.313266844F,  0.323157489F,
      0.333178788F,  0.343324661F,  0.353589F,     0.363965631F,  0.37444827F,
      0.385030657F,  0.395706385F,  0.406469047F,  0.417312145F,  0.428229123F,
      0.439213425F,  0.450258464F,  0.461357534F,  0.472504F,     0.483691096F,
      0.494912118F,  0.506160319F,  0.517428875F,  0.528711F,     0.54F,
      0.551288962F,  0.562571108F,  0.573839724F,  0.585087895F,  0.596308887F,
      0.607496F,     0.618642449F,  0.629741549F,  0.640786588F,  0.65177089F,
      0.662687898F,  0.673530936F,  0.684293628F,  0.694969356F,  0.705551744F,
      0.716034353F,  0.726411F,     0.736675322F,  0.746821225F,  0.756842494F,
      0.76673317F,   0.776487291F,  0.786098897F,  0.795562327F,  0.804871798F,
      0.814021707F,  0.823006511F,  0.831820905F,  0.840459526F,  0.848917127F,
      0.857188642F,  0.865269125F,  0.873153687F,  0.8808375F,    0.888316095F,
      0.895584822F,  0.902639329F,  0.909475446F,  0.916089F,     0.922476F,
      0.928632617F,  0.934555173F,  0.94024F,      0.945683777F,  0.95088315F,
      0.955835104F,  0.96053648F,   0.964984596F,  0.96917671F,   0.973110259F,
      0.976783F,     0.980192542F,  0.983337F,     0.986214399F,  0.988823F,
      0.991161227F,  0.99322772F,   0.995021224F,  0.996540606F,  0.997785F,
      0.998753607F,  0.999445915F,  0.999861479F,  1.0F,          0.999861479F,
      0.999445915F,  0.998753607F,  0.997785F,     0.996540606F,  0.995021224F,
      0.99322772F,   0.991161227F,  0.988823F,     0.986214399F,  0.983337F,
      0.980192542F,  0.976783F,     0.973110259F,  0.96917671F,   0.964984596F,
      0.96053648F,   0.955835104F,  0.95088315F,   0.945683777F,  0.94024F,
      0.934555173F,  0.928632617F,  0.922476F,     0.916089F,     0.909475446F,
      0.902639329F,  0.895584822F,  0.888316095F,  0.8808375F,    0.873153687F,
      0.865269125F,  0.857188642F,  0.848917127F,  0.840459526F,  0.831820905F,
      0.823006511F,  0.814021707F,  0.804871798F,  0.795562327F,  0.786098897F,
      0.776487291F,  0.76673317F,   0.756842494F,  0.746821225F,  0.736675322F,
      0.726411F,     0.716034353F,  0.705551744F,  0.694969356F,  0.684293628F,
      0.673530936F,  0.662687898F,  0.65177089F,   0.640786588F,  0.629741549F,
      0.618642449F,  0.607496F,     0.596308887F,  0.585087895F,  0.573839724F,
      0.562571108F,  0.551288962F,  0.54F,         0.528711F,     0.517428875F,
      0.506160319F,  0.494912118F,  0.483691096F,  0.472504F,     0.461357534F,
      0.450258464F,  0.439213425F,  0.428229123F,  0.417312145F,  0.406469047F,
      0.395706385F,  0.385030657F,  0.37444827F,   0.363965631F,  0.353589F,
      0.343324661F,  0.333178788F,  0.323157489F,  0.313266844F,  0.303512752F,
      0.293901086F,  0.284437686F,  0.275128245F,  0.265978307F,  0.256993473F,
      0.248179093F,  0.239540488F,  0.231082886F,  0.222811356F,  0.214730874F,
      0.206846341F,  0.199162483F,  0.191683933F,  0.184415191F,  0.177360639F,
      0.170524538F,  0.163910985F,  0.157523975F,  0.151367366F,  0.14544484F,
      0.139759988F,  0.134316221F,  0.129116818F,  0.124164924F,  0.119463511F,
      0.115015417F,  0.110823311F,  0.106889732F,  0.103217036F,  0.0998074487F,
      0.0966630131F, 0.0937856212F, 0.0911770165F, 0.088838771F,  0.0867722854F,
      0.084978804F,  0.0834594145F, 0.082215026F,  0.0812463909F, 0.0805540904F,
      0.0801385418F};
  static float squaredBicoherenceMatrix[65536];
  coder::array<creal32_T, 1U> x0;
  coder::array<creal32_T, 1U> xIn;
  creal32_T pairProduct;
  double tmp_data[1024];
  float b_frequencyAxis[4096];
  float frequencyAxis[4096];
  float frequencyDeviation[4096];
  float normalizedCumulativePower[4096];
  float normalizedPowerSpectrum[4096];
  float powerSpectrum[4096];
  float window_data[1024];
  float f;
  float totalSpectralWeight;
  int tmp_size;
  int windowLength;
  //  EXTRACTFREQUENCYFEATURES
  //  从单个复数 IQ 段中提取 25 个频域统计特征。
  //
  //  Input:
  //    x           复数 IQ 向量。输入被视为一个完整的段。
  //    sampleRate  采样率
  //
  //  Output:
  //    features    1-by-25 单精度特征向量
  //
  //  Feature order:
  //     1  Normalized mean frequency       归一化平均频率
  //     2  Normalized median frequency     归一化中值频率
  //     3  Normalized spectral spread      归一化频谱扩展
  //     4  Normalized 99% occupied bandwidth       归一化 99% 占用带宽
  //     5  Normalized peak frequency       归一化峰值频率
  //     6  Log10 band power                对数带内功率
  //     7  Global normalized spectral entropy      全局归一化频谱熵
  //     8  Global spectral crest factor    全局频谱峰值因子
  //     9  Global spectral flatness        全局频谱平坦度
  //    10  Normalized spectral skewness    归一化频谱偏度
  //    11  PSD standard deviation          PSD 标准差
  //    12  Log mean bispectrum magnitude
  //                                       对数平均双谱幅值
  //    13  Log maximum bispectrum magnitude
  //                                       对数最大双谱幅值
  //    14  Log bispectral energy
  //                                       对数双谱能量
  //    15  Bispectral crest factor
  //                                       双谱峰值因子
  //    16  Normalized bispectral entropy
  //                                       归一化双谱熵
  //    17  Normalized bispectrum peak frequency 1
  //                                       双谱峰值频率 1
  //    18  Normalized bispectrum peak frequency 2
  //                                       双谱峰值频率 2
  //    19  Mean squared bicoherence
  //                                       平均平方双相干
  //    20  Maximum squared bicoherence
  //                                       最大平方双相干
  //    21  Standard deviation of squared bicoherence
  //                                       平方双相干标准差
  //    22  Normalized bicoherence entropy
  //                                       归一化双相干熵
  //    23  Bicoherence-weighted biphase concentration
  //                                       双相干加权双相位集中度
  //    24  Dominant biphase cosine
  //                                       主导双相位余弦
  //    25  Dominant biphase sine
  //                                       主导双相位正弦
  //  Frequency normalization:
  //    normalizedFrequency = frequencyHz / sampleRate
  //
  //  注意：
  //    - 进行PSD之前会去除直流分量
  //    - Welch 分段仅用于PSD估计
  //    - 输入最大长度限制为 16384 点
  //    - Welch 最大窗长为 1024 点
  //    - FFT 点数固定为 4096
  //    - PSD standard deviation 保留原始 PSD 数值尺度
  //  1. 输入验证
  windowLength = x.size(0);
  xIn.set_size(x.size(0));
  tmp_size = (x.size(0) < 800);
  if (tmp_size) {
    for (int i{0}; i < windowLength; i++) {
      xIn[i] = x[i];
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

    for (int i = 0; i < windowLength; i++) {
      xIn[i] = x[i];
    }
  }
  pairProduct = coder::blockedSummation(xIn, xIn.size(0));
  if (pairProduct.im == 0.0F) {
    pairProduct.re /= static_cast<float>(xIn.size(0));
    pairProduct.im = 0.0F;
  } else if (pairProduct.re == 0.0F) {
    pairProduct.re = 0.0F;
    pairProduct.im /= static_cast<float>(xIn.size(0));
  } else {
    pairProduct.re /= static_cast<float>(xIn.size(0));
    pairProduct.im /= static_cast<float>(xIn.size(0));
  }
  x0.set_size(windowLength);
  if (tmp_size) {
    for (int i1{0}; i1 < windowLength; i1++) {
      x0[i1].re = xIn[i1].re - pairProduct.re;
      x0[i1].im = xIn[i1].im - pairProduct.im;
    }
  } else {
#pragma omp parallel for num_threads(omp_get_max_threads())

    for (int i1 = 0; i1 < windowLength; i1++) {
      x0[i1].re = xIn[i1].re - pairProduct.re;
      x0[i1].im = xIn[i1].im - pairProduct.im;
    }
  }
  //  3.  Welch PSD 配置
  //
  //  使用最大长度为 1024 点的汉明窗。
  //  输入长度更短时则使用整个输入长度作为窗长。
  //
  //  FFT 点数固定为 4096
  //  输入长度增加时，Welch 可使用更多数据段进行平均
  windowLength =
      static_cast<int>(std::fmin(static_cast<double>(xIn.size(0)), 1024.0));
  tmp_size = coder::hamming(static_cast<double>(windowLength), tmp_data);
  for (int b_i{0}; b_i < tmp_size; b_i++) {
    window_data[b_i] = static_cast<float>(tmp_data[b_i]);
  }
  //  4. 计算中心化双边 Welch PSD
  coder::pwelch(x0, window_data, tmp_size,
                std::floor(static_cast<double>(windowLength) / 2.0), sampleRate,
                normalizedPowerSpectrum, frequencyAxis);
  // 避免由于数值误差产生极小负值
#pragma omp parallel for num_threads(omp_get_max_threads())

  for (int k = 0; k < 4096; k++) {
    powerSpectrum[k] = std::fmax(normalizedPowerSpectrum[k], 0.0F);
  }
  totalSpectralWeight = coder::b_combineVectorElements(powerSpectrum);
  //  5. 处理零信号或极弱信号
  std::memset(&features[0], 0, 25U * sizeof(float));
  if (totalSpectralWeight <= 1.0E-20F) {
    features[5] = -20.0F;
  } else {
    __m128 r;
    __m128 r1;
    float fMeanSquaredBicoherence;
    float logSpectrumSum;
    float meanFrequency;
    float spectralEntropy;
    float spectralSpread;
    int iindx;
    int lowerIndex;
    int medianIndex;
    int upperIndex;
    boolean_T exitg1;
    //  6. 归一化 PSD
    //
    //  将 PSD 转换为总和为 1 的功率权重。
    //
    //  normalizedPowerSpectrum 用于计算：
    //    - 频谱熵
    //    - 频谱偏度
    //
    //  这样这些统计量不会受到信号整体功率尺度的直接影响。
    //  6. 平均频率 / 频谱质心
    //  7. 中值频率
    for (int b_i{0}; b_i <= 4092; b_i += 4) {
      r = _mm_loadu_ps(&powerSpectrum[b_i]);
      _mm_storeu_ps(&normalizedPowerSpectrum[b_i],
                    _mm_div_ps(r, _mm_set1_ps(totalSpectralWeight)));
      r1 = _mm_loadu_ps(&frequencyAxis[b_i]);
      _mm_storeu_ps(&b_frequencyAxis[b_i], _mm_mul_ps(r1, r));
      _mm_storeu_ps(&normalizedCumulativePower[b_i], r);
    }
    meanFrequency =
        coder::b_combineVectorElements(b_frequencyAxis) / totalSpectralWeight;
    for (int b_i{0}; b_i < 4095; b_i++) {
      normalizedCumulativePower[b_i + 1] += normalizedCumulativePower[b_i];
    }
    for (int b_i{0}; b_i <= 4092; b_i += 4) {
      r = _mm_loadu_ps(&normalizedCumulativePower[b_i]);
      _mm_storeu_ps(&normalizedCumulativePower[b_i],
                    _mm_div_ps(r, _mm_set1_ps(totalSpectralWeight)));
    }
    medianIndex = 0;
    windowLength = 0;
    exitg1 = false;
    while ((!exitg1) && (windowLength < 4096)) {
      if (normalizedCumulativePower[windowLength] >= 0.5F) {
        medianIndex = windowLength;
        exitg1 = true;
      } else {
        windowLength++;
      }
    }
    //  8. 频谱扩展
    //
    //  以平均频率为中心的功率加权标准差。
    for (int b_i{0}; b_i <= 4092; b_i += 4) {
      r = _mm_loadu_ps(&frequencyAxis[b_i]);
      r = _mm_sub_ps(r, _mm_set1_ps(meanFrequency));
      _mm_storeu_ps(&frequencyDeviation[b_i], r);
      r1 = _mm_loadu_ps(&powerSpectrum[b_i]);
      _mm_storeu_ps(&b_frequencyAxis[b_i], _mm_mul_ps(_mm_mul_ps(r, r), r1));
    }
    spectralSpread = std::sqrt(std::fmax(
        coder::b_combineVectorElements(b_frequencyAxis) / totalSpectralWeight,
        0.0F));
    //  9. 99% 占用带宽
    //
    //  从累计功率 0.5% 到 99.5% 之间的频率范围，
    //  对应总功率的 99%。
    lowerIndex = 0;
    upperIndex = 4095;
    windowLength = 0;
    exitg1 = false;
    while ((!exitg1) && (windowLength < 4096)) {
      if (normalizedCumulativePower[windowLength] >= 0.005F) {
        lowerIndex = windowLength;
        exitg1 = true;
      } else {
        windowLength++;
      }
    }
    windowLength = 0;
    exitg1 = false;
    while ((!exitg1) && (windowLength < 4096)) {
      if (normalizedCumulativePower[windowLength] >= 0.995F) {
        upperIndex = windowLength;
        exitg1 = true;
      } else {
        windowLength++;
      }
    }
    //  10. 峰值频率
    coder::internal::maximum(powerSpectrum, iindx);
    //  11. 带内功率
    //
    //  对频率轴上的 PSD 进行积分
    features[5] = std::log10(std::fmax(
        totalSpectralWeight * std::abs(frequencyAxis[1] - frequencyAxis[0]),
        1.0E-20F));
    //  13. 全局归一化频谱熵
    //
    //  Spectral entropy:
    //
    //    H = -sum(p(k) * log2(p(k))) / log2(K)
    //
    //  p(k) 为归一化 PSD。
    //
    //  输出范围大致位于 [0, 1]：
    //
    //    较小：频谱能量集中
    //    较大：频谱能量分散
    spectralEntropy = 0.0F;
    for (int b_i{0}; b_i < 4096; b_i++) {
      logSpectrumSum = normalizedPowerSpectrum[b_i];
      if (logSpectrumSum > 0.0F) {
        spectralEntropy -= logSpectrumSum * coder::b_log2(logSpectrumSum);
      }
    }
    features[6] = spectralEntropy / 12.0F;
    //  14. 全局频谱峰值因子
    //
    //  Spectral crest factor:
    //
    //    max(PSD) / mean(PSD)
    //
    //  描述最大谱峰相对于平均谱水平的突出程度。
    totalSpectralWeight /= 4096.0F;
    if (totalSpectralWeight <= 1.0E-20F) {
      features[7] = 0.0F;
    } else {
      features[7] = coder::internal::maximum(powerSpectrum) /
                    (totalSpectralWeight + 1.0E-20F);
    }
    //  15. 全局频谱平坦度
    //
    //  Spectral flatness:
    //
    //    geometric mean(PSD) / arithmetic mean(PSD)
    //
    //  输出范围通常位于 [0, 1]：
    //
    //    接近 0：频谱存在明显峰值
    //    接近 1：频谱较为平坦
    //
    //  使用与平均 PSD 成比例的最小值，避免 log(0)。
    spectralEntropy = std::fmax(totalSpectralWeight * 1.0E-12F, 1.0E-20F);
    logSpectrumSum = 0.0F;
    for (int b_i{0}; b_i < 4096; b_i++) {
      logSpectrumSum +=
          std::log(std::fmax(powerSpectrum[b_i], spectralEntropy));
    }
    if (totalSpectralWeight <= 1.0E-20F) {
      features[8] = 0.0F;
    } else {
      features[8] =
          std::exp(logSpectrumSum / 4096.0F) / (totalSpectralWeight + 1.0E-20F);
    }
    //  16. 归一化频谱偏度
    //
    //  Spectral skewness:
    //
    //    sum(p(k) * (f(k)-meanFrequency)^3)
    //    ----------------------------------
    //               spectralSpread^3
    //
    //  该统计量本身是无量纲的，因此不需要再除以 sampleRate。
    //
    //    接近 0：频谱相对对称
    //    > 0：向正频率侧偏
    //    < 0：向负频率侧偏
    if (spectralSpread <= 1.0E-20F) {
      features[9] = 0.0F;
    } else {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(f)

      for (int b_k = 0; b_k < 4096; b_k++) {
        f = rt_powf_snf(frequencyDeviation[b_k], 3.0F) *
            normalizedPowerSpectrum[b_k];
        b_frequencyAxis[b_k] = f;
      }
      features[9] = coder::b_combineVectorElements(b_frequencyAxis) /
                    (rt_powf_snf(spectralSpread, 3.0F) + 1.0E-20F);
    }
    //  17. PSD 标准差
    //
    //  直接对 Welch PSD 的频率点数值计算标准差。
    //
    //  该特征描述 PSD 在不同频率位置上的整体起伏程度。
    //
    //  与归一化频谱统计不同，本特征保留 PSD 的原始功率尺度，
    //  因此会受到信号整体功率和接收增益的影响。
    features[10] = coder::b_std(powerSpectrum);
    //  18. 双谱特征初始化
    //
    //  双谱用于描述普通功率谱无法反映的三阶频率耦合关系。
    //
    //  Bispectrum:
    //
    //    B(f1,f2) =
    //        E[X(f1) X(f2) conj(X(f1+f2))]
    //
    //  双谱分析与前面的 Welch PSD 独立进行。
    //
    //  为控制二维双谱的计算量：
    //
    //    window length = 256
    //    overlap       = 50%
    //    FFT length    = 256
    //
    //  当输入长度小于 512 点时，不计算双谱特征，
    //  对应特征全部保持为 0。
    features[11] = 0.0F;
    features[12] = 0.0F;
    features[13] = 0.0F;
    features[14] = 0.0F;
    features[15] = 0.0F;
    features[16] = 0.0F;
    features[17] = 0.0F;
    fMeanSquaredBicoherence = 0.0F;
    features[19] = 0.0F;
    features[20] = 0.0F;
    features[21] = 0.0F;
    features[22] = 0.0F;
    features[23] = 0.0F;
    features[24] = 0.0F;
    //  19. 双谱配置
    //  最大输入长度 16384 点时：
    //
    //  floor((16384 - 256) / 128) + 1 = 127
    //
    //  因此固定预分配最多 127 个 FFT 子段。
    //  20. 双谱计算
    if (xIn.size(0) >= 512) {
      creal32_T dominantBiphaseBispectrumValue;
      double validBispectrumPointCount;
      float bispectrumEnergy;
      float bispectrumEntropy;
      float bispectrumMagnitude;
      float bispectrumMagnitudeSum;
      float maximumBispectrumMagnitude;
      float maximumSquaredBicoherence;
      float pairPowerSum;
      float squaredBicoherenceSquaredSum;
      float squaredBicoherenceSum;
      int b_index2;
      int frequencyBin3;
      int peakBispectrumIndex1;
      int peakBispectrumIndex2;
      int y;
      y = static_cast<int>(
          std::floor((static_cast<double>(xIn.size(0)) - 256.0) / 128.0));
      //  保存所有子段的中心化 FFT。
      //
      //  频率索引对应：
      //
      //    -Fs/2 ... 0 ... Fs/2
      //
      //  固定尺寸预分配有利于后续 MATLAB Coder。
      std::memset(&segmentSpectra[0], 0, 32512U * sizeof(creal32_T));
      //     %% 20.1 每个子段计算 FFT
      for (int segmentIndex{0}; segmentIndex <= y; segmentIndex++) {
        creal32_T b_x0[256];
        creal32_T segmentSpectrum[256];
        windowLength = segmentIndex << 7;
        //  再次去除当前子段的局部直流分量，
        //  减少零频附近对双谱的影响。
        xIn.set_size(256);
        std::copy(&x0[windowLength],
                  &x0[static_cast<int>(static_cast<unsigned int>(windowLength) +
                                       256U)],
                  &xIn[0]);
        pairProduct = coder::blockedSummation(xIn, 256);
        if (pairProduct.im == 0.0F) {
          pairProduct.re /= 256.0F;
          pairProduct.im = 0.0F;
        } else if (pairProduct.re == 0.0F) {
          pairProduct.re = 0.0F;
          pairProduct.im /= 256.0F;
        } else {
          pairProduct.re /= 256.0F;
          pairProduct.im /= 256.0F;
        }
        for (int b_i{0}; b_i < 256; b_i++) {
          tmp_size = windowLength + b_i;
          totalSpectralWeight = bispectrumWindow[b_i];
          b_x0[b_i].re =
              totalSpectralWeight * (x0[tmp_size].re - pairProduct.re);
          b_x0[b_i].im =
              totalSpectralWeight * (x0[tmp_size].im - pairProduct.im);
        }
        coder::fft(b_x0, segmentSpectrum);
        //  手动进行 fftshift。
        //
        //  对固定 256 点 FFT：
        //
        //    [129:256, 1:128]
        //  对窗函数增益进行简单归一化，
        //  提高不同信号之间原始双谱幅值的可比性。
        for (int b_i{0}; b_i < 128; b_i++) {
          totalSpectralWeight = segmentSpectrum[b_i + 128].re;
          spectralEntropy = segmentSpectrum[b_i + 128].im;
          if (spectralEntropy == 0.0F) {
            windowLength = b_i + (segmentIndex << 8);
            segmentSpectra[windowLength].re = totalSpectralWeight / 138.239975F;
            segmentSpectra[windowLength].im = 0.0F;
          } else if (totalSpectralWeight == 0.0F) {
            windowLength = b_i + (segmentIndex << 8);
            segmentSpectra[windowLength].re = 0.0F;
            segmentSpectra[windowLength].im = spectralEntropy / 138.239975F;
          } else {
            windowLength = b_i + (segmentIndex << 8);
            segmentSpectra[windowLength].re = totalSpectralWeight / 138.239975F;
            segmentSpectra[windowLength].im = spectralEntropy / 138.239975F;
          }
          totalSpectralWeight = segmentSpectrum[b_i].re;
          spectralEntropy = segmentSpectrum[b_i].im;
          if (spectralEntropy == 0.0F) {
            windowLength = (b_i + (segmentIndex << 8)) + 128;
            segmentSpectra[windowLength].re = totalSpectralWeight / 138.239975F;
            segmentSpectra[windowLength].im = 0.0F;
          } else if (totalSpectralWeight == 0.0F) {
            windowLength = (b_i + (segmentIndex << 8)) + 128;
            segmentSpectra[windowLength].re = 0.0F;
            segmentSpectra[windowLength].im = spectralEntropy / 138.239975F;
          } else {
            windowLength = (b_i + (segmentIndex << 8)) + 128;
            segmentSpectra[windowLength].re = totalSpectralWeight / 138.239975F;
            segmentSpectra[windowLength].im = spectralEntropy / 138.239975F;
          }
        }
      }
      //     %% 20.2 双谱和平方双相干矩阵
      //
      //  对中心化 FFT：
      //
      //    k = -NFFT/2, ..., NFFT/2-1
      //
      //  仅计算满足：
      //
      //    k3 = k1 + k2
      //
      //  且 k3 位于有效 FFT 频率范围内的点。
      //
      //  同时由于：
      //
      //    B(f1,f2) = B(f2,f1)
      //
      //  只计算 index2 >= index1 的区域，
      //  避免完全重复的频率组合。
      std::memset(&bispectrumMatrix[0], 0, 65536U * sizeof(creal32_T));
      std::memset(&squaredBicoherenceMatrix[0], 0, 65536U * sizeof(float));
      bispectrumMagnitudeSum = 0.0F;
      bispectrumEnergy = 0.0F;
      maximumBispectrumMagnitude = 0.0F;
      squaredBicoherenceSum = 0.0F;
      squaredBicoherenceSquaredSum = 0.0F;
      maximumSquaredBicoherence = 0.0F;
      validBispectrumPointCount = 0.0;
      peakBispectrumIndex1 = 1;
      peakBispectrumIndex2 = 0;
      dominantBiphaseBispectrumValue.re = 0.0F;
      dominantBiphaseBispectrumValue.im = 0.0F;
      for (int segmentIndex{0}; segmentIndex < 256; segmentIndex++) {
        int i2;
        i2 = 256 - segmentIndex;
        for (int index2{0}; index2 < i2; index2++) {
          b_index2 = segmentIndex + index2;
          frequencyBin3 = (segmentIndex + b_index2) - 256;
          //  f1 + f2 必须仍位于 Nyquist 范围内。
          if ((frequencyBin3 >= -128) && (frequencyBin3 <= 127)) {
            creal32_T tripleProductSum;
            float sumFrequencyPower;
            tripleProductSum.re = 0.0F;
            tripleProductSum.im = 0.0F;
            pairPowerSum = 0.0F;
            sumFrequencyPower = 0.0F;
            //                 %% 对所有数据段平均
            for (int b_i{0}; b_i <= y; b_i++) {
              int pairProduct_tmp;
              windowLength = b_i << 8;
              tmp_size = segmentIndex + windowLength;
              pairProduct_tmp = b_index2 + windowLength;
              totalSpectralWeight = segmentSpectra[tmp_size].re;
              spectralEntropy = segmentSpectra[pairProduct_tmp].im;
              logSpectrumSum = segmentSpectra[tmp_size].im;
              bispectrumEntropy = segmentSpectra[pairProduct_tmp].re;
              pairProduct.re = totalSpectralWeight * bispectrumEntropy -
                               logSpectrumSum * spectralEntropy;
              pairProduct.im = totalSpectralWeight * spectralEntropy +
                               logSpectrumSum * bispectrumEntropy;
              windowLength = (frequencyBin3 + windowLength) + 128;
              totalSpectralWeight = segmentSpectra[windowLength].re;
              spectralEntropy = -segmentSpectra[windowLength].im;
              tripleProductSum.re += pairProduct.re * totalSpectralWeight -
                                     pairProduct.im * spectralEntropy;
              tripleProductSum.im += pairProduct.re * spectralEntropy +
                                     pairProduct.im * totalSpectralWeight;
              totalSpectralWeight = coder::b_abs(pairProduct);
              pairPowerSum += totalSpectralWeight * totalSpectralWeight;
              totalSpectralWeight = coder::b_abs(segmentSpectra[windowLength]);
              sumFrequencyPower += totalSpectralWeight * totalSpectralWeight;
            }
            //                 %% 双谱
            if (tripleProductSum.im == 0.0F) {
              pairProduct.re =
                  tripleProductSum.re / (static_cast<float>(y) + 1.0F);
              pairProduct.im = 0.0F;
            } else if (tripleProductSum.re == 0.0F) {
              pairProduct.re = 0.0F;
              pairProduct.im =
                  tripleProductSum.im / (static_cast<float>(y) + 1.0F);
            } else {
              pairProduct.re =
                  tripleProductSum.re / (static_cast<float>(y) + 1.0F);
              pairProduct.im =
                  tripleProductSum.im / (static_cast<float>(y) + 1.0F);
            }
            windowLength = segmentIndex + (b_index2 << 8);
            bispectrumMatrix[windowLength] = pairProduct;
            bispectrumMagnitude = coder::b_abs(pairProduct);
            //                 %% 平方双相干
            //
            //  b^2(f1,f2) =
            //
            //  |sum X1 X2 conj(X3)|^2
            //  ----------------------------------
            //  sum|X1 X2|^2 * sum|X3|^2
            //
            //  理论范围为 [0,1]。
            totalSpectralWeight = coder::b_abs(tripleProductSum);
            totalSpectralWeight = std::fmin(
                std::fmax(totalSpectralWeight * totalSpectralWeight /
                              (pairPowerSum * sumFrequencyPower + 1.0E-20F),
                          0.0F),
                1.0F);
            squaredBicoherenceMatrix[windowLength] = totalSpectralWeight;
            //                 %% 双谱基本统计累积
            validBispectrumPointCount++;
            bispectrumMagnitudeSum += bispectrumMagnitude;
            bispectrumEnergy += bispectrumMagnitude * bispectrumMagnitude;
            //                 %% 最大双谱幅值及其位置
            if (bispectrumMagnitude > maximumBispectrumMagnitude) {
              maximumBispectrumMagnitude = bispectrumMagnitude;
              peakBispectrumIndex1 = segmentIndex + 1;
              peakBispectrumIndex2 = b_index2;
            }
            //                 %% 双相干统计累积
            squaredBicoherenceSum += totalSpectralWeight;
            squaredBicoherenceSquaredSum +=
                totalSpectralWeight * totalSpectralWeight;
            //                 %% 最大双相干对应的双相位
            if (totalSpectralWeight > maximumSquaredBicoherence) {
              maximumSquaredBicoherence = totalSpectralWeight;
              dominantBiphaseBispectrumValue = pairProduct;
            }
          }
        }
      }
      //     %% 21. 原始双谱幅值统计
      if (validBispectrumPointCount > 0.0) {
        totalSpectralWeight = bispectrumMagnitudeSum /
                              static_cast<float>(validBispectrumPointCount);
        // 双谱平均幅度的对数，反映整体耦合强度的平均水平
        features[11] = std::log10(std::fmax(totalSpectralWeight, 1.0E-20F));
        // 最大耦合峰值的对数，反映是否存在极强的单次耦合
        features[12] =
            std::log10(std::fmax(maximumBispectrumMagnitude, 1.0E-20F));
        // 双谱总能量的对数，反映高阶耦合总强度
        features[13] = std::log10(std::fmax(bispectrumEnergy, 1.0E-20F));
        if (totalSpectralWeight > 1.0E-20F) {
          // 双谱峰因子，类似于PSD的谱峰因子，值越大说明耦合能量越集中于少数频率对
          features[14] =
              maximumBispectrumMagnitude / (totalSpectralWeight + 1.0E-20F);
        }
        // 平均平方双相干，反映整体耦合显著性（0~1）。越接近1，说明大部分频率对都存在强耦合
        fMeanSquaredBicoherence = squaredBicoherenceSum /
                                  static_cast<float>(validBispectrumPointCount);
        // 最大耦合显著性，用于判断是否存在极其确定的单一耦合
        features[19] = maximumSquaredBicoherence;
      }
      //     %% 22. 双谱峰值频率
      //
      //  centered FFT 中：
      //
      //    f = k * Fs / NFFT
      //
      //  再除以 Fs 后：
      //
      //    normalizedFrequency = k / NFFT
      //
      //  因此无需显式使用 sampleRate。
      //  对于fNormalizedBispectrumPeakFrequency1 和
      //  fNormalizedBispectrumPeakFrequency2
      //  给出了最强的二次相位耦合发生在哪两个频率上。
      features[16] =
          ((static_cast<float>(peakBispectrumIndex1) - 128.0F) - 1.0F) / 256.0F;
      features[17] =
          (((static_cast<float>(peakBispectrumIndex2) + 1.0F) - 128.0F) -
           1.0F) /
          256.0F;
      //     %% 23. 双相干均值和标准差
      if (validBispectrumPointCount > 1.0) {
        // 平方双相干的标准差（样本标准差），反映耦合强度分布的均匀性。值大说明有的频率对耦合很强，有的很弱
        features[20] = std::sqrt(std::fmax(
            (squaredBicoherenceSquaredSum -
             static_cast<float>(validBispectrumPointCount) *
                 (fMeanSquaredBicoherence * fMeanSquaredBicoherence)) /
                static_cast<float>(validBispectrumPointCount - 1.0),
            0.0F));
      }
      //     %% 24. 双谱熵、双相干熵和双相位集中度
      bispectrumEntropy = 0.0F;
      pairPowerSum = 0.0F;
      pairProduct.re = 0.0F;
      pairProduct.im = 0.0F;
      for (int b_i{0}; b_i < 256; b_i++) {
        tmp_size = 256 - b_i;
        for (int segmentIndex{0}; segmentIndex < tmp_size; segmentIndex++) {
          b_index2 = b_i + segmentIndex;
          frequencyBin3 = (b_i + b_index2) - 256;
          if ((frequencyBin3 >= -128) && (frequencyBin3 <= 127)) {
            windowLength = b_i + (b_index2 << 8);
            bispectrumMagnitude = coder::b_abs(bispectrumMatrix[windowLength]);
            //                 %% 归一化双谱熵
            //
            //  使用 |B|^2 构造概率分布。
            if (bispectrumEnergy > 1.0E-20F) {
              totalSpectralWeight = bispectrumMagnitude * bispectrumMagnitude /
                                    (bispectrumEnergy + 1.0E-20F);
              if (totalSpectralWeight > 0.0F) {
                bispectrumEntropy -=
                    totalSpectralWeight * coder::b_log2(totalSpectralWeight);
              }
            }
            //                 %% 归一化双相干熵
            if (squaredBicoherenceSum > 1.0E-20F) {
              totalSpectralWeight = squaredBicoherenceMatrix[windowLength] /
                                    (squaredBicoherenceSum + 1.0E-20F);
              if (totalSpectralWeight > 0.0F) {
                pairPowerSum -=
                    totalSpectralWeight * coder::b_log2(totalSpectralWeight);
              }
            }
            //                 %% 双相干加权双相位
            //
            //  不直接对 angle(B) 计算普通平均值，
            //  而使用复平面单位向量表示双相位。
            if ((bispectrumMagnitude > 1.0E-20F) &&
                (squaredBicoherenceMatrix[windowLength] > 0.0F)) {
              totalSpectralWeight = bispectrumMatrix[windowLength].re;
              spectralEntropy = bispectrumMatrix[windowLength].im;
              if (spectralEntropy == 0.0F) {
                logSpectrumSum = totalSpectralWeight / bispectrumMagnitude;
                totalSpectralWeight = 0.0F;
              } else if (totalSpectralWeight == 0.0F) {
                logSpectrumSum = 0.0F;
                totalSpectralWeight = spectralEntropy / bispectrumMagnitude;
              } else {
                logSpectrumSum = totalSpectralWeight / bispectrumMagnitude;
                totalSpectralWeight = spectralEntropy / bispectrumMagnitude;
              }
              spectralEntropy = squaredBicoherenceMatrix[windowLength];
              pairProduct.re += spectralEntropy * logSpectrumSum;
              pairProduct.im += spectralEntropy * totalSpectralWeight;
            }
          }
        }
      }
      //     %% 25. 熵归一化
      if (validBispectrumPointCount > 1.0) {
        totalSpectralWeight =
            coder::b_log2(static_cast<float>(validBispectrumPointCount));
        if (totalSpectralWeight > 0.0F) {
          features[15] = bispectrumEntropy / totalSpectralWeight;
          features[21] = pairPowerSum / totalSpectralWeight;
        }
      }
      //     %% 26. 双相位集中度
      if (squaredBicoherenceSum > 1.0E-20F) {
        features[22] =
            coder::b_abs(pairProduct) / (squaredBicoherenceSum + 1.0E-20F);
      }
      //     %% 27. 主导双相位
      //
      //  主导双相位定义为最大平方双相干位置
      //  对应的双谱相位。
      //
      //  不直接输出 angle(B)，而输出：
      //
      //    cos(phi)
      //    sin(phi)
      //
      //  避免 -pi 与 +pi 的角度跳变。
      totalSpectralWeight = coder::b_abs(dominantBiphaseBispectrumValue);
      if (totalSpectralWeight > 1.0E-20F) {
        if (dominantBiphaseBispectrumValue.im == 0.0F) {
          pairProduct.re =
              dominantBiphaseBispectrumValue.re / totalSpectralWeight;
          pairProduct.im = 0.0F;
        } else if (dominantBiphaseBispectrumValue.re == 0.0F) {
          pairProduct.re = 0.0F;
          pairProduct.im =
              dominantBiphaseBispectrumValue.im / totalSpectralWeight;
        } else {
          pairProduct.re =
              dominantBiphaseBispectrumValue.re / totalSpectralWeight;
          pairProduct.im =
              dominantBiphaseBispectrumValue.im / totalSpectralWeight;
        }
        features[23] = pairProduct.re;
        features[24] = pairProduct.im;
      }
    }
    //  28. 固定尺寸输出
    features[0] = meanFrequency / static_cast<float>(sampleRate);
    features[1] = frequencyAxis[medianIndex] / static_cast<float>(sampleRate);
    features[2] = spectralSpread / static_cast<float>(sampleRate);
    features[3] =
        std::fmax(frequencyAxis[upperIndex] - frequencyAxis[lowerIndex], 0.0F) /
        static_cast<float>(sampleRate);
    features[4] = frequencyAxis[iindx - 1] / static_cast<float>(sampleRate);
    //  双谱特征输出
    features[18] = fMeanSquaredBicoherence;
    //  19. 数值保护
    for (int b_i{0}; b_i < 25; b_i++) {
      totalSpectralWeight = features[b_i];
      if (std::isnan(totalSpectralWeight) || std::isinf(totalSpectralWeight)) {
        features[b_i] = 0.0F;
      }
    }
  }
}

// End of code generation (extractFrequencyFeatures.cpp)
