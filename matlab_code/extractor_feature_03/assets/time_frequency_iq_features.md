# 28维 IQ 时频特征说明

本文档对应 `extractTimeFrequencyFeatures.m`。输出由 18 个 STFT 局部谱形状/活动特征、5 个 FSST 特征和 5 个 WSST 特征组成，共 `1×28 single`。

这些特征用于描述时频结构，不是调制或协议判决。STFT/FSST 脊线是每个时间位置的最大功率 bin；对多载波信号，它可能在不同分量之间切换，不能直接解释为严格瞬时频率。

## 1. 输入与公共预处理

- 输入长度：`32 <= N <= 16384`。
- 输入采样率：正有限标量，单位 Hz。
- 去直流：`x0 = x - mean(x)`。
- 极弱信号返回全零有限向量。
- 本文件所有序列标准差均使用总体标准差，即除以 `N`，由 `localMeanAndStd` 显式实现。

## 2. 变换配置

### STFT

```text
windowLength  = clamp(floor(N/4), 32, 256)，并调整为偶数
window        = periodic Hamming
overlap       = 50%
nfft          = 512
frequencyAxis = centered two-sided
```

### FSST

FSST 直接处理复数 IQ，使用与 STFT 相同长度的 double Hamming 窗。频率轴保持复基带正负方向。

### WSST

MATLAB `wsst` 只接受实数输入，因此分别计算 `WSST(I)` 与 `WSST(Q)`，再合成功率：

```text
Pwsst = |WSST(I)|^2 + |WSST(Q)|^2
```

WSST 频率轴为非负频率轴，不保留复基带正负频率方向。

## 3. 输出顺序

| 序号 | 英文名称 | 中文名称 |
|---:|---|---|
| 1 | SpectralKurtosisMean | 谱峰度均值 |
| 2 | SpectralKurtosisStd | 谱峰度标准差 |
| 3 | SpectralSkewnessMean | 谱偏度均值 |
| 4 | SpectralSkewnessStd | 谱偏度标准差 |
| 5 | SpectralCrestFactorMean | 谱峰因子均值 |
| 6 | SpectralCrestFactorStd | 谱峰因子标准差 |
| 7 | SpectralFlatnessMean | 谱平坦度均值 |
| 8 | SpectralFlatnessStd | 谱平坦度标准差 |
| 9 | SpectralEntropyMean | 谱熵均值 |
| 10 | SpectralEntropyStd | 谱熵标准差 |
| 11 | NormalizedSTFTRidgeFrequencyStd | 归一化 STFT 脊线频率标准差 |
| 12 | NormalizedSTFTRidgeSlope | 归一化 STFT 脊线斜率 |
| 13 | NormalizedSpectralCentroidStd | 归一化谱质心标准差 |
| 14 | NormalizedSpectralSpreadMean | 归一化频谱扩展均值 |
| 15 | NormalizedSpectralSpreadStd | 归一化频谱扩展标准差 |
| 16 | STFTFrameEnergyCV | STFT 帧能量变异系数 |
| 17 | STFTActiveFrameRatio | STFT 活动帧比例 |
| 18 | STFTNormalizedEnergyTransitionCount | STFT 能量状态跃迁比例 |
| 19 | NormalizedFSSTRidgeFrequencyStd | 归一化 FSST 脊线频率标准差 |
| 20 | NormalizedFSSTRidgeSlope | 归一化 FSST 脊线斜率 |
| 21 | NormalizedFSSTRidgeCurvatureRMS | 归一化 FSST 脊线曲率 RMS |
| 22 | FSSTRidgeEnergyRatio | FSST 脊线能量占比 |
| 23 | NormalizedFSSTTimeFrequencyEntropy | 归一化 FSST 时频熵 |
| 24 | NormalizedWSSTRidgeFrequencyStd | 归一化 WSST 脊线频率标准差 |
| 25 | NormalizedWSSTRidgeSlope | 归一化 WSST 脊线斜率 |
| 26 | NormalizedWSSTBandwidthMean | 归一化 WSST 带宽均值 |
| 27 | NormalizedWSSTBandwidthStd | 归一化 WSST 带宽标准差 |
| 28 | WSSTRidgeEnergyRatio | WSST 脊线能量占比 |

## 4. 特征解释

### 1. SpectralKurtosisMean

每个 STFT 帧功率谱关于局部谱质心的标准化四阶矩的时间均值。高值通常表示局部频谱更尖锐或尾部更重。

### 2. SpectralKurtosisStd

局部谱峰度序列的总体标准差，描述频谱尖锐程度随时间的变化。

### 3. SpectralSkewnessMean

每帧功率谱标准化三阶中心矩的时间均值，描述局部频谱平均左右不对称程度。

### 4. SpectralSkewnessStd

局部谱偏度序列的总体标准差。频偏、扫频、跳频和干扰都可能提高该值。

### 5. SpectralCrestFactorMean

每帧 `max(localPower)/mean(localPower)` 的均值，描述局部谱峰平均突出程度。

### 6. SpectralCrestFactorStd

局部谱峰因子序列的总体标准差，描述谱峰突出程度是否随时间变化。

### 7. SpectralFlatnessMean

每帧谱平坦度的均值。接近 1 表示局部频谱更平坦，接近 0 表示谱峰更突出。

### 8. SpectralFlatnessStd

局部谱平坦度序列的总体标准差，描述集中/分散状态的时间变化。

### 9. SpectralEntropyMean

每帧归一化谱熵的均值，描述局部频谱平均分散程度，不是通信信息熵。

### 10. SpectralEntropyStd

局部谱熵序列的总体标准差，描述频谱分散程度随时间的变化。

### 11. NormalizedSTFTRidgeFrequencyStd

每帧最大功率频率除以采样率后的总体标准差。它描述主峰位置波动，不是严格瞬时频率。

### 12. NormalizedSTFTRidgeSlope

在归一化时间轴 `[0,1]` 上，对 STFT 脊线归一化频率做线性回归得到的斜率。正值表示总体上升，负值表示总体下降。旧版因变量名错误恒为 0，现已修正。

### 13. NormalizedSpectralCentroidStd

每帧功率加权谱质心除以采样率后的总体标准差。对宽带、多载波信号，它通常比单峰脊线更稳定。

### 14. NormalizedSpectralSpreadMean

每帧功率谱围绕局部质心的标准差除以采样率，再对时间求均值。它不是协议规定带宽。

### 15. NormalizedSpectralSpreadStd

归一化局部频谱扩展序列的总体标准差，描述局部频谱宽度的时间变化。

### 16. STFTFrameEnergyCV

```text
populationStd(frameEnergy) / mean(frameEnergy)
```

描述 STFT 帧能量的相对起伏。连续稳定占用通常较低，突发或局部强弱变化通常较高。

### 17. STFTActiveFrameRatio

活动帧定义为 `frameEnergy >= 0.1*max(frameEnergy)`，本特征是活动帧数占总帧数的比例。它是片段内部相对活动度，不是绝对能量检测结果。

### 18. STFTNormalizedEnergyTransitionCount

相邻帧在活动/非活动状态之间的切换次数除以 `numFrames-1`。高值表示片段内能量状态切换更频繁。

### 19. NormalizedFSSTRidgeFrequencyStd

FSST 最大功率脊线归一化频率的总体标准差。对单主分量扫频较敏感，对多分量信号可能发生脊线切换。

### 20. NormalizedFSSTRidgeSlope

FSST 脊线归一化频率关于归一化时间的一阶线性斜率。接近 0 不表示没有非线性或往返变化。

### 21. NormalizedFSSTRidgeCurvatureRMS

FSST 脊线归一化频率二阶差分的 RMS，描述轨迹弯曲、跳变或错选程度。

### 22. FSSTRidgeEnergyRatio

每帧主脊线及上下各一个频率 bin 的能量之和除以 FSST 全局能量。高值表示能量更集中在单条主脊附近。

### 23. NormalizedFSSTTimeFrequencyEntropy

将整个 FSST 功率图归一化后计算的全局熵，再除以最大熵。高值表示时频能量更分散。

### 24. NormalizedWSSTRidgeFrequencyStd

I/Q 两路 WSST 合成功率图中主脊线归一化频率的总体标准差。频率轴为非负轴。

### 25. NormalizedWSSTRidgeSlope

WSST 主脊线归一化频率关于归一化时间的一阶斜率，不保留复基带正负频率方向。

### 26. NormalizedWSSTBandwidthMean

每帧 WSST 功率谱围绕功率加权谱质心的频率标准差除以采样率，再对时间求均值。旧文档所称“围绕主脊线”不正确，现已按实际实现修正。

### 27. NormalizedWSSTBandwidthStd

上述 WSST 局部带宽序列的总体标准差，描述多尺度频率扩展随时间的变化。

### 28. WSSTRidgeEnergyRatio

WSST 主脊线及相邻频率 bin 能量占总 WSST 能量的比例，描述多尺度时频能量集中度。

## 5. 解释边界

- 谱熵、平坦度、峰值因子及其时间统计量彼此相关，不应重复计为独立证据。
- STFT、FSST、WSST 脊线统计提供互补视角，但在多载波信号上都可能由最强局部分量主导。
- STFT 活动特征使用片段内部相对阈值，不能替代原始连续 IQ 上的独立突发检测。
- WSST 带宽围绕功率加权质心计算；STFT/FSST 使用双边频率轴，WSST 使用非负频率轴，三者不能直接比较符号。
- 所有特征都应结合采样率、片段长度、信噪比和幅度归一化方式解释。
