# 25维 IQ 时频特征说明

本文档用于说明 `extractTimeFrequencyFeatures.m` 提取的 25 个时频 IQ 特征，主要供大模型理解特征含义、输入输出格式、预处理方式、STFT / Spectrogram 配置、FSST 配置、WSST 配置和解释边界使用。

这些特征不是直接的调制类型标签，也不是通信协议或信号体制的判决规则，而是从一个复数 IQ 信号段的时频表示中提取出的统计描述。大模型在使用这些特征时，应结合多个特征的整体模式进行判断，不应只根据单个特征值直接断定信号体制。

## 1. 输入说明

函数输入为一个复数 IQ 向量和采样率：

```matlab
x
sampleRate
```

其中：

| 项目 | 说明 |
|---|---|
| `x` | 复数 IQ 序列 |
| `sampleRate` | 采样率，单位为 Hz |
| 输入形状 | `x` 为一维向量，行向量或列向量均可 |
| 输入含义 | 一个完整的 IQ 信号段 |
| 实部 | I 路采样 |
| 虚部 | Q 路采样 |
| 最小长度 | 当前建议实现中不小于 32 个采样点 |
| 最大长度 | 当前建议实现中限制为不超过 16384 个采样点 |

输入会被统一转换为列向量并转为复数 `single` 类型：

```matlab
xIn = complex(single(real(x(:))), single(imag(x(:))));
```

`sampleRate` 必须为正的有限标量。如果输入为空、长度不满足限制，或实部 / 虚部包含 `NaN` / `Inf`，函数应报错。

## 2. 预处理说明

### 2.1 去除直流分量

时频分析前先对整段 IQ 信号去除均值：

```matlab
x0 = xIn - mean(xIn);
```

这样可以减小直流偏置、接收机零点偏移或静态频谱中心泄漏对 STFT、FSST 和 WSST 特征的影响。

### 2.2 数值保护

实现中使用极小正数：

```matlab
tinyValue = single(1.0e-20);
```

用于避免除零、`log(0)`、极弱信号导致的非有限值，以及局部时频帧能量过小时的异常统计。最终输出前还应检查特征值，如果某个特征为 `NaN` 或 `Inf`，则置为 0。

### 2.3 零能量或极弱信号

如果去直流后的信号能量过低，或某个局部时频帧的功率和小于数值保护阈值，对应特征保持为 0。需要注意：输出为 0 通常表示该实现口径下无法稳定计算该统计量，不应直接解释为“信号一定没有对应的物理结构”。

## 3. STFT / Spectrogram 配置

前 15 个特征来自 STFT / Spectrogram 功率图。

当前实现采用自适应短时窗、固定 FFT 点数和中心化双边频率轴：

```text
windowLength        = adaptive value in [32, 256]
window              = periodic Hamming window
overlapLength       = floor(windowLength / 2)
nfft                = 512
overlap             = 50%
frequency axis      = centered two-sided frequency axis
```

MATLAB 实现口径可理解为：

```matlab
[stftMatrix, frequencyAxis, timeAxis] = spectrogram( ...
    x0, window, overlapLength, nfft, sampleRate, 'centered');

stftPowerMatrix = abs(stftMatrix).^2;
```

其中 `frequencyAxis` 为中心化双边频率轴，范围约为：

```text
-sampleRate/2 ... +sampleRate/2
```

STFT 频率类特征统一使用：

```text
normalizedFrequency = frequencyHz / sampleRate
```

因此 STFT 归一化频率大致落在 `[-0.5, 0.5)`。STFT 每个时间帧都会形成一条局部功率谱，前 15 项主要描述这些局部功率谱的形状、脊线位置、谱质心变化和频谱扩展变化。

## 4. FSST 配置

第 16 到第 20 个特征来自 Fourier Synchrosqueezed Transform，简称 FSST。

FSST 可以直接处理复数 IQ 信号，因此 FSST 部分对完整复数 `x0` 进行分析。为了与 STFT 的基本时间尺度保持一致，FSST 使用与 STFT 相同长度的 Hamming 分析窗：

```matlab
fsstWindow = double(hamming(windowLength, 'periodic'));

[fsstMatrix, fsstFrequencyAxis, ~] = fsst( ...
    x0, sampleRate, fsstWindow);

fsstPowerMatrix = abs(fsstMatrix).^2;
```

FSST 的同步压缩会把 STFT 中较分散的时频能量重新分配到更集中的频率位置，因此更适合描述主脊线、频率扫动、脊线曲率和时频能量集中度。

FSST 频率类特征同样使用：

```text
normalizedFrequency = frequencyHz / sampleRate
```

对于每个 FSST 时间帧，主脊线频率由该帧功率最大的频率 bin 给出：

```matlab
[~, ridgeIndex] = max(localPower);
ridgeFrequency = fsstFrequencyAxis(ridgeIndex);
```

FSST 脊线能量占比使用主脊线及其上下相邻频率 bin 的能量之和除以 FSST 全局时频能量。

## 5. WSST 配置

第 21 到第 25 个特征来自 Wavelet Synchrosqueezed Transform，简称 WSST。

MATLAB 的 `wsst` 输入要求为实值序列，不能直接把复数 IQ 作为一个整体输入。因此当前建议实现采用明确的工程口径：分别对 I 路和 Q 路计算 WSST，再将两路 WSST 功率相加：

```text
I 路 WSST ─┐
           ├─ |WSST(I)|^2 + |WSST(Q)|^2 -> WSST power matrix
Q 路 WSST ─┘
```

对应实现口径可理解为：

```matlab
[wsstI, wsstFrequencyAxis] = wsst(double(real(x0)), sampleRate);
[wsstQ, ~]                 = wsst(double(imag(x0)), sampleRate);

wsstPowerMatrix = abs(wsstI).^2 + abs(wsstQ).^2;
```

WSST 的频率轴为非负频率轴，通常从低频到 Nyquist 频率附近。由于 I/Q 两路被分别作为实值信号处理，这种 WSST 表示主要描述多尺度局部频率结构和能量集中度，不再保留复数基带 IQ 的正负频率方向信息。

WSST 频率类特征也使用：

```text
normalizedFrequency = frequencyHz / sampleRate
```

因此 WSST 的归一化频率通常位于 `[0, 0.5]` 附近。解释 WSST 特征时，应避免把它和 STFT / FSST 的中心化双边频率轴直接混为同一种频率符号。

## 6. 输出说明

函数输出为固定长度的单精度特征向量：

```matlab
features
```

输出格式：

| 项目 | 说明 |
|---|---|
| 输出类型 | `single` |
| 输出尺寸 | `1×25` |
| 输出含义 | 当前 IQ 段的 25 个时频统计特征 |
| 顺序要求 | 特征顺序固定，不能随意调换 |

固定输出顺序如下：

| 序号 | 英文名称 | 中文名称 | 特征组 |
|---:|---|---|---|
| 1 | SpectralKurtosisMean | 谱峰度均值 | STFT / Spectrogram |
| 2 | SpectralKurtosisStd | 谱峰度标准差 | STFT / Spectrogram |
| 3 | SpectralSkewnessMean | 谱偏度均值 | STFT / Spectrogram |
| 4 | SpectralSkewnessStd | 谱偏度标准差 | STFT / Spectrogram |
| 5 | SpectralCrestFactorMean | 谱峰因子均值 | STFT / Spectrogram |
| 6 | SpectralCrestFactorStd | 谱峰因子标准差 | STFT / Spectrogram |
| 7 | SpectralFlatnessMean | 谱平坦度均值 | STFT / Spectrogram |
| 8 | SpectralFlatnessStd | 谱平坦度标准差 | STFT / Spectrogram |
| 9 | SpectralEntropyMean | 谱熵均值 | STFT / Spectrogram |
| 10 | SpectralEntropyStd | 谱熵标准差 | STFT / Spectrogram |
| 11 | NormalizedSTFTRidgeFrequencyStd | 归一化 STFT 脊线频率标准差 | STFT / Spectrogram |
| 12 | NormalizedSTFTRidgeSlope | 归一化 STFT 脊线斜率 | STFT / Spectrogram |
| 13 | NormalizedSpectralCentroidStd | 归一化谱质心标准差 | STFT / Spectrogram |
| 14 | NormalizedSpectralSpreadMean | 归一化频谱扩展均值 | STFT / Spectrogram |
| 15 | NormalizedSpectralSpreadStd | 归一化频谱扩展标准差 | STFT / Spectrogram |
| 16 | NormalizedFSSTRidgeFrequencyStd | 归一化 FSST 脊线频率标准差 | FSST |
| 17 | NormalizedFSSTRidgeSlope | 归一化 FSST 脊线斜率 | FSST |
| 18 | NormalizedFSSTRidgeCurvatureRMS | 归一化 FSST 脊线曲率 RMS | FSST |
| 19 | FSSTRidgeEnergyRatio | FSST 脊线能量占比 | FSST |
| 20 | NormalizedFSSTTimeFrequencyEntropy | 归一化 FSST 时频熵 | FSST |
| 21 | NormalizedWSSTRidgeFrequencyStd | 归一化 WSST 脊线频率标准差 | WSST |
| 22 | NormalizedWSSTRidgeSlope | 归一化 WSST 脊线斜率 | WSST |
| 23 | NormalizedWSSTBandwidthMean | 归一化 WSST 带宽均值 | WSST |
| 24 | NormalizedWSSTBandwidthStd | 归一化 WSST 带宽标准差 | WSST |
| 25 | WSSTRidgeEnergyRatio | WSST 脊线能量占比 | WSST |

## 7. 特征解释

### 1. SpectralKurtosisMean

**计算对象：** STFT / Spectrogram 每个时间帧的局部功率谱。

**实现口径：** 对每个时间帧，先计算局部功率谱的功率加权谱质心和频谱扩展，再计算标准化四阶中心矩：

```matlab
p = localPower / sum(localPower)
localCentroid = sum(frequencyAxis .* p)
localSpread = sqrt(sum(((frequencyAxis - localCentroid).^2) .* p))
localKurtosis = sum((((frequencyAxis - localCentroid) / localSpread).^4) .* p)
SpectralKurtosisMean = mean(localKurtosisSequence)
```

**含义：** 描述 STFT 各时间帧频谱形状的平均尖锐程度。数值较高时，通常表示局部频谱能量更集中在少数频率附近，或存在较明显的频谱尖峰 / 重尾结构。

**解释作用：** 可辅助判断信号的局部频谱是否更像窄带尖峰、局部谱线结构，还是更分散的宽带结构。

**注意事项：** 这里的谱峰度是基于每帧频率分布的四阶统计量，不是调制类型标签，也不应直接等同于时域幅度峰度。局部噪声、干扰峰、窗函数泄漏和低能量帧都会影响该值。

### 2. SpectralKurtosisStd

**计算对象：** STFT / Spectrogram 各时间帧的局部谱峰度序列。

**实现口径：**

```matlab
SpectralKurtosisStd = std(localKurtosisSequence)
```

**含义：** 描述局部频谱尖锐程度随时间变化的稳定性。数值越大，说明不同时间帧之间的谱峰度差异越明显。

**解释作用：** 可用于观察信号是否存在时变的谱峰结构，例如某些时间段频谱很集中，而另一些时间段频谱更分散。

**注意事项：** 它反映的是谱峰度的时间波动，不是整体频谱峰度。短信号、低帧数或低信噪比条件下，该统计量可能不稳定。

### 3. SpectralSkewnessMean

**计算对象：** STFT / Spectrogram 每个时间帧的局部功率谱。

**实现口径：** 对每个时间帧计算标准化三阶中心矩，然后对时间求均值：

```matlab
p = localPower / sum(localPower)
localCentroid = sum(frequencyAxis .* p)
localSpread = sqrt(sum(((frequencyAxis - localCentroid).^2) .* p))
localSkewness = sum((((frequencyAxis - localCentroid) / localSpread).^3) .* p)
SpectralSkewnessMean = mean(localSkewnessSequence)
```

**含义：** 描述局部频谱围绕谱质心的平均左右不对称程度。正值通常表示局部频谱尾部更偏向正频率侧，负值通常表示尾部更偏向负频率侧。

**解释作用：** 可辅助理解复数基带频谱是否存在方向性偏移、单边扩展、镜像不平衡或局部频谱不对称。

**注意事项：** 符号依赖中心化频率轴方向。频偏、IQ 不平衡、滤波、截取位置和噪声都可能改变该值，不应单独用它判断具体制式。

### 4. SpectralSkewnessStd

**计算对象：** STFT / Spectrogram 各时间帧的局部谱偏度序列。

**实现口径：**

```matlab
SpectralSkewnessStd = std(localSkewnessSequence)
```

**含义：** 描述频谱不对称程度随时间变化的强弱。数值越大，说明不同时间帧的频谱偏斜方向或偏斜程度变化越明显。

**解释作用：** 可辅助识别时变频偏、扫频结构、跳频结构或局部干扰导致的频谱形状变化。

**注意事项：** 当局部频谱扩展很小或能量很弱时，偏度计算容易受数值误差影响。该特征应与脊线频率标准差、脊线斜率和谱质心标准差一起解释。

### 5. SpectralCrestFactorMean

**计算对象：** STFT / Spectrogram 每个时间帧的局部功率谱。

**实现口径：** 每个时间帧计算最大谱功率与平均谱功率的比值，再对时间求均值：

```matlab
localCrestFactor = max(localPower) / (mean(localPower) + tinyValue)
SpectralCrestFactorMean = mean(localCrestFactorSequence)
```

**含义：** 描述局部频谱中主峰相对于平均谱水平的平均突出程度。

**解释作用：** 可用于区分局部谱峰明显的信号和局部频谱较平坦、较分散的信号。

**注意事项：** 高谱峰因子可能来自真实窄带成分，也可能来自窄带干扰、残余直流、噪声尖峰或频率分辨率设置。不能仅凭该项判断是否为某种窄带调制。

### 6. SpectralCrestFactorStd

**计算对象：** STFT / Spectrogram 各时间帧的局部谱峰因子序列。

**实现口径：**

```matlab
SpectralCrestFactorStd = std(localCrestFactorSequence)
```

**含义：** 描述局部谱峰突出程度随时间变化的稳定性。数值越大，说明谱峰结构在不同时间帧之间起伏越明显。

**解释作用：** 可辅助发现时变谱峰、突发窄带成分或频率跳变导致的峰值强度变化。

**注意事项：** 该特征对单个异常帧敏感。解释时应结合 SpectralEntropyStd 和脊线特征，避免把偶然尖峰误判为稳定信号结构。

### 7. SpectralFlatnessMean

**计算对象：** STFT / Spectrogram 每个时间帧的局部功率谱。

**实现口径：** 对每个时间帧计算几何平均功率与算术平均功率之比，再对时间求均值：

```matlab
localGeometricMean = exp(mean(log(max(localPower, localPowerFloor))))
localFlatness = localGeometricMean / (mean(localPower) + tinyValue)
SpectralFlatnessMean = mean(localFlatnessSequence)
```

**含义：** 描述局部频谱平均是否接近平坦。数值接近 0 通常表示存在明显谱峰，数值接近 1 通常表示局部频谱更平坦或更噪声状。

**解释作用：** 可用于区分尖峰型局部频谱和宽带平坦型局部频谱。

**注意事项：** 平坦度与频谱熵相关但不相同。高平坦度不一定表示有用信号更复杂，也可能表示噪声占比较高或信号能量被扩散。

### 8. SpectralFlatnessStd

**计算对象：** STFT / Spectrogram 各时间帧的局部谱平坦度序列。

**实现口径：**

```matlab
SpectralFlatnessStd = std(localFlatnessSequence)
```

**含义：** 描述局部频谱平坦程度随时间变化的强弱。

**解释作用：** 可辅助判断信号在不同时间段是否呈现不同的频谱集中 / 分散状态。

**注意事项：** 当部分时间帧能量很低时，平坦度可能更接近噪声状。解释时应结合整体能量、谱熵和脊线能量占比。

### 9. SpectralEntropyMean

**计算对象：** STFT / Spectrogram 每个时间帧归一化后的局部功率谱。

**实现口径：** 每个时间帧将功率谱归一化为概率分布，并计算归一化香农熵：

```matlab
p = localPower / sum(localPower)
localEntropy = -sum(p .* log2(p + tinyValue)) / log2(numFrequencyBins)
SpectralEntropyMean = mean(localEntropySequence)
```

**含义：** 描述局部频谱能量在频率 bin 上的平均分散程度。数值越低，通常表示能量集中在少数频率；数值越高，表示能量分布更分散。

**解释作用：** 可辅助判断信号局部频谱是集中、稀疏，还是宽带、噪声状或多频成分较多。

**注意事项：** 高谱熵不等于高信息速率，也不直接等于 OFDM、Wi-Fi、LTE 或 5G。噪声、干扰和低信噪比也会提高谱熵。

### 10. SpectralEntropyStd

**计算对象：** STFT / Spectrogram 各时间帧的局部谱熵序列。

**实现口径：**

```matlab
SpectralEntropyStd = std(localEntropySequence)
```

**含义：** 描述局部频谱分散程度随时间变化的稳定性。

**解释作用：** 可用于观察信号是否在某些时间段更集中、某些时间段更分散，或是否存在突发频谱结构变化。

**注意事项：** 谱熵标准差受时间帧数量和局部信噪比影响。单独升高只能说明局部频谱复杂度随时间变化较明显，不能直接说明某种协议或调制存在。

### 11. NormalizedSTFTRidgeFrequencyStd

**计算对象：** STFT / Spectrogram 每个时间帧的主脊线频率序列。

**实现口径：** 每个时间帧取功率最大的频率 bin 作为 STFT 主脊线频率，再除以采样率并计算标准差：

```matlab
[~, ridgeIndex] = max(localPower)
ridgeFrequency = frequencyAxis(ridgeIndex)
normalizedRidgeFrequency = ridgeFrequency / sampleRate
NormalizedSTFTRidgeFrequencyStd = std(normalizedRidgeFrequencySequence)
```

**含义：** 描述 STFT 主能量频率位置随时间变化的强弱。

**解释作用：** 对扫频、跳频、FSK 类频率切换、LoRa chirp 频率移动或频偏变化有解释价值。

**注意事项：** 主脊线只是每帧最大功率点，可能被强干扰、残余直流或噪声峰吸引。它不是严格的瞬时频率，也不是协议中心频率。

### 12. NormalizedSTFTRidgeSlope

**计算对象：** STFT / Spectrogram 主脊线的归一化频率时间序列。

**实现口径：** 将时间轴归一化到 `[0, 1]`，对归一化主脊线频率做一阶线性拟合，输出斜率：

```matlab
tNorm = linspace(0, 1, numFrames)
fNorm = ridgeFrequencySequence / sampleRate
NormalizedSTFTRidgeSlope = linearSlope(tNorm, fNorm)
```

**含义：** 描述 STFT 主脊线频率随时间的整体上升或下降趋势。正值表示总体向高频方向移动，负值表示总体向低频方向移动。

**解释作用：** 可辅助描述 chirp、扫频、频率漂移或片段内频率趋势。

**注意事项：** 线性斜率只能描述整体趋势，不能完整描述非线性扫频、频率跳变或多分量交替。若脊线频率呈来回变化，斜率可能接近 0，但并不表示没有频率变化。

### 13. NormalizedSpectralCentroidStd

**计算对象：** STFT / Spectrogram 每个时间帧的功率加权谱质心序列。

**实现口径：** 每个时间帧计算局部谱质心，除以采样率后计算标准差：

```matlab
localCentroid = sum(frequencyAxis .* localPower) / sum(localPower)
normalizedCentroid = localCentroid / sampleRate
NormalizedSpectralCentroidStd = std(normalizedCentroidSequence)
```

**含义：** 描述 STFT 每个时间帧频谱能量中心位置随时间变化的程度。

**解释作用：** 可辅助理解信号整体频谱能量中心是否稳定，或是否随时间发生移动。

**注意事项：** 该特征是谱质心标准差，不是真正基于相位导数的瞬时频率标准差。对于多载波、宽带或多分量信号，谱质心通常比“单一瞬时频率”更适合作为整体时频描述。

### 14. NormalizedSpectralSpreadMean

**计算对象：** STFT / Spectrogram 每个时间帧的局部频谱扩展序列。

**实现口径：** 每个时间帧计算功率围绕局部谱质心的频率标准差，除以采样率后对时间求均值：

```matlab
localCentroid = sum(frequencyAxis .* localPower) / sum(localPower)
localSpread = sqrt( ...
    sum(((frequencyAxis - localCentroid).^2) .* localPower) / sum(localPower))
normalizedSpread = localSpread / sampleRate
NormalizedSpectralSpreadMean = mean(normalizedSpreadSequence)
```

**含义：** 描述每个 STFT 时间帧中频谱能量围绕谱质心的平均扩散程度。

**解释作用：** 可用于区分局部窄带结构和局部宽带结构，也可辅助理解信号主要能量在每个时间帧内占据的频率宽度。

**注意事项：** 它是功率加权频率标准差，不是严格协议带宽，也不是原先名称中的 instantaneous bandwidth。噪声底、旁瓣和多峰结构都会提高该值。

### 15. NormalizedSpectralSpreadStd

**计算对象：** STFT / Spectrogram 每个时间帧的归一化频谱扩展序列。

**实现口径：**

```matlab
NormalizedSpectralSpreadStd = std(normalizedSpreadSequence)
```

**含义：** 描述局部频谱宽度随时间变化的稳定性。

**解释作用：** 可辅助判断信号是否在不同时间段呈现不同的带宽占用，例如突发扩展、频率跳变、多分量出现或消失。

**注意事项：** 该特征反映的是局部频谱扩展的时间波动，不应直接解释为通信标准中的信道带宽变化。

### 16. NormalizedFSSTRidgeFrequencyStd

**计算对象：** FSST 功率图中每个时间帧的主脊线频率序列。

**实现口径：** 每个 FSST 时间帧取功率最大的频率 bin 作为主脊线频率，除以采样率后计算标准差：

```matlab
[~, ridgeIndex] = max(localPower)
ridgeFrequency = fsstFrequencyAxis(ridgeIndex)
fNorm = ridgeFrequency / sampleRate
NormalizedFSSTRidgeFrequencyStd = std(fNormSequence)
```

**含义：** 描述 FSST 同步压缩后主能量频率位置随时间变化的强弱。

**解释作用：** 相比普通 STFT，FSST 脊线通常更集中，适合辅助观察扫频、频率漂移、chirp 或主频随时间变化的结构。

**注意事项：** FSST 脊线仍然是最大能量脊线，不一定等于严格瞬时频率。多分量信号中，主脊线可能在不同分量之间切换。

### 17. NormalizedFSSTRidgeSlope

**计算对象：** FSST 主脊线归一化频率时间序列。

**实现口径：** 将 FSST 时间轴归一化到 `[0, 1]`，对归一化主脊线频率做一阶线性拟合：

```matlab
tNorm = linspace(0, 1, numFSSTTimeFrames)
fNorm = fsstRidgeFrequencySequence / sampleRate
NormalizedFSSTRidgeSlope = linearSlope(tNorm, fNorm)
```

**含义：** 描述 FSST 主脊线频率随时间的整体线性趋势。

**解释作用：** 可辅助判断主频是否存在持续上升、持续下降或整体频率漂移。

**注意事项：** 斜率接近 0 不代表没有频率结构变化；对称扫频、跳频或非线性变化可能抵消一阶趋势。解释时应结合 FSST 脊线标准差和曲率 RMS。

### 18. NormalizedFSSTRidgeCurvatureRMS

**计算对象：** FSST 主脊线归一化频率时间序列的二阶变化。

**实现口径：** 对归一化主脊线频率计算二阶差分或等价的离散曲率度量，并取 RMS：

```matlab
fNorm = fsstRidgeFrequencySequence / sampleRate
ridgeCurvature = diff(fNorm, 2)
NormalizedFSSTRidgeCurvatureRMS = sqrt(mean(ridgeCurvature.^2))
```

**含义：** 描述 FSST 主脊线频率轨迹的弯曲程度或非线性变化强弱。数值越大，说明脊线不是简单直线趋势，而是存在更明显的曲率、折返或加速度变化。

**解释作用：** 对非线性 chirp、频率弯曲、频率调制速率变化或复杂主脊线轨迹有解释价值。

**注意事项：** 曲率 RMS 对脊线跳变、噪声造成的局部错选和时间帧数量敏感。它不应单独解释为某种具体扫频调制，只能说明主脊线轨迹更弯曲或更不平滑。

### 19. FSSTRidgeEnergyRatio

**计算对象：** FSST 功率图中的主脊线邻域能量和全局时频能量。

**实现口径：** 对每个时间帧，取主脊线频率 bin 及其上下相邻一个频率 bin 的功率，累加后除以 FSST 全部时频 bin 的功率和：

```matlab
ridgeEnergy = sum(power around ridgeIndex - 1 : ridgeIndex + 1)
totalEnergy = sum(fsstPowerMatrix(:))
FSSTRidgeEnergyRatio = ridgeEnergy / (totalEnergy + tinyValue)
```

**含义：** 描述 FSST 时频能量有多大比例集中在主脊线附近。数值越高，通常说明主导时频轨迹越清晰、越集中。

**解释作用：** 可用于判断信号是否存在清晰主脊线，或能量是否分散在多个时频区域。

**注意事项：** 高脊线能量占比不一定表示信号简单，也可能表示强单音、窄带干扰或某个主导分量压过其他分量。低值也不一定表示无结构，可能表示多分量并存或能量分散。

### 20. NormalizedFSSTTimeFrequencyEntropy

**计算对象：** FSST 全局功率图。

**实现口径：** 将 FSST 全部时频 bin 的功率归一化为概率分布，计算归一化熵：

```matlab
p = fsstPowerMatrix(:) / sum(fsstPowerMatrix(:))
H = -sum(p .* log2(p + tinyValue)) / log2(numel(p))
NormalizedFSSTTimeFrequencyEntropy = H
```

**含义：** 描述 FSST 时频能量在整个时频平面上的分散程度。数值低表示能量集中在较少时频区域，数值高表示能量分布更广。

**解释作用：** 可辅助理解信号时频结构是集中、清晰，还是分散、复杂或噪声状。

**注意事项：** 高时频熵不等同于高通信信息熵，也不直接表示制式复杂。噪声、宽带干扰、多分量重叠和低信噪比都可能提高该值。

### 21. NormalizedWSSTRidgeFrequencyStd

**计算对象：** WSST 合成功率图中每个时间帧的主脊线频率序列。

**实现口径：** 对 I/Q 两路 WSST 功率相加后，每个时间帧取最大功率频率 bin 作为 WSST 主脊线频率，除以采样率后计算标准差：

```matlab
wsstPowerMatrix = abs(wsstI).^2 + abs(wsstQ).^2
[~, ridgeIndex] = max(localPower)
ridgeFrequency = wsstFrequencyAxis(ridgeIndex)
fNorm = ridgeFrequency / sampleRate
NormalizedWSSTRidgeFrequencyStd = std(fNormSequence)
```

**含义：** 描述 WSST 多尺度表示下主脊线频率随时间变化的强弱。

**解释作用：** 可辅助观察多尺度局部频率结构是否稳定，尤其适合补充 STFT / FSST 对非平稳信号的描述。

**注意事项：** WSST 这里基于 I/Q 两路实值变换的功率合成，频率轴为非负频率轴，因此不能把其符号解释为复数基带的正负频率方向。

### 22. NormalizedWSSTRidgeSlope

**计算对象：** WSST 主脊线归一化频率时间序列。

**实现口径：** 将 WSST 时间轴归一化到 `[0, 1]`，对归一化主脊线频率做一阶线性拟合：

```matlab
tNorm = linspace(0, 1, numWSSTTimeFrames)
fNorm = wsstRidgeFrequencySequence / sampleRate
NormalizedWSSTRidgeSlope = linearSlope(tNorm, fNorm)
```

**含义：** 描述 WSST 主脊线频率随时间的整体上升或下降趋势。

**解释作用：** 可辅助判断多尺度主频结构是否存在持续频率漂移或扫频趋势。

**注意事项：** WSST 斜率只反映非负频率轴上的趋势，不保留复数 IQ 的正负基带方向。线性斜率也无法完整描述非线性或跳变式频率轨迹。

### 23. NormalizedWSSTBandwidthMean

**计算对象：** WSST 合成功率图中每个时间帧的局部频率扩展序列。

**实现口径：** 对每个 WSST 时间帧，以该帧主脊线频率为中心，计算功率加权频率扩展，除以采样率后对时间求均值：

```matlab
localBandwidth = sqrt( ...
    sum(((wsstFrequencyAxis - ridgeFrequency).^2) .* localPower) / sum(localPower))
normalizedBandwidth = localBandwidth / sampleRate
NormalizedWSSTBandwidthMean = mean(normalizedBandwidthSequence)
```

**含义：** 描述 WSST 表示下局部频率能量围绕主脊线的平均扩散程度。

**解释作用：** 可用于理解信号在多尺度时频表示中是更集中在主脊线附近，还是在主脊线周围存在较宽的频率扩展。

**注意事项：** 该带宽是 WSST 时频图上的统计扩展，不是协议带宽或信道带宽。由于 WSST 频率轴为非负频率，不能与 STFT / FSST 的双边归一化频率范围直接按符号比较。

### 24. NormalizedWSSTBandwidthStd

**计算对象：** WSST 每个时间帧的归一化局部频率扩展序列。

**实现口径：**

```matlab
NormalizedWSSTBandwidthStd = std(normalizedBandwidthSequence)
```

**含义：** 描述 WSST 局部频率扩展随时间变化的稳定性。

**解释作用：** 可辅助判断多尺度时频结构是否存在时变带宽、局部扩展增强或突发宽带成分。

**注意事项：** 该特征对噪声、低能量帧和多分量叠加较敏感。数值升高只能说明 WSST 局部频率扩展变化更明显，不能直接说明信号占用带宽发生协议层面的变化。

### 25. WSSTRidgeEnergyRatio

**计算对象：** WSST 合成功率图中的主脊线邻域能量和全局 WSST 能量。

**实现口径：** 对每个 WSST 时间帧，取主脊线频率 bin 及其上下相邻一个频率 bin 的功率，累加后除以 WSST 全部时频 bin 的功率和：

```matlab
ridgeEnergy = sum(power around ridgeIndex - 1 : ridgeIndex + 1)
totalEnergy = sum(wsstPowerMatrix(:))
WSSTRidgeEnergyRatio = ridgeEnergy / (totalEnergy + tinyValue)
```

**含义：** 描述 WSST 能量集中在主脊线附近的比例。数值越高，通常表示多尺度主脊线越清晰。

**解释作用：** 可用于判断信号是否具有稳定、集中的多尺度时频主轨迹，或能量是否分散到多个频率区域。

**注意事项：** 高 WSST 脊线能量占比不一定代表某种具体调制，也可能来自强窄带分量或干扰。低值也不一定表示无有用信号，可能表示多分量、宽带或低信噪比条件。

## 8. 大模型使用建议与解释边界

大模型理解这 25 个时频特征时，建议优先按特征组解释，而不是孤立解释某一个数值：

| 分组 | 特征 | 主要描述 |
|---|---|---|
| STFT 局部频谱形状 | 1-10 | 每个时间帧的频谱是否尖锐、偏斜、平坦、分散，以及这些形状是否随时间变化 |
| STFT 主脊线与频谱扩展 | 11-15 | 主能量频率是否移动，谱质心是否稳定，局部频谱宽度是否较大或随时间变化 |
| FSST 主脊线动态 | 16-18 | 同步压缩后主脊线的波动、整体趋势和非线性弯曲程度 |
| FSST 时频集中度 | 19-20 | FSST 能量是否集中在主脊线附近，以及全局时频能量是否分散 |
| WSST 多尺度结构 | 21-25 | I/Q 两路合成后的多尺度主脊线、局部频率扩展和脊线能量集中度 |

更合理的解释方式是基于组合关系，例如：

- `SpectralEntropyMean` 高且 `SpectralFlatnessMean` 高：可能表示局部频谱更分散或更噪声状；
- `SpectralCrestFactorMean` 高且 `FSSTRidgeEnergyRatio` 高：可能表示存在较清晰、集中的主导时频轨迹；
- `NormalizedSTFTRidgeFrequencyStd` 高且 `NormalizedFSSTRidgeFrequencyStd` 高：可能表示主频随时间变化明显；
- `NormalizedFSSTRidgeSlope` 高且 `NormalizedFSSTRidgeCurvatureRMS` 低：可能表示较接近线性扫频趋势；
- `NormalizedFSSTRidgeCurvatureRMS` 高：可能表示主脊线存在非线性弯曲、跳变或局部错选；
- `NormalizedSpectralSpreadMean` 高且 `NormalizedWSSTBandwidthMean` 高：可能表示局部频率扩展较宽，但仍需结合能量占比判断是否为噪声或多分量结构。

不建议让大模型做以下推断：

| 不建议做法 | 原因 |
|---|---|
| 只根据某一个特征判断信号体制 | 单个统计量无法唯一对应 LoRa、Wi-Fi、Bluetooth、4G、5G 或其他体制 |
| 把 STFT / FSST / WSST 脊线直接等同于严格瞬时频率 | 当前脊线来自最大功率 bin，可能受干扰、噪声或多分量切换影响 |
| 把 `NormalizedSpectralCentroidStd` 解释为瞬时频率标准差 | 该特征实际是 STFT 每帧谱质心的时间标准差，不是相位导数定义的瞬时频率 |
| 把 `NormalizedSpectralSpreadMean` 或 WSST bandwidth 解释为协议带宽 | 它们是时频图上的功率加权频率扩展，不是信道带宽或标准规定带宽 |
| 直接比较 WSST 频率符号与 STFT / FSST 频率符号 | 当前 WSST 对 I/Q 实值路分别计算，频率轴为非负频率，不保留复数基带正负频率方向 |
| 把高谱熵或高时频熵等同于高信息熵 | 这里的熵描述时频能量分布，不是通信编码熵或信息速率 |

这 25 个特征应被理解为一个固定顺序的时频统计证据向量：

```text
[SpectralKurtosisMean,
 SpectralKurtosisStd,
 SpectralSkewnessMean,
 SpectralSkewnessStd,
 SpectralCrestFactorMean,
 SpectralCrestFactorStd,
 SpectralFlatnessMean,
 SpectralFlatnessStd,
 SpectralEntropyMean,
 SpectralEntropyStd,
 NormalizedSTFTRidgeFrequencyStd,
 NormalizedSTFTRidgeSlope,
 NormalizedSpectralCentroidStd,
 NormalizedSpectralSpreadMean,
 NormalizedSpectralSpreadStd,
 NormalizedFSSTRidgeFrequencyStd,
 NormalizedFSSTRidgeSlope,
 NormalizedFSSTRidgeCurvatureRMS,
 FSSTRidgeEnergyRatio,
 NormalizedFSSTTimeFrequencyEntropy,
 NormalizedWSSTRidgeFrequencyStd,
 NormalizedWSSTRidgeSlope,
 NormalizedWSSTBandwidthMean,
 NormalizedWSSTBandwidthStd,
 WSSTRidgeEnergyRatio]
```

在用于大模型理解时，应始终保留特征顺序、计算口径、采样率归一化方式和变换配置。更稳妥的表述应使用“可能提示”“倾向于说明”“需要结合其他特征判断”，避免把单个特征值过度解释为确定的调制方式、通信协议或信号体制。
