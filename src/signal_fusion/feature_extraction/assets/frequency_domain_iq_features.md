# 25维频域 IQ 特征说明

本文档说明一个单段复数 IQ 信号的 25 维频域特征提取器。特征由两部分组成：

- `1-11`: Welch PSD 频谱统计特征
- `12-25`: 双谱、平方双相干和双相位特征

这些特征用于描述一个 IQ 信号段的频域形状、频率分布、三阶频率耦合和相位耦合结构。它们是可解释的统计描述，不是调制方式或通信体制标签。

## 1. 输入说明

函数输入为一个复数 IQ 向量和采样率：

```matlab
x
sampleRate
```

其中：

| 项目         | 说明                                      |
| ------------ | ----------------------------------------- |
| `x`          | 复数 IQ 序列                              |
| `sampleRate` | 采样率，单位为 Hz                         |
| 输入形状     | `x` 为一维向量，行向量或列向量均可        |
| 输入含义     | 一个完整的 IQ 信号段                      |
| 实部         | I 路采样                                  |
| 虚部         | Q 路采样                                  |
| 最小长度     | 当前建议实现中不小于 8 个采样点           |
| 最大长度     | 当前建议实现中限制为不超过 16384 个采样点 |

输入会被统一转换为列向量并转为复数 `single` 类型：

```matlab
xIn = complex(single(real(x(:))), single(imag(x(:))));
```

`sampleRate` 必须为正的有限标量。如果输入为空、长度不满足限制，或实部 / 虚部包含 `NaN` / `Inf`，函数应报错。

## 2. 预处理说明

### 2.1 去除直流分量

所有频域分析前先去除整段 IQ 的直流分量：

```matlab
x0 = xIn - mean(xIn);
```

这样可以减少零频附近的直流偏置对 PSD、双谱和双相干统计的影响。

### 2.2 数值保护

实现中使用极小正数：

```matlab
tinyValue = single(1.0e-20);
```

用于避免除零、`log(0)` 和极弱信号导致的非有限值。最终输出会将 `NaN` 和 `Inf` 替换为 `0`。

### 2.3 极弱或零信号

如果 Welch PSD 总功率权重小于等于 `tinyValue`，输出保持固定 `1×25 single` 尺寸，其中第 6 项 `Log10 band power` 设为 `log10(tinyValue)`，其余项保持为 `0`。

## 3. Welch PSD 配置

PSD 部分使用中心化双边 Welch 功率谱估计。

```text
maximumWindowLength = 1024
windowLength        = min(signalLength, 1024)
window              = periodic Hamming window
overlapLength       = floor(windowLength / 2)
nfft                = 4096
PSD type            = centered two-sided Welch PSD
```

MATLAB 实现口径：

```matlab
[powerSpectrum, frequencyAxis] = pwelch( ...
    x0, window, overlapLength, nfft, sampleRate, 'centered');
```

`frequencyAxis` 是中心化双边频率轴，范围约为：

```text
-sampleRate/2 ... +sampleRate/2
```

频率类特征统一输出为：

```text
normalizedFrequency = frequencyHz / sampleRate
```

因此归一化频率大致落在 `[-0.5, 0.5)`，归一化带宽大致落在 `[0, 1]`。

## 4. 双谱配置

双谱部分独立于 Welch PSD 计算，不使用 PSD 的 `nfft = 4096`。

```text
minimumBispectrumSignalLength = 512
bispectrumWindowLength        = 256
bispectrumHopLength           = 128
bispectrumNfft                = 256
overlap                       = 50%
maximumBispectrumSegments     = 127
```

当 `length(x) < 512` 时，不计算双谱、双相干和双相位特征，第 `12-25` 项保持为 `0`。

## 5. 输出说明

函数输出为固定长度的单精度特征向量：

```matlab
features
```

输出格式：

| 项目     | 说明                          |
| -------- | ----------------------------- |
| 输出类型 | `single`                      |
| 输出尺寸 | `1×25`                        |
| 输出含义 | 当前 IQ 段的 25个频域统计特征 |
| 顺序要求 | 特征顺序固定，不能随意调换    |

固定输出顺序如下：

| 序号 | 英文名称 | 中文名称 | 特征组 |
|---:|---|---|---|
| 1 | Normalized mean frequency | 归一化平均频率 | PSD |
| 2 | Normalized median frequency | 归一化中值频率 | PSD |
| 3 | Normalized spectral spread | 归一化频谱扩展 | PSD |
| 4 | Normalized 99% occupied bandwidth | 归一化 99% 占用带宽 | PSD |
| 5 | Normalized peak frequency | 归一化峰值频率 | PSD |
| 6 | Log10 band power | 对数带内功率 | PSD |
| 7 | Global normalized spectral entropy | 全局归一化频谱熵 | PSD |
| 8 | Global spectral crest factor | 全局频谱峰值因子 | PSD |
| 9 | Global spectral flatness | 全局频谱平坦度 | PSD |
| 10 | Normalized spectral skewness | 归一化频谱偏度 | PSD |
| 11 | PSD standard deviation | PSD 标准差 | PSD |
| 12 | Log mean bispectrum magnitude | 对数平均双谱幅值 | Bispectrum |
| 13 | Log maximum bispectrum magnitude | 对数最大双谱幅值 | Bispectrum |
| 14 | Log bispectral energy | 对数双谱能量 | Bispectrum |
| 15 | Bispectral crest factor | 双谱峰值因子 | Bispectrum |
| 16 | Normalized bispectral entropy | 归一化双谱熵 | Bispectrum |
| 17 | Normalized bispectrum peak frequency 1 | 归一化双谱峰值频率 1 | Bispectrum |
| 18 | Normalized bispectrum peak frequency 2 | 归一化双谱峰值频率 2 | Bispectrum |
| 19 | Mean squared bicoherence | 平均平方双相干 | Bicoherence |
| 20 | Maximum squared bicoherence | 最大平方双相干 | Bicoherence |
| 21 | Standard deviation of squared bicoherence | 平方双相干标准差 | Bicoherence |
| 22 | Normalized bicoherence entropy | 归一化双相干熵 | Bicoherence |
| 23 | Bicoherence-weighted biphase concentration | 双相干加权双相位集中度 | Biphase |
| 24 | Dominant biphase cosine | 主导双相位余弦 | Biphase |
| 25 | Dominant biphase sine | 主导双相位正弦 | Biphase |

## 6. 特征解释

### 1. Normalized mean frequency

**计算对象：**中心化双边 Welch PSD。

**实现口径：**

```matlab
meanFrequency = sum(frequencyAxis .* powerSpectrum) / sum(powerSpectrum)
```

输出为 `meanFrequency / sampleRate`。

**含义：**PSD 的功率加权频谱质心。

**解释作用：**用于描述信号能量整体偏向负频率、零频附近还是正频率。

**注意事项：**它不是载波频率估计值。IQ 已经去直流，频率轴是中心化基带频率；频偏、镜像或不对称频谱都会改变该值。

### 2. Normalized median frequency

**计算对象：**中心化双边 Welch PSD 的累计功率分布。

**实现口径：**找到累计 PSD 功率首次达到 `50%` 的频率点，输出为该频率除以 `sampleRate`。

**含义：**将频谱总功率分成左右两半的频率位置。

**解释作用：**与平均频率一起判断频谱能量的中心位置和偏斜情况。

**注意事项：**该值受 PSD 频率网格影响，是离散 bin 上的结果；不能单独解释为通信信号的中心频点。

### 3. Normalized spectral spread

**计算对象：**中心化双边 Welch PSD。

**实现口径：**先计算平均频率，再计算 `sqrt(sum((f - meanFrequency)^2 .* P) / sum(P))`，输出除以 `sampleRate`。

**含义：**PSD 围绕频谱质心的功率加权标准差。

**解释作用：**用于描述频谱能量分布的宽窄程度。

**注意事项：**它描述围绕质心的扩散程度，不等同于 99% 占用带宽；离散谱线、噪声底和频偏都会影响它。

### 4. Normalized 99% occupied bandwidth

**计算对象：**中心化双边 Welch PSD 的累计功率分布。

**实现口径：**取累计功率达到 `0.5%` 和 `99.5%` 的两个频率点，二者差值除以 `sampleRate`。

**含义：**包含中间 99% 频谱功率的频率范围。

**解释作用：**用于描述信号主要能量占据的频带宽度。

**注意事项：**弱噪声底或旁瓣会扩大该值；它是统计带宽，不是协议带宽或信道带宽的直接测量。

### 5. Normalized peak frequency

**计算对象：**中心化双边 Welch PSD。

**实现口径：**找到 `powerSpectrum` 最大值所在的频率 bin，输出为 `peakFrequency / sampleRate`。

**含义：**最强 PSD 谱峰对应的归一化频率位置。

**解释作用：**用于定位最突出频谱成分的位置。

**注意事项：**最强峰可能来自真实信号结构，也可能来自窄带干扰、残余直流或估计噪声；需要结合带宽、熵和平坦度一起解释。

### 6. Log10 band power

**计算对象：**中心化双边 Welch PSD。

**实现口径：**

```matlab
bandPower = sum(powerSpectrum) * frequencyResolution
```

输出 `log10(max(bandPower, tinyValue))`。

含义：对 PSD 积分得到的总带内功率，并以 `log10` 压缩动态范围。

解释作用：用于描述信号段的整体频域能量尺度。

注意事项：这是非归一化功率特征，会受到输入幅度、接收增益和采样率单位影响；不应直接与归一化形状特征混为一类。

### 7. Global normalized spectral entropy

**计算对象：**归一化 Welch PSD。

**实现口径：**令 `p_k = P_k / sum(P)`，计算 `-sum(p_k log2 p_k) / log2(K)`。

**含义：**频谱能量在频率 bin 上的分散程度，通常约在 `[0, 1]`。

**解释作用：**低值表示能量集中在少数频点，高值表示能量分布更分散或更接近噪声状。

**注意事项：**高熵不等于调制更复杂；噪声、低信噪比或宽带干扰也会提高频谱熵。

### 8. Global spectral crest factor

**计算对象：**中心化双边 Welch PSD。

**实现口径：**

```matlab
max(powerSpectrum) / mean(powerSpectrum)
```

**含义：**最大谱峰相对于平均 PSD 水平的突出程度。

**解释作用：**用于区分尖峰明显的频谱和较平缓的频谱。

**注意事项：**该特征对窄带干扰、频率分辨率和窗函数泄漏敏感；它描述谱峰突出度，不直接代表信号功率大小。

### 9. Global spectral flatness

**计算对象：**中心化双边 Welch PSD。

**实现口径：**计算 PSD 的几何平均值与算术平均值之比，`geometricMeanPSD / meanPSD`；实现中使用谱地板避免 `log(0)`。

**含义：**频谱是否接近平坦，通常约在 `[0, 1]`。

**解释作用：**接近 `0` 表示存在明显谱峰，接近 `1` 表示频谱更平坦或更噪声状。

**注意事项：**平坦度和频谱熵相关但不相同。平坦度强调几何平均与算术平均的差异，熵强调概率分布的不确定性。

### 10. Normalized spectral skewness

**计算对象：**归一化 Welch PSD 和中心化频率轴。

**实现口径：**`sum(p_k * (f_k - meanFrequency)^3) / spectralSpread^3`，该值本身无量纲，不再除以 `sampleRate`。

**含义：**PSD 围绕频谱质心的左右不对称程度。

**解释作用：**接近 `0` 表示频谱相对对称；正值表示能量尾部偏向正频率侧；负值表示偏向负频率侧。

**注意事项：**符号依赖中心化频率轴方向；频偏、镜像不平衡或截取片段不同都可能改变偏度。

### 11. PSD standard deviation

**计算对象：**中心化双边 Welch PSD 的原始数值。

**实现口径：**

```matlab
std(powerSpectrum)
```

**含义：**PSD 在不同频率点上的功率起伏程度。

**解释作用：**用于描述频谱整体起伏是否明显，保留绝对 PSD 数值尺度。

**注意事项：**该特征未归一化，会受到接收增益、输入幅度和整体功率影响；使用时应与 `Log10 band power` 一起归为功率尺度类特征。

### 12. Log mean bispectrum magnitude

**计算对象：**有效双谱频率区域内的 `|B(f1,f2)|`。

**实现口径：**先计算所有有效点的平均双谱幅值，再输出 `log10(max(meanMagnitude, tinyValue))`。

**含义：**整体平均三阶频率耦合幅度。

**解释作用：**用于描述信号在双谱平面上是否存在普遍的三频耦合强度。

**注意事项：**这是原始双谱幅值统计，仍受信号幅度和功率尺度影响；不应与平方双相干的归一化强度混淆。

### 13. Log maximum bispectrum magnitude

**计算对象：**有效双谱频率区域内的最大 `|B(f1,f2)|`。

**实现口径：**找到最大双谱幅值，输出 `log10(max(maximumMagnitude, tinyValue))`。

**含义：**最强三频耦合点的原始双谱幅值。

**解释作用：**用于发现双谱平面中最突出的耦合峰。

**注意事项：**该值对局部峰值、强窄带分量和输入幅度敏感；单独升高不能直接说明某种调制类型存在。

### 14. Log bispectral energy

**计算对象：**有效双谱频率区域内的 `|B(f1,f2)|^2`。

**实现口径：**`bispectralEnergy = sum(|B|^2)`，输出 `log10(max(bispectralEnergy, tinyValue))`。

**含义：**双谱平面上的总三阶耦合能量。

**解释作用：**用于描述整体双谱结构强弱。

**注意事项：**该值与输入幅度、有效双谱点数量和双谱配置有关；本实现中 `bispectrumNfft` 固定为 256，因此同一配置内可比。

### 15. Bispectral crest factor

**计算对象：**有效双谱频率区域内的 `|B(f1,f2)|`。

**实现口径：**`maximumBispectrumMagnitude / meanBispectrumMagnitude`。

**含义：**最强双谱峰相对于平均双谱幅值的突出程度。

**解释作用：**用于判断三阶耦合结构是集中在少数频率组合，还是分布较平均。

**注意事项：**高值表示双谱峰更集中，不等同于双相干更强；原始双谱幅值和归一化双相干需要分开解释。

### 16. Normalized bispectral entropy

**计算对象：**有效双谱频率区域内的 `|B(f1,f2)|^2` 分布。

**实现口径：**令 `p = |B|^2 / sum(|B|^2)`，计算 `-sum(p log2 p) / log2(validPointCount)`。

**含义：**双谱能量在二维频率平面上的分散程度。

**解释作用：**低值表示双谱能量集中在少数耦合点，高值表示双谱能量分布更广。

**注意事项：**高双谱熵不等于信号更复杂，也可能来自噪声或较分散的弱耦合结构；应结合双谱能量和双谱峰值因子解释。

### 17. Normalized bispectrum peak frequency 1

**计算对象：**最大 `|B(f1,f2)|` 所在的频率组合。

**实现口径：**记录最大双谱幅值点的第一个频率 bin `k1`，输出 `k1 / bispectrumNfft`。

**含义：**最强双谱耦合点的第一个归一化频率坐标。

**解释作用：与**第 18 项一起定位最强三频耦合结构所在的二维频率位置。

**注意事项：**该特征必须与第 18 项成对解释；由于实现只计算 `index2 >= index1` 的半平面，两个坐标不能任意交换解释。

### 18. Normalized bispectrum peak frequency 2

**计算对象：**最大 `|B(f1,f2)|` 所在的频率组合。

**实现口径：**记录最大双谱幅值点的第二个频率 bin `k2`，输出 `k2 / bispectrumNfft`。

**含义：**最强双谱耦合点的第二个归一化频率坐标。

**解释作用：**与第 17 项共同描述最强耦合峰的位置，隐含第三个频率 `f3 = f1 + f2`。

**注意事项：**不要单独解释第 17 或第 18 项；二者共同表示一个频率组合，而不是两个独立峰值频率。

### 19. Mean squared bicoherence

**计算对象：**有效双谱频率区域内的平方双相干 `b2(f1,f2)`。

**实现口径：**对所有有效点的 `b2` 求平均。

**含义：**整体归一化三频相位耦合一致性，理论范围为 `[0, 1]`。

**解释作用：**用于判断信号是否在较多频率组合上存在稳定的二次相位耦合关系。

**注意事项：**低值不一定表示没有结构，可能只是耦合集中在少数点；高值也需要结合最大值、标准差和熵判断。

### 20. Maximum squared bicoherence

**计算对象：**有效双谱频率区域内的最大 `b2(f1,f2)`。

**实现口径：**在所有有效频率组合中取 `squaredBicoherence` 最大值。

**含义：**最强归一化三频相位耦合点的强度。

**解释作用：**用于识别是否存在局部非常稳定的相位耦合关系。

**注意事项：**单个最大值容易受低样本数、低功率频点或局部异常影响；不应只凭这一项判断信号体制。

### 21. Standard deviation of squared bicoherence

**计算对象：**有效双谱频率区域内的 `b2(f1,f2)`。

**实现口径：**使用有效点上的样本标准差，`sqrt(max(variance, 0))`。

**含义：**平方双相干在二维频率平面上的起伏程度。

**解释作用：**用于区分耦合强度是均匀分布，还是集中在少数显著频率组合。

**注意事项：**它需要结合平均值和最大值解释。平均值低但标准差高，可能表示少数耦合点突出。

### 22. Normalized bicoherence entropy

**计算对象：**有效双谱频率区域内的 `b2(f1,f2)` 分布。

**实现口径：**令 `q = b2 / sum(b2)`，计算 `-sum(q log2 q) / log2(validPointCount)`。

**含义：**归一化相位耦合强度在二维频率平面上的分散程度。

**解释作用：**低值表示双相干集中在少数频率组合，高值表示双相干分布更分散。

**注意事项：**如果整体 `b2` 很弱，熵值的解释意义会降低；需要同时查看平均平方双相干和最大平方双相干。

### 23. Bicoherence-weighted biphase concentration, 

**计算对象：**有效双谱点的双相位 `angle(B)` 和对应平方双相干 `b2`。

**实现口径：**用单位复数 `B / |B|` 表示双相位，计算 `abs(sum(b2 * B / |B|)) / sum(b2)`。

**含义：**强双相干点的双相位方向是否集中，范围通常为 `[0, 1]`。

**解释作用：**接近 `1` 表示强耦合点的双相位较一致，接近 `0` 表示双相位方向分散。

**注意事项：**它描述相位方向的一致性，不描述耦合强度大小；应与第 19-22 项一起解释。

### 24. Dominant biphase cosine

**计算对象：**最大平方双相干位置对应的双谱相位。

**实现口径：**取最大 `b2` 位置的 `B`，输出 `real(B / |B|)`，即 `cos(phi)`。

**含义：**主导耦合点双相位的余弦分量。

**解释作用：**与第 25 项一起表示主导双相位方向，避免直接输出角度带来的跳变。

**注意事项：**不能单独解释余弦分量；如需角度，应使用 `atan2(feature25, feature24)`。

### 25. Dominant biphase sine, 主导双相位正弦

计算对象：最大平方双相干位置对应的双谱相位。

实现口径：取最大 `b2` 位置的 `B`，输出 `imag(B / |B|)`，即 `sin(phi)`。

含义：主导耦合点双相位的正弦分量。

解释作用：与第 24 项共同表示主导双相位方向。

注意事项：不能单独解释正弦分量；第 24 和第 25 项应作为一个二维相位向量使用。

## 7. 大模型使用建议

### 按特征组解释

LLM 解释这些特征时，应优先按特征组理解，而不是孤立解释单个数值：

- `1-5`: 频率位置和频谱宽度，回答“能量在哪里、占多宽”。
- `6` 和 `11`: 原始功率尺度和 PSD 起伏，回答“整体能量和谱功率起伏有多大”。
- `7-10`: PSD 形状统计，回答“频谱是集中、平坦、分散还是偏斜”。
- `12-18`: 原始双谱幅值和双谱峰位置，回答“三阶频率耦合是否强、集中在哪里”。
- `19-22`: 平方双相干统计，回答“归一化三频相位耦合是否稳定、分布是否集中”。
- `23-25`: 双相位结构，回答“强耦合点的相位方向是否一致、主导双相位在哪里”。

### 避免过度解释单个特征

不要把某一个特征值直接等同于某种调制方式、协议或通信体制。例如：

- 高 PSD 熵不直接等于 OFDM。
- 高双相干不直接等于某一种调制。
- 主导双相位的某个角度不直接代表某个星座图。
- 峰值频率不直接等于载波频率。

更稳妥的解释方式是基于多个特征共同形成的模式：

```text
频谱宽度 + 频谱熵 + 频谱平坦度
原始功率尺度 + PSD 起伏
双谱能量 + 双谱熵 + 双谱峰值因子
平均双相干 + 最大双相干 + 双相干熵
双相位集中度 + 主导双相位向量
```

### 结合数据条件解释

解释时应同时考虑：

- 输入信号长度是否小于 512。若小于 512，第 `12-25` 项为 `0`，这表示未计算双谱特征，不表示没有双谱结构。
- 信号幅度、接收增益和归一化方式。第 `6`、`11`、`12`、`13`、`14` 对幅度尺度敏感。
- 信噪比和干扰。噪声会提高熵和平坦度，也可能制造不稳定峰值。
- 频偏和 IQ 不平衡。它们会影响平均频率、中值频率、偏度和峰值频率。
- 分段配置固定。PSD 使用 `4096` 点 FFT，双谱使用 `256` 点 FFT；解释结果时应保持同一配置下比较。

