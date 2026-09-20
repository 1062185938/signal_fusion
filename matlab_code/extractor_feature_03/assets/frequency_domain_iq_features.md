# 21维频域 IQ 特征说明

本文档对应 `extractFrequencyFeatures.m`。输出包含 11 个 Welch PSD 特征和 10 个双谱/双相干特征，共 `1×21 single`。这些特征描述频谱形状和高阶频率耦合，不是调制或协议标签。

## 1. 输入与配置

- 输入：复数 IQ 向量 `x` 与正有限采样率 `sampleRate`。
- 有效长度：`32 <= length(x) <= 16384`，与统一 C ABI 保持一致。
- 预处理：`x0 = x - mean(x)`。
- Welch PSD：周期 Hamming 窗，窗长 `min(N,1024)`，50% 重叠，`nfft=4096`，中心化双边频率轴。
- 双谱：仅在 `N>=512` 时计算，窗长 256、步长 128、`nfft=256`。
- 频率归一化：`frequencyHz/sampleRate`；有符号频率大致位于 `[-0.5,0.5)`。

当 `N<512` 时，第 12-21 项为 0，含义是“未计算双谱”，不是“确认不存在高阶耦合”。

旧版已删除以下项：

- `Bispectral crest factor`：可由 `10^(logMax-logMean)` 精确推导。
- `Standard deviation of squared bicoherence`：与均值/最大值/熵提供的信息高度重叠。
- `Dominant biphase cosine/sine`：会随整段 IQ 的任意整体相位旋转而改变，不适合作为稳定类别证据。

## 2. 输出顺序

| 序号 | 英文名称 | 中文名称 |
|---:|---|---|
| 1 | Normalized mean frequency | 归一化平均频率 |
| 2 | Normalized median frequency | 归一化中值频率 |
| 3 | Normalized spectral spread | 归一化频谱扩展 |
| 4 | Normalized 90% occupied bandwidth | 归一化 90% 占用带宽 |
| 5 | Normalized peak frequency | 归一化峰值频率 |
| 6 | Log10 band power | 对数带内功率 |
| 7 | Global normalized spectral entropy | 全局归一化频谱熵 |
| 8 | Global spectral crest factor | 全局频谱峰值因子 |
| 9 | Global spectral flatness | 全局频谱平坦度 |
| 10 | Normalized spectral skewness | 归一化频谱偏度 |
| 11 | PSD standard deviation | PSD 标准差 |
| 12 | Log mean bispectrum magnitude | 对数平均双谱幅值 |
| 13 | Log maximum bispectrum magnitude | 对数最大双谱幅值 |
| 14 | Log bispectral energy | 对数双谱能量 |
| 15 | Normalized bispectral entropy | 归一化双谱熵 |
| 16 | Normalized bispectrum peak frequency 1 | 归一化双谱峰值频率 1 |
| 17 | Normalized bispectrum peak frequency 2 | 归一化双谱峰值频率 2 |
| 18 | Mean squared bicoherence | 平均平方双相干 |
| 19 | Maximum squared bicoherence | 最大平方双相干 |
| 20 | Normalized bicoherence entropy | 归一化双相干熵 |
| 21 | Bicoherence-weighted biphase concentration | 双相干加权双相位集中度 |

## 3. 特征解释

### 1. Normalized mean frequency

Welch PSD 的功率加权质心除以采样率。正负号描述基带能量偏向，不是射频载波频率。

### 2. Normalized median frequency

累计 PSD 首次达到 50% 的频率除以采样率。它是离散频率 bin 上的统计中位位置。

### 3. Normalized spectral spread

PSD 围绕频谱质心的功率加权标准差除以采样率，描述频谱能量扩散程度。

### 4. Normalized 90% occupied bandwidth

累计功率 5% 到 95% 两个频率点的差除以采样率。相较旧版 99% 带宽，它对噪声底和边缘泄漏更稳健，但仍不是协议规定带宽。

### 5. Normalized peak frequency

最大 PSD bin 的中心化有符号频率除以采样率。窄带干扰或残余偏置也可能决定该值。

### 6. Log10 band power

```text
log10(sum(PSD) * frequencyResolution)
```

保留绝对功率尺度，受输入幅度、接收增益和归一化影响。

### 7. Global normalized spectral entropy

对归一化 PSD 计算香农熵并除以 `log2(4096)`。高值表示能量更分散，也可能由噪声造成。

### 8. Global spectral crest factor

`max(PSD)/mean(PSD)`，描述最强谱峰相对平均谱水平的突出程度。

### 9. Global spectral flatness

PSD 几何平均与算术平均之比。接近 1 表示更平坦，接近 0 表示峰值更明显。

### 10. Normalized spectral skewness

PSD 关于频谱质心的标准化三阶矩。它本身无量纲，符号描述频谱左右不对称方向。

### 11. PSD standard deviation

直接计算原始 Welch PSD bin 的样本标准差，保留功率尺度，应与 Log10 band power 一起解释。

### 12. Log mean bispectrum magnitude

有效双谱区域内平均 `|B(f1,f2)|` 的 `log10`。描述平均三阶耦合幅度，受信号尺度影响。

### 13. Log maximum bispectrum magnitude

有效双谱区域内最大 `|B|` 的 `log10`，描述最强局部耦合峰。

### 14. Log bispectral energy

`log10(sum(|B|^2))`，描述双谱平面的总高阶耦合能量。

### 15. Normalized bispectral entropy

以 `|B|^2` 归一化为概率分布后计算二维熵。低值表示能量更集中，高值表示更分散。

### 16. Normalized bispectrum peak frequency 1

最大双谱幅值点的第一个中心化频率 bin，输出 `k1/256`。它可以为负值。

### 17. Normalized bispectrum peak frequency 2

最大双谱幅值点的第二个中心化频率 bin，输出 `k2/256`。必须与第 16 项共同解释，并满足第三频率 `k3=k1+k2` 在 Nyquist 范围内。

### 18. Mean squared bicoherence

有效频率组合上的平方双相干均值，理论范围 `[0,1]`，描述整体归一化三频相位耦合一致性。

### 19. Maximum squared bicoherence

平方双相干最大值，描述最强局部耦合。短数据或低功率频点可能使最大值不稳定。

### 20. Normalized bicoherence entropy

将平方双相干归一化后计算熵，描述耦合强度在二维频率平面中的分散程度。

### 21. Bicoherence-weighted biphase concentration

```text
|sum(b2 * B/|B|)| / sum(b2)
```

描述强耦合点的双相位方向是否集中。该幅值对统一相位旋转更稳健，但仍需结合双相干强度解释。

## 4. 解释边界

- PSD 熵、平坦度和峰值因子相互相关，不能视为三条完全独立证据。
- 双谱幅值类特征受幅度尺度影响；平方双相干是归一化耦合指标，两者不能混为一类。
- 双谱峰值坐标使用中心化有符号频率，不能按未中心化 `0...1` 坐标解释。
- 对多载波、低信噪比或短片段，任何单一峰值和脊线都可能不稳定。
