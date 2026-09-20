# 15维 IQ 时域与相关性特征说明

本文档对应 `extractTimeFeatures.m`。输入是一个复数 IQ 片段，输出为固定顺序的 `1×15 single` 特征向量。所有特征都是统计证据，不是协议或调制类别的直接判决。

## 1. 输入与预处理

输入首先转换为列向量并去除复数均值：

```matlab
xIn = single(x(:));
x0 = xIn - mean(xIn);
absX0 = abs(x0);
```

实现使用 `eps('single')` 做数值保护，并在输出前将 `NaN` 或 `Inf` 替换为 0。

旧版的 `Standard deviation = std(x0)` 已删除。对去直流复数序列，它与 RMS 只相差样本/总体归一化因子，属于确定性冗余。`Envelope CV` 已修正为真正的包络标准差 `std(absX0)` 除以平均包络。

## 2. 输出顺序

| 序号 | 英文名称 | 中文名称 |
|---:|---|---|
| 1 | RMS | 均方根 |
| 2 | PAPR_dB | 峰均功率比 |
| 3 | Amplitude skewness | 幅度偏度 |
| 4 | Amplitude kurtosis | 幅度峰度 |
| 5 | Envelope CV | 包络变异系数 |
| 6 | Normalized amplitude entropy | 归一化幅度熵 |
| 7 | Normalized C20 | 归一化 C20 |
| 8 | Normalized C40 | 归一化 C40 |
| 9 | Normalized C41 | 归一化 C41 |
| 10 | Normalized C42 | 归一化 C42 |
| 11 | Differential phase standard deviation | 差分相位标准差 |
| 12 | Autocorrelation peak magnitude | 归一化自相关峰值 |
| 13 | Autocorrelation peak lag ratio | 自相关峰值延迟比例 |
| 14 | Mean autocorrelation magnitude | 平均归一化自相关幅值 |
| 15 | Envelope-power autocorrelation peak | 包络功率自相关峰值 |

## 3. 特征解释

### 1. RMS

`rms(x0)`，描述去直流后信号的整体幅度或功率尺度。它受接收增益、距离和外部归一化影响；若输入已做 RMS 归一化，应把它视为质量控制特征。

### 2. PAPR_dB

```text
10*log10(max(|x0|^2) / mean(|x0|^2))
```

描述瞬时峰值相对平均功率的突出程度。高值可见于 OFDM，也可由脉冲干扰或异常点产生，不能单独判定体制。

### 3. Amplitude skewness

包络 `|x0|` 的标准化三阶中心矩，描述幅度分布的不对称程度。对异常点和短片段较敏感。

### 4. Amplitude kurtosis

包络 `|x0|` 的标准化四阶中心矩。当前实现是普通峰度，不减 3；数值较高通常说明尖峰或重尾更明显。

### 5. Envelope CV

```text
std(|x0|) / mean(|x0|)
```

描述包络相对于自身平均值的起伏，理论上对统一幅度缩放不敏感。这里使用 MATLAB 默认的样本标准差口径。

### 6. Normalized amplitude entropy

将包络范围划分为 10 个等宽区间，计算直方图香农熵并除以 `log2(10)`。它描述幅度分布离散程度，不是通信信息熵。

### 7. Normalized C20

```text
|E[x0^2]| / E[|x0|^2]
```

描述复信号二阶非圆对称结构。应与其他高阶统计量组合使用。

### 8. Normalized C40

```text
|M40 - 3*M20^2| / M21^2
```

描述复平面四阶累积结构，受信噪比、频偏、滤波和样本长度影响。

### 9. Normalized C41

```text
|M41 - 3*M20*M21| / M21^2
```

从三次原信号和一次共轭信号的组合描述四阶统计结构。

### 10. Normalized C42

```text
|M42 - |M20|^2 - 2*M21^2| / M21^2
```

描述四阶幅度相关累积结构。C20、C40、C41、C42 应作为一组解释。

### 11. Differential phase standard deviation

```text
std(angle(x0[n] * conj(x0[n-1])))
```

描述相邻采样差分相位的波动。它消除恒定初始相位，但仍受载波频偏、采样率和低幅度噪声影响。

### 12. Autocorrelation peak magnitude

在延迟 `1...min(N/2,512)` 内计算能量归一化复自相关幅值并取最大值：

```text
max_lag |sum(conj(x[n])*x[n+lag])| / sqrt(E1*E2)
```

高值提示片段中存在较强重复或周期结构，但不能直接声称已检测到循环前缀或特定符号周期。

### 13. Autocorrelation peak lag ratio

最大归一化自相关对应的延迟除以片段长度 `N`。它给出候选重复周期在当前片段中的相对位置；比较不同样本时应保持 `seq_len` 和采样率一致。

### 14. Mean autocorrelation magnitude

对扫描范围内的归一化复自相关幅值求均值。它描述周期相关性是集中在单一延迟，还是在多个延迟上普遍存在。

### 15. Envelope-power autocorrelation peak

对中心化包络功率 `|x0|^2 - mean(|x0|^2)` 做多延迟归一化自相关并取峰值。它关注功率包络的重复结构，与复 IQ 相位相关性互为补充。

## 4. 解释边界

- RMS、PAPR 和幅度分布特征可能高度相关，不应当作多条完全独立证据重复计票。
- 自相关峰值只表示候选周期性；确认循环前缀、符号周期或协议仍需要专用检测和多片段验证。
- 高阶累积量应成组解释，低信噪比或过短样本会降低稳定性。
- 所有比较都应同时考虑采样率、片段长度、幅度归一化和接收增益。
