# LTE/DVB-T 周期门控证据说明

## 1. 用途与边界

本文档只解释 `periodicity_evidence` 和已经冻结的
`deterministic_fusion_result`。周期分支是一个受限的 LTE/DVB-T 物理证据门控，
不是新的三分类器，也不能识别 WiFi。

Hermes 只能解释代码已经给出的门控结果，不得重新选择候选周期、修改门限、
重算门控或改变最终标签与复核状态。

## 2. 测量方法

系统在完整连续 region 上去除复均值，然后在候选延迟附近计算能量归一化复
自相关幅值：

`|sum(conj(x[n]) * x[n + lag]))| / sqrt(E0 * Elag)`

每个候选在名义延迟的正负 2 个采样点范围内取最大值。归一化相关值位于
`0...1`，只表示当前延迟处的重复相关强度，不是类别概率或置信度。

当前 1 MS/s、4096 点合同使用三个候选：

| candidate_id | 名义周期 | 当前作用 |
|---|---:|---|
| `period_66p67us` | 66.67 μs | LTE 候选 score |
| `period_224us` | 224 μs | DVB-T 2K 候选 score |
| `period_896us` | 896 μs | DVB-T 8K 候选 score |

这些周期来自 OFDM 有效符号时间尺度，只构成候选物理证据。该测量不等同于
完整循环前缀检测、同步序列匹配、导频检测或协议解码。

## 3. Score、预测与 margin

- `lte_score`：`period_66p67us` 的归一化相关值；
- `dvbt_score`：`period_224us` 与 `period_896us` 相关值中的较大者；
- `prediction_label`：两者中 score 较大的一类；
- `margin`：`abs(lte_score - dvbt_score)`。

margin 只描述两个候选 score 的分离程度。它不是概率，也不能跨越当前采样率、
region 长度和预处理合同任意复用。

`reliability_margin` 只由开发验证集确定并冻结。`reliable=true` 表示当前 margin
达到该门限；不表示周期预测在所有环境和噪声条件下必然正确。

## 4. 门控状态

- `eligible`：三个 IQ 成员的 Top1 只包含 LTE/DVB-T，且两类都出现；
- `reliable`：周期 margin 达到冻结门限；
- `resolved`：同时满足 `eligible` 和 `reliable`；
- `changed`：resolved 后周期标签与 ensemble 标签不同，代码已经完成改判。

周期分支不得修改：

- 三个成员一致的判决；
- ensemble 的 WiFi 判决；
- 任一成员投票包含 WiFi 的冲突。

## 5. 确定性结果

`deterministic_fusion_result` 是代码产生的唯一操作结论：

- `accept`：存在可自动接受的 `final_label`；
- `review_required`：`final_label` 为 null，ensemble 标签只作为
  `provisional_label`，需要外部复核；
- `ensemble_unanimous`：采用一致的 ensemble 标签；
- `ensemble_confirmed_by_periodicity`：成员有 LTE/DVB-T 分歧，但可靠周期证据
  与 ensemble 标签一致；
- `periodicity_changed_label`：可靠周期证据已经改变 ensemble 标签；
- `unresolved_review`：周期门控不能解决成员分歧。

64 维全局特征可以说明物理背景、异常和局限，但不能覆盖这个冻结结果，也不能
把 `review_required` 私自变成 accept。

## 6. 解释限制

1. 4.096 ms / 1 MHz 只提供局部短时观测。
2. 相关峰可能受到噪声、干扰、信道、频偏和有限样本影响。
3. 当前候选只服务于已验证的 LTE/DVB-T 分歧门控，不可外推到其他体制。
4. `reliable=false` 表示证据不足，不表示相反类别成立。
5. WiFi 保护规则是有意的安全边界，不能利用周期预测绕过。
