# Technology Recognition V2：全区域基线

## 目的

V1 从每个原始文件的 268 个完整 4096 点区域中均匀抽取 32 个。排查发现这种抽样会与部分 WiFi 周期活动产生相位偏置，因此 V2 改为使用全部完整区域，验证模型是否能跨地点泛化。

本阶段只比较 512 和 4096 两种窗口长度，不修改模型，也不加入人工 AWGN。

## 数据

- 类别：LTE、WiFi、DVB-T
- 地点：gentbrugge、merelbeke、rabot、reep
- 原始文件：每个地点、每个类别各 10 个，共 120 个
- 采样率：1 MS/s
- IQ 格式：little-endian complex64
- region：4096 个复样本
- 每个文件使用 `floor(1,100,000 / 4096) = 268` 个连续完整 region
- 每个文件末尾不足 4096 点的 2272 个样本丢弃
- 不跳过文件开头，不做能量筛选
- 512 profile：每个 region 非重叠切成 8 个窗口
- 4096 profile：每个 region 对应 1 个窗口

每个 profile 共 32,160 个 region。512 profile 共 257,280 个窗口，4096 profile 共 32,160 个窗口。

## 地点轮换

| Fold | 训练地点 | 验证地点 | 测试地点 |
| --- | --- | --- | --- |
| fold1 | gentbrugge、merelbeke | rabot | reep |
| fold2 | gentbrugge、reep | merelbeke | rabot |
| fold3 | rabot、reep | gentbrugge | merelbeke |
| fold4 | merelbeke、rabot | reep | gentbrugge |

同一原始文件只属于一个 split。每折各类别数量相等：512 点训练/验证/测试窗口数为 128,640/64,320/64,320；4096 点为 16,080/8,040/8,040。

## 训练设置

- 模型：`deepconvnet_1d`
- 优化器：Adam，学习率 0.001
- 损失：交叉熵
- batch size：256
- 最大 30 epoch，early stopping patience 8
- seed：44
- 设备：CUDA，AMP 开启
- 训练 AWGN：关闭（probability = 0）
- checkpoint：按最低验证损失选择

## Clean 测试结果

| Window | Test location | 窗口准确率 | Region 准确率 |
| ---: | --- | ---: | ---: |
| 512 | reep | 96.214% | 99.900% |
| 512 | rabot | 96.706% | 99.154% |
| 512 | merelbeke | 82.993% | 90.460% |
| 512 | gentbrugge | 76.465% | 75.087% |
| 4096 | reep | 99.726% | 99.726% |
| 4096 | rabot | 99.677% | 99.677% |
| 4096 | merelbeke | 96.517% | 96.517% |
| 4096 | gentbrugge | 100.000% | 100.000% |

四折汇总：

| Window | 窗口准确率 mean ± std | Region 准确率 mean ± std |
| ---: | ---: | ---: |
| 512 | 88.094% ± 8.680% | 91.150% ± 9.989% |
| 4096 | 98.980% ± 1.427% | 98.980% ± 1.427% |

512 点的主要错误来自 DVB-T：四折类别平均准确率为 69.209%，其中 merelbeke 为 59.627%，gentbrugge 为 29.841%。同一 region 内的窗口若发生系统性误判，平均概率投票无法纠正。4096 点的四折类别平均准确率为 LTE 99.813%、WiFi 99.664%、DVB-T 97.463%。

V1 与 V2 的 4096 点四折均值几乎相同（98.958% 与 98.980%），说明 4096 点结论不依赖 32-region 抽样。V1 的 512 点结果则被抽样明显高估：V1 窗口均值 94.619%，V2 为 88.094%。

## 产物

- V2 数据：`data/external/processed/technology_recognition_lte_wifi_dvbt/v2/`
- 训练 manifest：`configs/training/technology_recognition_v2/`
- 模型与 ONNX：`outputs/technology_recognition_v2_all_regions/window_<size>/fold<1-4>/`
- 评估 JSON：`outputs/technology_recognition_v2_all_regions/clean_evaluation/ablation_evaluation.json`
- 评估 CSV：`outputs/technology_recognition_v2_all_regions/clean_evaluation/ablation_runs.csv`

## 当前结论

公开数据集的主基线应使用 4096 点输入。512 点保留为短窗口消融，不适合作为当前主模型。

## 4096 点增量 AWGN 测试

测试只在各折固定测试集推理前动态加入圆复高斯白噪声，不修改训练数据和模型。SNR 表示“原测试窗口总功率 / 新增噪声功率”，不是对原始空口 SNR 的估计。每个噪声条件使用 seed 44–48 共 5 次试验。

| 条件 | 四折准确率 mean ± std | 相对 clean |
| --- | ---: | ---: |
| clean | 98.980% ± 1.427% | 0.000 pp |
| 10 dB | 99.251% ± 0.661% | +0.271 pp |
| 7.5 dB | 99.093% ± 0.561% | +0.113 pp |
| 5 dB | 98.216% ± 1.457% | -0.764 pp |

逐地点结果：

| 测试地点 | clean | 10 dB | 7.5 dB | 5 dB |
| --- | ---: | ---: | ---: | ---: |
| reep | 99.726% | 99.296% | 98.505% | 95.726% |
| rabot | 99.677% | 99.512% | 99.129% | 98.674% |
| merelbeke | 96.517% | 98.194% | 98.751% | 99.132% |
| gentbrugge | 100.000% | 100.000% | 99.985% | 99.331% |

5 dB 下四折类别平均准确率为 LTE 99.371%、WiFi 95.507%、DVB-T 99.769%。主要退化来自 reep 的 WiFi，其准确率从 clean 的 99.552% 降到 87.851%。

Merelbeke 的 DVB-T 在 clean 下为 90.000%，加入 5 dB 噪声后反而达到 99.261%。这说明 clean 错误更可能来自模型对地点特有确定性细节的敏感，而非单纯缺少信噪比；噪声扰动在这里产生了平滑效果。该现象不能解释为噪声通常会提升性能。

总体上，当前 4096 点模型在新增 5 dB AWGN 下仍保持 98.216% 四折均值，暂时没有证据要求立即进行噪声增强重训。AWGN 汇总结果位于 `outputs/technology_recognition_v2_all_regions/awgn_summary_4096/`。

## 未使用 1 MS/s 录制的外部测试

复用现有 benchmark 数据入口，将此前未进入训练、验证或测试的 44 个原始文件组成独立 clean 测试集：UZ 和 iGent 的 40 个 LTE/DVB-T 文件、iGent 的 1 个 WiFi 文件，以及 Merelbeke 5180 MHz 的 3 个 WiFi 文件。全部文件按连续非重叠 4096 点 region 处理，不抽样，共 11,641 个样本。

| 模型 | 全集准确率 | 类别宏平均 |
| --- | ---: | ---: |
| fold1 | 99.691% | 99.726% |
| fold2 | 84.546% | 88.507% |
| fold3 | 85.826% | 89.445% |
| fold4 | 87.853% | 90.952% |
| 四模型概率平均 | 86.419% | 89.886% |

四模型概率平均在 iGent 为 99.708%，Merelbeke 5180 MHz WiFi 为 100%，但在 UZ 只有 70.802%。UZ 的 LTE 为 99.963%，主要问题是 UZ DVB-T 只有 41.642%；其中多个原始文件几乎全部被误判。四模型在全集上的预测完全一致率为 83.266%，在 UZ 只有 65.709%。

因此，该测试没有显示普遍的类别识别失败，而是揭示了显著的 UZ DVB-T 录制批次偏移。fold1 对同一批数据仍达到 99.691%，也说明四个训练地点组合学到的判别依据并不一致。下一步应先检查 UZ DVB-T 与训练地点 DVB-T 的数据分布和采集条件，再决定是否把这些外部文件加入训练；不应直接用这 44 个文件重训并同时继续把它们当外部测试集。

- 外部测试数据：`data/external/processed/technology_recognition_lte_wifi_dvbt/external_1msps_4096/test.npz`
- 评估结果：`outputs/technology_recognition_v2_all_regions/external_1msps_evaluation/`

## 四地点联合训练

为排除两地点训练组合造成的偶然性，使用四个核心地点共同训练一个 4096 点模型。每个地点、每个类别按原始文件拆分：r01–r08 训练，r09 验证，r10 内部测试。三个 split 分别包含 96、12、12 个互不重叠的源文件，对应 25,728、3,216、3,216 个类别均衡样本。UZ、iGent 和 Merelbeke 5180 MHz WiFi 仍只用于外部测试。

- 内部 clean 测试准确率：98.912%
- 外部 clean 准确率：85.293%
- 外部 10 dB 增量 AWGN 准确率：85.242%
- 外部 clean 类别准确率：LTE 99.459%、WiFi 100%、DVB-T 67.691%
- 外部 clean 地点准确率：iGent 99.507%、Merelbeke 100%、UZ 68.563%
- UZ clean 类别准确率：LTE 99.925%、DVB-T 37.201%

四地点联合训练没有修复 UZ DVB-T：r02、r03、r05、r06 仍为 0%，r04 为 5.970%，且错误预测置信度很高。10 dB AWGN 对外部准确率没有实质影响，支持“确定性的频谱/相位域偏移”而不是单纯噪声不足这一判断。

- 数据清单：`configs/training/technology_recognition_v2/window_4096_all_locations.json`
- 固定数据集：`data/external/processed/technology_recognition_lte_wifi_dvbt/v2_all_locations/window_4096/`
- 模型：`outputs/technology_recognition_v2_all_locations/window_4096/`
- 外部评估：`outputs/technology_recognition_v2_all_locations/external_1msps_evaluation/`

## 62 维特征分类先导实验

为判断现有 MATLAB/C ABI 物理特征能否修复跨地点错误，保持特征提取算法不变，对 4096 点完整 region 做了一次小规模 clean 基线。核心四地点仍按 r01–r08/r09/r10 划分训练、验证和内部测试；每个训练源文件均匀取 4 个 region，每个验证、测试源文件也取 4 个。外部 44 个源文件各均匀取 8 个 region，以便逐文件观察 UZ 和 iGent。

- 核心训练：384 个 region，每类 128 个
- 核心验证：48 个 region，每类 16 个
- 核心内部测试：48 个 region，每类 16 个
- 外部测试：352 个 region；LTE 160、WiFi 32、DVB-T 160
- 分类器：训练集均值/标准差归一化后的线性 `62 -> 3` 分类器
- 训练设备：CUDA

| 测试范围 | 总体 | LTE | WiFi | DVB-T |
| --- | ---: | ---: | ---: | ---: |
| 核心内部测试 | 100.000% | 100.000% | 100.000% | 100.000% |
| 外部测试 | 62.216% | 60.000% | 100.000% | 56.875% |

外部结果呈明显的地点依赖：iGent DVB-T 为 100%，但 iGent LTE 只有 20%；UZ LTE 为 100%，但 UZ DVB-T 只有 13.75%。UZ DVB-T 的 r01–r06、r09 均为 0%，只有 r07 为 37.5%、r08 为 100%。这不是随机少量错误，而是地点与类别组合发生了系统性翻转。

在相同的 352 个外部 region 上对齐现有 IQ 模型后，IQ 模型准确率为 85.227%，特征模型为 62.216%，固定 0.5/0.5 概率平均为 82.102%。两分支标签不一致的 94 个样本中，IQ 分支准确率为 86.170%，特征分支为 0%。UZ DVB-T 上，IQ 模型为 36.25%，特征模型为 13.75%，等权融合为 31.25%。

因此，当前 62 维线性特征分支没有为 UZ DVB-T 提供互补纠错证据，直接融合反而降低准确率。LLM 也不能从两个同向错误的分类分支中可靠恢复真值。下一步应先改善跨地点训练与验证设计，再决定是否扩大特征样本或进入正式融合；本先导实验的 4/8-region 抽样结果不能替代完整特征数据集实验。

- 核心特征数据：`data/external/processed/technology_recognition_lte_wifi_dvbt/v2_all_locations/features_4096_sample4/`
- 外部特征数据：`data/external/processed/technology_recognition_lte_wifi_dvbt/external_1msps_4096/features_sample8/`
- 特征模型及评估：`outputs/technology_recognition_v2_all_locations/feature_classifier_sample4/`

## 训练时频域增强对照

在四地点联合训练的同一固定数据集上保持模型和训练参数不变，分别测试两种训练时增强。验证、内部测试和 44 文件外部测试均保持原始 clean 数据；AWGN 训练增强关闭。

| 训练增强 | 内部 clean | 外部 clean | 外部 DVB-T | UZ DVB-T | 外部 10 dB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 无增强 | 98.912% | 85.293% | 67.691% | 37.201% | 85.242% |
| 50% 随机频移，最大 ±0.1Fs | 99.907% | 85.835% | 68.708% | 39.328% | 86.436% |
| 50% 频谱翻转 | 99.969% | 92.114% | 82.453% | 66.007% | 95.155% |

随机频移只带来 0.542 个百分点的外部 clean 提升，而且主要把一部分 DVB-T 错误从 LTE 转移到了 WiFi，没有实质修复地点偏移。频谱翻转则把外部 clean 提高 6.821 个百分点，UZ DVB-T 提高 28.806 个百分点，同时保持 LTE 99.925%、WiFi 100%。这支持接收链路频谱方向是主要干扰因素之一。

频谱翻转仍未彻底解决 UZ：seed 44 下 DVB-T r02 为 3.731%、r04 为 13.433%、r05 为 25.373%，说明还存在频谱方向之外的域偏移。

使用相同配置补充 seed 45 和 46 后，三次训练结果如下：

| Seed | 内部 clean | 外部 clean | 外部 10 dB | 外部 DVB-T | UZ DVB-T |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 44 | 99.969% | 92.114% | 95.155% | 82.453% | 66.007% |
| 45 | 99.969% | 97.096% | 99.064% | 93.588% | 87.537% |
| 46 | 99.969% | 91.264% | 92.363% | 80.514% | 62.164% |
| mean ± std | 99.969% ± 0.000 | 93.491% ± 2.573 | 95.527% ± 2.748 | 85.519% ± 5.761 | 71.903% ± 11.166 |

三个 seed 的外部 clean 都高于无增强基线，证明频谱翻转的收益可重复；但 UZ DVB-T 的标准差达到 11.166 个百分点，说明同地点内部验证无法可靠选择跨地点最优 checkpoint。三个模型直接平均概率的外部 clean 为 92.544%，UZ DVB-T 为 67.724%，没有消除该问题。seed 45 的 97.096% 不能因为外部测试最好就被选作最终模型，否则会把外部测试标签用于模型选择。

因此频谱翻转应保留为候选训练策略，但下一步应建立地点隔离的验证方式，再冻结模型；暂不需要用 LLM 替代分类器纠错。

- 频移模型：`outputs/technology_recognition_v2_all_locations/window_4096_frequency_shift_0p1_p0p5/`
- 频谱翻转模型：`outputs/technology_recognition_v2_all_locations/window_4096_spectral_inversion_p0p5/`
- 频谱翻转 seed 45：`outputs/technology_recognition_v2_all_locations/window_4096_spectral_inversion_p0p5_seed45/`
- 频谱翻转 seed 46：`outputs/technology_recognition_v2_all_locations/window_4096_spectral_inversion_p0p5_seed46/`

## 地点隔离验证

同一地点内按 r01–r08/r09/r10 划分可以防止文件泄漏，但训练和验证仍共享接收环境。为检验频谱翻转是否真正跨地点，复用既有四折轮换：每折训练、验证和测试地点完全不同，UZ/iGent 不参与训练或 checkpoint 选择。每折都使用 seed 44、50% 频谱翻转，频移和训练 AWGN 关闭。

| Fold | 验证地点 | 测试地点 | 无增强 clean | 频谱翻转 clean | 变化 |
| --- | --- | --- | ---: | ---: | ---: |
| fold1 | rabot | reep | 99.726% | 99.975% | +0.249 pp |
| fold2 | merelbeke | rabot | 99.677% | 99.826% | +0.149 pp |
| fold3 | gentbrugge | merelbeke | 96.517% | 98.619% | +2.102 pp |
| fold4 | reep | gentbrugge | 100.000% | 100.000% | 0.000 pp |
| mean ± std | — | — | 98.980% ± 1.427 | 99.605% ± 0.573 | +0.625 pp |

频谱翻转在四个独立测试地点上均未退化，并将地点间标准差从 1.427 降到 0.573 个百分点。四折类别平均准确率为 LTE 100%、WiFi 99.972%、DVB-T 98.843%；主要收益来自原来较弱的 Merelbeke。该结果比同地点内部验证更能支持“频谱翻转是合理的不变性增强”，但 UZ/iGent 仍应保持为外部测试，不能用于选择 checkpoint。

- 地点隔离模型与逐折评估：`outputs/technology_recognition_v2_location_validation/window_4096_spectral_inversion_p0p5/`
