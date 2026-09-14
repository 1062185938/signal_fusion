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
