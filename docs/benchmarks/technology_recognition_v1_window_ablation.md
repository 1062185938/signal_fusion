# Technology Recognition V1：地点轮换与窗口长度消融

## 实验目标

验证两个问题：

1. 模型是否能泛化到训练阶段未见过的采集地点；
2. 在固定 4096 点 region 内，128、512、1024、2048、4096 点窗口对识别能力有什么影响。

所有实验使用相同的 120 个公开数据源和相同的 3,840 个 region。训练阶段没有额外添加 AWGN，唯一变化是地点 Fold 和窗口长度。

## 地点轮换

| Fold | Train | Validation | Test |
| --- | --- | --- | --- |
| fold1 | gentbrugge、merelbeke | rabot | reep |
| fold2 | gentbrugge、reep | merelbeke | rabot |
| fold3 | rabot、reep | gentbrugge | merelbeke |
| fold4 | merelbeke、rabot | reep | gentbrugge |

每个地点恰好作为一次 validation 和一次 test。每个 split 的 LTE、WiFi、DVB-T 数量完全相同，源文件和 region 在 split 之间没有交集。

## 窗口契约

| Window | 每个 region 的窗口数 | Train windows | Validation windows | Test windows |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 32 | 61,440 | 30,720 | 30,720 |
| 512 | 8 | 15,360 | 7,680 | 7,680 |
| 1024 | 4 | 7,680 | 3,840 | 3,840 |
| 2048 | 2 | 3,840 | 1,920 | 1,920 |
| 4096 | 1 | 1,920 | 960 | 960 |

五套 profile 均从同一批原始 `.bin` 重新生成。对全部 120 个源文件逐一验证后，较短窗口重新拼接得到的 4096 点 IQ 与 4096 profile 逐样本完全一致。

训练数据装配阶段统一执行窗口级去直流和复数 RMS 归一化。4096 点 region 结果使用 region 内全部窗口的平均类别概率，不丢弃低能量窗口。

## 固定训练设置

- 模型：DeepConvNet1D
- 优化器：Adam
- 学习率：0.001
- 损失：交叉熵
- batch size：256
- 最大 epoch：30
- early stopping patience：8
- seed：44
- AMP：开启
- 设备：NVIDIA GeForce RTX 3060 / CUDA
- 人工 AWGN：关闭

每种窗口长度仍让模型在一个 epoch 内看完对应 split 的所有窗口，因此原始 IQ 总覆盖量保持一致。

## 四折结果

### 每个 Fold 的窗口准确率

| Window | fold1 / reep | fold2 / rabot | fold3 / merelbeke | fold4 / gentbrugge |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 78.02% | 79.89% | 71.18% | 76.22% |
| 512 | 92.41% | 96.24% | 94.58% | 95.25% |
| 1024 | 97.92% | 98.52% | 92.84% | 99.77% |
| 2048 | 99.58% | 99.79% | 89.38% | 100.00% |
| 4096 | 97.71% | 99.38% | 98.75% | 100.00% |

### 四折汇总

这里的标准差是四个地点 Fold 之间的离散程度，不是多随机种子误差。

| Window | 窗口准确率 | 4096 点 region 准确率 |
| ---: | ---: | ---: |
| 128 | 76.33% ± 3.24% | 91.22% ± 13.82% |
| 512 | 94.62% ± 1.41% | **99.66% ± 0.20%** |
| 1024 | 97.26% ± 2.64% | 99.32% ± 1.06% |
| 2048 | 97.19% ± 4.51% | 98.00% ± 3.41% |
| 4096 | **98.96% ± 0.85%** | 98.96% ± 0.85% |

4096 点只有一个窗口，所以它的窗口结果与 region 结果相同。

### 窗口级分类别四折结果

| Window | LTE | WiFi | DVB-T |
| ---: | ---: | ---: | ---: |
| 128 | 90.79% ± 4.93% | 77.50% ± 8.83% | 60.70% ± 22.89% |
| 512 | 98.40% ± 1.70% | 89.54% ± 6.56% | 95.92% ± 5.80% |
| 1024 | 99.26% ± 1.20% | 97.77% ± 2.42% | 94.75% ± 7.80% |
| 2048 | 99.34% ± 1.15% | 99.57% ± 0.51% | 92.66% ± 12.63% |
| 4096 | 99.45% ± 0.95% | 98.20% ± 2.94% | 99.22% ± 0.78% |

## 512 / 4096 增量 AWGN 实验

使用原有 clean checkpoint，不重新训练。每个测试窗口分别加入 10、7.5、
5 dB 圆复高斯白噪声；每个噪声条件使用种子 44–48 重复五次。这里的 SNR
是“原窗口功率 / 新增噪声功率”，不等于包含原始接收机噪声在内的真实空口
SNR。

下表为四个 held-out test location fold 的均值和总体标准差：

| 输入与输出粒度 | clean | 10 dB | 7.5 dB | 5 dB |
| --- | ---: | ---: | ---: | ---: |
| 512 window | 94.62% ± 1.41% | 93.62% ± 2.23% | 92.63% ± 2.84% | 90.56% ± 3.99% |
| 512 region（8 个窗口概率均值） | **99.66% ± 0.20%** | **99.36% ± 0.53%** | **98.84% ± 1.22%** | 97.34% ± 3.03% |
| 4096 direct | 98.96% ± 0.85% | 98.83% ± 1.37% | 98.59% ± 1.80% | **97.61% ± 3.47%** |

5 dB 的 region 分类别结果：

| 方案 | LTE | WiFi | DVB-T |
| --- | ---: | ---: | ---: |
| 512 region | 99.20% | 92.84% | 99.98% |
| 4096 direct | 99.33% | 93.58% | 99.92% |

主要退化集中在 WiFi 和 reep 测试地点。5 dB 时 reep 的 region 准确率为
92.10%（512）和 91.63%（4096），其他三个地点约为 98.98%–99.90%。
同一地点五个噪声种子的 region 标准差最大只有 0.34 个百分点（512）和
0.45 个百分点（4096），所以当前差异主要来自地点/采集分布，而不是噪声
随机种子。

这组结果支持继续保留 512 region 聚合和 4096 direct 两种方案。下一步应先
检查并扩充跨地点 WiFi 数据；当前没有证据要求修改网络结构或 62 维 MATLAB
特征定义。

## 结论

128 点窗口过短。它不仅窗口准确率最低，在 `fold3` 的 merelbeke 测试地点还把大量 DVB-T region 系统性判断成 WiFi，导致该 Fold 的 region 准确率只有 67.29%。这说明增加窗口投票不能修复所有系统性短窗口错误。

512 点是当前最合适的多窗口方案。每个 4096 点 region 有 8 个局部判断，四折 region 准确率为 99.66%，Fold 标准差只有 0.20%，比 128 点稳定得多，同时保留了局部一致性和异常窗口分析能力。

4096 点是当前最好的单次整段分类方案。它的四折直接准确率为 98.96%，且三个类别都接近 99%。代价是每个 region 只有一个模型输出，不能再利用多窗口一致性。

1024 点可作为中间方案，但当前 region 均值和稳定性均没有超过 512 点。2048 点在 merelbeke 的 DVB-T 上仍有明显下降，因此不建议作为默认值。

当前推荐：

- 主流程需要“局部判断 + region 聚合”时使用 512 点窗口；
- 只需要对完整 4096 点 region 做一次模型判断时使用 4096 点输入；
- 不再把 128 点作为这个公开数据集的默认推理窗口。

这些结论仅覆盖一个模型和一个随机种子，也不能直接证明模型能泛化到该公开数据集之外的采集系统。

## 复现

生成五套 profile 和 20 个 manifest：

```bash
PYTHONPATH=src python -m signal_fusion.benchmarks.technology_recognition_cli \
  --dataset-root "/mnt/sda2/mydata/Technology Recognition (LTE, Wi-Fi and DVB-T)" \
  --output-dir data/external/processed/technology_recognition_lte_wifi_dvbt/v1 \
  --prepare \
  --window-sizes 128 512 1024 2048 4096 \
  --manifests-dir configs/training/technology_recognition_v1_ablation
```

每个 manifest 继续使用 `signal_fusion.training_data.cli` 装配固定 split，再使用 `signal_fusion.training.cli` 训练。统一评估命令：

```bash
MPLCONFIGDIR=/tmp/matplotlib-signal-fusion PYTHONPATH=src \
  /home/dianci/miniconda3/envs/radioml_amc_clean/bin/python \
  -m signal_fusion.benchmarks.technology_recognition_ablation_cli \
  --dataset-root data/external/processed/technology_recognition_lte_wifi_dvbt/v1/ablation/folds \
  --model-root outputs/technology_recognition_v1_ablation \
  --output-dir outputs/technology_recognition_v1_ablation/evaluation \
  --device cuda \
  --batch-size 256
```

单个 fold 的增量 AWGN 评估命令：

```bash
MPLCONFIGDIR=/tmp/matplotlib-signal-fusion PYTHONPATH=src \
  /home/dianci/miniconda3/envs/radioml_amc_clean/bin/python \
  -m signal_fusion.evaluation.cli \
  --dataset_dir data/external/processed/technology_recognition_lte_wifi_dvbt/v1/ablation/folds/window_512/fold1 \
  --model_path outputs/technology_recognition_v1_ablation/window_512/fold1/best_model.pth \
  --output_dir outputs/technology_recognition_v1_ablation/window_512/fold1/awgn_evaluation \
  --snr_db 10 7.5 5 \
  --trials 5 \
  --seed 44 \
  --batch_size 256 \
  --device cuda \
  --model_name deepconvnet_1d
```

八个 fold 结果生成完毕后，跨 fold 汇总命令：

```bash
MPLCONFIGDIR=/tmp/matplotlib-signal-fusion PYTHONPATH=src \
  /home/dianci/miniconda3/envs/radioml_amc_clean/bin/python \
  -m signal_fusion.benchmarks.technology_recognition_awgn_cli \
  --model-root outputs/technology_recognition_v1_ablation \
  --output-dir outputs/technology_recognition_v1_ablation/awgn_summary
```

## 产物

- Prepared profiles：`data/external/processed/technology_recognition_lte_wifi_dvbt/v1/profiles`
- 固定数据集：`data/external/processed/technology_recognition_lte_wifi_dvbt/v1/ablation/folds`
- 20 个 manifest：`configs/training/technology_recognition_v1_ablation`
- 20 个 checkpoint、ONNX 和曲线：`outputs/technology_recognition_v1_ablation/window_*`
- 完整机器可读结果：`outputs/technology_recognition_v1_ablation/evaluation/ablation_evaluation.json`
- 逐实验 CSV：`outputs/technology_recognition_v1_ablation/evaluation/ablation_runs.csv`
- 消融图：`outputs/technology_recognition_v1_ablation/evaluation/ablation_accuracy.png`
- AWGN 跨 fold 汇总：`outputs/technology_recognition_v1_ablation/awgn_summary/awgn_summary.json`
- AWGN 可交互报告：`outputs/technology_recognition_v1_ablation/awgn_summary/report.html`
