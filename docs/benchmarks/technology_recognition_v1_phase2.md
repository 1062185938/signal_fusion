# Technology Recognition 公开基准 V1：Phase 2

## 目标

在 Phase 1 prepared source 不变的前提下，建立地点隔离的固定数据划分，训练第一个不添加人工噪声的 128 点 IQ 基线。

这里的 `clean` 仅表示训练时没有额外加入 AWGN。公开数据本身包含真实采集噪声和信道影响。

## Fold 1 固定划分

| Split | 地点 | 源文件数 | Region 数 | Window 数 |
| --- | --- | ---: | ---: | ---: |
| train | gentbrugge、merelbeke | 60 | 1,920 | 61,440 |
| validation | rabot | 30 | 960 | 30,720 |
| test | reep | 30 | 960 | 30,720 |

每个 split 内三个类别完全均衡。每个 4096 点 region 保留全部 32 个非重叠 128 点窗口。训练数据装配阶段对每个窗口去直流并进行复数 RMS 归一化。

地点在 split 之间完全隔离，原始采集文件 `source_id` 也完全隔离。测试集只在模型选择完成后用于最终评价。

固定配置：

- `configs/training/technology_recognition_v1_ablation/window_128_fold1.json`

生成配置：

```bash
PYTHONPATH=src python -m signal_fusion.benchmarks.technology_recognition_cli \
  --dataset-root "/mnt/sda2/mydata/Technology Recognition (LTE, Wi-Fi and DVB-T)" \
  --output-dir "data/external/processed/technology_recognition_lte_wifi_dvbt/v1" \
  --window-sizes 128 \
  --manifests-dir configs/training/technology_recognition_v1_ablation
```

装配固定 split：

```bash
PYTHONPATH=src python -m signal_fusion.training_data.cli \
  --manifest configs/training/technology_recognition_v1_ablation/window_128_fold1.json \
  --output-dir data/external/processed/technology_recognition_lte_wifi_dvbt/v1/ablation/folds/window_128/fold1
```

数据审计结果位于 `fold1_clean_128/assembly_report.json`。审计确认：

- train、validation、test 的 `source_id` 无交集；
- selected region 无交集；
- 每个 region 恰好包含 32 个窗口；
- 所有窗口去直流后的复均值接近零；
- 所有窗口的复数 RMS 为 1；
- 三个类别的样本数完全相同。

## Clean 128 点基线

训练配置：

- 模型：DeepConvNet1D
- 输入：`[batch, 2, 128]`
- 类别：LTE、WiFi、DVB-T
- 优化器：Adam，学习率 0.001
- 损失：交叉熵
- batch size：256
- 最多 30 epoch，early stopping patience 为 8
- seed：44
- AMP：开启
- 设备：NVIDIA GeForce RTX 3060 / CUDA
- 训练 AWGN probability：0

复现命令：

```bash
MPLCONFIGDIR=/tmp/matplotlib-signal-fusion PYTHONPATH=src \
  /home/dianci/miniconda3/envs/radioml_amc_clean/bin/python \
  -m signal_fusion.training.cli \
  --dataset_dir data/external/processed/technology_recognition_lte_wifi_dvbt/v1/ablation/folds/window_128/fold1 \
  --save_dir outputs/technology_recognition_v1_ablation/window_128/fold1 \
  --onnx_filename technology_recognition_window_128_fold1.onnx \
  --model_name deepconvnet_1d \
  --class_num 3 \
  --batch_size 256 \
  --optimizer adam \
  --lr_model 0.001 \
  --loss ce \
  --max_epoch 30 \
  --patience 8 \
  --seed 44 \
  --num_workers 2 \
  --device cuda \
  --awgn_probability 0
```

训练在第 11 个 epoch 触发 early stopping，最佳 checkpoint 来自第 3 个 epoch：

- 最佳 validation window accuracy：83.60%
- test window accuracy：78.02%
- test mean predicted confidence：0.864
- ONNX 导出与推理验证：通过

测试集的窗口级结果：

| 真实类别 | Accuracy | 预测 LTE | 预测 WiFi | 预测 DVB-T |
| --- | ---: | ---: | ---: | ---: |
| LTE | 85.47% | 8,752 | 33 | 1,455 |
| WiFi | 72.60% | 275 | 7,434 | 2,531 |
| DVB-T | 76.00% | 404 | 2,054 | 7,782 |

WiFi 与 DVB-T 是当前最明显的混淆方向。训练准确率继续上升到 98.52%，而地点隔离的验证集没有同步提升，说明继续增加 epoch 不能解决跨地点泛化问题。

## Region 聚合结果

对同一个 4096 点 region 的全部 32 个窗口概率取平均，再选择 Top1：

- test region accuracy：99.48%（955 / 960）
- LTE：99.38%（318 / 320）
- WiFi：100%（320 / 320）
- DVB-T：99.06%（317 / 320）

region 混淆矩阵：

| 真实类别 | 预测 LTE | 预测 WiFi | 预测 DVB-T |
| --- | ---: | ---: | ---: |
| LTE | 318 | 0 | 2 |
| WiFi | 0 | 320 | 0 |
| DVB-T | 0 | 3 | 317 |

Region 结果验证了“128 点局部识别 + 4096 点长信号聚合”的可行性，但它不能替代窗口级结果。该结论目前只覆盖 Fold 1，仍需地点轮换交叉验证。

## 输出

装配数据：

- `data/external/processed/technology_recognition_lte_wifi_dvbt/v1/ablation/folds/window_128/fold1/train.npz`
- `data/external/processed/technology_recognition_lte_wifi_dvbt/v1/ablation/folds/window_128/fold1/validation.npz`
- `data/external/processed/technology_recognition_lte_wifi_dvbt/v1/ablation/folds/window_128/fold1/test.npz`
- `data/external/processed/technology_recognition_lte_wifi_dvbt/v1/ablation/folds/window_128/fold1/assembly_report.json`

训练输出：

- `outputs/technology_recognition_v1_ablation/window_128/fold1/best_model.pth`
- `outputs/technology_recognition_v1_ablation/window_128/fold1/technology_recognition_window_128_fold1.onnx`
- `outputs/technology_recognition_v1_ablation/window_128/fold1/training_result.json`
- `outputs/technology_recognition_v1_ablation/window_128/fold1/training_curves.png`
