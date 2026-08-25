---
name: radioml-iq-modulation
description: |
  IQ 调制识别 workflow skill，用于基于已有 pkl/mat/npz/npy IQ 数据集训练 PyTorch 模型、导出 ONNX、运行 ONNX Runtime 推理，并可选支持从 SigMF IQ 文件构建 NPZ 数据集和可视化检查切片效果。Use this skill for IQ modulation recognition, SigMF IQ, LoRa IQ slicing, RadioML2016.10A, NPZ signal datasets, model selection, ONNX export, ONNX inference, label_map.json, seq_len, split_mode, and batch/vote inference.
---

# radioml-iq-modulation

## 用途

本 skill 用于 IQ 调制识别相关脚本流程，核心能力包括：

- 使用已有 `pkl` / `mat` / `npz` / `npy` IQ 数据集训练模型。
- 将训练后的 PyTorch 模型导出为 ONNX。
- 使用 ONNX Runtime 对 IQ 信号或固定长度样本数据集进行推理。
- 可选：从 SigMF IQ 文件构建固定长度 NPZ 数据集。
- 可选：检查 SigMF 切片边界、样本功率和窗口质量。

推荐主流程是：

```text
已有 IQ 数据集
  -> 训练并导出 ONNX
  -> 运行 ONNX 推理
```

如果只有 SigMF 原始 IQ 文件，可以先执行可选前置流程：

```text
SigMF IQ 文件
  -> 构建 SigMF NPZ 数据集
  -> 可选：检查生成的切片
  -> 训练并导出 ONNX
  -> 运行 ONNX 推理
```

## 何时使用

当用户需要完成以下任务时使用本 skill：

- 基于已有 IQ 数据集训练调制识别模型。
- 使用 `deepconvnet_1d` 或 `lstm_iq` 切换模型结构。
- 导出 ONNX 模型。
- 使用 ONNX 模型进行 IQ 信号推理。
- 从 SigMF `.sigmf-data` / `.sigmf-meta` 文件构建固定长度 NPZ 数据集。
- 可视化检查 SigMF burst 边界和固定长度 IQ 切片。
- 排查 `seq_len`、`label_map`、`split_mode`、`inference_mode` 或数据格式问题。

## 路径约定

在Linux环境中的文件组织形式：

```text
/home/dianci/
├── amc_workspace/
│   ├── 01_raw_sigmf/
│   ├── 02_public_datasets/
│   ├── 03_processed_npz/
│   ├── 04_outputs/          # 训练输出、pth、onnx、曲线
│   └── 05_inspection/       # 可视化检查图片、功率分布图
└── .hermes/
    └── skills/
        └── radioml-iq-modulation/
            |── SKILL.md
            └── scripts/
```

- 不要依赖脚本内置默认路径。正式使用 skill 时，必须显式传入数据、模型、输出路径。
- 不要假设数据一定存放在 skill 目录内部。
- 正式运行或跨机器测试时，建议使用绝对路径。
- 如果使用相对路径，需要从项目根目录执行命令。

使用路径占位符，后面是示例路径 ：

```text
<RAW_SIGMF_DATA_PATH>   /home/dianci/amc_workspace/01_raw_sigmf/sigmf_lora.sigmf-data
<RAW_SIGMF_META_PATH>   /home/dianci/amc_workspace/01_raw_sigmf/sigmf_lora.sigmf-meta
<PUBLIC_DATASET_PATH>   /home/dianci/amc_workspace/02_public_datasets/RML2016.10a_dict.pkl
<NPZ_DATASET_PATH>      /home/dianci/amc_workspace/03_processed_npz/sigmf_lora_dataset_128.npz
<LABEL_MAP_PATH>        /home/dianci/amc_workspace/03_processed_npz/sigmf_lora_dataset_128_label_map.json
<OUTPUT_ROOT>            /home/dianci/amc_workspace/04_outputs/
<RUN_NAME>               dataset_train01
<RUN_OUTPUT_DIR>   /home/dianci/amc_workspace/04_outputs/dataset_train01
<ONNX_PATH>        /home/dianci/amc_workspace/04_outputs/dataset_train01/deep_iq_cnn.onnx
<INSPECTION_DIR>        /home/dianci/amc_workspace/05_inspection/sigmf_lora_slice_preview
<SKILL_DIR>             /home/dianci/.hermes/skills/radioml-iq-modulation
```
路径使用规则：
- 对输入文件，直接使用完整文件路径占位符，例如 `<NPZ_DATASET_PATH>`。
- 对训练输出，使用 `<RUN_OUTPUT_DIR>` 表示本次实验输出目录。
- 如需新建一次训练实验，只修改 `<RUN_NAME>`，并同步得到：
  `<RUN_OUTPUT_DIR>` = `<OUTPUT_ROOT>`/`<RUN_NAME>`
  `<ONNX_PATH>` = `<RUN_OUTPUT_DIR>`/deep_iq_cnn.onnx

## 环境

在linux环境中如果遇到CUDA库冲突，不要默认设置自定义的 LD_LIBRARY_PATH，建议优先清除从环境中继承的库路径：
env -u LD_LIBRARY_PATH


根据部署环境选择可用的 Python 解释器。示例命令统一使用 `python`，实际运行时可以替换为 `<PYTHON>` 或某个虚拟环境中的解释器路径。

在linux环境中使用下面这个python环境：

`<PYTHON>` /home/dianci/miniconda3/envs/radioml_amc_clean/bin/python

Linux 是后续优先测试环境，所以 Linux Bash 示例放在前面。
Linux 示例使用 `/` 路径和 `\` 换行符；

Windows 主要用于本地脚本调试，如需运行请将 / 路径改为 \，并使用 PowerShell 反引号换行。

核心依赖：

- 数据集构建和可视化检查：`numpy`、`scipy`、`matplotlib`
- 训练、ONNX 导出和推理：`torch`、`onnx`、`onnxruntime`



## 核心脚本

- `train_and_export.py`：训练模型并导出 ONNX，是主训练入口。
- `onnx_inference.py`：使用 ONNX Runtime 进行推理，是主推理入口。
- `sigmf_dataset_builder.py`：可选前置脚本，从 SigMF 文件构建固定长度 NPZ 切片数据集。
- `inspect_sigmf_slices.py`：可选可视化脚本，检查生成的切片和 burst 边界。

## 可选检查脚本

- `inspect_sigmf_slices.py`：检查 SigMF 切片和 burst 边界是否合理。
- `check_npz_power_distribution.py`：检查 NPZ 样本功率分布，辅助判断低能量或疑似噪声窗口。

## 内部依赖

- `data_loaders.py`：训练和推理共用的数据加载层，只作为内部依赖说明，不作为独立 workflow 调用。

## 工作流 1：训练并导出 ONNX

本流程用于直接使用已有 IQ 训练数据集进行模型训练，并导出 ONNX。训练数据可以是 `npz`、`mat`、`npy`、`pkl` 等当前 `train_and_export.py` 支持的格式。

输入：

- 训练数据集，例如 `<NPZ_DATASET_PATH>` ,如果要更换公开数据集，则使用 `<PUBLIC_DATASET_PATH>`，并设置相应的`--data_format` 。
- 数据集中应包含标签 `y`，或通过 `--label_path` 提供外部标签。

输出：

- `<RUN_OUTPUT_DIR>/best_model.pth`
- `<RUN_OUTPUT_DIR>/deep_iq_cnn.onnx`，或 `--onnx_filename` 指定的 ONNX 文件名
- 训练曲线和指标文件，例如 `training_curves.png`、`train_acc.npy`、`train_loss.npy`、`val_acc.npy`、`val_loss.npy`

Linux Bash:

```bash
<PYTHON> <SKILL_DIR>/scripts/train_and_export.py \
  --data_path <NPZ_DATASET_PATH> \
  --data_format npz \
  --x_key X \
  --y_key y \
  --save_dir <RUN_OUTPUT_DIR> \
  --onnx_filename deep_iq_cnn.onnx \
  --model_name deepconvnet_1d \
  --class_num 1 \
  --batch_size 128 \
  --max_epoch 30 \
  --split_mode group \
  --device cuda \
  --no_plot
```


模型切换：

- `--model_name deepconvnet_1d`：默认 1D CNN 模型。
- `--model_name lstm_iq`：LSTM IQ 模型。

优化器切换：

- `--optimizer sgd`：默认 sgd 优化器。
- `--optimizer adam`：对于 LSTM 模型更适合使用adam。

AMP开启或关闭：

AMP默认开启，使用`--no_amp`关闭AMP。


数据划分：

- `--split_mode random`：按样本随机划分，默认模式，适合普通数据集和 smoke test。
- `--split_mode group`：按 `burst_id` 分组划分，适合 SigMF 切片数据集，避免同一个 burst 的相邻窗口同时进入训练集和测试集。
- `--split_mode group` 要求 NPZ 中存在 `burst_id` 元数据，且当前不支持 `--max_samples`。

对于标准 NPZ 文件，如果字段名就是 `X` 和 `y`，`--x_key X --y_key y` 可以省略；当字段名不标准时再显式传入。

## 工作流 2：运行 ONNX 推理

本流程用于使用导出的 ONNX 模型进行推理。

输入：

- 推理输入文件或数据集，通过 `--signal_path` 传入。
- ONNX 模型，通过 `--model_path` 传入。
- 标签映射 JSON，通过 `--label_map_path` 传入。

Linux Bash:

```bash
<PYTHON> <SKILL_DIR>/scripts/onnx_inference.py \
  --signal_path <NPZ_DATASET_PATH> \
  --data_format npz \
  --model_path <ONNX_PATH> \
  --label_map_path <LABEL_MAP_PATH> \
  --batch_size 64 \
  --seq_len 128 \
  --x_key X \
  --inference_mode vote
```


推理模式：

- `--inference_mode vote`：默认模式。先读取全部候选样本，再随机抽取 `batch_size` 个样本进行推理和最终投票。
- `--inference_mode batch`：批量直接推理。读取前 `batch_size` 个样本，逐样本输出 Top-K，不执行最终投票。`batch_size=1` 时就是单样本推理。

推理要求：

- `--seq_len` 必须和 ONNX 导出时的输入长度一致。
- `--label_map_path` 中的标签数量必须和模型输出类别数一致。
- 对 `dat` / `bin` 输入，可按需要提供 `--sample_mode` 和 `--iq_format`。

## 工作流 3：可选，从 SigMF 构建 NPZ 数据集

这是可选前置流程。当用户已有 `npz` / `mat` / `npy` / `pkl` 数据集时，可以跳过本步骤，直接进入训练。

输入：

- SigMF data 文件：`<RAW_SIGMF_DATA_PATH>`
- SigMF meta 文件：`<RAW_SIGMF_META_PATH>`

输出：

- NPZ 数据集：`<NPZ_DATASET_PATH>`
- 与 NPZ 同目录的 dataset summary JSON

Linux Bash:

```bash
<PYTHON> <SKILL_DIR>/scripts/sigmf_dataset_builder.py \
  --data_path <RAW_SIGMF_DATA_PATH> \
  --meta_path <RAW_SIGMF_META_PATH> \
  --output_path <NPZ_DATASET_PATH> \
  --label 0 \
  --class_name LoRa \
  --seq_len 128 \
  --hop_len 128 \
  --chunk_size 1000000 \
  --window_ms 1.0 \
  --start_threshold_db 6.0 \
  --end_threshold_db 5.0 \
  --min_signal_ms 8.0 \
  --min_gap_ms 2.0 \
  --pad_before_ms 0.5 \
  --pad_after_ms 0.2 \
  --normalize rms \
  --remove_dc \
  --remainder drop \
  --release_windows 2 \
  --noise_percentile 20 \
  --noise_probe_count 8 \
  --ignore_initial_ms 0 \
  --window_power_ratio 0.05
```

生成的 NPZ 至少应包含 `X`、`y`、burst/window 元数据、`sample_rate`、`seq_len`、`hop_len`、`label`、`class_name` 和源文件路径。模型输入约定为 `X.shape == [N, 2, seq_len]`。

## 工作流 4：可选，检查生成的 SigMF 切片

这是可选检查流程，不参与训练，也不会重新生成数据集。它只使用 NPZ 中已有的边界和切片元数据进行可视化。

Linux Bash:

```bash
<PYTHON> <SKILL_DIR>/scripts/inspect_sigmf_slices.py \
  --npz_path <NPZ_DATASET_PATH> \
  --data_path <RAW_SIGMF_DATA_PATH> \
  --meta_path <RAW_SIGMF_META_PATH> \
  --output_dir <INSPECTION_DIR> \
  --max_bursts 5 \
  --windows_per_burst 3 \
  --max_plot_points 20000
```



如果 NPZ 中保存的 `source_data_path` 和 `source_meta_path` 在当前环境有效，可以省略 `--data_path` 和 `--meta_path`。图片会保存到 `--output_dir`。

## 重要参数

`sigmf_dataset_builder.py`:

- `--data_path`：输入 `.sigmf-data` 文件。
- `--meta_path`：输入 `.sigmf-meta` 文件。
- `--output_path`：输出 `.npz` 数据集路径。
- `--label`、`--class_name`：写入 NPZ 的数字标签和类别名称。
- `--seq_len`、`--hop_len`：固定窗口长度和滑窗步长。
- `--window_ms`：能量检测窗口时长。
- `--start_threshold_db`、`--end_threshold_db`：滞回检测阈值。
- `--min_signal_ms`、`--min_gap_ms`：burst 过滤和合并相关参数。
- `--pad_before_ms`、`--pad_after_ms`：确定性 padding。
- `--normalize`、`--remove_dc`、`--no_remove_dc`：预处理控制参数。
- `--ignore_initial_ms`：检测前忽略开头指定毫秒数的数据。
- `--window_power_ratio`：基于窗口功率的 burst refinement 阈值。

`train_and_export.py`:

- `--data_path`：训练数据集路径。
- `--data_format`：数据格式，可选 `auto`、`pkl`、`mat`、`npz`、`npy`。
- `--x_key`、`--y_key`：`mat` / `npz` 中的输入和标签字段名。
- `--label_path`：`npy` 数据集对应的外部标签文件。
- `--save_dir`：模型权重、ONNX 和训练曲线输出目录。
- `--onnx_filename`：导出的 ONNX 文件名。
- `--class_num`：模型输出类别数。
- `--batch_size`：训练 batch 大小。
- `--max_epoch`：训练 epoch 数。
- `--max_samples`：快速 smoke test 的样本上限，仅支持 `split_mode=random`。
- `--model_name`：模型名称，当前支持 `deepconvnet_1d` 和 `lstm_iq`。
- `--split_mode`：数据划分方式，支持 `random` 和 `group`。
- `--device`：训练设备，支持 `auto`、`cpu`、`cuda`。
- `--loss`：损失函数，默认`logit_norm`，可选 `ce`、`logit_norm`、`ls`。分别表示交叉熵、LogitNormLoss、Label Smoothing。。
- `--no_amp`：用于关闭AMP模式，默认开启AMP。
- `--optimizer`：优化器类型，可选 `sgd` 和 `adam`。

`onnx_inference.py`:

- `--signal_path`：推理输入信号或数据集路径。
- `--model_path`：ONNX 模型路径。
- `--label_map_path`：标签映射 JSON 路径。
- `--data_format`：输入格式，例如 `npz` 或 `dat`。
- `--batch_size`：推理样本数量。`vote` 模式下表示随机抽取多少个样本投票；`batch` 模式下表示读取前多少个样本直接推理。
- `--seq_len`：模型输入序列长度，必须与 ONNX 导出时一致。
- `--x_key`：`mat` / `npz` 输入中的 X 字段名。
- `--inference_mode`：推理模式，支持 `vote` 和 `batch`。
- `--sample_mode`、`--iq_format`：`dat` / `bin` 输入的可选控制参数。

`inspect_sigmf_slices.py`:

- `--npz_path`：待检查的 NPZ 数据集路径。
- `--data_path`：可选的原始 `.sigmf-data` 路径。
- `--meta_path`：可选的原始 `.sigmf-meta` 路径。
- `--output_dir`：PNG 图片输出目录。
- `--max_bursts`：最多绘制多少个 burst。
- `--windows_per_burst`：每个 burst 绘制多少个窗口。
- `--max_plot_points`：长 burst 总览图的确定性降采样上限。

## 常见错误

- `npz` 中找不到 `X`：检查 NPZ 字段名，或通过 `--x_key` 指定正确字段。
- `label_map 与模型输出标签类别不匹配`：确认 `label_map.json` 的标签数量等于 `class_num`，且顺序与训练标签编码一致。
- `seq_len` 不一致：数据集制作、训练导出和推理必须使用一致的 `seq_len`。
- `split_mode=group requires burst_id in dataset metadata`：说明 NPZ 中缺少 `burst_id`，请使用 SigMF 构建脚本生成的 NPZ，或改用 `--split_mode random`。
- `split_mode=group` 同时使用 `--max_samples`：当前 group split 不支持 `max_samples`，快速测试请使用 random split。
- `class_num` 与训练数据标签数量不一致：`--class_num` 必须匹配训练标签类别数量，也要和推理时的 `label_map.json` 对齐。
- 单类别模型可以跑通流程，但分类置信度解释有限，推理时只会显示 Top-1。
- `batch` 推理模式不会输出最终综合判定；需要投票结论时使用 `--inference_mode vote`。
- Windows 路径包含空格或中文时需要加引号。
- 使用相对路径时，如果工作目录不是 `<PROJECT_ROOT>`，容易找不到文件。
