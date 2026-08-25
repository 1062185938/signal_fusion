---
name: iq_feature_extraction
description: 使用 MATLAB Coder 动态库和 Python ctypes 从固定长度 IQ 样本提取 62 维时域、频域及时频域特征，并将特征值与中英文名称和参考文档位置组装供 Agent 分析。Use when the user needs IQ feature extraction, extractAllFeatures DLL/SO loading, NPZ/MAT/PKL/NPY/DAT IQ input, 62-dimensional feature mapping, or interpretation of extracted IQ features.
---

# IQ 特征提取

使用当前 skill 内的动态库，将统一格式的 IQ 样本提取为 62 维特征。需要解释特征时，先用分析脚本组装数值和映射，再按 `reference` 读取对应说明文档。

## 路径原则

- 将输入数据和输出 NPZ 放在 skill 目录之外，并通过命令行参数传入。
- 从任意工作目录运行时，优先使用绝对路径。
- Windows 默认加载 `native/windows/extractAllFeatures.dll`。
- Linux 默认加载 `native/linux/libextractAllFeatures.so`；该文件需要单独编译并放入目录。
- 用户显式传入 `--dll_path` 或 `--dll_dir` 时，以显式路径为准。

## 提取特征

运行 `scripts/feature_extractor.py`：

```bash
python <SKILL_DIR>/scripts/feature_extractor.py \
  --data_path <IQ_DATASET_PATH> \
  --output_path <FEATURE_NPZ_PATH> \
  --data_format npz \
  --x_key X \
  --sample_rate 1000000
```

输入由 `scripts/data_loaders.py` 统一转换为 `[N, 2, seq_len]`。支持 `pkl/mat/npz/npy/dat/bin`；对于原始 `dat/bin` 必须提供 `--seq_len`。采样率优先使用 `--sample_rate`，否则尝试读取输入 NPZ/MAT 中的 `sample_rate/sampling_rate/fs/Fs` 标量字段。

输出 NPZ 只包含：

```text
features       [N, 62], float32
feature_names  [62]
feature_count  scalar, int64
sample_rate    scalar, float64
seq_len        scalar, int64
```

`feature_names` 必须与 `references/feature_map.json` 的 `code_name` 顺序一致。

## 组装特征结果

运行 `scripts/analyze_feature_results.py`，选择一个样本并输出结构化 JSON：

```bash
python <SKILL_DIR>/scripts/analyze_feature_results.py \
  --feature_path <FEATURE_NPZ_PATH> \
  --sample_index 0
```

该脚本输出 `sample_rate`、`signal_length`、`feature_count`，以及每个特征的索引、中英文名称、分组、参考文档位置和数值。它不直接解释特征，也不推断调制方式或协议。

## 解释规则

用户要求解释特征结果时：

1. 先运行 `analyze_feature_results.py` 获取结构化结果。
2. 根据每项的 `group` 和 `reference` 读取对应文档：
   - `time_domain`：`references/time_domain_iq_features.md`
   - `frequency_domain`：`references/frequency_domain_iq_features.md`
   - `time_frequency`：`references/time_frequency_iq_features.md`
3. 结合采样率、序列长度和多个相关特征解释结果。
4. 不根据单个特征值直接断言调制方式、协议类型或设备类别。

## 关键文件

- `scripts/feature_extractor.py`：数据读取、DLL 调用和特征 NPZ 保存入口。
- `scripts/ctypes_backend.py`：动态库加载及 C API 调用。
- `scripts/data_loaders.py`：内部多格式 IQ 数据加载依赖。
- `scripts/feature_map_utils.py`：特征映射读取和一致性校验。
- `scripts/analyze_feature_results.py`：特征数值与映射组装入口。
- `references/feature_map.json`：62 维数组到特征名称、分组和说明文档位置的程序映射。

## 常见错误

- 找不到动态库：检查 `native/<platform>`，或显式传入 `--dll_path`。
- 缺少采样率：传入 `--sample_rate`，或在 NPZ/MAT 中提供采样率标量字段。
- `seq_len` 不一致：确认输入样本真实长度与显式参数一致。
- 特征数或名称不匹配：确认 NPZ、`feature_map.json` 和动态库均为同一版 62 维接口。
