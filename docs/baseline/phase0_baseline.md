# Signal Fusion Phase 0 基线

记录日期：2026-08-21（Asia/Shanghai）

> 历史基线说明：本文记录重构前的 62 维 `matlab_iq_features_62_v1`
> 行为，不是当前使用手册。当前运行时已经切换到 64 维
> `matlab_iq_features_64_v2`；旧特征文件、旧动态库和旧分类器不得与新 schema
> 混用。

## 1. 目的与边界

本基线用于在后续目录重构前固定“当前可以工作的行为”，重点保护 MATLAB 特征提取、C ABI 和 ONNX 推理行为。Phase 0 不实现 `signal_fusion`，不定义最终 Agent/Skill 协议，也不把当前 SigMF 能量切片算法固化为以后所有信号的统一算法。

当前工程按用途分成两条路径：

- 主分析路径：使用已经处理好的 IQ 数据集，执行 62 维特征提取或小模型推理；后续再接入融合与 LLM 解释。
- 可选数据准备路径：从 SigMF、MAT、DAT 等真实采集文件生成切片数据集。不同信号允许使用不同切片策略；该路径不是主分析的前置强制步骤。

Phase 0 只新增本文档、验证说明和少量 fixture。现有 Python/MATLAB/C++/模型代码、两个 `SKILL.md`、原始数据及已有处理结果均保持原位且不修改。

## 2. 当前 Python 入口

### 2.1 主分析入口

| 入口 | 类型 | 当前职责 |
| --- | --- | --- |
| `iq_feature_extraction/scripts/feature_extractor.py` | CLI + Python API | `extract_features_from_dataset(...)` 读取 IQ 数据，调用 ctypes/C ABI，生成 62 维特征 NPZ。 |
| `iq_feature_extraction/scripts/analyze_feature_results.py` | CLI + Python API | 读取特征 NPZ，按照 `feature_map.json` 分组并输出统计摘要；不参与特征计算。 |
| `radioml-iq-modulation/scripts/onnx_inference.py` | CLI + async Python API | `recognize_iq_modulation(...)` 加载数据、运行 ONNX、做 softmax，并以 batch 或 vote 模式输出分类文字；同步核心为 `_sync_signal_inference(...)`。 |
| `radioml-iq-modulation/scripts/train_and_export.py` | CLI + Python API | 加载带标签数据，训练已注册的小模型，保存 PTH、标签映射并导出 ONNX。 |
| `radioml-iq-modulation/scripts/models/model_factory.py` | Python API | `build_model(...)` 按名称构建 `deepconvnet_1d` 或 `lstm_iq`。当前 `models` 包导入可用。 |

### 2.2 数据访问与可选数据准备入口

| 入口 | 类型 | 当前职责 | Phase 0 定位 |
| --- | --- | --- | --- |
| 两个 skill 下各自的 `scripts/data_loaders.py` | Python API | 将多种文件形式统一成 `[N, 2, L]` float32。 | 现状记录；后续应合并为业务层公共数据接口。 |
| `radioml-iq-modulation/scripts/sigmf_dataset_builder.py` | CLI + Python API | 对 SigMF 做能量检测、burst 修整、滑窗和归一化，生成切片 NPZ。 | 可选数据准备；只代表旧 LoRa 流程。 |
| `radioml-iq-modulation/scripts/inspect_sigmf_slices.py` | CLI + Python API | 检查 SigMF 切片结果并生成统计/可视化。 | 保留的诊断工具；不是主分析入口。 |

旧测试 `test_sigmf_dataset_builder.py` 仍在原位，但不纳入 Phase 0 的长期接口承诺；旧 `radioml-iq-modulation/SKILL.md` 也不在本阶段重写。

## 3. 当前 IQ 输入约定

业务内部统一张量为：

```text
X: float32, shape = [N, 2, L]
X[:, 0, :] = I
X[:, 1, :] = Q
```

- `N` 是样本数，`L` 是每个样本的 IQ 点数。
- 特征 C ABI 的单条输入最终为连续的 I、Q `float32` 数组。
- 当前公共 loader 声明支持 `pkl/pickle`、`mat`、`npz`、`npy`、`dat`、`bin` 和 `sigmf`；具体 CLI 的 choices 比底层 loader 窄。例如特征提取 CLI 当前没有暴露 `sigmf` 选项。
- MAT/NPZ 可通过 `x_key`、`y_key` 指定字段；NPY 可用独立 `label_path`；DAT/BIN 需要明确或推断 IQ 编码和 `seq_len`。
- 真实采集的 SigMF、MAT、DAT 都属于未来统一数据准备入口的输入来源，但它们不要求共享同一种 burst 检测算法。

Phase 0 固定的输入实例定义在 `tests/fixtures/phase0_sources.json`。其中 `source_id` 是稳定的人类可读标识，不使用 SHA256。

## 4. 62 维特征契约

### 4.1 数量与顺序

`iq_feature_extraction/references/feature_map.json` 是特征顺序的基线：

| 分组 | 索引范围 | 数量 |
| --- | --- | ---: |
| 时域 `time_domain` | 0–11 | 12 |
| 频域 `frequency_domain` | 12–36 | 25 |
| 时频域 `time_frequency` | 37–61 | 25 |
| 合计 | 0–61 | 62 |

索引连续、`code_name` 唯一。首项为 `time_rms_amplitude`，末项为 `time_frequency_wsst_ridge_energy_ratio`。后续重构必须保持数量、索引顺序和名称一致，除非另行做带版本的特征契约迁移。

### 4.2 Python 特征输入/输出

`extract_features_from_dataset(...)` 输入是数据路径及格式参数，内部标准化成 `[N, 2, L]`。当前 C ABI 允许 `32 <= L <= 16384`，采样率单位为 Hz。

输出为压缩 NPZ，固定字段如下：

| 字段 | dtype | shape/含义 |
| --- | --- | --- |
| `features` | float32 | `[N, 62]` |
| `sample_rate` | float64 scalar | Hz |
| `seq_len` | int64 scalar | 单样本 IQ 点数 |
| `feature_count` | int64 scalar | 固定为 62 |
| `feature_names` | Unicode | `[62]`，顺序与特征列一致 |

当前输出不会自动携带输入标签 `y` 或逐样本来源信息。这是后续业务数据契约需要补足的内容，不在 Phase 0 修改。

### 4.3 固定回放结果

基于 `wifi5_raw_mat_0005` 的前 5 个连续 4096 点窗口：

- 输入 MAT 字段：`iq`，原始 shape `(1000000, 1)`，dtype `complex64`。
- 显式采样率：100 MHz。
- 输出 shape：`(5, 62)`，dtype `float32`，所有值有限。
- 与 `wifi_5_features_4096_smoke.npz` 的字段、元数据、特征名称一致。
- 本机回放最大绝对差：`1.1920928955078125e-07`；因此数值验收使用 `rtol=1e-5, atol=1e-6`，不要求逐 bit 相同。

## 5. 动态库 C ABI

Linux 头文件与实现位于 `iq_feature_extraction/native/linux/`，Windows 对应文件位于 `iq_feature_extraction/native/windows/`。稳定接口为：

```c
void iqFeatureInitialize(void);

int32_t iqFeatureExtract(
    const float *iData,
    const float *qData,
    int32_t signalLength,
    double sampleRate,
    float *features
);

void iqFeatureTerminate(void);
```

缓冲区与返回码约定：

- `iData`、`qData` 各至少分配 16384 个 `float`，前 `signalLength` 个点有效。
- `signalLength` 范围为 32–16384。
- `features` 至少分配 62 个 `float`。
- 返回 `0` 表示成功，`-1` 表示长度非法，`-2` 表示采样率非法，`-3` 表示输入含 NaN/Inf。
- 一般在进程内初始化一次、处理多条信号、退出前终止一次。

当前 Linux `.so` 只导出 `iqFeatureInitialize`、`iqFeatureExtract`、`iqFeatureTerminate` 三个业务符号；动态依赖包括 `libstdc++`、`libm`、`libgomp`、`libgcc_s` 和 `libc`。

## 6. ONNX 模型契约

Phase 0 使用旧的单类别 LoRa 模型 `lora_deep_iq_cnn_single_class` 做行为回归。实际 ONNX 图为 IR 8、opset 18：

| 名称 | dtype | shape | 语义 |
| --- | --- | --- | --- |
| 输入 `input` | float32 | `[batch_size, 2, 128]` | I/Q 双通道窗口 |
| 输出 `output` | float32 | `[batch_size, 1]` | 分类 logits，不是概率 |
| 输出 `feature` | float32 | `[batch_size, 300]` | 网络池化后的中间特征 |

`label_map.json` 为 `{"0": "LORA"}`。当前推理层对 logits 做 softmax，并按模型实际类别数选择 Top-K。因为模型只有一个类别，任意有限 logit 的 softmax 都是 100% LORA；该结果只能证明推理链路没有变化，不能证明模型具有区分 LoRa 与其他信号的能力。

固定参考 `lora_single_class_onnx_reference.npz` 包含 LoRa 切片数据集的前 8 条输入及其 ONNX 输出：

- `X`: `(8, 2, 128)` float32
- `logits`: `(8, 1)` float32
- `features`: `(8, 300)` float32
- `sample_id`: `0..7`
- `source_id`: `lora_slices_128_example`
- `model_id`: `lora_deep_iq_cnn_single_class`

在 PyTorch 2.5.1+cu124 与 ONNX Runtime 1.23.2 下，PTH/ONNX 的最大绝对差为：logits `2.2649765014648438e-06`，feature `8.761882781982422e-06`。验收使用 `rtol=1e-4, atol=1e-5`。

旧 PTH state dict 额外包含 THOP 产生的 `total_ops`、`total_params` 两个键；在全新构建的模型上回放时需忽略这两个非参数键。Phase 0 不改写该模型文件。

## 7. 已处理 LoRa 数据集格式

`sigmf_lora_dataset_128.npz` 当前主要字段为：

- `X`: `(4804, 2, 128)` float32
- `y`: `(4804,)` int64，当前全为类别 0
- 窗口定位：`burst_id`、`window_id`、`window_start_sample`、`window_end_sample`
- burst 定位：`burst_start_sample`、`burst_end_sample`、`raw_burst_start_sample`、`raw_burst_end_sample`
- 统计/归一化：`burst_rms`、`burst_mean_power`、`burst_peak_power`、`normalization_scale`
- 标量元数据：采样率 1 MHz、`seq_len=128`、`hop_len=64`、`label=0`、`class_name=LoRa`

其中 4804 个窗口、20 个 burst 及旧能量阈值只描述现有示例数据集。未来重写 SigMF/MAT/DAT 统一切片入口时，不以这些数字作为新切片器的兼容性要求。

NPZ 内部 `source_data_path`、`source_meta_path` 仍是旧 Windows 风格相对路径，且缺少当前的 `lora/` 层级。它们不可作为当前文件定位依据；Phase 0 fixture 清单记录真实路径，但不修改已有 NPZ。

## 8. 当前已知限制

- 两个 skill 仍混有业务实现、模型、工具和旧测试；职责拆分留到编码重构阶段。
- 两份 `data_loaders.py` 重复，格式支持和调用入口存在轻微差异。
- 当前 SigMF builder 是 LoRa 能量检测实现，不适合作为 BLE/MAT/DAT 的统一固定策略。
- `radioml-iq-modulation/SKILL.md` 暂时保持旧内容，待业务代码重构完成后重写。
- 单类别 LoRa ONNX 仅用于回归，不能作为模型准确率或泛化能力基准。
- 在本机 conda 环境中，同步核心 `_sync_signal_inference(...)` 可在 CPU 完成 8 条 batch 推理；CLI/async 包装经 10 秒仍未退出。该现象被记录为既有行为，Phase 0 不修改推理代码。重构后应单独验证并决定是否修复异步边界。
- 公开 RadioML 2016.10a 数据约 612 MB，Phase 0 只登记，不进入默认 smoke 测试。
