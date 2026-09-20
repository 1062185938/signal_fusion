# signal_fusion 模块与架构说明

本文档简要说明 `src/signal_fusion` 核心包中各模块的职责、文件组织方式以及模块之间的调用关系。

## 1. 项目定位

`signal_fusion` 当前提供无线 IQ 信号分析所需的基础能力，包括：

- 原始采集文件的可选预处理；
- 已处理 IQ 数据集的统一读写；
- MATLAB/C ABI 64 维特征提取；
- ONNX 小模型推理；
- PyTorch 模型定义、训练与 ONNX 导出；
- 面向后续融合分析的统一数据契约。

当前尚未实现真正的 LLM 融合编排模块。现阶段的重点是让特征提取和小模型推理都能产生结构化、可组合的 `Evidence`，为后续 `signal_fusion` 推理层提供稳定输入。

## 2. 总体流程

主分析流程使用已经处理好的数据集：

```text
已处理数据集
    │
    ▼
io.load_prepared_dataset
    │
    ▼
PreparedDataset [N, 2, L]
    ├──► feature_extraction ──► FeatureResult ──► Evidence
    └──► model_inference    ──► ModelInferenceResult ──► Evidence
                                                        │
                                                        ▼
                                             后续 LLM / 融合分析
```

原始信号准备是一条独立、可选的离线流程，不属于主分析必经步骤：

```text
SigMF / MAT / DAT 原始采集文件
    │
    ▼
preparation.readers ──► RawSignal
    │
    ▼
信号区域检测 ──► 区域整理 ──► 窗口切片 ──► 归一化
    │
    ▼
PreparedDataset ──► io.writers ──► NPZ 数据集
```

模型开发也是独立的离线流程。不同采集源先由 `training_data` 固定装配成互不泄漏的训练、验证和测试文件，再交给训练模块：

```text
PreparedDataset 切片 ──► training_data ──► train/validation/test.npz
                                               │
                                               ▼
                                     training.splitting / DataLoader
                         │
modeling 模型定义 ───────┤
                         ▼
                   training.trainer
                         │
                         ├──► PTH 权重和训练指标
                         └──► training.exporter ──► ONNX 模型
```

## 3. 根模块

### `signal_fusion/__init__.py`

核心包的公共入口，集中导出常用契约、数据加载、数据准备、特征提取和模型推理接口。调用方通常优先从这里导入稳定接口，而不是依赖模块内部实现。

`modeling` 和 `training` 没有在根模块中导入，因为它们依赖 PyTorch、ONNX 等离线开发依赖，不应影响只使用基础数据或运行时能力的环境。

### `signal_fusion/contracts.py`

定义跨模块共享的数据契约：

| 契约 | 作用 |
| --- | --- |
| `PreparedDataset` | 标准 IQ 数据集；`X` 固定为 `float32 [N, 2, L]`，`y` 可选且为 `int64 [N]` |
| `Evidence` | 特征或模型推理产生的一条可解释证据，供后续融合模块组合 |
| `ModelManifest` | 描述模型路径、输入、输出、标签和版本等运行时信息 |

这些契约是模块间协作的边界。算法模块通过契约交换数据，避免直接依赖彼此的内部结构。

## 4. `io`：已处理数据集的统一读写

`io` 面向已经形成样本的数据，例如 `[N, 2, L]` 的 NPZ、MAT、NPY 或 PKL。它不负责信号区域检测和切片。

| 文件 | 作用 |
| --- | --- |
| `io/__init__.py` | 汇总并导出公共 Loader 和 Writer |
| `io/loaders.py` | 识别数据格式、选择数据键、统一 IQ 排布和类型，并生成 `PreparedDataset`；同时保留旧 CLI 使用的字典返回接口 |
| `io/writers.py` | 将 `PreparedDataset` 写成无 pickle 的 NPZ，并生成 JSON 摘要；负责 NumPy 元数据的安全序列化 |

主要接口：

- `load_prepared_dataset()`：推荐的标准入口，返回 `PreparedDataset`；
- `load_signal_dataset()`：兼容旧代码，返回 `{"X", "y", "meta"}`；
- `load_signal_for_inference()`：模型推理兼容入口，支持限制样本数量；
- `write_prepared_dataset()`：保存标准数据集；
- `write_dataset_summary()`：保存数据集摘要。

`io` 对 DAT/BIN 的直接加载只进行固定长度、记录对齐的样本转换，不执行能量检测。需要先检测有效信号区域时，应使用 `preparation`。

## 5. `preparation`：原始信号准备

`preparation` 负责把连续原始采集文件转换成可分析的数据集。读取方式、检测策略、切片策略和归一化策略彼此独立，因此以后可以为 BLE、LoRa 等信号选择不同的检测方法。

### 5.1 顶层文件

| 文件 | 作用 |
| --- | --- |
| `preparation/__init__.py` | 汇总公共接口；绘图检查功能采用延迟导入，避免普通准备流程强制加载 Matplotlib |
| `preparation/contracts.py` | 定义连续原始信号 `RawSignal`、检测区域 `SignalRegion`、格式无关配置 `PreparationConfig`，以及版本化重采样配置 `ResamplingConfig` |
| `preparation/pipeline.py` | 通用组合流程：检测、区域整理、可选 region 重采样、目标坐标窗口切片和归一化，最终返回 `PreparedDataset`；不传重采样配置时保持旧行为 |
| `preparation/resampling.py` | 建立有理数采样率计划、复数 IQ polyphase FIR 重采样、带滤波保护区的 region 读取，以及原生/目标双坐标映射 |
| `preparation/dataset.py` | 文件级编排：打开原始文件、执行通用流程、写入 NPZ 和摘要 |
| `preparation/segmentation.py` | 对检测区域进行最小长度过滤、邻近区域合并和前后扩展 |
| `preparation/windowing.py` | 将区域转换为定长窗口，定义尾部 `drop` 或 `pad` 行为 |
| `preparation/normalization.py` | 可选去直流，以及 `none`、RMS、峰值归一化 |
| `preparation/cli.py` | `signal-prepare` 命令入口，负责解析 Reader、Detector 和切片配置 |
| `preparation/inspection.py` | `signal-inspect` 命令及切片可视化检查；可把生成窗口映射回原始采集坐标 |
| `preparation/energy_v1_dataset.py` | 保留原有能量检测数据集生成行为的专用路径，并兼容历史 SigMF 输出结构 |
| `preparation/legacy_burst.py` | 保留旧 `smart_extract_bursts` 调用接口 |
| `preparation/legacy_sigmf.py` | 保留旧 SigMF 类型、读取器和 CLI 的兼容入口 |

### 5.2 `readers` 子模块

Reader 只负责将不同原始文件暴露为统一的、可按区间读取的 `RawSignal`，不负责检测或切片。

| 文件 | 作用 |
| --- | --- |
| `readers/base.py` | 定义原始信号 Reader 抽象接口 |
| `readers/dat.py` | 读取复数或交织 IQ 的 DAT/BIN 文件 |
| `readers/mat.py` | 从 MAT 中选择 IQ、采样率和中心频率等字段 |
| `readers/sigmf.py` | 解析 SigMF metadata/data 文件及其数据类型 |
| `readers/__init__.py` | 格式识别和 Reader 注册入口，提供 `open_raw_signal()` |

### 5.3 `detectors` 子模块

Detector 输入 `RawSignal`，只输出 `SignalRegion`，不直接生成训练样本。

| 文件 | 作用 |
| --- | --- |
| `detectors/base.py` | 定义 `SignalDetector` 抽象接口 |
| `detectors/full_signal.py` | 把整个采集文件视为一个信号区域，适合无需检测的情况 |
| `detectors/fixed_blocks.py` | 把连续记录划成固定长度 block，并可在全文件范围内均匀选择指定数量的 region |
| `detectors/energy_v1.py` | 当前 LoRa 能量检测策略及其配置、噪声估计和区域细化逻辑 |
| `detectors/ble_packet_v1.py` | 使用 BLE 前导码、Access Address、包头与 CRC 约束定位完整 BLE 包 |
| `detectors/registry.py` | Detector 注册表和按名称构建策略的入口 |
| `detectors/__init__.py` | 导出 Detector 公共接口 |

通用内部组织顺序为：

```text
Reader → RawSignal → Detector → SignalRegion
       → segmentation → windowing → normalization → PreparedDataset
```

V2-B 已把重采样接入通用 pipeline，处理顺序固定为：

```text
Reader → native-rate detection → segmentation
       → guarded region resampling → target-rate windowing
       → normalization → PreparedDataset
```

Python 调用 `prepare_signal()`、`prepare_file()` 或 `build_prepared_dataset()` 时，可通过 `resampling=ResamplingConfig(target_sample_rate=4_000_000)` 启用；CLI 可通过 `--target_sample_rate 4000000` 启用。默认值仍为 `None`，因此不传该配置或参数时，已有算法和数据集输出保持不变。

启用重采样后，检测、区域整理及其参数仍使用原始采样率坐标；`seq_len`、`hop_len` 和窗口切片使用目标采样率坐标。输出数据集使用 `dual_rate_v1` 双坐标契约，同时保存 `source_*` 原始坐标和 `target_*` 目标坐标。`signal-inspect` 会自动在原始 IQ 概览中使用 `source_*` 坐标，在 NPZ 窗口图中使用目标坐标。

### 5.4 `benchmarks`：公开数据集实验编排

`benchmarks` 保存公开数据集特有的文件筛选和可复现实验入口，不改变通用 Reader、Detector 或训练模块。

| 文件 | 作用 |
| --- | --- |
| `benchmarks/technology_recognition.py` | 解析 LTE/WiFi/DVB-T 数据集文件名，审计 190 个源文件，冻结四地点 V1 子集，并调用通用 `fixed_blocks` pipeline |
| `benchmarks/technology_recognition_cli.py` | 生成 `inventory.json`，批量生成不同窗口长度的 prepared profile，并生成四地点轮换 manifest |
| `benchmarks/technology_recognition_ablation.py` | 统一评估 4 Fold × 5 窗口长度模型，计算窗口级、4096 点 region 级和分类别结果 |
| `benchmarks/technology_recognition_ablation_cli.py` | 公开数据集消融评估命令入口，输出 JSON、CSV 和曲线图 |

V1 使用原生 1 Msps、4096 点 region、每个源文件均匀选择 32 个 region。公开数据集的审计和初始生成结果记录在 `docs/benchmarks/technology_recognition_v1_phase0_phase1.md`，地点隔离的 clean 128 点基线记录在 `docs/benchmarks/technology_recognition_v1_phase2.md`，完整四地点轮换和 128–4096 点窗口消融记录在 `docs/benchmarks/technology_recognition_v1_window_ablation.md`。

## 6. `feature_extraction`：64 维 IQ 特征提取

该模块封装现有 MATLAB 生成的 C/C++ 动态库。Python 层只负责输入校验、C ABI 调用、特征名称映射和结果组织，不改变原特征算法。

| 文件 | 作用 |
| --- | --- |
| `feature_extraction/__init__.py` | 导出特征提取公共接口 |
| `feature_extraction/contracts.py` | 定义固定 64 维的 `FeatureResult`、特征数量和 schema ID `matlab_iq_features_64_v2`，并支持转换为 `Evidence` |
| `feature_extraction/backend.py` | 使用 `ctypes` 加载动态库，声明 C ABI 参数并逐个样本提取 64 维特征 |
| `feature_extraction/service.py` | 面向 `PreparedDataset` 的业务层；解析采样率、控制样本范围并组织 `FeatureResult` |
| `feature_extraction/feature_map.py` | 加载并校验 `feature_map.json`，保证 64 个特征名称唯一、顺序稳定且分组为 15/21/28 |
| `feature_extraction/resource_paths.py` | 统一定位打包后的动态库、C 头文件、feature map 和说明文档 |
| `feature_extraction/cli.py` | `signal-extract-features` 命令入口，加载数据集并写出特征 NPZ |

`assets` 保存运行时必需且与算法版本绑定的资源：

| 资源 | 作用 |
| --- | --- |
| `assets/__init__.py` | 将资源目录声明为可随 Python 包发布的子包 |
| `feature_map.json` | 64 维特征的索引、代码名、分组和说明映射 |
| `include/iq_feature_c_api.h` | 动态库公开的 C ABI 声明 |
| `native/linux/*.so` | Linux 特征提取动态库 |
| `*_iq_features.md` | 时域、频域、时频域特征说明 |

内部调用关系：

```text
CLI → io.load_prepared_dataset → FeatureExtractionService
    → IQFeatureCtypesBackend → C ABI 动态库
    → FeatureResult → Evidence
```

### 6.1 `feature_classifier`：region 级 64 维特征分类

该模块使用完整连续 region 的 64 维特征训练独立分类器，不使用窗口特征统计。
特征数据构建时通过装配数据的来源信息回读完整 region，统一去直流和复 RMS
归一化，超过 16384 点时保留前 16384 点，然后实际调用特征动态库一次。

| 文件 | 作用 |
| --- | --- |
| `feature_classifier/dataset.py` | 构建 region 级特征 split；训练集可生成 5–20 dB AWGN 特征副本，测试集可生成固定 5 dB 条件 |
| `feature_classifier/model.py` | 定义最小的 `Linear(64, class_count)` 分类器 |
| `feature_classifier/trainer.py` | 仅使用训练 split 计算标准化参数，执行训练、评估并导出 PTH、ONNX、scaler 和 manifest |
| `feature_classifier/service.py` | 加载 scaler 和 ONNX，对一行或多行 64 维特征输出类别概率和 Top-K |
| `feature_classifier/cli.py` | 提供 `build-dataset` 和 `train` 两个子命令 |

```text
固定 IQ split → 完整连续 region → 一次 64 维特征提取
              → 训练集 mean/std 标准化 → Linear(64, 3)
              → ONNX 概率输出
```

## 7. `model_inference`：ONNX 小模型推理

该模块只负责运行已经导出的 ONNX 模型，不依赖 PyTorch 模型定义或训练代码。

| 文件 | 作用 |
| --- | --- |
| `model_inference/__init__.py` | 导出推理公共接口和兼容接口 |
| `model_inference/contracts.py` | 定义单个排序结果 `RankedPrediction` 和批量结果 `ModelInferenceResult`；提供 Top-K 和 `Evidence` 转换 |
| `model_inference/backend.py` | 封装 ONNX Runtime，读取真实输入输出契约，并按 CUDA 优先、CPU 回退策略执行模型 |
| `model_inference/service.py` | 校验 `PreparedDataset` 与 `ModelManifest`，执行批量推理、稳定 softmax，并保留 feature 等辅助输出 |
| `model_inference/ensemble.py` | 对同一 region 的多个模型结果先做模型内窗口概率平均，再做模型间概率平均；按成员 Top1 是否一致输出 `accept` 或 `review_required` |
| `model_inference/labels.py` | 加载和校验按类别索引排序的 label map |
| `model_inference/legacy.py` | 适配旧推理函数、文本输出和异步调用方式 |
| `model_inference/cli.py` | `signal-infer` 命令入口，并保留旧 CLI 输出行为 |
| `model_inference/ensemble_cli.py` | `signal-infer-ensemble` 命令入口，针对一个 `group_id` 运行多个 ONNX 模型并写出结构化 JSON |

内部调用关系：

```text
CLI / Python API → io → PreparedDataset
                 → ModelManifest + ONNXRuntimeBackend
                 → ModelInferenceService
                 → logits + probabilities + auxiliary outputs
                 → ModelInferenceResult → Top-K / Evidence
```

## 8. `modeling`：PyTorch 模型定义

`modeling` 只保存可训练模型结构，不处理数据、不执行训练，也不参与 ONNX 运行时推理。

| 文件 | 作用 |
| --- | --- |
| `modeling/__init__.py` | 导出模型类、兼容别名、注册表和工厂函数 |
| `modeling/registry.py` | 保存模型名称到模型类的映射，通过 `build_model()` 统一实例化 |
| `architectures/__init__.py` | 汇总具体模型结构 |
| `architectures/deepconvnet_1d.py` | 现有一维 DeepConvNet；返回分类 logits 和 300 维 feature |
| `architectures/lstm_iq.py` | LSTM IQ 模型；返回分类 logits 和 128 维 feature |

模型保持双输出约定：

```text
forward(X) → (logits, feature)
```

这使训练、ONNX 导出和运行时辅助特征输出保持一致。

## 9. `training_data`：训练数据装配

`training_data` 是不依赖 PyTorch 的离线数据装配层。它不读取连续原始信号、不重新执行检测或切片，也不训练模型；输入是 `preparation` 已生成的带标签 NPZ，输出是固定的 `train.npz`、`validation.npz`、`test.npz` 和审计报告。

| 文件 | 作用 |
| --- | --- |
| `training_data/__init__.py` | 导出 manifest 契约和装配 API |
| `training_data/contracts.py` | 严格校验 split manifest、标签映射、输入来源和装配参数 |
| `training_data/assembly.py` | 统一 `burst_id/region_id`，按区域和类别平衡抽样，执行共同的窗口标准化，写出固定 split 并检查泄漏 |
| `training_data/cli.py` | `signal-assemble-training` 命令入口 |

装配顺序为：

```text
split manifest
    └──► 已切片 NPZ 校验
         └──► source_id + 原始 region 分组
              └──► 每类等量区域 + 每区域等量窗口
                   └──► 每窗口去直流和 RMS 归一化
                        └──► 固定 split NPZ + assembly_report.json
```

输出中的 `group_id` 是整个装配包内唯一的区域编号，`burst_id` 是它的训练兼容别名；原文件中的区域编号保存在 `source_region_id`，来源保存在 `sample_source_id`，因此不会丢失回溯关系。

## 10. `training`：离线训练与 ONNX 导出

`training` 是离线开发模块，依赖 PyTorch、Matplotlib、ONNX 和 ONNX Runtime，不属于部署时主分析流程的必需部分。

| 文件 | 作用 |
| --- | --- |
| `training/__init__.py` | 导出训练循环、损失函数、早停和 ONNX 导出接口 |
| `training/fixed_splits.py` | 读取 `training_data` 的固定三个 split，复核标签、shape 和来源/区域隔离，并直接构建 DataLoader |
| `training/augmentation.py` | 训练批次中的随机频移、频谱翻转和复 AWGN；不改写固定数据集 |
| `training/losses.py` | 标准训练使用的 LogitNorm 和 Label Smoothing 损失 |
| `training/splitting.py` | 随机样本划分和基于 `burst_id` 的分组划分，防止同一 burst 跨集合泄漏 |
| `training/trainer.py` | 设备选择、优化器、训练/验证循环、早停、指标保存、测试和曲线绘制 |
| `training/exporter.py` | 导出带 `output` 和 `feature` 双输出的 ONNX，并使用 ONNX Runtime 验证 |
| `training/cli.py` | `signal-train` 命令和 `train_and_export_model()` Python 入口 |

内部调用关系：

```text
training.cli
    ├──► io.load_signal_dataset
    ├──► training.splitting
    └──► training.trainer
             ├──► modeling.build_model
             ├──► training.losses
             └──► training.exporter
```

### 10.1 `evaluation`：离线模型评估

`evaluation` 对已经训练好的模型执行可重复的离线评估。它读取 `training_data` 生成的固定 `test.npz` 和已有 PyTorch 权重，不重新训练模型，也不修改数据集。V2-D.1 当前实现的是受控复高斯白噪声 SNR 扫描。

| 文件 | 作用 |
| --- | --- |
| `evaluation/__init__.py` | 导出数值 API，并延迟导出端到端评估入口 |
| `evaluation/snr.py` | 添加复高斯白噪声，重新执行逐窗口去直流/RMS 归一化，计算窗口、来源和区域指标 |
| `evaluation/snr_runner.py` | 加载固定测试集与 PTH 权重，执行多次 SNR 试验，写出 JSON、CSV 和曲线图 |
| `evaluation/cli.py` | `signal-evaluate-snr` 命令入口 |

评估顺序为：

```text
固定 test.npz + best_model.pth
    ├──► 干净基线推理
    └──► 各 SNR 下添加受控复高斯白噪声
             └──► 去直流 + RMS 归一化
                  └──► 窗口推理
                       ├──► 窗口/类别/来源指标
                       └──► 区域内平均概率投票
                            └──► JSON + CSV + SNR 曲线
```

### 10.2 `fusion`：Hermes 证据包

`fusion` 不训练新的融合模型，也不直接调用 LLM。当前主路径从未做窗口级归一化的 `prepared_sources` 重建完整连续 region，对整段统一去直流和 RMS 归一化，然后只提取一次 64 维特征。IQ 分支继续使用装配测试集中的局部窗口，由三个独立 ONNX 模型先在模型内平均窗口概率、再跨模型平均。冻结的周期门控只在成员发生 LTE/DVB-T 分歧时介入；64 维特征不参与改判，只作为最终解释的全局物理背景。

| 文件 | 作用 |
| --- | --- |
| `fusion/region_dataset.py` | 从未归一化 prepared source 窗口校验并重建连续 region，执行一次 region 级标准化 |
| `fusion/region_cli.py` | `signal-rebuild-regions` 命令入口 |
| `fusion/periodicity_gate.py` | 定义 LTE/DVB-T 候选周期、冻结 manifest 合同、region 周期评分和保守门控 |
| `fusion/ensemble_evidence.py` | 组合三模型 ensemble、冻结周期门控和每个 region 的单个 64 维向量，隔离盲输入与私有审计信息 |
| `fusion/references/ofdm_periodicity_gate_reference.md` | 说明周期 score、margin、门限、适用范围和 Hermes 解释边界 |
| `fusion/cli.py` | `signal-build-evidence` 命令入口 |

```text
未归一化 prepared source 窗口
    └──► 按坐标连续拼接完整 region
             └──► region 级去直流 + RMS 归一化
                      └──► 每个 region 只提取一次 64 维特征

装配测试集中的 2048 点窗口
    └──► 三个独立 IQ ONNX 模型
             └──► 模型内窗口均值 → 跨模型均值 + 一致性风险门控

完整 4096 点 region
    └──► 66.67 / 224 / 896 μs 归一化自相关
             └──► 冻结 LTE/DVB-T 周期门控

ensemble + 周期门控
    └──► accept final_label 或 review_required provisional_label

确定性结果 + 全局 64 维特征
    ├──► 匿名盲输入 JSON
    └──► 含真值和来源信息的私有 audit.json
```

### 10.3 `signal-fusion-analysis`：Agent 编排层

项目内 Skill 位于 `skills/signal-fusion-analysis/SKILL.md`。业务侧证据合同为“三模型 IQ ensemble + 冻结周期门控 + 全局 64 维原始特征”，不再包含特征分类器或概率权重融合。Skill 会完整读取三篇特征定义、三类别物理解释参考和周期门控参考。分类与 review 状态由代码冻结：`accept` 才有 `final_label`；`review_required` 只有 `provisional_label`。Hermes 只能解释证据、异常和限制，不能重算门控、修改标签或将 review 私自转为 accept。真值和来源信息只保留在独立私有 audit 中。

## 11. 模块间依赖原则

| 上层模块 | 允许依赖 | 不应承担的职责 |
| --- | --- | --- |
| `contracts` | NumPy 和标准库 | 文件读取、算法和框架运行时 |
| `io` | `contracts` | 原始信号检测、模型或特征算法 |
| `preparation` | `contracts`、`io.writers` | 模型推理和特征提取 |
| `feature_extraction` | `contracts`，CLI 可使用 `io` | 数据切片、模型推理 |
| `model_inference` | `contracts`，入口可使用 `io` | PyTorch 模型定义和训练 |
| `modeling` | PyTorch | 数据加载、训练编排和 ONNX 推理 |
| `training_data` | `contracts`、`io` | 原始信号检测、PyTorch 训练和随机划分 |
| `training` | `io`、`modeling` | 在线分析和 LLM 融合 |
| `evaluation` | `io`、`modeling`、PyTorch；绘图时使用 Matplotlib | 重新训练、重新划分或修改源数据集 |
| `fusion` | `feature_extraction`、`model_inference`、`io` | 训练新模型或生成自然语言结论 |
| `skills/signal-fusion-analysis` | 读取 `fusion` 生成的公开盲输入 | 证据生成、特征提取、模型推理、训练、评估或数据准备业务代码 |

几个关键边界：

1. `PreparedDataset` 是准备、特征提取、训练和推理之间的统一数据接口。
2. `preparation.readers` 读取连续原始采集；`io.loaders` 读取已经形成样本的数据集，两者用途不同。
3. `modeling` 管模型结构，`training` 管模型训练，`model_inference` 管部署后的 ONNX 推理。
4. `FeatureResult` 和 `ModelInferenceResult` 保留完整数值结果，同时可以投影成统一 `Evidence`。
5. 当前业务实现只以 `src/signal_fusion` 为准；旧目录不属于当前运行时合同。

## 12. 命令入口

| 命令 | 所属模块 | 作用 |
| --- | --- | --- |
| `signal-prepare` | `preparation.cli` | 从原始采集生成切片数据集 |
| `signal-inspect` | `preparation.inspection` | 检查切片和原始坐标关系 |
| `signal-extract-features` | `feature_extraction.cli` | 提取 64 维特征 |
| `signal-infer` | `model_inference.cli` | 执行 ONNX 调制识别 |
| `signal-infer-ensemble` | `model_inference.ensemble_cli` | 对一个 region 执行多模型概率平均和一致性风险门控 |
| `signal-rebuild-regions` | `fusion.region_cli` | 从未归一化 prepared source 重建并统一标准化完整 region |
| `signal-assemble-training` | `training_data.cli` | 从已切片数据构建固定、平衡且可审计的训练集合 |
| `signal-train` | `training.cli` | 训练模型并导出 ONNX |
| `signal-evaluate-snr` | `evaluation.cli` | 在干净和受控加噪固定测试集上评估 PTH 模型 |
| `signal-build-evidence` | `fusion.cli` | 组合三模型 ensemble、冻结周期门控与整段特征，生成隔离的盲输入和私有审计 JSON |
| `signal-select-fusion-weight` | `fusion.weight_cli` | 在validation选择融合权重并用冻结权重评估test |
| `signal-feature-classifier` | `feature_classifier.cli` | 构建 region 特征数据集并训练 64 维线性分类器 |

## 13. CLI 使用与参数参考

### 13.1 两种等价的运行方式

执行 `python -m pip install -e .` 后，可以使用简短命令。未安装项目时，可以从项目根目录设置 `PYTHONPATH=src` 并直接运行模块。两种方式调用的是同一个 `main()` 函数：

| 简短命令 | 未安装项目时的等价命令 |
| --- | --- |
| `signal-prepare` | `PYTHONPATH=src python -m signal_fusion.preparation.cli` |
| `signal-inspect` | `PYTHONPATH=src python -m signal_fusion.preparation.inspection` |
| `signal-extract-features` | `PYTHONPATH=src python -m signal_fusion.feature_extraction.cli` |
| `signal-infer` | `PYTHONPATH=src python -m signal_fusion.model_inference.cli` |
| `signal-infer-ensemble` | `PYTHONPATH=src python -m signal_fusion.model_inference.ensemble_cli` |
| `signal-rebuild-regions` | `PYTHONPATH=src python -m signal_fusion.fusion.region_cli` |
| `signal-assemble-training` | `PYTHONPATH=src python -m signal_fusion.training_data.cli` |
| `signal-train` | `PYTHONPATH=src python -m signal_fusion.training.cli` |
| `signal-evaluate-snr` | `PYTHONPATH=src python -m signal_fusion.evaluation.cli` |
| `signal-build-evidence` | `PYTHONPATH=src python -m signal_fusion.fusion.cli` |
| `signal-select-fusion-weight` | `PYTHONPATH=src python -m signal_fusion.fusion.weight_cli` |
| `signal-feature-classifier` | `PYTHONPATH=src python -m signal_fusion.feature_classifier.cli` |

以下示例使用简短命令。需要查看当前代码提供的帮助时，可以执行：

```bash
signal-prepare --help
signal-inspect --help
signal-extract-features --help
signal-infer --help
signal-infer-ensemble --help
signal-rebuild-regions --help
signal-assemble-training --help
signal-train --help
signal-evaluate-snr --help
signal-build-evidence --help
signal-select-fusion-weight --help
```

### 13.2 `signal-prepare`

用途：读取 SigMF、MAT、DAT 或 BIN 原始连续 IQ，执行区域检测、分段、定长窗口切片和归一化，输出标准 NPZ 数据集及 JSON 摘要。

基本格式：

```bash
signal-prepare \
  --input_path <原始文件> \
  --output_path <输出数据集.npz> \
  --source_id <可读来源标识> \
  [其他参数]
```

#### 必需参数

| 参数 | 说明 |
| --- | --- |
| `--input_path` | 原始文件路径。SigMF 可传 `.sigmf-data` 或 `.sigmf-meta` |
| `--output_path` | 输出 NPZ 路径；同时生成 `<文件名>_dataset_summary.json` |
| `--source_id` | 用户定义的可读来源标识，例如 `lora_capture_001`，不要求使用哈希 |

#### 格式与 Reader 参数

| 参数 | 默认值 | 支持值或说明 |
| --- | --- | --- |
| `--data_format` | `auto` | `auto`、`sigmf`、`mat`、`dat`、`bin` |
| `--meta_path` | 无 | SigMF metadata 路径；无法根据文件名自动配对时使用 |
| `--x_key` | 无 | MAT 中保存连续 IQ 的字段名；未指定时由 Reader 自动选择 |
| `--sample_rate` | 无 | 采样率 Hz；DAT/BIN 必填，MAT 缺少采样率字段时也应提供 |
| `--center_frequency` | 无 | 可选中心频率 Hz，主要用于 MAT、DAT/BIN 元数据 |
| `--iq_format` | `complex64` | 当前 DAT/BIN Reader 仅支持 `complex64` |

#### Detector 与输出窗口参数

| 参数 | 默认值 | 支持值或说明 |
| --- | --- | --- |
| `--detector` | `energy_v1` | `energy_v1`、`ble_packet_v1`、`full_signal` 或 `fixed_blocks` |
| `--label` | 无 | 写入所有窗口的整数类别；`energy_v1` 未指定时使用 `0` |
| `--class_name` | 无 | 可读类别名；`energy_v1` 未指定时使用 `LoRa` |
| `--seq_len` | `128` | 每个输出窗口的 IQ 点数 |
| `--hop_len` | `128` | 相邻窗口起点之间的 IQ 点数；小于 `seq_len` 时产生重叠窗口 |
| `--normalize` | `rms` | `rms`、`none`、`peak`；`energy_v1` 不支持 `peak` |
| `--remainder` | `drop` | `drop`、`pad`、`zero_pad`；`energy_v1` 使用 `drop` 或 `zero_pad`，通用流程使用 `drop` 或 `pad` |
| `--remove_dc` | 默认启用 | 去除每个窗口的复数均值 |
| `--no_remove_dc` | — | 关闭去直流，与 `--remove_dc` 互斥 |

#### 可选重采样参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--target_sample_rate` | 无 | 目标采样率 Hz；兼容写法为 `--target-sample-rate`。不提供时不重采样，提供后在检测和区域整理之后、窗口切片之前对完整区域执行多相重采样 |

启用重采样时需注意两个坐标域：

- Detector、`min_region_samples`、`merge_gap_samples`、`pad_before_samples` 和 `pad_after_samples` 均按原始采样率解释。
- `seq_len` 和 `hop_len` 均按目标采样率解释。
- 每个窗口在 NPZ 中同时记录原始采样率的 `source_*` 坐标和目标采样率的 `target_*` 坐标；无前缀的 `window_*`、`region_*` 坐标等同于目标坐标。
- `energy_v1` 未指定目标采样率时继续使用旧版 burst 数据集生成路径；指定目标采样率时使用通用 region pipeline，以便输出双坐标元数据。

#### `full_signal` 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--start_sample` | `0` | 把全文件方式的区域起点限制在该绝对采样点 |
| `--end_sample` | 文件末尾 | 把全文件方式的区域终点限制在该绝对采样点 |

#### `fixed_blocks` 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--block_size_samples` | `4096` | 每个固定 region 的复数 IQ 点数 |
| `--block_count` | 全部完整 block | 在完整文件范围内均匀选择的 region 数量；不够一个 block 的尾部丢弃 |

`fixed_blocks` 保留相邻 region 的边界，不会因为两个 block 首尾相接而被 segmentation 层重新合并。若 `seq_len=128`、`hop_len=128`、`block_size_samples=4096`，每个 region 会输出全部 32 个非重叠窗口。

#### 通用区域整理参数

这些参数用于通用 region pipeline；Detector 输出区域后，由 segmentation 层执行过滤、合并和扩展。`energy_v1` 只有在启用目标采样率、转入通用 pipeline 时才使用这些参数。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--min_region_samples` | `1` | 保留区域所需的最少 IQ 点数 |
| `--merge_gap_samples` | `0` | 两个区域间隔不超过该点数时合并 |
| `--pad_before_samples` | `0` | 区域起点向前扩展的 IQ 点数 |
| `--pad_after_samples` | `0` | 区域终点向后扩展的 IQ 点数 |

#### `energy_v1` 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--chunk_size` | `1000000` | 分块检测时每块读取的 IQ 点数 |
| `--window_ms` | `1.0` | 移动平均功率窗口长度，单位 ms |
| `--start_threshold_db` | `6.0` | 高于噪声底多少 dB 时开始信号区域 |
| `--end_threshold_db` | `5.0` | 低于相应阈值时准备结束信号区域 |
| `--min_signal_ms` | `8.0` | 最短信号持续时间，单位 ms |
| `--min_gap_ms` | `2.0` | 小于该间隔的候选区域可被合并，单位 ms |
| `--pad_before_ms` | `0.5` | 检测区域向前扩展时间，单位 ms |
| `--pad_after_ms` | `0.2` | 检测区域向后扩展时间，单位 ms |
| `--release_windows` | `2` | 连续满足结束条件的窗口数 |
| `--noise_percentile` | `20.0` | 估计噪声底时使用的功率百分位 |
| `--noise_probe_count` | `8` | 在录制范围内均匀抽取的噪声探测段数量 |
| `--ignore_initial_ms` | `0.0` | 检测时忽略开头的时间，单位 ms |
| `--window_power_ratio` | `0.05` | 区域细化时的窗口功率比例阈值 |

#### `ble_packet_v1` 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--ble_channel` | `37` | BLE 信道编号 |
| `--ble_symbol_rate` | `1000000` | BLE 符号率，单位 symbols/s |
| `--ble_access_address` | `0x8E89BED6` | Access Address，接受十进制或 `0x` 十六进制写法 |
| `--ble_crc_init` | `0x555555` | 24 位 CRC 初值，接受十进制或 `0x` 十六进制写法 |
| `--ble_chunk_size` | `2000000` | 分块扫描时每块读取的 IQ 点数 |
| `--ble_min_sync_matches` | `35` | 同步序列允许的最少匹配位数 |
| `--ble_max_payload_length` | `37` | 允许的最大 payload 字节数 |

使用 `full_signal` 对 MAT 全文件切片：

```bash
signal-prepare \
  --input_path "data/raw/example.mat" \
  --output_path "data/processed/slices/example_128.npz" \
  --source_id "example_capture" \
  --data_format mat \
  --detector full_signal \
  --seq_len 128 \
  --hop_len 128 \
  --normalize rms
```

使用 `energy_v1` 处理 SigMF：

```bash
signal-prepare \
  --input_path "data/raw/lora/example.sigmf-data" \
  --output_path "data/processed/slices/lora_128.npz" \
  --source_id "lora_capture_001" \
  --data_format sigmf \
  --detector energy_v1 \
  --seq_len 128 \
  --hop_len 128 \
  --normalize rms \
  --remainder drop
```

以 LoRa 为例，先在原始 1 MS/s 信号上进行能量检测，再将检测区域重采样到统一的 4 MS/s，并在目标采样率上切出 128 点窗口：

```bash
signal-prepare \
  --input_path "data/raw/lora/example.sigmf-data" \
  --output_path "data/processed/slices/lora_4msps_128.npz" \
  --source_id "lora_capture_001_4msps" \
  --data_format sigmf \
  --detector energy_v1 \
  --target_sample_rate 4000000 \
  --seq_len 128 \
  --hop_len 128 \
  --normalize rms \
  --remainder drop
```

### 13.3 `signal-inspect`

用途：读取 preparation 生成的 NPZ，绘制区域概览和窗口图，检查切片坐标、窗口间距及其与原始信号的对应关系。

对于 `dual_rate_v1` 数据集，命令会自动区分两个坐标域：读取和绘制原始 IQ 区域时使用 `source_*` 坐标及 `source_sample_rate`，绘制 NPZ 内窗口时使用目标坐标及 `sample_rate`。通常无需手工换算坐标；若显式传入 `--sample_rate`，它必须与原始采样率元数据一致。

基本格式：

```bash
signal-inspect \
  --npz_path <切片数据集.npz> \
  --output_dir <图片输出目录> \
  [原始文件参数]
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--npz_path` | 必填 | 要检查的标准或旧版切片 NPZ |
| `--output_dir` | 必填 | PNG 图片输出目录 |
| `--raw_path` | 无 | 可选原始采集文件；提供后可生成原始区域概览。兼容别名为 `--data_path` |
| `--metadata_path` | 无 | 可选 SigMF metadata 路径。兼容别名为 `--meta_path` |
| `--data_format` | `auto` | 原始文件格式：`auto`、`sigmf`、`mat`、`dat`、`bin` |
| `--sample_rate` | 无 | 原始 DAT/BIN 的采样率，或覆盖其他格式采样率 |
| `--center_frequency` | 无 | 可选原始文件中心频率 |
| `--x_key` | 无 | 原始 MAT 中的 IQ 字段名 |
| `--max_regions` | `5` | 最多绘制多少个区域；兼容别名为 `--max_bursts` |
| `--windows_per_region` | `3` | 每个区域最多绘制多少个窗口；兼容别名为 `--windows_per_burst` |
| `--max_plot_points` | `20000` | 区域概览图最多绘制的原始 IQ 点数，较长区域会降采样显示 |

只检查 NPZ 内的窗口：

```bash
signal-inspect \
  --npz_path "data/processed/slices/lora_128.npz" \
  --output_dir "outputs/lora_inspection"
```

同时与 SigMF 原始数据对照：

```bash
signal-inspect \
  --npz_path "data/processed/slices/lora_128.npz" \
  --raw_path "data/raw/lora/example.sigmf-data" \
  --metadata_path "data/raw/lora/example.sigmf-meta" \
  --data_format sigmf \
  --output_dir "outputs/lora_inspection" \
  --max_regions 5 \
  --windows_per_region 3
```

没有提供或无法找到原始采集文件时，命令仍会生成切片窗口图，但跳过原始区域概览。

### 13.4 `signal-extract-features`

用途：读取已经形成样本的数据集，通过 MATLAB 生成的 C ABI 动态库逐样本提取 64 维特征，并写出特征 NPZ。

基本格式：

```bash
signal-extract-features \
  --data_path <IQ数据集> \
  --output_path <特征结果.npz> \
  [其他参数]
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--data_path` | 必填 | 输入 IQ 数据集路径 |
| `--output_path` | 必填 | 输出特征 NPZ 路径 |
| `--data_format` | `auto` | `auto`、`pkl`、`pickle`、`mat`、`npz`、`npy`、`dat`、`bin` |
| `--seq_len` | 无 | 可选序列长度校验；DAT/BIN 输入必须提供该参数以形成定长样本 |
| `--sample_rate` | 无 | 采样率 Hz；未提供时尝试读取 `sample_rate`、`sampling_rate`、`fs` 或 `Fs` |
| `--label_path` | 无 | NPY 输入可使用的独立标签 `.npy` 路径 |
| `--x_key` | 无 | MAT/NPZ 中 IQ 样本数组的字段名 |
| `--y_key` | 无 | MAT/NPZ 中标签数组的字段名 |
| `--max_samples` | 无 | 只提取前 N 个样本，用于 smoke test |
| `--dll_dir` | 包内平台目录 | 指定动态库所在目录 |
| `--dll_path` | 自动选择包内库 | 显式指定 `extractAllFeatures` 动态库文件 |
| `--dependency_dir` | 无 | 附加动态库依赖目录；参数可以重复传入，主要供 Windows 使用 |
| `--progress_every` | `100` | 每处理 N 个样本打印一次进度；`0` 表示关闭进度输出 |

输入必须能被转换为：

```text
X: float32 [N, 2, seq_len]
```

采样率是特征算法的必需输入。输入 NPZ/MAT 已包含采样率字段时可以省略 `--sample_rate`，否则必须显式提供。

示例：

```bash
signal-extract-features \
  --data_path "data/processed/slices/lora_128.npz" \
  --data_format npz \
  --output_path "data/processed/extraction/lora_features.npz" \
  --sample_rate 1000000 \
  --max_samples 100
```

其中 `1000000` 仅为命令格式示例，实际使用时必须替换为采集数据的真实采样率。

输出 NPZ 固定包含：

```text
features       float32 [N, 64]
sample_rate    float64 标量
seq_len        int64 标量
feature_count  int64 标量，固定为 64
feature_schema_id Unicode 标量，固定为 matlab_iq_features_64_v2
feature_names  64 个按索引排列的特征名
```

### 13.5 `signal-infer`

用途：读取已经形成样本的 IQ 数据，加载 ONNX 模型和 label map，输出逐样本 Top-K 结果；`vote` 模式还会执行文件级投票。

基本格式：

```bash
signal-infer \
  --signal_path <IQ数据集> \
  --model_path <模型.onnx> \
  --label_map_path <label_map.json> \
  [其他参数]
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--signal_path` | `./data/processed/radioml2016_infer.dat` | 输入 IQ 数据路径 |
| `--data_format` | `dat` | 推荐使用 `auto`、`pkl`、`mat`、`npz`、`npy`、`dat` 或 `bin`；运行时 Loader 不直接读取原始 SigMF |
| `--model_path` | `./outputs_logit_norm/deep_iq_cnn.onnx` | ONNX 模型路径 |
| `--label_map_path` | `./data/processed/label_map.json` | 类别索引到名称的 JSON 映射；类别数量必须与模型 logits 宽度一致 |
| `--batch_size` | `64` | `batch` 模式最多读取的前置样本数；`vote` 模式随机抽取的样本数量 |
| `--seq_len` | `128` | 每个样本的 IQ 点数，必须与数据和模型输入一致 |
| `--sample_mode` | 无 | DAT/BIN 当前仅支持 `record_aligned`；不传时自动使用该值 |
| `--iq_format` | 无 | DAT/BIN 当前仅支持 `complex64`；不传时自动使用该值 |
| `--x_key` | 无 | MAT/NPZ 中输入数组的字段名 |
| `--inference_mode` | `vote` | `batch` 或 `vote` |

两种推理模式：

| 模式 | 行为 |
| --- | --- |
| `batch` | 使用前 `batch_size` 个样本，显示逐样本 Top-K，不进行文件级投票 |
| `vote` | 有放回地随机抽取 `batch_size` 个样本，只统计 Top-1 置信度不低于 80% 的结果并形成文件级判定 |

使用项目现有单类别 LoRa 模型验证推理链路：

```bash
signal-infer \
  --signal_path "data/processed/slices/sigmf_lora_dataset_128.npz" \
  --data_format npz \
  --model_path "radioml-iq-modulation/training/sigmf_lora/deep_iq_cnn.onnx" \
  --label_map_path "radioml-iq-modulation/training/sigmf_lora/label_map.json" \
  --seq_len 128 \
  --batch_size 5 \
  --inference_mode batch
```

该历史模型只有一个 LoRa 类别，只适合验证 ONNX 加载、输入格式和输出流程，不代表多类别识别性能。

### 13.5.1 `signal-infer-ensemble`

用途：从已经带有 region/window 边界元数据的窗口数据集中选出一个完整 region，使用三个输入契约和标签顺序一致的 ONNX 模型分别推理。每个模型先对 region 内窗口概率取平均，再对三个模型的 region 概率取平均。三个成员 Top1 一致时输出 `accept`，否则输出 `review_required`。该命令不调用特征分类器或 Hermes，也不自动修改集成标签。

```bash
signal-infer-ensemble \
  --dataset_path <包含group_id的test.npz> \
  --model_path <seed44.onnx> <seed45.onnx> <seed46.onnx> \
  --label_map_path <label_map.json> \
  --group_id <整数> \
  --output <结果.json>
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--dataset_path` | 必填 | 标准 PreparedDataset NPZ；必须含 `group_id` 以及完整的 region/window 起止位置元数据 |
| `--model_path` | 必填 | 恰好三个 ONNX 路径；模型必须具有相同输入长度和标签顺序 |
| `--label_map_path` | 必填 | 所有成员共同使用的标签映射 |
| `--group_id` | 必填 | 本次推理的 region 标识；命令会选取该 region 的全部窗口 |
| `--output` | 必填 | 结构化 JSON 输出路径 |
| `--batch_size` | `64` | 每个模型的窗口推理批大小 |
| `--overwrite` | 关闭 | 允许覆盖已有输出 |

输出中的 `ensemble.region_top3` 是模型间平均概率结果；`members` 保存每个模型的 region Top3 和窗口 Top1；`decision_status` 只表达模型一致性风险，不是真实标签正确性的保证。该 JSON 会保留模型路径、数据路径和 `source_id` 以便实验审计，不应原样传给 Hermes；后续若调用 Hermes，应另外生成去除路径等内部信息的最小证据输入。

### 13.6 `signal-assemble-training`

用途：根据显式 JSON manifest，把多个已经切片的单类别 NPZ 装配成固定、类别平衡且无采集源泄漏的训练、验证和测试文件。该命令只依赖项目基础依赖，不需要安装 PyTorch。

基本格式：

```bash
signal-assemble-training \
  --manifest <装配配置.json> \
  --output-dir <输出目录> \
  [--overwrite]
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--manifest` | 必填 | split manifest；其中的相对数据路径按 manifest 所在目录解析 |
| `--output-dir` | 必填 | 输出 `train.npz`、`validation.npz`、`test.npz` 和 `assembly_report.json` 的目录 |
| `--overwrite` | 默认关闭 | 明确允许覆盖上述已有装配结果；源切片 NPZ 始终只读 |

当前 LoRa/Zigbee/BLE 第一版配置可以直接执行：

```bash
signal-assemble-training \
  --manifest configs/training/three_system_v1.json \
  --output-dir data/processed/training/lora_zigbee_ble_v1
```

未安装项目时使用：

```bash
PYTHONPATH=src python -m signal_fusion.training_data.cli \
  --manifest configs/training/three_system_v1.json \
  --output-dir data/processed/training/lora_zigbee_ble_v1
```

也可以直接调用 Python API：

```python
from signal_fusion.training_data import assemble_training_dataset

result = assemble_training_dataset(
    "configs/training/three_system_v1.json",
    "data/processed/training/lora_zigbee_ble_v1",
)
print(result.report_path)
```

manifest 负责显式指定每个来源属于 `train`、`validation` 或 `test`。`regions_per_class: "minimum"` 按集合内区域最少的类别确定每类区域数；`windows_per_region` 固定每个区域贡献的窗口数。`window_selection` 支持 `uniform` 和 `top_energy`：后者根据源切片中 RMS 归一化前的 `normalization_scale`，选择每个 region 内能量最高的窗口，入选窗口仍按时间顺序写出，并以 `source_window_rms` 保留原始尺度。两种策略都是确定性的，不进行样本级随机拆分。

`signal-train --dataset_dir` 可以直接使用这三个文件，并在训练前再次检查采集源和区域是否交叉。不要把三个文件重新拼接后再随机划分，否则会破坏这里建立的采集源隔离。

### 13.7 `signal-train`

用途：训练 PyTorch 模型、保存最佳 PTH 权重和指标并导出 ONNX。可以传入一个待划分的数据文件，也可以直接使用 `training_data` 生成的固定三个 split。

基本格式：

```bash
signal-train \
  (--data_path <带标签数据集> | --dataset_dir <固定split目录>) \
  --save_dir <训练输出目录> \
  --class_num <类别数量> \
  [其他参数]
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--data_path` | `./data/processed/radioml2016_train.mat` | 带标签训练数据路径 |
| `--dataset_dir` | 无 | 包含 `train.npz`、`validation.npz`、`test.npz` 的装配目录；与 `--data_path` 互斥，且不会再次划分 |
| `--data_format` | `mat` | `auto`、`pkl`、`mat`、`npz`、`npy` |
| `--label_path` | 无 | NPY 输入对应的独立标签文件 |
| `--x_key` | 无 | MAT/NPZ 中训练样本字段名 |
| `--y_key` | 无 | MAT/NPZ 中标签字段名 |
| `--save_dir` | `./outputs_logit_norm` | PTH、ONNX、指标和曲线输出目录 |
| `--onnx_filename` | `deep_iq_cnn.onnx` | 导出的 ONNX 文件名 |
| `--model_name` | `deepconvnet_1d` | 当前支持 `deepconvnet_1d` 和 `lstm_iq` |
| `--class_num` | `11` | 模型输出类别数量，必须与标签集合一致 |
| `--batch_size` | `128` | DataLoader 批次大小 |
| `--lr_model` | `0.008` | 初始学习率 |
| `--optimizer` | `sgd` | `sgd` 或 `adam` |
| `--max_epoch` | `30` | 最大训练轮数 |
| `--max_samples` | 无 | 随机划分模式下最多使用多少个样本，适合 smoke test；分组模式不支持 |
| `--split_mode` | `random` | `random` 或 `group` |
| `--patience` | `15` | 验证损失连续多少轮未改善后早停 |
| `--seed` | `44` | NumPy/PyTorch 数据选择和划分使用的随机种子 |
| `--num_workers` | `2` | DataLoader 工作进程数；调试时可设为 `0` |
| `--device` | `auto` | `auto`、`cpu`、`cuda`；请求 CUDA 但不可用时回退 CPU |
| `--no_plot` | 默认关闭 | 不弹出 Matplotlib 窗口，但仍保存 `training_curves.png` |
| `--no_amp` | 默认关闭 | 关闭 CUDA 自动混合精度；CPU 下 AMP 本来就不会启用 |
| `--loss` | `logit_norm` | `ce`、`logit_norm` 或 `ls` |
| `--temp` | `0.2` | `logit_norm` 损失的温度系数 |
| `--epsilon` | `0.1` | `ls` 标签平滑损失的平滑系数 |
| `--frequency_shift_probability` | `0.0` | 训练窗口执行随机频移的概率；`0` 表示关闭 |
| `--frequency_shift_max_fraction` | `0.0` | 最大频移占采样率的比例；`0.1` 表示在 `[-0.1Fs,+0.1Fs]` 中均匀采样 |
| `--spectral_inversion_probability` | `0.0` | 训练窗口执行复共轭频谱翻转的概率；`0` 表示关闭 |
| `--awgn_probability` | `0.0` | 仅在训练批次中动态加入 AWGN 的逐窗口概率；`0` 表示关闭 |
| `--awgn_snr_min` | `5.0` | 训练 AWGN 随机 SNR 下界，单位 dB |
| `--awgn_snr_max` | `20.0` | 训练 AWGN 随机 SNR 上界，单位 dB |

三种增强都只作用于送入模型的当前训练批次，不改写 `train.npz`，也不作用于验证集和测试集。随机频移对每个入选窗口独立采样频移量，随后重新去直流并执行复信号 RMS 归一化；频谱翻转通过复共轭实现，即 `I` 不变、`Q` 取反。AWGN 的 SNR 表示“原窗口总功率 / 本次新增噪声功率”，加噪后同样重新标准化。随机过程共同使用 `--seed`，不再增加第二套种子配置。

数据划分行为：

| 模式 | 说明 |
| --- | --- |
| `random` | 按样本随机划分约 70%/15%/15% 的训练、验证、测试集合 |
| `group` | 按 NPZ 元数据中的 `burst_id` 分组，保证同一 burst 不跨集合；不能同时使用 `--max_samples` |
| 固定 split | 使用 `--dataset_dir` 时直接读取三个集合，并复核 `dataset_id`、标签、`source_id`、`group_id` 和原始区域；不使用 `--split_mode` |

训练示例：

```bash
signal-train \
  --data_path "data/processed/training/modulation_dataset.npz" \
  --data_format npz \
  --save_dir "outputs/modulation_training" \
  --onnx_filename "modulation.onnx" \
  --model_name deepconvnet_1d \
  --class_num 11 \
  --batch_size 128 \
  --optimizer adam \
  --lr_model 0.001 \
  --loss logit_norm \
  --max_epoch 30 \
  --device auto \
  --no_plot
```

当前三体制固定集合的 CUDA smoke training 示例：

```bash
signal-train \
  --dataset_dir data/processed/training/lora_zigbee_ble_v1 \
  --save_dir outputs/lora_zigbee_ble_v1_smoke \
  --onnx_filename lora_zigbee_ble_smoke.onnx \
  --model_name deepconvnet_1d \
  --class_num 3 \
  --batch_size 64 \
  --optimizer adam \
  --lr_model 0.001 \
  --loss ce \
  --max_epoch 3 \
  --device cuda \
  --no_plot
```

三体制固定集合的动态 AWGN 训练示例（约一半训练窗口保持干净，另一半随机加入 5–20 dB AWGN）：

```bash
signal-train \
  --dataset_dir data/processed/training/lora_zigbee_ble_v2_4msps \
  --save_dir outputs/lora_zigbee_ble_v3_awgn_5_20 \
  --onnx_filename lora_zigbee_ble_v3_awgn_5_20.onnx \
  --model_name deepconvnet_1d \
  --class_num 3 \
  --batch_size 64 \
  --optimizer adam \
  --lr_model 0.001 \
  --loss ce \
  --max_epoch 30 \
  --patience 15 \
  --seed 44 \
  --device cuda \
  --awgn_probability 0.5 \
  --awgn_snr_min 5 \
  --awgn_snr_max 20 \
  --no_plot
```

训练目录最终包含：

```text
best_model.pth       验证损失最低的 PyTorch 权重
<onnx_filename>      双输出 ONNX 模型
train_acc.npy        每轮训练准确率
train_loss.npy       每轮训练损失
val_acc.npy          每轮验证准确率
val_loss.npy         每轮验证损失
training_curves.png  准确率与损失曲线
label_map.json        固定 split 对应的模型类别映射
training_result.json  固定 split 训练的设备、测试指标和数据集信息
```

导出的 ONNX 保持统一双输出契约：

```text
input    [batch, 2, seq_len]
output   [batch, class_num]
feature  [batch, feature_dimension]
```

其中 DeepConvNet 的 `feature_dimension` 为 300，LSTM IQ 模型为 128。

### 13.8 `signal-evaluate-snr`

用途：使用已有 PTH 权重和固定 `test.npz`，先得到干净基线，再在多个受控加噪 SNR 下重复推理，衡量窗口级识别和长信号区域投票的鲁棒性。该命令不训练模型，也不会覆盖或修改测试数据。

基本格式：

```bash
signal-evaluate-snr \
  --dataset_dir <固定split目录> \
  --model_path <best_model.pth> \
  --output_dir <评估输出目录> \
  [其他参数]
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--dataset_dir` | 必填 | 包含固定 `test.npz` 的装配数据集目录 |
| `--model_path` | 必填 | 与测试集类别和输入长度匹配的 PyTorch 权重 |
| `--output_dir` | 必填 | JSON、CSV 和曲线图输出目录 |
| `--snr_db` | `20 15 10 5 0 -5 -10` | 一个或多个新增复高斯白噪声 SNR，单位 dB |
| `--trials` | `5` | 每个 SNR 使用不同随机噪声重复评估的次数 |
| `--seed` | `44` | 第一轮噪声随机种子；后续轮次依次加一 |
| `--batch_size` | `256` | 推理批次大小 |
| `--device` | `auto` | `auto`、`cpu` 或 `cuda`；明确请求 CUDA 但不可用时会报错 |
| `--model_name` | `deepconvnet_1d` | 用于重建 PTH 对应的模型结构 |
| `--overwrite` | 默认关闭 | 允许覆盖已有同名评估结果 |
| `--no_plot` | 默认关闭 | 不生成 `snr_accuracy_curve.png`，仍生成 JSON 和 CSV |

当前三体制 V2 模型的 CUDA 评估命令：

```bash
PYTHONPATH=src \
/home/dianci/miniconda3/envs/radioml_amc_clean/bin/python \
  -m signal_fusion.evaluation.cli \
  --dataset_dir data/processed/training/lora_zigbee_ble_v2_4msps \
  --model_path outputs/lora_zigbee_ble_v2_4msps/best_model.pth \
  --output_dir outputs/lora_zigbee_ble_v2_4msps/snr_evaluation \
  --snr_db 20 15 10 5 0 -5 -10 \
  --trials 5 \
  --seed 44 \
  --batch_size 256 \
  --device cuda
```

输出文件：

```text
snr_evaluation.json      完整配置、逐次试验、混淆矩阵、分类别和来源指标
snr_accuracy.csv         适合表格分析的逐试验核心指标
snr_accuracy_curve.png   窗口准确率与区域投票准确率的 SNR 曲线
```

这里的 SNR 是“原测试窗口总功率 / 本次新增噪声功率”，不是对真实空口 SNR 的估计，因为测试窗口本身已经包含采集噪声。噪声按每个窗口的实际复信号功率标定；加噪后重新执行与训练数据装配一致的去直流和 RMS 归一化。区域级结果对相同 `group_id` 的类别概率取平均后再选择类别，用于模拟长信号场景下的多窗口投票。

### 13.9 `signal-build-evidence`

先用 `signal-rebuild-regions` 从未归一化 prepared source 构造连续 region 数据集：

```bash
PYTHONPATH=src python -m signal_fusion.fusion.region_cli \
  --dataset_path <窗口测试集.npz> \
  --prepared_root <prepared_sources目录> \
  --output_path <continuous_regions.npz>
```

随后对 `continuous_regions.npz` 提取特征，再生成 evidence：

```bash
PYTHONPATH=src python -m signal_fusion.fusion.cli \
  --dataset_path <窗口测试集.npz> \
  --region_dataset_path <continuous_regions.npz> \
  --feature_path <continuous_region_features.npz> \
  --periodicity_gate_manifest <periodicity_gate_manifest.json> \
  --model_path <seed44.onnx> <seed45.onnx> <seed46.onnx> \
  --label_map_path <label_map.json> \
  --output_dir <evidence输出目录> \
  --all_groups
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--dataset_path` | 必填 | 保留局部窗口的装配测试集 |
| `--region_dataset_path` | 必填 | `signal-rebuild-regions` 生成的连续 region 数据集 |
| `--feature_path` | 必填 | 对连续 region 数据集提取的 64 维特征 NPZ；包含 `group_id`、来源 ID 与来源 region ID，用于逐行对齐校验 |
| `--periodicity_gate_manifest` | 必填 | 由 validation 冻结、包含候选周期和可靠性门限的 manifest |
| `--model_path` | 必填 | 恰好三个输入和标签顺序一致的 ONNX 模型 |
| `--label_map_path` | 必填 | 连续类别索引到名称的 JSON |
| `--output_dir` | 必填 | `blind/` 和 `audit.json` 的共同输出目录 |
| `--group_id` / `--all_groups` | 二选一 | 处理一个指定 region 或全部 region |
| `--case_id_start` | `1` | 连续匿名案例编号的起点 |
| `--batch_size` | `64` | ONNX 推理批次大小 |
| `--overwrite` | 默认关闭 | 允许覆盖已有证据文件 |

输出位置固定分离：

```text
<output_dir>/blind/case_0001.json  交给 Hermes 的公开盲输入
<output_dir>/audit.json             实验结束后使用的私有映射和真值
```

公开文件只保留安全的采样率、时长、窗口配置、匿名化的三模型 ensemble 证据、周期测量、冻结门控状态和整段 region 的一个 64 维特征向量。真实标签、`group_id`、来源文件、地点、增益和中心频率只写入私有 `audit.json`。代码已经生成确定性的 `decision_status`：`accept` case 提供冻结 `final_label`；review case 的 ensemble 候选只作为 `provisional_label`。

生成前会校验窗口数据、连续 region 和特征文件的普通行标识完全一致，并校验 64 个特征名严格遵循 `feature_map.json` 顺序；这些标识仅用于本地装配检查，不写入 blind case。

公开输入共引用五篇文档：三篇特征定义、一篇 LTE／WiFi／DVB-T 物理解释和一篇冻结周期门控说明。Hermes Skill 只解释已经存在的公开文件，不在同一会话中生成实验条件，也不能重新选择标签。正式盲测应开启新的 Hermes 会话，只向它提供 `blind/case_XXXX.json` 中性路径；得到并保存解释后，评估端才能读取私有 `audit.json` 中相同 case 编号的记录。

### 13.10 `signal-feature-classifier`

先从固定 IQ split 构建 region 级 64 维特征数据集：

```bash
PYTHONPATH=src python -m signal_fusion.feature_classifier.cli build-dataset \
  --dataset-dir data/processed/training/lora_zigbee_ble_v3_top_energy_4msps \
  --output-dir data/processed/features/lora_zigbee_ble_v3_top_energy_4msps_region_features \
  --train-awgn-copies 1 \
  --awgn-snr-min 5 \
  --awgn-snr-max 20 \
  --test-awgn-snr 5 \
  --seed 44
```

然后训练线性特征分类器并导出ONNX：

```bash
PYTHONPATH=src python -m signal_fusion.feature_classifier.cli train \
  --dataset-dir data/processed/features/lora_zigbee_ble_v3_top_energy_4msps_region_features \
  --output-dir outputs/lora_zigbee_ble_v3_top_energy_feature_linear_awgn_5_20 \
  --device cuda \
  --batch-size 64 \
  --learning-rate 0.01 \
  --max-epochs 300 \
  --patience 30 \
  --seed 44
```

`build-dataset`只在训练split中加入随机5–20 dB噪声副本；validation保持clean；
test同时保存clean与固定5 dB记录，便于分别报告结果。`train`输出PTH、ONNX、仅由
训练split计算的mean/std scaler、运行时manifest和完整评估结果。

### 13.11 `signal-select-fusion-weight`

该命令在validation的clean和指定压力条件上运行两个分类分支，遍历候选权重；选定后冻结权重，再评估test。test结果不参与权重选择。

```bash
PYTHONPATH=src python -m signal_fusion.fusion.weight_cli \
  --dataset_dir data/processed/training/lora_zigbee_ble_v3_top_energy_4msps \
  --model_path outputs/lora_zigbee_ble_v3_awgn_5_20/lora_zigbee_ble_v3_awgn_5_20.onnx \
  --label_map_path outputs/lora_zigbee_ble_v3_awgn_5_20/label_map.json \
  --feature_classifier_manifest outputs/lora_zigbee_ble_v3_top_energy_feature_linear_awgn_5_20/feature_classifier_manifest.json \
  --feature_dataset_dir data/processed/features/lora_zigbee_ble_v3_top_energy_4msps_region_features \
  --output_dir outputs/lora_zigbee_ble_v3_top_energy_fusion_weight_v1 \
  --stress_snr_db 5 \
  --weight_step 0.1 \
  --seed 44
```

选择顺序固定为：宏平均准确率、总体准确率、分支冲突样本准确率、最接近等权。输出：

```text
fusion_manifest.json    正式推理加载的冻结权重
fusion_evaluation.json  候选权重以及validation/test详细指标
```

当前数据上validation的228个clean/5 dB案例中两分支均为100%，且没有Top1分歧，因此所有候选权重准确率相同；按并列规则选择IQ `0.5`、特征分类器 `0.5`。test整体准确率三者均为99.702%，两分支在同一个5 dB LoRa案例上同时出错，因此本轮融合没有带来准确率提升。
