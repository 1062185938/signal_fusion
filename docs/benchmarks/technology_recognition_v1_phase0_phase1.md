# Technology Recognition 公开基准 V1：Phase 0 和 Phase 1

## 范围

- 数据集：Technology Recognition (LTE, Wi-Fi and DVB-T)
- 原始数据目录：`/mnt/sda2/mydata/Technology Recognition (LTE, Wi-Fi and DVB-T)`
- 生成数据目录：`data/external/processed/technology_recognition_lte_wifi_dvbt/v1`
- 类别：LTE、WiFi、DVB-T
- 原生采样率：1 Msps
- 地点：gentbrugge、merelbeke、rabot、reep

## 官方示例核查

没有发现丢弃前 100,000 个样本的说明或操作。

- `script_for_accessing_bin_files.m` 从第一个样本开始读取，示例只按指定总长度截取。
- `Processing .bin files and add SNR/inp_bin_ext_snr.m` 从第一个样本开始，仅删除末尾不能组成完整 4096 复数样本块的 remainder。
- 因此 V1 明确记录 `discard_initial_samples = 0`。

均匀选择 region 不代表第一个入选 region 必须从样本零开始。每个文件有 268 个完整源 block，均匀选择 32 个时，第一个入选的是 block 4，从样本 16,384 开始。这是在完整记录范围内抽样，不是固定删除前缀。

## Phase 0 结果

- 扫描到 `.bin` 文件：190
- V1 选中：120
- 每个地点、每个类别：10 个文件
- 每个选中文件：1,100,000 个复数样本
- 包含 NaN 或 Inf 的选中文件：0
- 排除：70
  - 地点不属于 V1：41
  - 非 1 Msps：25
  - Merelbeke 额外 WiFi 频点：3
  - 文件名地点与目录不一致：1

机器可读的完整审计结果为生成目录下的 `inventory.json`。

## Phase 1 数据契约

- Region：4096 个复数样本
- 每个源文件：从完整、非重叠源 block 中均匀选择 32 个 region
- 原始数据起始偏移：0
- Window：128 个复数样本
- Window hop：128
- 每个 region 的窗口数：32
- 窗口选择：保留全部窗口，不再选择 Top-7
- Remainder：drop
- Preparation 归一化：none
- Preparation 去直流：关闭

归一化和去直流推迟到训练数据装配阶段，使后续窗口长度和归一化消融可以复用同一份原始数据契约。

## 生成结果

- Prepared source：120
- Region：3,840
- Window：122,880
- 每个源 NPZ 的 shape：`[1024, 2, 128]`
- 生成数据大小：约 53 MiB
- 源文件摘要 JSON：120

机器可读的完整运行报告为生成目录下的 `preparation_report.json`。

复现命令：

```bash
PYTHONPATH=src python -m signal_fusion.benchmarks.technology_recognition_cli \
  --dataset-root "/mnt/sda2/mydata/Technology Recognition (LTE, Wi-Fi and DVB-T)" \
  --output-dir "data/external/processed/technology_recognition_lte_wifi_dvbt/v1" \
  --prepare \
  --window-sizes 128
```

为避免意外覆盖，目标目录已有 prepared NPZ 时命令会直接失败。

## 验证

全部 NPZ 都经过 shape、dtype、标签、region 分组、窗口顺序、非重叠关系和坐标检查。从 LTE、WiFi、DVB-T 各选择一个 region，使用保存的 32 个窗口重建后与原始 `.bin` 逐样本比较，三者完全一致。

Phase 0 和 Phase 1 不生成训练、验证、测试划分，也不训练模型。下一阶段是按地点装配固定 split。
