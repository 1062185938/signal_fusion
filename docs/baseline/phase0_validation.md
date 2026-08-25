# Phase 0 重构前后验证方法

## 1. 验证原则

后续重构允许目录、包名和 Agent/Skill 入口变化，但不能无意改变三类既有行为：

1. 同一固定 IQ 输入的 62 维特征数量、顺序和数值。
2. MATLAB 生成动态库对外暴露的 C ABI、缓冲区约定和返回码。
3. 同一固定 `[N, 2, 128]` 输入的 ONNX logits 与 300 维网络特征。

重构前执行一次本页流程，重构后在新入口执行等价流程，再与 Phase 0 fixture 比较。测试应优先比较结构和数值，不应依赖日志文字、绝对路径或目录布局。

以下命令均从项目根目录运行。特征提取使用当前默认 Python；PTH/ONNX 校验使用已有环境：

```bash
/home/dianci/miniconda3/envs/radioml_amc_clean/bin/python
```

设置 `PYTHONDONTWRITEBYTECODE=1` 可避免验证过程在源码目录生成 `__pycache__`。

## 2. Fixture 完整性

解析 source catalog：

```bash
env PYTHONDONTWRITEBYTECODE=1 python -c "import json; d=json.load(open('tests/fixtures/phase0_sources.json', encoding='utf-8')); ids=[x['source_id'] for x in d['sources']]; assert len(ids)==len(set(ids)); print(ids)"
```

检查 ONNX 参考文件：

```bash
env PYTHONDONTWRITEBYTECODE=1 python -c "import numpy as np; d=np.load('tests/fixtures/lora_single_class_onnx_reference.npz', allow_pickle=False); assert d.files==['X','logits','features','sample_id','source_id','model_id']; assert d['X'].shape==(8,2,128); assert d['logits'].shape==(8,1); assert d['features'].shape==(8,300); print({k:(d[k].shape,str(d[k].dtype)) for k in d.files})"
```

验收条件：所有 `source_id` 唯一；固定参考文件不含 pickle/object 数组；字段、shape 和 dtype 与基线一致。

## 3. 62 维 feature map

```bash
env PYTHONDONTWRITEBYTECODE=1 python -c "import json,collections; d=json.load(open('iq_feature_extraction/references/feature_map.json', encoding='utf-8')); f=d['features']; assert d['feature_count']==len(f)==62; assert [x['index'] for x in f]==list(range(62)); assert len({x['code_name'] for x in f})==62; assert collections.Counter(x['group'] for x in f)=={'time_domain':12,'frequency_domain':25,'time_frequency':25}; print(f[0]['code_name'], f[-1]['code_name'])"
```

验收条件：总数 62，索引为 0–61，名称唯一，分组为 12/25/25，首尾名称与基线一致。

## 4. MATLAB 特征提取回放

先用固定 MAT 的前 5 个窗口生成临时结果；输出写到 `/tmp`，不覆盖项目数据：

```bash
env PYTHONDONTWRITEBYTECODE=1 python iq_feature_extraction/scripts/feature_extractor.py \
  --data_path 'data/raw/wifi/WIFI_5_batch;Freq=5230 MHz;Span=80 MHz;Rate=100.0 MHz;0005.mat' \
  --output_path /tmp/signal_fusion_phase0_wifi5_features.npz \
  --data_format mat \
  --x_key iq \
  --seq_len 4096 \
  --sample_rate 100000000 \
  --max_samples 5 \
  --progress_every 0
```

再与 golden NPZ 比较：

```bash
env PYTHONDONTWRITEBYTECODE=1 python -c "import numpy as np; a=np.load('/tmp/signal_fusion_phase0_wifi5_features.npz', allow_pickle=False); b=np.load('data/processed/extraction/wifi_5_features_4096_smoke.npz', allow_pickle=False); assert a.files==b.files; assert a['features'].shape==b['features'].shape==(5,62); assert a['features'].dtype==b['features'].dtype==np.float32; assert np.array_equal(a['feature_names'],b['feature_names']); assert float(a['sample_rate'])==float(b['sample_rate'])==1e8; assert int(a['seq_len'])==int(b['seq_len'])==4096; assert int(a['feature_count'])==int(b['feature_count'])==62; assert np.isfinite(a['features']).all(); assert np.allclose(a['features'],b['features'],rtol=1e-5,atol=1e-6); print(float(np.max(np.abs(a['features']-b['features']))))"
```

Phase 0 实测最大绝对差为 `1.1920928955078125e-07`。重构后达到上述容差即通过，不要求不同机器或编译器下逐 bit 相等。

## 5. C ABI 验证

检查 Linux 导出符号：

```bash
nm -D --defined-only iq_feature_extraction/native/linux/libextractAllFeatures.so
```

必须包含且只暴露以下三个业务入口：

```text
iqFeatureInitialize
iqFeatureExtract
iqFeatureTerminate
```

检查运行时依赖：

```bash
ldd iq_feature_extraction/native/linux/libextractAllFeatures.so
```

不得出现 `not found`。

错误码回放：

```bash
env PYTHONDONTWRITEBYTECODE=1 python -c "import ctypes,numpy as np; lib=ctypes.CDLL('iq_feature_extraction/native/linux/libextractAllFeatures.so'); fp=ctypes.POINTER(ctypes.c_float); f=lib.iqFeatureExtract; f.argtypes=[fp,fp,ctypes.c_int32,ctypes.c_double,fp]; f.restype=ctypes.c_int32; i=np.zeros(16384,np.float32); q=np.zeros(16384,np.float32); out=np.zeros(62,np.float32); lib.iqFeatureInitialize(); assert f(i.ctypes.data_as(fp),q.ctypes.data_as(fp),31,1e6,out.ctypes.data_as(fp))==-1; assert f(i.ctypes.data_as(fp),q.ctypes.data_as(fp),32,0.0,out.ctypes.data_as(fp))==-2; i[0]=np.nan; assert f(i.ctypes.data_as(fp),q.ctypes.data_as(fp),32,1e6,out.ctypes.data_as(fp))==-3; lib.iqFeatureTerminate(); print('C ABI status codes OK')"
```

有效输入的成功路径由上一节真实 MAT 回放覆盖。

## 6. ONNX 图接口验证

```bash
env PYTHONDONTWRITEBYTECODE=1 /home/dianci/miniconda3/envs/radioml_amc_clean/bin/python -c "import onnx; m=onnx.load('radioml-iq-modulation/training/sigmf_lora/deep_iq_cnn.onnx'); dims=lambda v:[d.dim_param or d.dim_value for d in v.type.tensor_type.shape.dim]; assert m.ir_version==8; assert m.opset_import[0].version==18; assert m.graph.input[0].name=='input' and dims(m.graph.input[0])==['batch_size',2,128]; assert [(x.name,dims(x)) for x in m.graph.output]==[('output',['batch_size',1]),('feature',['batch_size',300])]; print('ONNX graph contract OK')"
```

验收条件：输入名、双通道 shape、两个输出名及 shape 不变。后续若更换为多类别生产模型，应建立新的带版本模型基线，不能静默覆盖这个单类别 fixture。

## 7. ONNX 数值回放

用保存的 8 条输入重新运行当前 ONNX，并比较 logits 和中间特征：

```bash
env PYTHONDONTWRITEBYTECODE=1 /home/dianci/miniconda3/envs/radioml_amc_clean/bin/python -c "import numpy as np,onnxruntime as ort; d=np.load('tests/fixtures/lora_single_class_onnx_reference.npz',allow_pickle=False); s=ort.InferenceSession('radioml-iq-modulation/training/sigmf_lora/deep_iq_cnn.onnx',providers=['CPUExecutionProvider']); logits,features=s.run(['output','feature'],{'input':np.ascontiguousarray(d['X'],dtype=np.float32)}); assert np.allclose(logits,d['logits'],rtol=1e-4,atol=1e-5); assert np.allclose(features,d['features'],rtol=1e-4,atol=1e-5); print(float(np.max(np.abs(logits-d['logits']))),float(np.max(np.abs(features-d['features']))))"
```

验收条件：logits 和 feature 均在 `rtol=1e-4, atol=1e-5` 内。固定使用 CPU provider 可减少不同硬件 provider 带来的偏差。

若重构涉及模型构建/导出，还应加载 `best_model.pth`，忽略现有 state dict 的 `total_ops`、`total_params` 两个 THOP 键，并验证 PyTorch 与 ONNX：

- logits 最大差不超过上述容差；Phase 0 实测 `2.2649765014648438e-06`。
- feature 最大差不超过上述容差；Phase 0 实测 `8.761882781982422e-06`。

## 8. 推理包装层验证

当前同步核心可完成实际 loader、label map、softmax 和格式化链路：

```bash
env PYTHONDONTWRITEBYTECODE=1 /home/dianci/miniconda3/envs/radioml_amc_clean/bin/python -c "import sys; sys.path.insert(0,'radioml-iq-modulation/scripts'); import onnx_inference as m; text=m._sync_signal_inference('data/processed/slices/sigmf_lora_dataset_128.npz','radioml-iq-modulation/training/sigmf_lora/deep_iq_cnn.onnx','radioml-iq-modulation/training/sigmf_lora/label_map.json',8,128,'npz',None,None,'X','batch'); assert '数据维度: (4804, 2, 128)' in text; assert '特征维度: (8, 300)' in text; assert text.count('LORA (100.0%)')==8; print(text)"
```

这里的 `100.0%` 只来自单类别 softmax，验证的是包装层行为，不是模型效果。

Phase 0 环境中，CLI/async 包装在下面的调用中没有在 10 秒内退出，而同步核心在 1 秒内完成：

```bash
timeout 10s env PYTHONDONTWRITEBYTECODE=1 \
  /home/dianci/miniconda3/envs/radioml_amc_clean/bin/python \
  radioml-iq-modulation/scripts/onnx_inference.py \
  --signal_path data/processed/slices/sigmf_lora_dataset_128.npz \
  --data_format npz \
  --model_path radioml-iq-modulation/training/sigmf_lora/deep_iq_cnn.onnx \
  --label_map_path radioml-iq-modulation/training/sigmf_lora/label_map.json \
  --batch_size 8 \
  --seq_len 128 \
  --x_key X \
  --inference_mode batch
```

在重构前保留这条诊断；重构允许修复异步边界，但修复不能改变同步推理的数值输出。

## 9. 重构前后验收矩阵

| 验证项 | 重构前基线 | 重构后通过条件 |
| --- | --- | --- |
| 数据内部形状 | `[N, 2, L]` float32 | 新业务入口传给算法/模型的数据等价 |
| 特征定义 | 62 列，12/25/25 分组 | 数量、顺序、名称完全一致 |
| Wi‑Fi 特征值 | 固定 5 条 golden | `allclose(rtol=1e-5, atol=1e-6)` |
| 特征 NPZ | 5 个固定字段 | 字段语义、shape、dtype 保持；新增版本化字段可另议 |
| C ABI | 三个 C linkage 函数 | 函数签名、缓冲区长度和 0/-1/-2/-3 语义不变 |
| ONNX 输入 | `input: [B,2,128]` | 完全一致 |
| ONNX 输出 | logits `[B,1]`、feature `[B,300]` | 名称、shape、数值在容差内 |
| 推理解释 | 单类只显示 Top-1 LORA | 不把单类 100% 当作性能结论 |
| 数据准备 | 旧 LoRa 输出 4804 窗口 | 不作为新统一切片器的硬性回归条件 |
| Skill | 当前入口描述尚未重写 | 编码重构完成后单独重写与验收 |

## 10. 不属于 Phase 0 回归门槛的内容

- 不要求新切片器复现旧 SigMF builder 的 20 个 burst 或 4804 个窗口。
- 不要求 BLE 使用 LoRa 的能量检测策略。
- 不运行或延续准备删除的旧测试作为长期测试设计。
- 不用 RadioML 2016.10a 大文件作为默认 smoke fixture。
- 不评估单类 LoRa 模型的准确率、召回率或未知类拒识能力。
- 不检查尚未实现的 signal fusion 与 LLM 输出。

