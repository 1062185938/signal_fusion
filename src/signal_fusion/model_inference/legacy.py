"""Compatibility behavior for the historical RadioML inference API."""

from __future__ import annotations

from collections import Counter
import os
from pathlib import Path

import numpy as np

from signal_fusion.contracts import PreparedDataset
from signal_fusion.io.loaders import load_signal_for_inference
from signal_fusion.model_inference.backend import ONNXRuntimeBackend
from signal_fusion.model_inference.labels import load_label_map
from signal_fusion.model_inference.service import ModelInferenceService


def load_mod_labels(label_map_path):
    """Return the historical list form of the canonical label map."""

    return list(load_label_map(label_map_path))


def run_onnx_inference(processed_data_path, model_path, batch_size, log_buffer):
    """Preserve the older raw-complex batch inference helper."""

    try:
        backend = ONNXRuntimeBackend(model_path)
    except Exception as exc:
        log_buffer.append(f"ONNX 模型加载失败: {exc}")
        return

    expected_shape = backend.input_shape
    elements_per_sample = np.prod(expected_shape[1:])
    required_complex_len = int(elements_per_sample // 2)
    data = np.fromfile(processed_data_path, dtype=np.complex64)
    num_vectors = len(data) // required_complex_len
    if num_vectors == 0:
        log_buffer.append("错误: 提取的数据量不足以构成一个完整的推理样本。")
        return

    data = data[: num_vectors * required_complex_len]
    interleaved_data = np.empty((len(data) * 2,), dtype=np.float32)
    interleaved_data[0::2] = np.real(data)
    interleaved_data[1::2] = np.imag(data)
    target_shape = [num_vectors, *expected_shape[1:]]
    all_batched_input = interleaved_data.reshape(target_shape)

    probability_chunks = []
    for start in range(0, num_vectors, batch_size):
        outputs = backend.run(all_batched_input[start : start + batch_size])
        logits = outputs[backend.output_names[0]]
        exp_outputs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probability_chunks.extend(
            exp_outputs / np.sum(exp_outputs, axis=1, keepdims=True)
        )
    probabilities = np.asarray(probability_chunks)
    top_k = min(5, probabilities.shape[1])
    top_indices = np.argsort(probabilities, axis=1)[:, -top_k:][:, ::-1]

    log_buffer.append(
        f"[推  理] 硬件: {backend.provider}, 总样本数: {num_vectors}, "
        f"批次大小: {batch_size}"
    )
    log_buffer.append(
        f"\n--- Top-{top_k} 预测结果与置信度分布 (展示前 25 个样本) ---"
    )
    headers = " | ".join(
        f"{f'Top-{rank + 1}':<12}" for rank in range(top_k)
    )
    log_buffer.append(f"{'样本序号':<10} | {headers}")
    log_buffer.append("-" * 80)
    for sample_index in range(min(25, num_vectors)):
        items = [
            f"类{class_index}({probabilities[sample_index, class_index]:.1%})"
            for class_index in top_indices[sample_index]
        ]
        formatted_items = " | ".join(f"{item:<12}" for item in items)
        log_buffer.append(f"#{sample_index:<9} | {formatted_items}")


def _sync_signal_inference(
    signal_path: str,
    model_path: str,
    label_map_path: str,
    batch_size: int = 64,
    seq_len: int = 128,
    data_format: str = "auto",
    sample_mode: str | None = None,
    iq_format: str | None = None,
    x_key: str | None = None,
    inference_mode: str = "vote",
) -> str:
    """Run the legacy batch/vote interface on the structured core runtime."""

    log_buffer: list[str] = []
    inference_mode = (inference_mode or "vote").lower()
    if inference_mode not in {"batch", "vote"}:
        return "错误: inference_mode 必须是 'batch' 或 'vote'"
    if batch_size <= 0:
        return "错误: batch_size 必须大于 0"

    if not os.path.exists(signal_path):
        return f"错误: 找不到输入文件 {signal_path}"
    if not os.path.exists(model_path):
        return f"错误: 找不到模型文件 {model_path}"
    if not os.path.exists(label_map_path):
        return f"错误: 找不到标签映射文件 {label_map_path}"

    try:
        labels = load_label_map(label_map_path)
    except Exception as exc:
        return f"label_map 加载失败: {exc}"

    try:
        backend = ONNXRuntimeBackend(model_path)
    except Exception as exc:
        return f"ONNX 模型加载失败: {exc}"

    try:
        loaded = load_signal_for_inference(
            path=signal_path,
            data_format=data_format,
            seq_len=seq_len,
            sample_mode=sample_mode,
            iq_format=iq_format,
            max_samples=None,
            x_key=x_key,
        )
    except Exception as exc:
        return f"推理数据加载失败: {exc}"

    all_x = loaded["X"].astype(np.float32, copy=False)
    if all_x.ndim != 3:
        return f"错误: 推理输入必须是 [N, 2, seq_len]，当前 shape={all_x.shape}"
    if all_x.shape[1] != 2 or all_x.shape[2] != seq_len:
        return (
            f"错误: 推理输入 shape 与参数不匹配，期望 [N, 2, {seq_len}]，"
            f"当前 shape={all_x.shape}"
        )
    total_samples = int(all_x.shape[0])
    if total_samples == 0:
        return "错误: 推理输入样本数为 0"

    if inference_mode == "vote":
        sample_indices = np.random.randint(
            0,
            total_samples,
            size=batch_size,
        ).astype(np.int64, copy=False)
    else:
        sample_indices = np.arange(
            min(batch_size, total_samples),
            dtype=np.int64,
        )
    actual_batch_size = int(sample_indices.shape[0])
    meta = loaded.get("meta", {})
    log_buffer.append(
        f"[预处理] 成功读取 {total_samples} 个候选样本，数据维度: {all_x.shape}"
    )
    if inference_mode == "vote":
        log_buffer.append(
            f"[预处理] 随机抽取 {actual_batch_size} 个样本用于推理投票"
        )
    else:
        log_buffer.append(
            f"[预处理] 读取前 {actual_batch_size} 个样本用于 batch 推理"
        )
    log_buffer.append(
        f"[预处理] 数据格式: {meta.get('data_format')}, "
        f"seq_len: {meta.get('seq_len')}"
    )

    manifest = backend.build_manifest(
        model_id=Path(model_path).stem,
        labels=labels,
    )
    dataset = PreparedDataset(
        X=all_x,
        meta=meta,
        source_id=Path(signal_path).stem,
    )
    try:
        result = ModelInferenceService(manifest, backend=backend).predict(
            dataset,
            batch_size=actual_batch_size,
            sample_indices=sample_indices,
        )
    except ValueError as exc:
        if "label_map 与模型输出标签类别不匹配" in str(exc):
            return str(exc)
        raise

    top_k = min(3, result.num_classes, len(labels))
    top_indices = np.argsort(result.probabilities, axis=1)[:, -top_k:][:, ::-1]
    features = next(iter(result.auxiliary_outputs.values()), None)
    log_buffer.append(
        f"[推  理] 模式: {inference_mode}, 硬件: {result.provider}, "
        f"批次大小: {actual_batch_size}"
    )
    if features is not None:
        log_buffer.append(f"[特  征] 成功提取网络底层特征，特征维度: {features.shape}")
    log_buffer.append(f"\n--- Top-{top_k} 预测结果与置信度分布 ---")
    top_headers = " | ".join(f"{f'Top-{rank + 1}':<16}" for rank in range(top_k))
    log_buffer.append(f"{'样本序号':<8} | {top_headers}")
    log_buffer.append("-" * 75)

    valid_predictions: list[tuple[int, float]] = []
    for sample_index in range(actual_batch_size):
        indices = top_indices[sample_index]
        confidences = result.probabilities[sample_index, indices]
        top1_index = int(indices[0])
        top1_confidence = float(confidences[0])
        if inference_mode == "vote" and top1_confidence >= 0.80:
            valid_predictions.append((top1_index, top1_confidence))

        items = []
        for rank, class_index in enumerate(indices):
            label = labels[int(class_index)]
            confidence = confidences[rank]
            items.append(f"Top-{rank + 1}: {label} ({confidence:.1%})")
        top_text = " | ".join(f"{item:<16}" for item in items)
        if inference_mode == "batch":
            log_buffer.append(f"#{sample_index:<7} | {top_text}")
        elif top1_confidence < 0.80:
            log_buffer.append(
                f"#{sample_index:<7} | {top_text} (置信度不足，丢弃)"
            )
        else:
            log_buffer.append(f"#{sample_index:<7} | {top_text} ✓")

    if inference_mode == "batch":
        log_buffer.append("\n[推  理] batch 模式已完成逐样本推理，不执行最终投票。")
        return "\n".join(log_buffer)

    log_buffer.append("\n" + "=" * 50)
    log_buffer.append("               【文 件 最 终 综 合 判 定】")
    log_buffer.append("=" * 50)
    if not valid_predictions:
        log_buffer.append("结果: [未知信号] (原因: 所有样本的最高置信度均不足 80%)")
    else:
        class_votes = [prediction[0] for prediction in valid_predictions]
        final_class_index, final_votes = Counter(class_votes).most_common(1)[0]
        winning_confidences = [
            prediction[1]
            for prediction in valid_predictions
            if prediction[0] == final_class_index
        ]
        average_confidence = sum(winning_confidences) / len(winning_confidences)
        log_buffer.append(f"识别调制方式 : {labels[final_class_index]}")
        log_buffer.append(f"平均高置信度 : {average_confidence:.2%}")
        log_buffer.append(
            f"有效投票占比 : {final_votes} / {actual_batch_size} "
            "(仅统计置信度≥80%且判定一致的样本)"
        )
    log_buffer.append("=" * 50 + "\n")
    return "\n".join(log_buffer)


async def recognize_iq_modulation(
    signal_path="./data/processed/radioml2016_infer.dat",
    model_path="./outputs_logit_norm/deep_iq_cnn.onnx",
    label_map_path="./data/processed/label_map.json",
    batch_size=64,
    seq_len=128,
    data_format="dat",
    sample_mode=None,
    iq_format=None,
    x_key=None,
    inference_mode="vote",
):
    """Compatibility coroutine without ONNX Runtime worker-thread leakage.

    The runtime is intentionally synchronous. Higher-level orchestration may
    choose its own process or task isolation instead of hiding it here.
    """

    return _sync_signal_inference(
        signal_path=signal_path,
        model_path=model_path,
        label_map_path=label_map_path,
        batch_size=batch_size,
        seq_len=seq_len,
        data_format=data_format,
        sample_mode=sample_mode,
        iq_format=iq_format,
        x_key=x_key,
        inference_mode=inference_mode,
    )


__all__ = [
    "_sync_signal_inference",
    "load_mod_labels",
    "recognize_iq_modulation",
    "run_onnx_inference",
]
