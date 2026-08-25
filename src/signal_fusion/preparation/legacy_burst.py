"""Compatibility home for the pre-Phase-2 adaptive burst extractor."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def smart_extract_bursts(
    input_path: str,
    output_path: str,
    burst_len: int,
    log_buffer: list[str],
) -> bool:
    """Preserve the historical helper while keeping it out of ONNX inference."""

    if not Path(input_path).exists():
        log_buffer.append(f"错误: 找不到输入文件 {input_path}")
        return False

    data = np.fromfile(input_path, dtype=np.complex64)
    abs_data = np.abs(data)
    noise_samples = np.sort(abs_data)[: max(1, int(len(abs_data) * 0.05))]
    mu_noise = np.mean(noise_samples)
    sigma_noise = np.std(noise_samples)
    dynamic_threshold = mu_noise * 2 + 5 * sigma_noise

    window_size = 32
    window = np.ones(window_size) / window_size
    smoothed_abs = np.convolve(abs_data, window, mode="same")
    bursts: list[np.ndarray] = []
    index = 0
    total_samples = len(data)
    in_signal = False

    while index <= total_samples - burst_len - 30:
        current_energy = smoothed_abs[index]
        if current_energy > dynamic_threshold and not in_signal:
            if index + burst_len + 30 >= total_samples:
                break
            burst = data[index + 30 : index + burst_len + 30]
            max_value = np.max(np.abs(burst))
            bursts.append(burst / max_value if max_value > 0 else burst)
            in_signal = True
            index += burst_len
        elif current_energy < dynamic_threshold and in_signal:
            in_signal = False
            index += 32
        else:
            index += 64

    if not bursts:
        log_buffer.append("未检测到有效信号，请检查原始数据。")
        return False

    np.concatenate(bursts).astype(np.complex64).tofile(output_path)
    log_buffer.append(
        f"[预处理] 自适应底噪均值: {mu_noise:.4f}, "
        f"动态阈值: {dynamic_threshold:.4f}"
    )
    log_buffer.append(f"[预处理] 提取完成！共截取 {len(bursts)} 段有效突发信号。")
    return True


__all__ = ["smart_extract_bursts"]
