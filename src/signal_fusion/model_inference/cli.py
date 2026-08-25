"""Command-line entry point for ONNX modulation inference."""

from __future__ import annotations

import argparse

from signal_fusion.model_inference.legacy import _sync_signal_inference


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RadioML ONNX IQ modulation inference")
    parser.add_argument(
        "--signal_path", type=str, default="./data/processed/radioml2016_infer.dat"
    )
    parser.add_argument("--data_format", type=str, default="dat")
    parser.add_argument(
        "--model_path", type=str, default="./outputs_logit_norm/deep_iq_cnn.onnx"
    )
    parser.add_argument(
        "--label_map_path", type=str, default="./data/processed/label_map.json"
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--sample_mode", type=str, default=None)
    parser.add_argument("--iq_format", type=str, default=None)
    parser.add_argument("--x_key", type=str, default=None)
    parser.add_argument(
        "--inference_mode",
        type=str,
        default="vote",
        choices=["batch", "vote"],
        help=(
            "Inference mode: vote randomly samples batch_size candidates and "
            "votes; batch runs the first batch_size samples without voting."
        ),
    )
    return parser


def _run_from_argv(argv: list[str] | None = None) -> str:
    args = build_arg_parser().parse_args(argv)
    return _sync_signal_inference(
        signal_path=args.signal_path,
        model_path=args.model_path,
        label_map_path=args.label_map_path,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        data_format=args.data_format,
        sample_mode=args.sample_mode,
        iq_format=args.iq_format,
        x_key=args.x_key,
        inference_mode=args.inference_mode,
    )


def run_legacy_cli(argv: list[str] | None = None) -> str:
    """Preserve the historical script's string return value."""

    result = _run_from_argv(argv)
    print(result)
    return result


def main(argv: list[str] | None = None) -> int:
    """Run as an installed console script with a successful integer exit code."""

    print(_run_from_argv(argv))
    return 0


__all__ = ["build_arg_parser", "main", "run_legacy_cli"]


if __name__ == "__main__":
    main()
