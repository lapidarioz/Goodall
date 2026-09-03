from __future__ import annotations

import argparse
from pathlib import Path

from .config import ExperimentConfig
from .experiment import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce the NeoPrego Audio 2026 temporal-support publication experiment."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root containing the audio files.",
    )
    parser.add_argument(
        "--events-csv",
        type=Path,
        required=True,
        help="CSV with Media_file and Approximation columns.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/publication"))
    parser.add_argument(
        "--fold-manifest", type=Path, help="Optional frozen recording-to-fold CSV."
    )
    parser.add_argument(
        "--panns-checkpoint", type=Path, help="Optional official Cnn14_16k checkpoint."
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10_000,
        help="Use a smaller value only for a smoke test (publication: 10000).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ExperimentConfig(
        dataset_root=args.dataset_root,
        events_csv=args.events_csv,
        output_dir=args.output_dir,
        fold_manifest=args.fold_manifest,
        panns_checkpoint=args.panns_checkpoint,
        bootstrap_samples=args.bootstrap_samples,
    )
    summary = run_experiment(config)
    print(summary.to_string(index=False))
    print(f"\nResults written to {config.output_dir.resolve()}")
    return 0
