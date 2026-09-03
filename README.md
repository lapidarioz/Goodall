# Temporal Aggregation Support for Sparse Behavioral Acoustic Event Detection

[GitHub repository](https://github.com/lapidarioz/neoprego_audio2026)

This branch contains the code needed to reproduce the experiments in **“Temporal Aggregation Support for Sparse Behavioral Acoustic Event Detection.”** It compares three ways of selecting frames from the same five-second audio candidates:

- `GLOBAL100`: all frames;
- `RANDOM20`: 20% of frames sampled with 20 deterministic seeds; and
- `RMS20`: the 20% of frames with the highest RMS energy.

The implementation also reproduces the RMS fraction sensitivity analysis, the 64-dimensional representation control, recording-grouped cross-validation, paired recording-cluster bootstrap intervals, and the optional frozen PANN Cnn14_16k baseline reported in the article.

The research dataset is not included. Raw recordings and behavioral annotations are external research data and may be subject to access, privacy, conservation, and institutional restrictions.

## Requirements

- Python 3.11–3.13
- [uv](https://docs.astral.sh/uv/)
- Audio recordings accessible from a local dataset directory
- A derived CSV containing at least:

```csv
Media_file,Approximation,Behavior
audios/example.wav,12.340,Strike
```

`Media_file` may be absolute or relative to `--dataset-root`. `Approximation` is the event time in seconds. If `Behavior` is present, only `Strike` rows are used. See `examples/events.example.csv` for a synthetic schema example.

## Reproduce the experiment

Install the locked environment:

```bash
uv sync
```

Run the CPU experiment:

```bash
uv run neoprego-audio2026 \
  --dataset-root /path/to/dataset \
  --events-csv /path/to/strikes_events.csv \
  --output-dir outputs/publication
```

The published protocol uses five recording-disjoint folds, a 300-tree balanced Random Forest, probability threshold 0.50, event tolerance ±1 second, seed 42, and 10,000 paired recording-cluster bootstrap resamples. These settings are frozen in code. For a quick infrastructure check, `--bootstrap-samples 100` shortens only the bootstrap stage.

If the exact fold manifest distributed with the research data is available, pass it with `--fold-manifest`. Otherwise, the program generates the same deterministic recording-level assignment algorithm and exports it as `fold_manifest.csv`.

### Optional PANN baseline

Install the optional dependencies, download the verified external checkpoint, and pass it to the experiment:

```bash
uv sync --extra panns
uv run python scripts/download_panns_checkpoint.py
uv run neoprego-audio2026 \
  --dataset-root /path/to/dataset \
  --events-csv /path/to/strikes_events.csv \
  --panns-checkpoint checkpoints/Cnn14_16k_mAP=0.438.pth
```

The checkpoint is not stored in this repository. The downloader verifies the official MD5 and SHA-256 checksums before accepting it. The PANN adapter is derived from the MIT-licensed reference implementation; see `models/PANNs_LICENSE.MIT`.

## Outputs

Each run writes plain CSV and JSON artifacts:

- `candidates.csv`: shared candidate windows and labels;
- `fold_manifest.csv`: recording-to-fold assignments;
- `predictions.csv`: out-of-fold segment predictions for every fitted branch;
- `recording_metrics.csv`: event counts and metrics by recording;
- `summary.csv`: segment and event point estimates;
- `bootstrap_differences.csv`: paired 95% intervals; and
- `metadata.json`: frozen configuration, partition counts, seeds, and runtime versions.

No annotation is used to place proposals or choose support frames. Annotations are used only to label proposals, reject supplemental negative windows, determine their count, and score predictions, matching the article protocol.

## Tests

```bash
uv run python -m unittest discover -s tests
```

The tests use synthetic arrays and temporary files; the private research dataset is not required.

## Container

```bash
docker compose run --rm neoprego_audio2026 \
  --events-csv /data/strikes_events.csv
```

Set `NEOPREGO_AUDIO2026_DATASET_ROOT` before running Compose to mount the intended dataset directory read-only at `/data`.

## Citation and license

Citation metadata is in `CITATION.cff`. The repository is licensed under the Apache License 2.0; see `LICENSE` for the complete terms. Source code and updates are available on [GitHub](https://github.com/lapidarioz/neoprego_audio2026).
