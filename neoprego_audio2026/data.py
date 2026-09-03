from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_ANNOTATION_COLUMNS = {"Media_file", "Approximation"}


def load_event_rows(path: Path) -> list[dict[str, str]]:
    """Load the derived event table used by the experiment.

    ``Media_file`` identifies an audio recording relative to the dataset root and
    ``Approximation`` is the event time in seconds. If a ``Behavior`` column is
    present, only rows labelled ``Strike`` are retained.
    """
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or ())
        missing = REQUIRED_ANNOTATION_COLUMNS - columns
        if missing:
            raise ValueError(
                "Annotation CSV is missing columns: " + ", ".join(sorted(missing))
            )
        rows = [
            row
            for row in reader
            if not row.get("Behavior") or row["Behavior"] == "Strike"
        ]
    if not rows:
        raise ValueError("Annotation CSV contains no Strike events.")
    for row in rows:
        if not row["Media_file"].strip():
            raise ValueError("Every annotation row must identify a Media_file.")
        event_time = float(row["Approximation"])
        if event_time < 0:
            raise ValueError("Event times must be non-negative.")
    return rows


def split_by_recording(
    rows: list[dict[str, str]],
    train_ratio: float = 0.85,
    template_ratio: float = 0.08,
    random_state: int = 42,
) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, list[float]]]:
    """Return evaluation, development, and template partitions without leakage."""
    by_recording: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_recording.setdefault(row["Media_file"], []).append(row)
    targets = {
        "development": (train_ratio - template_ratio) * len(rows),
        "template": template_ratio * len(rows),
        "evaluation": (1.0 - train_ratio) * len(rows),
    }
    if len(by_recording) < len(targets):
        raise ValueError("At least three recordings are required for the frozen split.")
    items = sorted(by_recording.items())
    np.random.default_rng(random_state).shuffle(items)
    items.sort(key=lambda item: len(item[1]), reverse=True)
    counts = {name: 0 for name in targets}
    assigned: dict[str, list[tuple[str, list[dict[str, str]]]]] = {
        name: [] for name in targets
    }
    for recording, recording_rows in items:
        partition = max(
            targets,
            key=lambda name: (
                (targets[name] - counts[name]) / targets[name],
                targets[name] - counts[name],
            ),
        )
        assigned[partition].append((recording, recording_rows))
        counts[partition] += len(recording_rows)

    def as_group(partition: str) -> dict[str, list[float]]:
        return {
            recording: [float(row["Approximation"]) for row in recording_rows]
            for recording, recording_rows in assigned[partition]
        }

    return as_group("evaluation"), as_group("development"), as_group("template")


def resolve_audio_path(dataset_root: Path, recording: str) -> Path:
    supplied = Path(recording)
    direct = supplied if supplied.is_absolute() else dataset_root / supplied
    candidates = (direct, direct.with_suffix(".mp3") if not direct.suffix else direct)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matching = [
        path
        for path in dataset_root.rglob("*")
        if path.is_file() and (path.name == supplied.name or path.stem == supplied.stem)
    ]
    if not matching:
        raise FileNotFoundError(
            f"Audio recording not found under {dataset_root}: {recording}"
        )
    if len(matching) > 1:
        options = ", ".join(
            str(path.relative_to(dataset_root)) for path in matching[:5]
        )
        raise ValueError(
            f"Audio recording is ambiguous under {dataset_root}: {recording}. "
            f"Matches include: {options}"
        )
    return matching[0]


def build_fold_manifest(
    recording_ids: list[str], n_splits: int = 5, random_state: int = 42
) -> pd.DataFrame:
    recordings = np.asarray(sorted(set(map(str, recording_ids))), dtype=object)
    if len(recordings) < n_splits:
        raise ValueError(
            f"The publication protocol requires at least {n_splits} recordings."
        )
    shuffled = recordings[
        np.random.default_rng(random_state).permutation(len(recordings))
    ]
    assignments = {
        recording: index % n_splits + 1 for index, recording in enumerate(shuffled)
    }
    return pd.DataFrame(
        {
            "recording_id": sorted(assignments),
            "fold": [assignments[key] for key in sorted(assignments)],
        }
    )


def load_or_build_fold_manifest(
    recording_ids: list[str], path: Path | None, n_splits: int, random_state: int
) -> pd.DataFrame:
    manifest = (
        pd.read_csv(path)
        if path is not None
        else build_fold_manifest(
            recording_ids, n_splits=n_splits, random_state=random_state
        )
    )
    if not {"recording_id", "fold"}.issubset(manifest.columns):
        raise ValueError("Fold manifest must contain recording_id and fold columns.")
    manifest = manifest[["recording_id", "fold"]].copy()
    manifest["recording_id"] = manifest["recording_id"].astype(str)
    manifest["fold"] = manifest["fold"].astype(int)
    if manifest["recording_id"].duplicated().any():
        raise ValueError("Fold manifest contains duplicate recording IDs.")
    expected = set(map(str, recording_ids))
    observed = set(manifest["recording_id"])
    if expected != observed:
        raise ValueError(
            "Fold manifest does not match development recordings: "
            f"missing={sorted(expected - observed)[:5]}, unexpected={sorted(observed - expected)[:5]}"
        )
    if manifest["fold"].nunique() != n_splits:
        raise ValueError(f"Fold manifest must contain exactly {n_splits} folds.")
    return manifest.sort_values("recording_id").reset_index(drop=True)
