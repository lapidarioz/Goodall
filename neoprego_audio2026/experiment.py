from __future__ import annotations

import json
import platform
import sys
from collections import defaultdict

import librosa
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .audio import extract_candidate, generate_candidates, preprocess_audio
from .config import ExperimentConfig
from .data import (
    load_event_rows,
    load_or_build_fold_manifest,
    resolve_audio_path,
    split_by_recording,
)
from .evaluation import score_events, summarize
from .features import extract_publication_features


def _random_support_seeds(config: ExperimentConfig) -> tuple[int, ...]:
    rng = np.random.default_rng(config.random_state)
    return tuple(map(int, rng.integers(0, 100_000, size=config.random_support_repeats)))


def _extract_dataset(
    recordings: dict[str, list[float]],
    config: ExperimentConfig,
    random_seeds: tuple[int, ...],
) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, list[str]], dict]:
    blocks: dict[str, list[np.ndarray]] = defaultdict(list)
    feature_names: dict[str, list[str]] = {}
    rows: list[dict] = []
    pann_encoder = None
    if config.panns_checkpoint is not None:
        from models.PANNs import PANNsEncoder, PANNsEncoderConfig

        pann_encoder = PANNsEncoder(
            PANNsEncoderConfig(checkpoint_path=str(config.panns_checkpoint))
        )

    for recording_id in sorted(recordings):
        audio_path = resolve_audio_path(config.dataset_root, recording_id)
        audio, sr = preprocess_audio(audio_path, config.sample_rate)
        candidates = generate_candidates(
            audio,
            sr,
            recordings[recording_id],
            window_seconds=config.window_seconds,
            margin_seconds=config.annotation_margin_seconds,
        )
        waveforms = [
            extract_candidate(audio, sr, row["start_time"], row["end_time"])
            for row in candidates
        ]
        if pann_encoder is not None and waveforms:
            blocks["pann"].extend(pann_encoder.embed(waveforms))
            feature_names["pann"] = [f"pann_embedding_{index}" for index in range(2048)]
        for candidate, waveform in zip(candidates, waveforms):
            vectors, names = extract_publication_features(
                waveform, sr, config.rms_fractions, random_seeds
            )
            for branch, vector in vectors.items():
                public_branch = (
                    branch if branch.startswith("64only__") else f"fixed6__{branch}"
                )
                blocks[public_branch].append(vector)
                expected_names = names[branch]
                if (
                    public_branch in feature_names
                    and feature_names[public_branch] != expected_names
                ):
                    raise ValueError(
                        f"Feature ordering changed within branch {public_branch}."
                    )
                feature_names[public_branch] = expected_names
            rows.append(
                {
                    "recording_id": str(recording_id),
                    "segment_id": int(candidate["segment_id"]),
                    "start_time": float(candidate["start_time"]),
                    "end_time": float(candidate["end_time"]),
                    "center_time": float(candidate["center_time"]),
                    "candidate_source": candidate["source"],
                    "recording_duration_s": float(len(audio) / sr),
                    "label": int(candidate["label"]),
                }
            )
    row_frame = pd.DataFrame(rows)
    matrices = {
        branch: np.vstack(values).astype(np.float32)
        for branch, values in blocks.items()
    }
    if row_frame.empty or row_frame["label"].nunique() != 2:
        raise ValueError(
            "Candidate extraction must produce both positive and negative rows."
        )
    expected_rows = len(row_frame)
    mismatched = {
        name: len(matrix)
        for name, matrix in matrices.items()
        if len(matrix) != expected_rows
    }
    if mismatched:
        raise ValueError(f"Feature branches do not share candidate rows: {mismatched}")
    metadata = {
        "recordings": int(row_frame["recording_id"].nunique()),
        "candidates": len(row_frame),
        "positives": int(row_frame["label"].sum()),
        "negatives": int((row_frame["label"] == 0).sum()),
        "features_by_branch": {
            name: matrix.shape[1] for name, matrix in matrices.items()
        },
    }
    if pann_encoder is not None:
        metadata["panns"] = pann_encoder.metadata()
    return matrices, row_frame, feature_names, metadata


def _fit_oof(
    branch: str,
    matrix: np.ndarray,
    rows: pd.DataFrame,
    assignments: dict[str, int],
    config: ExperimentConfig,
) -> pd.DataFrame:
    labels = rows["label"].to_numpy(dtype=int)
    groups = rows["recording_id"].astype(str).to_numpy()
    output = []
    for fold in sorted(set(assignments.values())):
        test = np.asarray([assignments[group] == fold for group in groups])
        train = ~test
        if not test.any() or not train.any() or np.unique(labels[train]).size != 2:
            raise ValueError(f"Fold {fold} is invalid for branch {branch}.")
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        train_values = imputer.fit_transform(matrix[train])
        scaler = StandardScaler().fit(train_values)
        model = RandomForestClassifier(
            n_estimators=config.n_trees,
            class_weight="balanced",
            random_state=config.random_state,
            n_jobs=-1,
        )
        model.fit(scaler.transform(train_values), labels[train])
        test_values = scaler.transform(imputer.transform(matrix[test]))
        probabilities = model.predict_proba(test_values)
        positive_index = list(model.classes_).index(1)
        scores = probabilities[:, positive_index]
        indices = np.flatnonzero(test)
        fold_rows = rows.iloc[indices].copy()
        fold_rows.insert(0, "branch", branch)
        fold_rows.insert(1, "fold", fold)
        fold_rows["y_true"] = labels[test]
        fold_rows["y_score"] = scores
        fold_rows["y_pred"] = (scores >= 0.5).astype(int)
        output.append(fold_rows)
    predictions = pd.concat(output, ignore_index=True)
    if len(predictions) != len(rows) or set(predictions["recording_id"]) != set(
        rows["recording_id"]
    ):
        raise ValueError(f"Incomplete out-of-fold coverage for {branch}.")
    return predictions


def _branch_result(
    branch: str,
    predictions: pd.DataFrame,
    recordings: dict[str, list[float]],
    tolerance: float,
) -> tuple[dict, pd.DataFrame]:
    recording_metrics = score_events(predictions, recordings, tolerance)
    return {
        "system": branch,
        **summarize(predictions, recording_metrics),
    }, recording_metrics


def _mean_random_result(
    representation: str, results: pd.DataFrame, random_branches: list[str]
) -> dict:
    subset = results[results["system"].isin(random_branches)]
    numeric = subset.select_dtypes(include=[np.number]).mean().to_dict()
    return {"system": f"{representation}__random20", **numeric}


def _confusion_by_recording(
    predictions: pd.DataFrame, recording_ids: list[str]
) -> np.ndarray:
    counts = []
    for recording_id in recording_ids:
        group = predictions[predictions["recording_id"].astype(str) == recording_id]
        truth, predicted = group["y_true"].to_numpy(), group["y_pred"].to_numpy()
        counts.append(
            [
                np.sum((truth == 0) & (predicted == 0)),
                np.sum((truth == 0) & (predicted == 1)),
                np.sum((truth == 1) & (predicted == 0)),
                np.sum((truth == 1) & (predicted == 1)),
            ]
        )
    return np.asarray(counts, dtype=float)


def _macro_f1_from_counts(counts: np.ndarray) -> np.ndarray:
    tn, fp, fn, tp = counts.T
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_positive = np.divide(
            2 * tp,
            2 * tp + fp + fn,
            out=np.zeros_like(tp),
            where=(2 * tp + fp + fn) > 0,
        )
        f1_negative = np.divide(
            2 * tn,
            2 * tn + fp + fn,
            out=np.zeros_like(tn),
            where=(2 * tn + fp + fn) > 0,
        )
    return (f1_positive + f1_negative) / 2.0


def _event_f1_from_counts(counts: np.ndarray) -> np.ndarray:
    true, predicted, matched = counts.T
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.divide(
            matched, predicted, out=np.zeros_like(matched), where=predicted > 0
        )
        recall = np.divide(matched, true, out=np.ones_like(matched), where=true > 0)
        return np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) > 0,
        )


def _bootstrap_differences(
    predictions: dict[str, pd.DataFrame],
    recording_metrics: dict[str, pd.DataFrame],
    recording_ids: list[str],
    random_seeds: tuple[int, ...],
    config: ExperimentConfig,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.random_state)
    sampled = rng.integers(
        0, len(recording_ids), size=(config.bootstrap_samples, len(recording_ids))
    )
    distributions: dict[str, dict[str, np.ndarray]] = {}
    for branch, frame in predictions.items():
        confusion = _confusion_by_recording(frame, recording_ids)
        event_frame = (
            recording_metrics[branch].set_index("recording_id").reindex(recording_ids)
        )
        event_counts = event_frame[
            ["n_true_events", "n_predicted_events", "matched_events"]
        ].to_numpy(dtype=float)
        distributions[branch] = {
            "segment_macro_f1": _macro_f1_from_counts(confusion[sampled].sum(axis=1)),
            "event_f1": _event_f1_from_counts(event_counts[sampled].sum(axis=1)),
        }

    random_branches = [f"fixed6__random20_seed{seed}" for seed in random_seeds]
    random_expected = {
        metric: np.mean(
            [distributions[branch][metric] for branch in random_branches], axis=0
        )
        for metric in ("segment_macro_f1", "event_f1")
    }
    comparisons = [
        ("fixed6__rms20", "fixed6__global100", "RMS20 - GLOBAL100"),
    ]
    comparisons.extend(
        (
            f"fixed6__rms{round(fraction * 100):02d}",
            "fixed6__global100",
            f"RMS{round(fraction * 100)} - GLOBAL100",
        )
        for fraction in config.rms_fractions
        if fraction != 0.20
    )
    rows = []
    for left, right, label in comparisons:
        for metric in ("segment_macro_f1", "event_f1"):
            delta = distributions[left][metric] - distributions[right][metric]
            rows.append(
                {
                    "comparison": label,
                    "metric": metric,
                    "mean_delta": float(np.mean(delta)),
                    "lower_95": float(np.quantile(delta, 0.025)),
                    "upper_95": float(np.quantile(delta, 0.975)),
                    "bootstrap_samples": config.bootstrap_samples,
                }
            )
    for left, right, label in (
        ("fixed6__rms20", None, "RMS20 - RANDOM20"),
        ("fixed6__global100", None, "GLOBAL100 - RANDOM20"),
    ):
        for metric in ("segment_macro_f1", "event_f1"):
            delta = distributions[left][metric] - random_expected[metric]
            rows.append(
                {
                    "comparison": label,
                    "metric": metric,
                    "mean_delta": float(np.mean(delta)),
                    "lower_95": float(np.quantile(delta, 0.025)),
                    "upper_95": float(np.quantile(delta, 0.975)),
                    "bootstrap_samples": config.bootstrap_samples,
                }
            )
    return pd.DataFrame(rows)


def run_experiment(config: ExperimentConfig) -> pd.DataFrame:
    """Run the publication protocol and write auditable tabular artifacts."""
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_event_rows(config.events_csv)
    evaluation, development, templates = split_by_recording(
        rows, random_state=config.random_state
    )
    manifest = load_or_build_fold_manifest(
        list(development), config.fold_manifest, config.n_splits, config.random_state
    )
    manifest.to_csv(config.output_dir / "fold_manifest.csv", index=False)
    assignments = dict(zip(manifest["recording_id"], manifest["fold"]))
    random_seeds = _random_support_seeds(config)
    matrices, candidate_rows, _, extraction = _extract_dataset(
        development, config, random_seeds
    )
    candidate_rows.to_csv(config.output_dir / "candidates.csv", index=False)

    # The paper reports all Fixed6 fractions, the three 64-D control supports,
    # and an optional frozen PANN baseline. Other extracted 64-D fractions are
    # intentionally not fitted because they are not part of the article.
    fitted_branches = [name for name in matrices if not name.startswith("64only__")]
    fitted_branches.extend(
        [
            "64only__global100",
            "64only__rms20",
            *[f"64only__random20_seed{seed}" for seed in random_seeds],
        ]
    )
    fitted_branches = list(dict.fromkeys(fitted_branches))
    all_predictions: dict[str, pd.DataFrame] = {}
    all_recording_metrics: dict[str, pd.DataFrame] = {}
    result_rows = []
    for branch in fitted_branches:
        predictions = _fit_oof(
            branch, matrices[branch], candidate_rows, assignments, config
        )
        result, recording_metrics = _branch_result(
            branch, predictions, development, config.match_tolerance_seconds
        )
        result_rows.append(result)
        all_predictions[branch] = predictions
        all_recording_metrics[branch] = recording_metrics.assign(system=branch)

    raw_results = pd.DataFrame(result_rows)
    fixed_random = [f"fixed6__random20_seed{seed}" for seed in random_seeds]
    control_random = [f"64only__random20_seed{seed}" for seed in random_seeds]
    summary_rows = [
        *result_rows,
        _mean_random_result("fixed6", raw_results, fixed_random),
        _mean_random_result("64only", raw_results, control_random),
    ]
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(config.output_dir / "summary.csv", index=False)
    pd.concat(all_predictions.values(), ignore_index=True).to_csv(
        config.output_dir / "predictions.csv", index=False
    )
    pd.concat(all_recording_metrics.values(), ignore_index=True).to_csv(
        config.output_dir / "recording_metrics.csv", index=False
    )
    bootstrap = _bootstrap_differences(
        all_predictions,
        all_recording_metrics,
        sorted(map(str, development)),
        random_seeds,
        config,
    )
    bootstrap.to_csv(config.output_dir / "bootstrap_differences.csv", index=False)
    metadata = {
        "config": config.to_dict(),
        "random_support_seeds": random_seeds,
        "partition_counts": {
            "development_recordings": len(development),
            "development_events": sum(map(len, development.values())),
            "template_recordings": len(templates),
            "template_events": sum(map(len, templates.values())),
            "evaluation_recordings": len(evaluation),
            "evaluation_events": sum(map(len, evaluation.values())),
        },
        "extraction": extraction,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "librosa": librosa.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    (config.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    return summary
