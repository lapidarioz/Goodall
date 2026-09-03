from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def segment_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    truth = predictions["y_true"].to_numpy(dtype=int)
    labels = predictions["y_pred"].to_numpy(dtype=int)
    scores = predictions["y_score"].to_numpy(dtype=float)
    return {
        "segment_macro_f1": float(
            f1_score(truth, labels, average="macro", zero_division=0)
        ),
        "segment_precision": float(precision_score(truth, labels, zero_division=0)),
        "segment_recall": float(recall_score(truth, labels, zero_division=0)),
        "segment_roc_auc": (
            float(roc_auc_score(truth, scores))
            if np.unique(truth).size == 2
            else np.nan
        ),
    }


def one_to_one_event_matching(
    prediction_times: Sequence[float],
    truth_times: Sequence[float],
    tolerance: float = 1.0,
) -> list[dict]:
    """Maximize valid matches, then minimize their total timing error."""
    predictions = np.asarray(prediction_times, dtype=float)
    truths = np.asarray(truth_times, dtype=float)
    if predictions.size == 0 or truths.size == 0:
        return []
    distances = np.abs(predictions[:, None] - truths[None, :])
    # Invalid edges cost more than any possible sum of valid edges, so the
    # assignment first maximizes the number of matches and only then timing fit.
    invalid_cost = (max(len(predictions), len(truths)) + 1) * (tolerance + 1.0)
    size = len(predictions) + len(truths)
    costs = np.zeros((size, size), dtype=float)
    costs[: len(predictions), : len(truths)] = np.where(
        distances <= tolerance, distances, invalid_cost
    )
    costs[: len(predictions), len(truths) :] = tolerance + 1.0
    costs[len(predictions) :, : len(truths)] = tolerance + 1.0
    row_indices, column_indices = linear_sum_assignment(costs)
    matches = []
    for prediction_index, truth_index in zip(row_indices, column_indices):
        if prediction_index >= len(predictions) or truth_index >= len(truths):
            continue
        error = distances[prediction_index, truth_index]
        if error <= tolerance:
            matches.append(
                {
                    "prediction_index": int(prediction_index),
                    "truth_index": int(truth_index),
                    "timing_error_s": float(error),
                    "signed_timing_error_s": float(
                        predictions[prediction_index] - truths[truth_index]
                    ),
                }
            )
    return matches


def score_events(
    predictions: pd.DataFrame,
    truth_by_recording: Mapping[str, Sequence[float]],
    tolerance: float = 1.0,
) -> pd.DataFrame:
    rows = []
    for recording_id, truth_values in truth_by_recording.items():
        selected = predictions[
            (predictions["recording_id"].astype(str) == str(recording_id))
            & (predictions["y_score"] >= 0.5)
        ]
        truth_times = [float(value) for value in truth_values]
        matches = one_to_one_event_matching(
            selected["center_time"], truth_times, tolerance
        )
        errors = [row["timing_error_s"] for row in matches]
        predicted, true, matched = len(selected), len(truth_times), len(matches)
        precision = matched / predicted if predicted else 0.0
        recall = matched / true if true else 1.0
        event_f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        rows.append(
            {
                "recording_id": str(recording_id),
                "n_true_events": true,
                "n_predicted_events": predicted,
                "matched_events": matched,
                "missed_events": true - matched,
                "false_detections": predicted - matched,
                "event_precision": precision,
                "event_recall": recall,
                "event_f1": event_f1,
                "timing_error_mean_s": float(np.mean(errors)) if errors else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize(
    predictions: pd.DataFrame, recording_metrics: pd.DataFrame
) -> dict[str, float]:
    true = int(recording_metrics["n_true_events"].sum())
    predicted = int(recording_metrics["n_predicted_events"].sum())
    matched = int(recording_metrics["matched_events"].sum())
    precision = matched / predicted if predicted else 0.0
    recall = matched / true if true else 1.0
    return {
        **segment_metrics(predictions),
        "event_f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "event_precision": precision,
        "event_recall": recall,
        "n_true_events": true,
        "n_predicted_events": predicted,
        "matched_events": matched,
        "missed_events": true - matched,
        "false_detections": predicted - matched,
        "false_detections_per_recording": float(
            recording_metrics["false_detections"].mean()
        ),
    }
