from __future__ import annotations

import librosa
import numpy as np
from scipy import signal

EPSILON = 1e-12


def _detect_spikes(audio: np.ndarray, z_threshold: float = 8.0) -> np.ndarray:
    if audio.size < 3:
        return np.array([], dtype=int)
    expected = 0.5 * (audio[:-2] + audio[2:])
    residual = audio[1:-1] - expected
    median = np.median(residual)
    mad = np.median(np.abs(residual - median))
    if mad < EPSILON:
        centered = np.abs(residual - median)
        nonzero = centered[centered > EPSILON]
        if nonzero.size == 0:
            return np.array([], dtype=int)
        threshold = max(float(np.std(residual)) * min(z_threshold, 3.0), EPSILON)
        return np.flatnonzero(centered > threshold) + 1
    robust_z = 0.6745 * (residual - median) / mad
    return np.flatnonzero(np.abs(robust_z) > z_threshold) + 1


def preprocess_audio(path, sample_rate: int = 16_000) -> tuple[np.ndarray, int]:
    """Apply the publication's DC, spike, high-pass, RMS, and peak processing."""
    audio, sr = librosa.load(path, sr=sample_rate, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return audio, sr
    audio = audio - np.mean(audio)
    spikes = _detect_spikes(audio)
    if spikes.size:
        audio[spikes] = 0.5 * (audio[spikes - 1] + audio[spikes + 1])
    sos = signal.butter(2, 20.0 / (sr / 2.0), btype="highpass", output="sos")
    padlen = min(audio.size - 1, 3 * (2 * len(sos) + 1))
    if padlen > 0:
        audio = signal.sosfiltfilt(sos, audio, padlen=padlen)
    active_mask = np.zeros(audio.size, dtype=bool)
    for start, end in librosa.effects.split(audio, top_db=40.0):
        active_mask[start:end] = True
    if not active_mask.any():
        active_mask[:] = True
    active_rms = np.sqrt(np.mean(np.square(audio[active_mask])))
    if active_rms >= EPSILON:
        audio = audio * (10 ** (-20.0 / 20.0) / active_rms)
    peak = np.max(np.abs(audio))
    ceiling = 10 ** (-1.0 / 20.0)
    if peak > ceiling:
        audio = audio * (ceiling / peak)
    return audio.astype(np.float32, copy=False), sr


def detect_active_regions(
    audio: np.ndarray,
    sr: int,
    frame_length_ms: float = 25.0,
    hop_length_ms: float = 10.0,
    k_mad: float = 3.0,
    merge_gap_ms: float = 200.0,
    min_duration_ms: float = 150.0,
) -> list[tuple[float, float]]:
    if audio.size == 0:
        return []
    frame_length = int(sr * frame_length_ms / 1000.0)
    hop_length = int(sr * hop_length_ms / 1000.0)
    rms = librosa.feature.rms(
        y=audio, frame_length=frame_length, hop_length=hop_length
    )[0]
    flux = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop_length)
    common = min(len(rms), len(flux))
    rms, flux = rms[:common], flux[:common]

    def robust_scale(values: np.ndarray) -> np.ndarray:
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        return values - median if mad == 0 else (values - median) / (1.4826 * mad)

    combined = robust_scale(rms) + robust_scale(flux)
    median = np.median(combined)
    mad = np.median(np.abs(combined - median))
    active = combined > median + k_mad * 1.4826 * max(mad, 1e-6)
    regions: list[list[int]] = []
    start = None
    for index, value in enumerate(active):
        if value and start is None:
            start = index
        elif not value and start is not None:
            regions.append([start, index])
            start = None
    if start is not None:
        regions.append([start, len(active)])
    merge_gap = int(merge_gap_ms / hop_length_ms)
    merged: list[list[int]] = []
    for region in regions:
        if merged and region[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = region[1]
        else:
            merged.append(region)
    min_frames = int(min_duration_ms / hop_length_ms)
    return [
        (start_index * hop_length / sr, end_index * hop_length / sr)
        for start_index, end_index in merged
        if end_index - start_index >= min_frames
    ]


def _overlaps_event(
    start: float, end: float, events: list[float], margin: float
) -> bool:
    return any(start - margin <= event <= end + margin for event in events)


def generate_candidates(
    audio: np.ndarray,
    sr: int,
    events: list[float],
    window_seconds: float = 5.0,
    margin_seconds: float = 0.5,
) -> list[dict]:
    """Generate audio-only proposals and deterministic supplemental negatives."""
    duration = len(audio) / sr
    candidates: list[dict] = []
    for region_start, region_end in detect_active_regions(audio, sr):
        event_start = max(0.0, region_start - 0.3)
        event_end = min(duration, region_end + 0.3)
        center = (event_start + event_end) / 2.0
        start, end = center - window_seconds / 2.0, center + window_seconds / 2.0
        candidates.append({"start_time": start, "end_time": end, "source": "proposal"})

    positive_count = sum(
        _overlaps_event(row["start_time"], row["end_time"], events, margin_seconds)
        for row in candidates
    )
    negative_target = min(20, max(1, positive_count))
    pool: list[tuple[float, float, str]] = []
    for region_start, region_end in detect_active_regions(audio, sr):
        center = (region_start + region_end) / 2.0
        start = max(0.0, min(center - window_seconds / 2.0, duration - window_seconds))
        pool.append((start, start + window_seconds, "active_negative"))
    start = 0.0
    while start < duration and len(pool) < negative_target * 4:
        pool.append((start, start + window_seconds, "grid_negative"))
        start += window_seconds
    seen = {
        (round(row["start_time"], 3), round(row["end_time"], 3)) for row in candidates
    }
    added = 0
    for start, end, source in pool:
        key = (round(start, 3), round(end, 3))
        if key in seen or _overlaps_event(start, end, events, margin_seconds):
            continue
        seen.add(key)
        candidates.append({"start_time": start, "end_time": end, "source": source})
        added += 1
        if added >= negative_target:
            break
    for index, row in enumerate(candidates):
        row["segment_id"] = index
        row["center_time"] = (row["start_time"] + row["end_time"]) / 2.0
        row["label"] = int(
            _overlaps_event(row["start_time"], row["end_time"], events, margin_seconds)
        )
    return candidates


def extract_candidate(
    audio: np.ndarray, sr: int, start: float, end: float
) -> np.ndarray:
    start_sample, end_sample = int(start * sr), int(end * sr)
    pad_start, pad_end = max(0, -start_sample), max(0, end_sample - len(audio))
    values = audio[max(0, start_sample) : min(len(audio), end_sample)]
    if pad_start or pad_end:
        values = np.pad(values, (pad_start, pad_end))
    return np.asarray(values, dtype=np.float32)
