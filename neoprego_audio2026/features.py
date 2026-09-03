from __future__ import annotations

import librosa
import numpy as np

FRAME_LENGTH = 2048
HOP_LENGTH = 512
N_MFCC = 13


def select_frame_indices(
    scores: np.ndarray,
    fraction: float,
    *,
    strategy: str = "rms",
    random_state: int | None = None,
    min_frames: int = 3,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if scores.size == 0:
        return np.array([], dtype=int)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("Support fraction must be in (0, 1].")
    count = min(scores.size, max(min_frames, int(np.ceil(scores.size * fraction))))
    indices = np.arange(scores.size)
    if strategy == "rms":
        safe = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
        ranked = np.lexsort((indices, -safe))
    elif strategy == "random":
        ranked = np.random.default_rng(random_state).permutation(indices)
    else:
        raise ValueError(f"Unknown support strategy: {strategy}")
    return np.sort(ranked[:count]).astype(int)


def _safe_delta(values: np.ndarray) -> np.ndarray:
    frame_count = values.shape[1]
    if frame_count < 3:
        return np.zeros_like(values)
    width = min(9, frame_count if frame_count % 2 else frame_count - 1)
    return librosa.feature.delta(values, width=max(3, width))


def extract_frame_features(audio: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        audio = np.zeros(1, dtype=np.float32)
    kwargs = {"y": audio, "hop_length": HOP_LENGTH, "n_fft": FRAME_LENGTH}
    mfcc = librosa.feature.mfcc(**kwargs, sr=sr, n_mfcc=N_MFCC)
    delta = _safe_delta(mfcc)
    features: dict[str, np.ndarray] = {}
    for index in range(N_MFCC):
        features[f"mfcc_{index}"] = mfcc[index]
        features[f"mfcc_delta_{index}"] = delta[index]
    features.update(
        {
            "spectral_centroid": librosa.feature.spectral_centroid(**kwargs, sr=sr)[0],
            "spectral_bandwidth": librosa.feature.spectral_bandwidth(**kwargs, sr=sr)[
                0
            ],
            "spectral_rolloff": librosa.feature.spectral_rolloff(**kwargs, sr=sr)[0],
            "spectral_flatness": librosa.feature.spectral_flatness(**kwargs)[0],
            "zcr": librosa.feature.zero_crossing_rate(
                y=audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
            )[0],
            "rms": librosa.feature.rms(
                y=audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
            )[0],
        }
    )
    common = min(map(len, features.values()))
    return {
        name: np.nan_to_num(np.asarray(values[:common], dtype=float))
        for name, values in features.items()
    }


def fixed_candidate_features(
    audio: np.ndarray, frame_features: dict[str, np.ndarray]
) -> dict[str, float]:
    absolute = np.abs(audio)
    rms = float(np.sqrt(np.mean(np.square(audio))))
    frame_rms = frame_features["rms"]
    noise_floor = float(np.percentile(frame_rms, 10))
    threshold = max(rms * 0.1, 1e-8)
    return {
        "peak_amplitude": float(np.max(absolute)),
        "crest_factor": float(np.max(absolute) / max(rms, 1e-8)),
        "silence_ratio": float(np.mean(frame_rms < threshold)),
        "clipping_ratio": float(np.mean(absolute >= 0.999)),
        "noise_floor_rms_p10": noise_floor,
        "snr_proxy_db": float(20.0 * np.log10(max(rms, 1e-8) / max(noise_floor, 1e-8))),
    }


def summarize_support(
    frame_features: dict[str, np.ndarray],
    indices: np.ndarray,
    fixed_features: dict[str, float] | None,
) -> tuple[np.ndarray, list[str]]:
    values: dict[str, float] = {}
    for name, frames in frame_features.items():
        selected = frames[indices]
        values[f"{name}_mean"] = float(np.nan_to_num(np.mean(selected)))
        values[f"{name}_std"] = float(np.nan_to_num(np.std(selected)))
    if fixed_features is not None:
        values.update(fixed_features)
    names = sorted(values)
    return np.asarray([values[name] for name in names], dtype=np.float32), names


def extract_publication_features(
    audio: np.ndarray,
    sr: int,
    rms_fractions: tuple[float, ...],
    random_seeds: tuple[int, ...],
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    """Extract Fixed6 and 64-only branches from one common frame matrix."""
    frame_features = extract_frame_features(audio, sr)
    rms = frame_features["rms"]
    all_indices = np.arange(len(rms), dtype=int)
    fixed = fixed_candidate_features(audio, frame_features)
    vectors: dict[str, np.ndarray] = {}
    names: dict[str, list[str]] = {}

    def add(branch: str, indices: np.ndarray) -> None:
        vectors[branch], names[branch] = summarize_support(
            frame_features, indices, fixed
        )
        diagnostic = f"64only__{branch}"
        vectors[diagnostic], names[diagnostic] = summarize_support(
            frame_features, indices, None
        )

    add("global100", all_indices)
    for fraction in rms_fractions:
        add(
            f"rms{round(fraction * 100):02d}",
            select_frame_indices(rms, fraction, strategy="rms"),
        )
    for seed in random_seeds:
        add(
            f"random20_seed{seed}",
            select_frame_indices(rms, 0.20, strategy="random", random_state=seed),
        )
    return vectors, names
