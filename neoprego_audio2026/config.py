from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    """Frozen settings reported in the publication."""

    dataset_root: Path
    events_csv: Path
    output_dir: Path = Path("outputs/publication")
    fold_manifest: Path | None = None
    panns_checkpoint: Path | None = None
    sample_rate: int = 16_000
    window_seconds: float = 5.0
    annotation_margin_seconds: float = 0.5
    match_tolerance_seconds: float = 1.0
    n_splits: int = 5
    n_trees: int = 300
    random_state: int = 42
    bootstrap_samples: int = 10_000
    rms_fractions: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40, 0.60)
    random_support_repeats: int = 20

    def validate(self) -> None:
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {self.dataset_root}")
        if not self.events_csv.is_file():
            raise FileNotFoundError(f"Annotation CSV does not exist: {self.events_csv}")
        if self.fold_manifest is not None and not self.fold_manifest.is_file():
            raise FileNotFoundError(
                f"Fold manifest does not exist: {self.fold_manifest}"
            )
        if self.panns_checkpoint is not None and not self.panns_checkpoint.is_file():
            raise FileNotFoundError(
                f"PANNs checkpoint does not exist: {self.panns_checkpoint}"
            )
        if self.sample_rate != 16_000:
            raise ValueError("The publication protocol requires 16 kHz audio.")
        if self.window_seconds != 5.0:
            raise ValueError("The publication protocol requires 5-second candidates.")
        if self.n_splits != 5 or self.n_trees != 300 or self.random_state != 42:
            raise ValueError(
                "Fold count, forest size, and seed are frozen at 5, 300, and 42."
            )
        if self.bootstrap_samples < 1:
            raise ValueError("bootstrap_samples must be positive.")
        if self.random_support_repeats < 1:
            raise ValueError("random_support_repeats must be positive.")

    def to_dict(self) -> dict:
        values = asdict(self)
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in values.items()
        }
