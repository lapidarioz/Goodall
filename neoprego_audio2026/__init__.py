"""Reproducible code for the NeoPrego Audio 2026 publication experiment."""

from .config import ExperimentConfig
from .experiment import run_experiment

__all__ = ["ExperimentConfig", "run_experiment"]
__version__ = "1.0.0"
