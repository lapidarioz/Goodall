"""Frozen PANNs Cnn14_16k embedding adapter.

The architecture follows the MIT-licensed reference implementation at
https://github.com/qiuqiangkong/audioset_tagging_cnn.  The pretrained weights
are external research artifacts and are never downloaded implicitly.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PANN_CNN14_16K_EMBEDDING_DIM = 2048
PANN_CNN14_16K_OFFICIAL_MD5 = "362fc5ff18f1d6ad2f6d464b45893f2c"
PANN_CNN14_16K_OFFICIAL_SHA256 = (
    "e2ee543a27919542c2ea03eabaa70b24dcd4e6c8e05621de6b67a94e4c5058e6"
)
PANN_CNN14_16K_FILENAME = "Cnn14_16k_mAP=0.438.pth"
PANN_CNN14_16K_URL = (
    "https://zenodo.org/records/3987831/files/Cnn14_16k_mAP%3D0.438.pth?download=1"
)


def file_digest(path: str | Path, algorithm: str = "sha256") -> str:
    digest = (
        hashlib.md5(usedforsecurity=False)
        if algorithm.lower() == "md5"
        else hashlib.new(algorithm)
    )
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(
    path: str | Path,
    expected_md5: str = PANN_CNN14_16K_OFFICIAL_MD5,
    expected_sha256: str = PANN_CNN14_16K_OFFICIAL_SHA256,
) -> dict[str, str | int]:
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"PANNs checkpoint not found: {checkpoint}. Download "
            f"{PANN_CNN14_16K_FILENAME} with scripts/download_panns_checkpoint.py "
            "or set NEOPREGO_AUDIO2026_PANNS_CHECKPOINT."
        )
    observed_md5 = file_digest(checkpoint, "md5")
    if expected_md5 and observed_md5.lower() != expected_md5.lower():
        raise ValueError(
            "PANNs checkpoint checksum mismatch: "
            f"expected md5={expected_md5}, observed md5={observed_md5}."
        )
    observed_sha256 = file_digest(checkpoint, "sha256")
    if expected_sha256 and observed_sha256.lower() != expected_sha256.lower():
        raise ValueError(
            "PANNs checkpoint checksum mismatch: "
            f"expected sha256={expected_sha256}, observed sha256={observed_sha256}."
        )
    return {
        "path": str(checkpoint.resolve()),
        "filename": checkpoint.name,
        "size_bytes": checkpoint.stat().st_size,
        "md5": observed_md5,
        "sha256": observed_sha256,
    }


@dataclass(frozen=True)
class PANNsEncoderConfig:
    checkpoint_path: str
    expected_md5: str = PANN_CNN14_16K_OFFICIAL_MD5
    expected_sha256: str = PANN_CNN14_16K_OFFICIAL_SHA256
    device: str = "auto"
    batch_size: int = 16
    sample_rate: int = 16000

    def validate(self) -> None:
        if not self.checkpoint_path:
            raise ValueError(
                "NEOPREGO_AUDIO2026_PANNS_CHECKPOINT must point to the official "
                f"{PANN_CNN14_16K_FILENAME} checkpoint."
            )
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("PANNs device must be auto, cpu, or cuda.")
        if self.batch_size < 1:
            raise ValueError("PANNs batch size must be positive.")
        if self.sample_rate != 16000:
            raise ValueError("The frozen Cnn14_16k encoder requires 16 kHz audio.")


def _build_cnn14_16k():
    try:
        import torch
        from torch import nn
        from torch.nn import functional
        from torchlibrosa.stft import LogmelFilterBank, Spectrogram
    except ImportError as exc:
        raise RuntimeError(
            "PANNs dependencies are unavailable. Install them with "
            "`uv sync --extra panns`."
        ) from exc

    def init_layer(layer):
        nn.init.xavier_uniform_(layer.weight)
        if getattr(layer, "bias", None) is not None:
            layer.bias.data.fill_(0.0)

    def init_bn(layer):
        layer.bias.data.fill_(0.0)
        layer.weight.data.fill_(1.0)

    class ConvBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1, bias=False
            )
            self.conv2 = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, padding=1, bias=False
            )
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
            init_layer(self.conv1)
            init_layer(self.conv2)
            init_bn(self.bn1)
            init_bn(self.bn2)

        def forward(self, values, pool_size=(2, 2)):
            values = functional.relu_(self.bn1(self.conv1(values)))
            values = functional.relu_(self.bn2(self.conv2(values)))
            return functional.avg_pool2d(values, kernel_size=pool_size)

    class Cnn14(nn.Module):
        def __init__(self):
            super().__init__()
            self.spectrogram_extractor = Spectrogram(
                n_fft=512,
                hop_length=160,
                win_length=512,
                window="hann",
                center=True,
                pad_mode="reflect",
                freeze_parameters=True,
            )
            self.logmel_extractor = LogmelFilterBank(
                sr=16000,
                n_fft=512,
                n_mels=64,
                fmin=50,
                fmax=8000,
                ref=1.0,
                amin=1e-10,
                top_db=None,
                freeze_parameters=True,
            )
            self.bn0 = nn.BatchNorm2d(64)
            self.conv_block1 = ConvBlock(1, 64)
            self.conv_block2 = ConvBlock(64, 128)
            self.conv_block3 = ConvBlock(128, 256)
            self.conv_block4 = ConvBlock(256, 512)
            self.conv_block5 = ConvBlock(512, 1024)
            self.conv_block6 = ConvBlock(1024, 2048)
            self.fc1 = nn.Linear(2048, 2048)
            self.fc_audioset = nn.Linear(2048, 527)
            init_bn(self.bn0)
            init_layer(self.fc1)
            init_layer(self.fc_audioset)

        def forward(self, waveform):
            values = self.spectrogram_extractor(waveform)
            values = self.logmel_extractor(values)
            values = values.transpose(1, 3)
            values = self.bn0(values)
            values = values.transpose(1, 3)
            values = self.conv_block1(values)
            values = self.conv_block2(values)
            values = self.conv_block3(values)
            values = self.conv_block4(values)
            values = self.conv_block5(values)
            values = self.conv_block6(values, pool_size=(1, 1))
            values = torch.mean(values, dim=3)
            maximum, _ = torch.max(values, dim=2)
            values = maximum + torch.mean(values, dim=2)
            values = functional.relu_(self.fc1(values))
            return {
                "embedding": values,
                "clipwise_output": torch.sigmoid(self.fc_audioset(values)),
            }

    return torch, Cnn14()


class PANNsEncoder:
    """Load one frozen Cnn14_16k checkpoint and batch segment embeddings."""

    embedding_dim = PANN_CNN14_16K_EMBEDDING_DIM

    def __init__(self, config: PANNsEncoderConfig):
        config.validate()
        self.config = config
        self.checkpoint = verify_checkpoint(
            config.checkpoint_path, config.expected_md5, config.expected_sha256
        )
        self._torch, self._model = _build_cnn14_16k()
        if config.device == "auto":
            device_name = "cuda" if self._torch.cuda.is_available() else "cpu"
        else:
            device_name = config.device
        if device_name == "cuda" and not self._torch.cuda.is_available():
            raise RuntimeError("PANNs CUDA was requested but CUDA is unavailable.")
        self.device = device_name
        self._device = self._torch.device(device_name)
        try:
            # The published checkpoint contains NumPy metadata in addition to the
            # state dict. It is loaded only after the configured checksum passes.
            checkpoint = self._torch.load(
                config.checkpoint_path,
                map_location=self._device,
                weights_only=False,
            )
        except TypeError:
            checkpoint = self._torch.load(
                config.checkpoint_path, map_location=self._device
            )
        state = checkpoint.get("model", checkpoint)
        if any(str(name).startswith("module.") for name in state):
            state = {
                str(name).removeprefix("module."): value
                for name, value in state.items()
            }
        self._model.load_state_dict(state, strict=True)
        self._model.to(self._device)
        self._model.eval()
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)

    def embed(self, waveforms: Iterable[np.ndarray]) -> np.ndarray:
        arrays = [np.asarray(item, dtype=np.float32).reshape(-1) for item in waveforms]
        if not arrays:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        expected_samples = self.config.sample_rate * 5
        normalized = []
        for waveform in arrays:
            if waveform.size < expected_samples:
                waveform = np.pad(waveform, (0, expected_samples - waveform.size))
            elif waveform.size > expected_samples:
                waveform = waveform[:expected_samples]
            normalized.append(waveform)
        results = []
        with self._torch.inference_mode():
            for start in range(0, len(normalized), self.config.batch_size):
                batch = np.stack(
                    normalized[start : start + self.config.batch_size]
                ).astype(np.float32)
                tensor = self._torch.from_numpy(batch).to(self._device)
                embedding = self._model(tensor)["embedding"]
                results.append(embedding.detach().cpu().numpy().astype(np.float32))
        matrix = np.vstack(results)
        if matrix.shape != (len(arrays), self.embedding_dim):
            raise RuntimeError(
                "Unexpected PANNs embedding shape: "
                f"{matrix.shape}, expected {(len(arrays), self.embedding_dim)}."
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError("PANNs produced NaN or infinite embeddings.")
        return matrix

    def metadata(self) -> dict:
        return {
            "architecture": "PANNs Cnn14_16k",
            "embedding_dim": self.embedding_dim,
            "sample_rate": self.config.sample_rate,
            "batch_size": self.config.batch_size,
            "device": self.device,
            "checkpoint": self.checkpoint,
            "frozen": True,
        }
