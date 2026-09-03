#!/usr/bin/env python3
"""Download and verify the official PANNs Cnn14_16k checkpoint."""

import argparse
import hashlib
import urllib.request
from pathlib import Path

URL = "https://zenodo.org/records/3987831/files/Cnn14_16k_mAP%3D0.438.pth?download=1"
EXPECTED_MD5 = "362fc5ff18f1d6ad2f6d464b45893f2c"
EXPECTED_SHA256 = "e2ee543a27919542c2ea03eabaa70b24dcd4e6c8e05621de6b67a94e4c5058e6"
FILENAME = "Cnn14_16k_mAP=0.438.pth"


def md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "destination",
        nargs="?",
        default=f"checkpoints/{FILENAME}",
        help="Destination checkpoint path.",
    )
    args = parser.parse_args()
    destination = Path(args.destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if (
        destination.exists()
        and md5(destination) == EXPECTED_MD5
        and sha256(destination) == EXPECTED_SHA256
    ):
        print(f"Verified existing checkpoint: {destination}")
        return

    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {URL}")
    urllib.request.urlretrieve(URL, temporary)
    observed = md5(temporary)
    if observed != EXPECTED_MD5:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checkpoint checksum mismatch: expected {EXPECTED_MD5}, observed {observed}."
        )
    observed_sha256 = sha256(temporary)
    if observed_sha256 != EXPECTED_SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            "Checkpoint checksum mismatch: "
            f"expected SHA-256 {EXPECTED_SHA256}, observed {observed_sha256}."
        )
    temporary.replace(destination)
    print(f"Verified checkpoint: {destination}")


if __name__ == "__main__":
    main()
