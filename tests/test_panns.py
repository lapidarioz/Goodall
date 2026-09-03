import hashlib
import tempfile
import unittest
from pathlib import Path

from models.PANNs import verify_checkpoint


class PANNsCheckpointTests(unittest.TestCase):
    def test_checkpoint_requires_both_published_digests(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth"
            path.write_bytes(b"synthetic checkpoint")
            expected_md5 = hashlib.md5(
                path.read_bytes(), usedforsecurity=False
            ).hexdigest()
            expected_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            metadata = verify_checkpoint(path, expected_md5, expected_sha256)
            self.assertEqual(metadata["md5"], expected_md5)
            self.assertEqual(metadata["sha256"], expected_sha256)
            with self.assertRaises(ValueError):
                verify_checkpoint(path, expected_md5, "0" * 64)
