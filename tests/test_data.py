import tempfile
import unittest
from pathlib import Path

from neoprego_audio2026.data import build_fold_manifest, load_event_rows, split_by_recording


class DatasetTests(unittest.TestCase):
    def test_csv_filter_and_recording_disjoint_split(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.csv"
            path.write_text(
                "Media_file,Approximation,Behavior\n"
                + "\n".join(
                    f"audio_{index}.wav,{index + 0.25},Strike" for index in range(12)
                )
                + "\naudio_ignored.wav,1.0,Other\n",
                encoding="utf-8",
            )
            rows = load_event_rows(path)
        evaluation, development, templates = split_by_recording(rows)
        self.assertEqual(len(rows), 12)
        self.assertFalse(set(evaluation) & set(development))
        self.assertFalse(set(evaluation) & set(templates))
        self.assertFalse(set(development) & set(templates))
        self.assertEqual(
            set(evaluation) | set(development) | set(templates),
            {f"audio_{i}.wav" for i in range(12)},
        )

    def test_fold_manifest_is_deterministic_and_recording_disjoint(self):
        recordings = [f"recording_{index}" for index in range(12)]
        first = build_fold_manifest(recordings)
        second = build_fold_manifest(list(reversed(recordings)))
        self.assertTrue(first.equals(second))
        self.assertEqual(first["recording_id"].nunique(), len(recordings))
        self.assertEqual(first["fold"].nunique(), 5)
