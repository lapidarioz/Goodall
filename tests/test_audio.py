import unittest

import numpy as np

from neoprego_audio2026.audio import extract_candidate, generate_candidates


class CandidateGenerationTests(unittest.TestCase):
    def test_candidates_are_fixed_length_and_labels_do_not_place_proposals(self):
        sr = 16_000
        audio = np.zeros(sr * 8, dtype=np.float32)
        rng = np.random.default_rng(7)
        audio[sr * 3 : sr * 4] = rng.normal(0, 0.25, sr)
        candidates = generate_candidates(audio, sr, [3.5])

        self.assertTrue(candidates)
        self.assertTrue(any(row["label"] == 1 for row in candidates))
        for row in candidates:
            segment = extract_candidate(audio, sr, row["start_time"], row["end_time"])
            self.assertEqual(len(segment), sr * 5)

    def test_boundary_candidates_are_zero_padded(self):
        audio = np.ones(100, dtype=np.float32)
        segment = extract_candidate(audio, 10, -2.0, 3.0)
        self.assertEqual(len(segment), 50)
        np.testing.assert_array_equal(segment[:20], 0.0)
