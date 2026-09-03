import unittest

import numpy as np

from neoprego_audio2026.features import extract_publication_features, select_frame_indices


class FeatureTests(unittest.TestCase):
    def test_equal_cardinality_supports(self):
        scores = np.linspace(0.0, 1.0, 157)
        rms = select_frame_indices(scores, 0.20, strategy="rms")
        random = select_frame_indices(scores, 0.20, strategy="random", random_state=42)
        self.assertEqual(len(rms), 32)
        self.assertEqual(len(random), 32)
        self.assertFalse(np.array_equal(rms, random))

    def test_fixed6_dimensions_and_shared_candidate_statistics(self):
        sr = 16_000
        time = np.arange(sr * 5) / sr
        audio = (0.1 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
        vectors, names = extract_publication_features(audio, sr, (0.20,), (42,))
        self.assertEqual(vectors["global100"].shape, (70,))
        self.assertEqual(vectors["64only__global100"].shape, (64,))
        fixed_names = {
            "peak_amplitude",
            "crest_factor",
            "silence_ratio",
            "clipping_ratio",
            "noise_floor_rms_p10",
            "snr_proxy_db",
        }
        for feature in fixed_names:
            global_value = vectors["global100"][names["global100"].index(feature)]
            rms_value = vectors["rms20"][names["rms20"].index(feature)]
            random_value = vectors["random20_seed42"][
                names["random20_seed42"].index(feature)
            ]
            self.assertEqual(global_value, rms_value)
            self.assertEqual(global_value, random_value)
