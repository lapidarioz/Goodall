import unittest

import pandas as pd

from neoprego_audio2026.evaluation import one_to_one_event_matching, score_events, summarize


class EvaluationTests(unittest.TestCase):
    def test_matching_is_one_to_one_and_minimizes_error(self):
        matches = one_to_one_event_matching([1.1, 1.9, 8.0], [1.0, 2.0], tolerance=1.0)
        self.assertEqual(len(matches), 2)
        self.assertEqual({row["prediction_index"] for row in matches}, {0, 1})
        self.assertEqual({row["truth_index"] for row in matches}, {0, 1})

    def test_segment_and_event_summary(self):
        predictions = pd.DataFrame(
            {
                "recording_id": ["a", "a", "b", "b"],
                "center_time": [1.0, 4.0, 2.0, 6.0],
                "y_true": [1, 0, 1, 0],
                "y_pred": [1, 1, 1, 0],
                "y_score": [0.9, 0.7, 0.8, 0.1],
            }
        )
        event_metrics = score_events(predictions, {"a": [1.1], "b": [2.1]})
        summary = summarize(predictions, event_metrics)
        self.assertEqual(summary["matched_events"], 2)
        self.assertEqual(summary["false_detections"], 1)
        self.assertAlmostEqual(summary["event_recall"], 1.0)
