import unittest

from neoprego_audio2026.cli import build_parser


class CommandLineTests(unittest.TestCase):
    def test_required_inputs_parse(self):
        args = build_parser().parse_args(
            ["--dataset-root", "/data", "--events-csv", "/data/events.csv"]
        )
        self.assertEqual(str(args.dataset_root), "/data")
        self.assertEqual(args.bootstrap_samples, 10_000)
