"""Unit tests for train_probe evaluation protocol enforcement."""

import unittest
from benchmarks.train_probe import main


class TestTrainProbeProtocol(unittest.TestCase):
    def test_missing_protocol_exits(self):
        """Omitting both --val-split and --legacy-test-as-val raises SystemExit."""
        with self.assertRaises(SystemExit) as ctx:
            main(["--dry-run"])
        self.assertIn("Evaluation protocol required", str(ctx.exception))

    def test_both_protocols_specified_exits(self):
        """Specifying both --val-split and --legacy-test-as-val raises SystemExit."""
        with self.assertRaises(SystemExit) as ctx:
            main(["--val-split", "dummy.npz", "--legacy-test-as-val"])
        self.assertIn("Cannot specify both", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
