import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "run_asv.py"


class TestRunAsv(unittest.TestCase):
    def test_help(self):
        with tempfile.TemporaryDirectory() as home:
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--help"],
                env={**os.environ, "HOME": home},
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage: run_asv.py", result.stdout)
        self.assertIn("--bench REGEX", result.stdout)
        self.assertIn("--quick", result.stdout)
        self.assertIn("python tools/run_asv.py --quick", result.stdout)
        self.assertIn(
            "python tools/run_asv.py --bench MatMul main^!",
            result.stdout,
        )
        self.assertNotIn("--environment", result.stdout)
        self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
