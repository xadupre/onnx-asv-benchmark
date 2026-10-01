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
        self.assertIn("usage: asv run", result.stdout)
        self.assertIn("Created ASV profile", result.stderr)


if __name__ == "__main__":
    unittest.main()
