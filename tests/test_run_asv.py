import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.run_asv import shard_benchmark_regex


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
        self.assertIn("--publish", result.stdout)
        self.assertIn("--shard NAME", result.stdout)
        self.assertIn("python tools/run_asv.py --quick", result.stdout)
        self.assertIn(
            "python tools/run_asv.py --bench MatMul main^!",
            result.stdout,
        )
        self.assertIn(
            "--publish --shard models/llm/tiny_llm HEAD^!",
            result.stdout,
        )
        self.assertNotIn("--environment", result.stdout)
        self.assertEqual(result.stderr, "")

    def test_quick_cannot_be_published(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--quick", "--publish"],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("--publish cannot be combined with --quick", result.stderr)

    def test_bench_cannot_be_combined_with_shard(self):
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--bench",
                "MatMul",
                "--shard",
                "ops/math/add",
            ],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("--bench cannot be combined with --shard", result.stderr)

    def test_bench_cannot_be_published(self):
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--bench",
                "MatMul",
                "--publish",
            ],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("--bench cannot be combined with --publish", result.stderr)

    def test_shard_benchmark_regex(self):
        self.assertEqual(
            shard_benchmark_regex("models/llm/tiny_llm"),
            r"^models\.llm\.tiny_llm\.",
        )


if __name__ == "__main__":
    unittest.main()
