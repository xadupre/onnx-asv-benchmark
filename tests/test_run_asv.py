import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.scheduled_shards import shard_benchmark_regex
from tools.run_asv import (
    final_status,
    require_main_commit,
    resolve_commit,
)

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
            "python tools/run_asv.py --bench MatMul",
            result.stdout,
        )
        self.assertIn(
            "--publish --shard models/llm/tiny_llm",
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
        import re

        regex = shard_benchmark_regex("cpu_backend_cases/math/cases/mul")
        self.assertIsNotNone(
            re.match(regex, "cpu_backend_cases.math.cases.MulUint32Inputs2Shard4.time_run")
        )
        self.assertIsNone(
            re.match(regex, "cpu_backend_cases.math.cases.MulIntegerFloat32.time_run")
        )
        self.assertIsNone(
            re.match(regex, "cpu_backend_cases.math.cases.AddFloat32.time_run")
        )

    def test_resolve_single_commit(self):
        root = SCRIPT.parents[1]
        expected = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
        ).strip()

        self.assertEqual(resolve_commit(root, "HEAD"), expected)
        with self.assertRaises(subprocess.CalledProcessError):
            resolve_commit(root, "HEAD^!")

    def test_commit_must_be_on_main(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "-b", "main"], cwd=root, check=True)
            subprocess.run(
                ["git", "config", "user.name", "Test User"], cwd=root, check=True
            )
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=root,
                check=True,
            )
            (root / "file.txt").write_text("main\n", encoding="utf-8")
            subprocess.run(["git", "add", "file.txt"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-m", "main"], cwd=root, check=True)
            main_hash = resolve_commit(root, "HEAD")
            subprocess.run(["git", "switch", "-c", "feature"], cwd=root, check=True)
            (root / "file.txt").write_text("feature\n", encoding="utf-8")
            subprocess.run(["git", "commit", "-am", "feature"], cwd=root, check=True)
            feature_hash = resolve_commit(root, "HEAD")

            require_main_commit(root, main_hash)
            with self.assertRaisesRegex(
                ValueError, "not part of the local main branch"
            ):
                require_main_commit(root, feature_hash)

    def test_published_benchmark_failures_are_successful(self):
        self.assertEqual(final_status(2, published=True), 0)
        self.assertEqual(final_status(2, published=False), 2)
        self.assertEqual(final_status(1, published=True), 1)


if __name__ == "__main__":
    unittest.main()
