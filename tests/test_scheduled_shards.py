import re
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from benchmarks.cpu_backend_cases._manifest import CASE_SHARDS
from tools.scheduled_shards import (
    batch_count,
    case_shard_classes,
    discover_shards,
    shard_benchmark_regex,
)


class TestScheduledShards(unittest.TestCase):
    def test_cli_lists_all_shards_without_bucket(self):
        root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [sys.executable, str(root / "tools" / "scheduled_shards.py")],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )

        shards = completed.stdout.splitlines()
        self.assertIn("ops/math/add", shards)
        self.assertIn("models/llm/tiny_llm", shards)
        self.assertEqual(len(shards), len(discover_shards(root)))

    def test_cli_runs_without_repository_on_pythonpath(self):
        root = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        completed = subprocess.run(
            [sys.executable, str(root / "tools" / "scheduled_shards.py"), "0"],
            cwd=root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        matrix = json.loads(completed.stdout)
        self.assertTrue(matrix["shard"])

    def test_oversized_case_shards_are_batched(self):
        root = Path(__file__).resolve().parents[1]
        for shard in (
            "cpu_backend_cases/math/cases/pow",
            "cpu_backend_cases/logical/cases/less",
        ):
            completed = subprocess.run(
                [
                    sys.executable,
                    str(root / "tools" / "scheduled_shards.py"),
                    "--matrix",
                    "--requested",
                    shard,
                ],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            batches = json.loads(completed.stdout)["include"]
            self.assertEqual(
                batches,
                [{"shard": shard, "batch": index} for index in range(batch_count(shard))],
            )
            classes = case_shard_classes()[shard]
            for index, name in enumerate(classes):
                matching = [
                    batch
                    for batch in batches
                    if re.match(
                        shard_benchmark_regex(shard, batch["batch"]),
                        f"cpu_backend_cases.{shard.split('/')[1]}.cases.{name}.time_case",
                    )
                ]
                self.assertEqual(matching, [batches[index // 10]])
        self.assertEqual(batch_count("cpu_backend_cases/math/cases/mul"), 1)
        self.assertEqual(
            shard_benchmark_regex("cpu_backend_cases/math/cases/mul", 0),
            shard_benchmark_regex("cpu_backend_cases/math/cases/mul"),
        )
        with self.assertRaises(ValueError):
            shard_benchmark_regex(
                "cpu_backend_cases/math/cases/pow",
                batch_count("cpu_backend_cases/math/cases/pow"),
            )

    def test_discover_shards(self):
        root = Path(__file__).resolve().parents[1]
        shards = discover_shards(root)

        self.assertIn("ops/math/add", shards)
        self.assertIn("ops/nn/attention", shards)
        self.assertIn("ops/nn/linear_attention", shards)
        self.assertNotIn("ops/nn", shards)
        self.assertIn("models/llm/qwen2", shards)
        self.assertIn("models/llm/tiny_llm", shards)
        self.assertIn("models/dummies/mlp", shards)
        self.assertIn("builder/pattern/pattern_fusion", shards)
        self.assertNotIn("builder/_onnx_io", shards)
        self.assertIn("builder/builder/graph_builder", shards)
        self.assertIn("builder/load/onnx_io", shards)
        self.assertIn("builder/save/onnx_io", shards)
        self.assertIn("builder/serialize/onnx_io", shards)
        self.assertIn("builder/parse/onnx_io", shards)
        self.assertTrue(
            any(shard.startswith("cpu_backend_cases/math/cases/") for shard in shards)
        )
        self.assertTrue(
            any(
                shard.startswith("cpu_backend_cases/nn/cases/")
                for shard in shards
            )
        )
        self.assertNotIn("__pycache__", shards)
        self.assertEqual(len(shards), len(set(shards)))
        operator_shards = [shard for shard in shards if shard.startswith("ops/")]
        self.assertGreater(len(operator_shards), 50)
        self.assertTrue(all(len(shard.split("/")) == 3 for shard in operator_shards))
        cpu_case_shards = [
            shard for shard in shards if shard.startswith("cpu_backend_cases/")
        ]
        self.assertTrue(cpu_case_shards)
        self.assertTrue(all(len(shard.split("/")) == 4 for shard in cpu_case_shards))
        self.assertIn("cpu_backend_cases/math/cases/mul", cpu_case_shards)
        self.assertEqual(
            [
                name for name in case_shard_classes()["cpu_backend_cases/math/cases/mul"]
                if name.startswith("MulUint32Inputs2Shard")
            ],
            [f"MulUint32Inputs2Shard{index}" for index in range(1, 5)],
        )
        self.assertEqual(len(cpu_case_shards), len(case_shard_classes()))
        self.assertCountEqual(
            [
                (shard.split("/")[1], name)
                for shard, classes in case_shard_classes().items()
                for name in classes
            ],
            [(category, name) for category, name, _, _, _ in CASE_SHARDS],
        )

        buckets = [
            {shard for index, shard in enumerate(shards) if index % 7 == bucket}
            for bucket in range(7)
        ]
        self.assertEqual(set().union(*buckets), set(shards))
        for left, left_bucket in enumerate(buckets):
            for right_bucket in buckets[left + 1 :]:
                self.assertFalse(left_bucket & right_bucket)

    def test_documented_shard_list_is_current(self):
        root = Path(__file__).resolve().parents[1]
        documentation = (root / "SHARDS.md").read_text(encoding="utf-8")
        section = documentation.split("<!-- BEGIN CURRENT SHARDS -->", 1)[1].split(
            "<!-- END CURRENT SHARDS -->", 1
        )[0]
        documented = section.split("```text\n", 1)[1].split("\n```", 1)[0].splitlines()

        self.assertEqual(documented, discover_shards(root))

    def test_each_bucket_is_scheduled_twice(self):
        root = Path(__file__).resolve().parents[1]
        workflow = (root / ".github" / "workflows" / "weekly-benchmarks.yml").read_text(
            encoding="utf-8"
        )
        scheduled = re.findall(r'- cron: "([^"]+)"', workflow)
        mappings = dict(re.findall(r'"([^"]+)"\) bucket=([0-6]) ;;', workflow))

        self.assertEqual(len(scheduled), 14)
        self.assertEqual(set(scheduled), set(mappings))
        self.assertRegex(
            workflow,
            r"(?s)  benchmark:.*?    timeout-minutes: "
            r"\$\{\{ matrix\.shard == 'ops/tensor/grid_sample' && 120 \|\| 60 \}\}",
        )
        self.assertIn('scheduled_shards.py "$bucket" --matrix', workflow)
        self.assertIn('scheduled_shards.py --matrix --requested "$REQUESTED_SHARD"', workflow)
        self.assertIn('scheduled_shards.py --regex "$SHARD" --batch "$BATCH"', workflow)
        self.assertIn("group: benchmark-shard-${{ matrix.shard }}", workflow)
        for bucket in map(str, range(7)):
            self.assertEqual(list(mappings.values()).count(bucket), 2)


if __name__ == "__main__":
    unittest.main()
