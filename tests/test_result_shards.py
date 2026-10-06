import json
import tempfile
import unittest
from pathlib import Path

from tools.result_shards import (
    benchmark_shard,
    canonical_benchmark_name,
    merge_shards,
    migrate_legacy_results,
    migrate_shard_hierarchy,
    write_shards,
)


class TestResultShards(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    @staticmethod
    def write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def make_results(self, root):
        benchmarks = {
            "version": 2,
            "maths.add.Add.time_run": {
                "type": "time",
                "legacy": True,
                "name": "maths.add.Add.time_run",
            },
            "ops.math.add.Add.time_run": {
                "type": "time",
                "name": "ops.math.add.Add.time_run",
            },
            "models.tiny_llm.TinyLLM.time_prefill": {
                "type": "time",
                "legacy": True,
            },
            "models.llm.tiny_llm.TinyLLM.time_prefill": {"type": "time"},
        }
        self.write_json(root / "benchmarks.json", benchmarks)
        self.write_json(
            root / "xadupre2025" / "machine.json",
            {
                "version": 1,
                "machine": "xadupre2025",
                "cpu": "Example CPU",
                "num_cpu": "8",
                "instruction_sets": "AVX, AVX2",
            },
        )
        self.write_json(
            root / "xadupre2025" / "result.json",
            {
                "commit_hash": "abc",
                "env_name": "existing",
                "date": 1,
                "params": {
                    "machine": "xadupre2025",
                    "cpu": "Example CPU",
                    "num_cpu": "8",
                    "instruction_sets": "AVX, AVX2",
                },
                "python": "3.12",
                "requirements": {},
                "env_vars": {},
                "result_columns": ["result"],
                "results": {
                    "maths.add.Add.time_run": [0],
                    "ops.math.add.Add.time_run": [1],
                    "models.tiny_llm.TinyLLM.time_prefill": [0],
                    "models.llm.tiny_llm.TinyLLM.time_prefill": [2],
                    "machine.track_processor": ["Example CPU"],
                },
                "durations": {
                    "maths.add.Add.time_run": 1,
                    "ops.math.add.Add.time_run": 3,
                    "models.tiny_llm.TinyLLM.time_prefill": 1,
                    "models.llm.tiny_llm.TinyLLM.time_prefill": 4,
                    "machine.track_processor": 5,
                },
                "version": 2,
            },
        )

    def test_benchmark_shard(self):
        self.assertEqual(benchmark_shard("ops.math.add.Add.time_run"), "ops/math")
        self.assertEqual(benchmark_shard("maths.add.Add.time_run"), "ops/math")
        self.assertEqual(
            benchmark_shard("models.llm.tiny_llm.TinyLLM.time_prefill"),
            "models/llm/tiny_llm",
        )
        self.assertEqual(
            benchmark_shard("models.tiny_llm.TinyLLM.time_prefill"),
            "models/llm/tiny_llm",
        )
        self.assertEqual(
            benchmark_shard("builder.load.onnx_io.OnnxLoad.time_run"),
            "builder/load/onnx_io",
        )

    def test_canonical_benchmark_name(self):
        self.assertEqual(
            canonical_benchmark_name("math.add.Add.time_run"),
            "ops.math.add.Add.time_run",
        )
        self.assertEqual(
            canonical_benchmark_name("models.matmul_add.MatMulAdd.time_run"),
            "models.dummies.matmul_add.MatMulAdd.time_run",
        )
        self.assertEqual(
            canonical_benchmark_name("models.llm.tiny_llm.TinyLLM.time_prefill"),
            "models.llm.tiny_llm.TinyLLM.time_prefill",
        )

    def test_write_and_merge_shards(self):
        source = self.root / "source"
        shards = self.root / "shards"
        merged = self.root / "merged"
        self.make_results(source)
        self.write_json(
            shards / "ops" / "math" / "cpu" / "machine.json",
            {"machine": "cpu", "cpu": "anonymous"},
        )
        self.write_json(shards / "ops" / "math" / "cpu" / "old.json", {})

        self.assertEqual(
            write_shards(source, shards),
            {"ops/math", "models/llm/tiny_llm"},
        )
        self.assertFalse((shards / "ops" / "math" / "cpu").exists())
        self.assertTrue(
            (
                shards
                / "ops"
                / "math"
                / "Example CPU (8 vCPU)"
                / "result.json"
            ).is_file()
        )
        math_result_path = (
            shards
            / "ops"
            / "math"
            / "Example CPU (8 vCPU)"
            / "result.json"
        )
        math_result = json.loads(math_result_path.read_text(encoding="utf-8"))
        math_result["params"]["num_cpu"] = "4"
        self.write_json(math_result_path, math_result)
        model_result_path = (
            shards
            / "models"
            / "llm"
            / "tiny_llm"
            / "Example CPU (8 vCPU)"
            / "result.json"
        )
        model_result = json.loads(model_result_path.read_text(encoding="utf-8"))
        model_result["params"]["instruction_sets"] = "AVX, AVX2, AVX-512F"
        self.write_json(model_result_path, model_result)
        serialized = "\n".join(
            path.read_text(encoding="utf-8") for path in shards.rglob("*.json")
        )
        self.assertNotIn("xadupre2025", serialized)
        self.assertIn("Example CPU", serialized)

        merge_shards(shards, merged)
        results = json.loads(
            (merged / "Example CPU (8 vCPU)" / "result.json").read_text(
                encoding="utf-8"
            )
        )
        merged_benchmarks = json.loads(
            (merged / "benchmarks.json").read_text(encoding="utf-8")
        )
        self.assertEqual(merged_benchmarks["version"], 2)
        self.assertNotIn("maths.add.Add.time_run", merged_benchmarks)
        self.assertNotIn("models.tiny_llm.TinyLLM.time_prefill", merged_benchmarks)
        self.assertEqual(
            set(results["results"]),
            {
                "ops.math.add.Add.time_run",
                "models.llm.tiny_llm.TinyLLM.time_prefill",
            },
        )
        self.assertEqual(results["params"]["machine"], "Example CPU (8 vCPU)")
        self.assertEqual(results["params"]["cpu"], "Example CPU")
        self.assertEqual(results["params"]["num_cpu"], "4; 8")
        self.assertEqual(
            results["params"]["instruction_sets"],
            "AVX,AVX2;AVX,AVX2,AVX-512F",
        )
        machine = json.loads(
            (merged / "Example CPU (8 vCPU)" / "machine.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(machine["num_cpu"], "anonymous")
        self.assertNotIn("instruction_sets", machine)

        second_source = self.root / "second-source"
        self.make_results(second_source)
        machine_path = second_source / "xadupre2025" / "machine.json"
        machine = json.loads(machine_path.read_text(encoding="utf-8"))
        machine["num_cpu"] = "4"
        self.write_json(machine_path, machine)
        result_path = second_source / "xadupre2025" / "result.json"
        second_results = json.loads(result_path.read_text(encoding="utf-8"))
        second_results["params"]["num_cpu"] = "4"
        self.write_json(result_path, second_results)

        write_shards(second_source, shards)
        merge_shards(shards, merged)

        self.assertTrue(
            (merged / "Example CPU (4 vCPU)" / "result.json").is_file()
        )
        self.assertTrue(
            (merged / "Example CPU (8 vCPU)" / "result.json").is_file()
        )

    def test_migrate_legacy_results(self):
        destination = self.root / "cache-data"
        self.make_results(destination)

        migrated = migrate_legacy_results(destination)

        self.assertEqual(migrated, {"ops/math", "models/llm/tiny_llm"})
        self.assertFalse((destination / "benchmarks.json").exists())
        self.assertFalse((destination / "xadupre2025").exists())
        self.assertTrue(
            (
                destination
                / "shards"
                / "models"
                / "llm"
                / "tiny_llm"
                / "Example CPU (8 vCPU)"
                / "result.json"
            ).is_file()
        )

    def test_migrate_shard_hierarchy(self):
        source = self.root / "source"
        shards = self.root / "shards"
        self.make_results(source)
        write_shards(source, shards)
        (shards / "ops" / "math").rename(shards / "maths")
        (shards / "models" / "llm" / "tiny_llm").rename(shards / "models" / "tiny_llm")

        migrated = migrate_shard_hierarchy(shards)

        self.assertEqual(migrated, {"ops/math", "models/llm/tiny_llm"})
        self.assertFalse((shards / "maths").exists())
        self.assertFalse((shards / "models" / "tiny_llm").exists())
        self.assertTrue((shards / "ops" / "math" / "benchmarks.json").is_file())
        self.assertTrue(
            (shards / "models" / "llm" / "tiny_llm" / "benchmarks.json").is_file()
        )
        benchmarks = json.loads(
            (shards / "ops" / "math" / "benchmarks.json").read_text(encoding="utf-8")
        )
        self.assertIn("ops.math.add.Add.time_run", benchmarks)
        self.assertNotIn("maths.add.Add.time_run", benchmarks)
        self.assertEqual(
            benchmarks["ops.math.add.Add.time_run"]["name"],
            "ops.math.add.Add.time_run",
        )


if __name__ == "__main__":
    unittest.main()
