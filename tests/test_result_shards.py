import json
import tempfile
import unittest
from pathlib import Path

from tools.result_shards import (
    benchmark_shard,
    merge_shards,
    migrate_legacy_results,
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
            "math.add.Add.time_run": {"type": "time"},
            "models.tiny_llm.TinyLLM.time_prefill": {"type": "time"},
        }
        self.write_json(root / "benchmarks.json", benchmarks)
        self.write_json(
            root / "xadupre2025" / "machine.json",
            {
                "version": 1,
                "machine": "xadupre2025",
                "cpu": "Example CPU",
            },
        )
        self.write_json(
            root / "xadupre2025" / "result.json",
            {
                "commit_hash": "abc",
                "env_name": "existing",
                "date": 1,
                "params": {"machine": "xadupre2025", "cpu": "Example CPU"},
                "python": "3.12",
                "requirements": {},
                "env_vars": {},
                "result_columns": ["result"],
                "results": {
                    "math.add.Add.time_run": [1],
                    "models.tiny_llm.TinyLLM.time_prefill": [2],
                    "machine.track_processor": ["Example CPU"],
                },
                "durations": {
                    "math.add.Add.time_run": 3,
                    "models.tiny_llm.TinyLLM.time_prefill": 4,
                    "machine.track_processor": 5,
                },
                "version": 2,
            },
        )

    def test_benchmark_shard(self):
        self.assertEqual(benchmark_shard("math.add.Add.time_run"), "math")
        self.assertEqual(
            benchmark_shard("models.tiny_llm.TinyLLM.time_prefill"),
            "models/tiny_llm",
        )

    def test_write_and_merge_shards(self):
        source = self.root / "source"
        shards = self.root / "shards"
        merged = self.root / "merged"
        self.make_results(source)
        self.write_json(
            shards / "math" / "cpu" / "machine.json",
            {"machine": "cpu", "cpu": "anonymous"},
        )
        self.write_json(shards / "math" / "cpu" / "old.json", {})

        self.assertEqual(
            write_shards(source, shards),
            {"math", "models/tiny_llm"},
        )
        self.assertFalse((shards / "math" / "cpu").exists())
        self.assertTrue((shards / "math" / "Example CPU" / "result.json").is_file())
        serialized = "\n".join(
            path.read_text(encoding="utf-8") for path in shards.rglob("*.json")
        )
        self.assertNotIn("xadupre2025", serialized)
        self.assertIn("Example CPU", serialized)

        merge_shards(shards, merged)
        results = json.loads(
            (merged / "Example CPU" / "result.json").read_text(encoding="utf-8")
        )
        merged_benchmarks = json.loads(
            (merged / "benchmarks.json").read_text(encoding="utf-8")
        )
        self.assertEqual(merged_benchmarks["version"], 2)
        self.assertEqual(
            set(results["results"]),
            {
                "math.add.Add.time_run",
                "models.tiny_llm.TinyLLM.time_prefill",
            },
        )
        self.assertEqual(results["params"]["machine"], "Example CPU")
        self.assertEqual(results["params"]["cpu"], "Example CPU")

        second_source = self.root / "second-source"
        self.make_results(second_source)
        machine_path = second_source / "xadupre2025" / "machine.json"
        machine = json.loads(machine_path.read_text(encoding="utf-8"))
        machine["cpu"] = "Other CPU"
        self.write_json(machine_path, machine)
        result_path = second_source / "xadupre2025" / "result.json"
        second_results = json.loads(result_path.read_text(encoding="utf-8"))
        second_results["params"]["cpu"] = "Other CPU"
        self.write_json(result_path, second_results)

        write_shards(second_source, shards)
        merge_shards(shards, merged)

        self.assertTrue((merged / "Other CPU" / "result.json").is_file())
        self.assertTrue((merged / "Example CPU" / "result.json").is_file())

    def test_migrate_legacy_results(self):
        destination = self.root / "cache-data"
        self.make_results(destination)

        migrated = migrate_legacy_results(destination)

        self.assertEqual(migrated, {"math", "models/tiny_llm"})
        self.assertFalse((destination / "benchmarks.json").exists())
        self.assertFalse((destination / "xadupre2025").exists())
        self.assertTrue(
            (
                destination
                / "shards"
                / "models"
                / "tiny_llm"
                / "Example CPU"
                / "result.json"
            ).is_file()
        )


if __name__ == "__main__":
    unittest.main()
