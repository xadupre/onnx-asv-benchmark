import re
import unittest
from pathlib import Path

from tools.scheduled_shards import discover_shards


class TestScheduledShards(unittest.TestCase):
    def test_discover_shards(self):
        root = Path(__file__).resolve().parents[1]
        shards = discover_shards(root)

        self.assertIn("ops/math", shards)
        self.assertIn("ops/nn", shards)
        self.assertIn("models/llm/qwen2", shards)
        self.assertIn("models/llm/tiny_llm", shards)
        self.assertIn("models/dummies/mlp", shards)
        self.assertIn("onnx/builder/graph_builder", shards)
        self.assertNotIn("__pycache__", shards)
        self.assertEqual(len(shards), len(set(shards)))

        buckets = [
            {shard for index, shard in enumerate(shards) if index % 7 == bucket}
            for bucket in range(7)
        ]
        self.assertEqual(set().union(*buckets), set(shards))
        for left, left_bucket in enumerate(buckets):
            for right_bucket in buckets[left + 1 :]:
                self.assertFalse(left_bucket & right_bucket)

    def test_each_bucket_is_scheduled_twice(self):
        root = Path(__file__).resolve().parents[1]
        workflow = (
            root / ".github" / "workflows" / "weekly-benchmarks.yml"
        ).read_text(encoding="utf-8")
        scheduled = re.findall(r'- cron: "([^"]+)"', workflow)
        mappings = dict(
            re.findall(r'"([^"]+)"\) bucket=([0-6]) ;;', workflow)
        )

        self.assertEqual(len(scheduled), 14)
        self.assertEqual(set(scheduled), set(mappings))
        for bucket in map(str, range(7)):
            self.assertEqual(list(mappings.values()).count(bucket), 2)


if __name__ == "__main__":
    unittest.main()
