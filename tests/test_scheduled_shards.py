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


if __name__ == "__main__":
    unittest.main()
