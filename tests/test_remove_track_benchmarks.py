import json
import tempfile
import unittest
from pathlib import Path

from tools.remove_track_benchmarks import remove_track_benchmarks


class TestRemoveTrackBenchmarks(unittest.TestCase):
    def test_remove_track_benchmarks(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "benchmarks.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 2,
                        "math.add.Add.time_run": {"type": "time"},
                        "machine.track_processor": {"type": "track"},
                    }
                ),
                encoding="utf-8",
            )

            remove_track_benchmarks(path)

            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {
                    "version": 2,
                    "math.add.Add.time_run": {"type": "time"},
                },
            )


if __name__ == "__main__":
    unittest.main()
