import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from asv.machine import Machine


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "setup_machine.py"


class TestSetupMachine(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.TemporaryDirectory()
        self.addCleanup(self.home.cleanup)
        self.path = Path(self.home.name) / ".asv-machine.json"
        self.env = {**os.environ, "HOME": self.home.name}
        self.profile = Machine.get_defaults()
        self.processor = self.profile["cpu"]
        self.profile["machine"] = self.processor

    def run_setup(self, *args):
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            env=self.env,
            capture_output=True,
            text=True,
        )

    def test_create_and_check_profile(self):
        result = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), self.processor)
        self.assertEqual(json.loads(self.path.read_text())[self.processor], self.profile)

        original = self.path.read_bytes()
        result = self.run_setup("--check")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), self.processor)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(self.run_setup().returncode, 0)
        self.assertEqual(self.path.read_bytes(), original)

    def test_repair_incorrect_profile_without_touching_other_machines(self):
        other = {"machine": "other", "cpu": "other CPU"}
        self.path.write_text(
            json.dumps(
                {
                    "version": 1,
                    self.processor: {**self.profile, "machine": "wrong", "cpu": "wrong"},
                    "other": other,
                }
            )
        )
        before = self.path.read_bytes()
        result = self.run_setup("--check")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing or outdated", result.stderr)
        self.assertEqual(self.path.read_bytes(), before)

        result = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stderr)
        machines = json.loads(self.path.read_text())
        self.assertEqual(machines[self.processor], self.profile)
        self.assertEqual(machines["other"], other)

    def test_check_missing_profile_does_not_create_file(self):
        result = self.run_setup("--check")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.path.exists())

    def test_repair_malformed_processor_entry(self):
        self.path.write_text(json.dumps({"version": 1, self.processor: "wrong"}))
        result = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(self.path.read_text())[self.processor], self.profile)


if __name__ == "__main__":
    unittest.main()
