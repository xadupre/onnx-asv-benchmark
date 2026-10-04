import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from asv.machine import Machine

from tools.setup_machine import detect_instruction_sets, instruction_sets, machine_name

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "setup_machine.py"


class TestSetupMachine(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.TemporaryDirectory()
        self.addCleanup(self.home.cleanup)
        self.path = Path(self.home.name) / ".asv-machine.json"
        self.env = {**os.environ, "HOME": self.home.name}
        self.profile = Machine.get_defaults()
        self.processor = self.profile["cpu"]
        self.machine = machine_name(self.processor, self.profile["num_cpu"])
        self.profile["machine"] = self.machine
        self.profile["instruction_sets"] = detect_instruction_sets()

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
        self.assertEqual(result.stdout.strip(), self.machine)
        self.assertIn(
            f"Created ASV profile {self.machine!r} in {self.path}", result.stderr
        )
        self.assertEqual(json.loads(self.path.read_text())[self.machine], self.profile)

        original = self.path.read_bytes()
        result = self.run_setup("--check")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), self.machine)
        self.assertIn(
            f"Verified ASV profile {self.machine!r} in {self.path}", result.stderr
        )
        self.assertEqual(self.path.read_bytes(), original)
        result = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"Already correct ASV profile {self.machine!r}", result.stderr)
        self.assertEqual(self.path.read_bytes(), original)

    def test_repair_incorrect_profile_without_touching_other_machines(self):
        other = {"machine": "other", "cpu": "other CPU"}
        self.path.write_text(
            json.dumps(
                {
                    "version": 1,
                    self.machine: {**self.profile, "machine": "wrong", "cpu": "wrong"},
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
        self.assertIn(
            f"Updated ASV profile {self.machine!r} in {self.path}", result.stderr
        )
        machines = json.loads(self.path.read_text())
        self.assertEqual(machines[self.machine], self.profile)
        self.assertEqual(machines["other"], other)

    def test_check_missing_profile_does_not_create_file(self):
        result = self.run_setup("--check")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(f"in {self.path} is missing or outdated", result.stderr)
        self.assertFalse(self.path.exists())

    def test_repair_malformed_processor_entry(self):
        self.path.write_text(json.dumps({"version": 1, self.machine: "wrong"}))
        result = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(self.path.read_text())[self.machine], self.profile)

    def test_instruction_sets(self):
        cpuinfo = """\
processor : 0
flags : sse2 pni ssse3 sse4_1 sse4_2 avx avx2 avx512f avx512_vnni
"""
        self.assertEqual(
            instruction_sets(cpuinfo),
            "SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX, AVX2, "
            "AVX-512F, AVX-512VNNI",
        )

    def test_machine_name_includes_available_logical_cores(self):
        self.assertEqual(machine_name("Example CPU", "4"), "Example CPU (4 vCPU)")


if __name__ == "__main__":
    unittest.main()
