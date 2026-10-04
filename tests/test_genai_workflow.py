import unittest
from pathlib import Path

WORKFLOW = (
    Path(__file__).resolve().parents[1] / ".github" / "workflows" / "genai-compatibility.yml"
).read_text(encoding="utf-8")


class TestGenAIWorkflow(unittest.TestCase):
    def test_main_branches_cover_both_python_abi_modes(self):
        self.assertIn("dependencies: main-branches-abi3", WORKFLOW)
        self.assertIn("dependencies: main-branches-native", WORKFLOW)
        self.assertIn("-C wheel.py-api=cp312", WORKFLOW)
        self.assertIn(
            "-C cmake.define.ONNX_LIGHT_PYTHON_STABLE_ABI=OFF",
            WORKFLOW,
        )
        self.assertIn(
            "-DONNX_LIGHT_CPU_PYTHON_STABLE_ABI=${{ matrix.stable-abi }}",
            WORKFLOW,
        )


if __name__ == "__main__":
    unittest.main()
