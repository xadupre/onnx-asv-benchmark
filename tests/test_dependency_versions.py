import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestDependencyVersions(unittest.TestCase):
    def test_pinned_release_wheels(self):
        config = json.loads((ROOT / "asv.conf.json").read_text(encoding="utf-8"))
        workflow = (ROOT / ".github" / "workflows" / "tests.yml").read_text(
            encoding="utf-8"
        )
        for name, version in (("onnx_light", "0.2.0"), ("onnx_light_cpu", "0.2.0")):
            env_name = f"{name.upper()}_VERSION"
            self.assertEqual(config["matrix"]["env"][env_name], [version])
            wheel = (
                f"{name}-{version}-cp312-cp312-"
                "manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
            )
            url = (
                f"https://github.com/xadupre/{name.replace('_', '-')}"
                f"/releases/download/{version}/{wheel}"
            )
            self.assertIn(f"python -m pip install {url}", config["install_command"])
            self.assertIn(url, workflow)


if __name__ == "__main__":
    unittest.main()
