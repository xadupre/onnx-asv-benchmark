import tempfile
import unittest
from pathlib import Path

from tools.show_backend_labels import show_backend_labels


class TestShowBackendLabels(unittest.TestCase):
    def test_single_backend_is_labeled(self):
        with tempfile.TemporaryDirectory() as directory:
            html = Path(directory)
            script = html / "graphdisplay.js"
            script.write_text(
                "else if (params[axis-1].length > 1) {\n"
                "    labels[param_names[axis-1]] = value;\n"
                "}\n",
                encoding="utf-8",
            )

            show_backend_labels(html)

            self.assertIn(
                "params[axis-1].length > 1 || "
                "param_names[axis-1] == 'backend'",
                script.read_text(encoding="utf-8"),
            )

    def test_unexpected_asv_output_is_not_modified(self):
        with tempfile.TemporaryDirectory() as directory:
            html = Path(directory)
            script = html / "graphdisplay.js"
            script.write_text("unexpected", encoding="utf-8")

            with self.assertRaises(ValueError):
                show_backend_labels(html)
            self.assertEqual(script.read_text(encoding="utf-8"), "unexpected")


if __name__ == "__main__":
    unittest.main()
