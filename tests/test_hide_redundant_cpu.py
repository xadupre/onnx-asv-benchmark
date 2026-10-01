import tempfile
import unittest
from pathlib import Path

from tools.hide_redundant_cpu import hide_redundant_cpu


class TestHideRedundantCpu(unittest.TestCase):
    def test_cpu_selector_is_hidden(self):
        with tempfile.TemporaryDirectory() as directory:
            html = Path(directory)
            source = (
                "$.each(index.params, function(param, values) {\n"
                "    if (values.length > 1 && param != 'machine') {\n"
                "    }\n"
                "});\n"
            )
            for name in ("graphdisplay.js", "summarylist.js"):
                (html / name).write_text(source, encoding="utf-8")

            hide_redundant_cpu(html)

            for name in ("graphdisplay.js", "summarylist.js"):
                script = (html / name).read_text(encoding="utf-8")
                self.assertIn(
                    "param != 'machine' && param != 'cpu'",
                    script,
                )

    def test_unexpected_asv_output_is_not_modified(self):
        with tempfile.TemporaryDirectory() as directory:
            html = Path(directory)
            graph = html / "graphdisplay.js"
            summary = html / "summarylist.js"
            graph.write_text("unexpected", encoding="utf-8")
            summary.write_text("unexpected", encoding="utf-8")

            with self.assertRaises(ValueError):
                hide_redundant_cpu(html)
            self.assertEqual(graph.read_text(encoding="utf-8"), "unexpected")
            self.assertEqual(summary.read_text(encoding="utf-8"), "unexpected")


if __name__ == "__main__":
    unittest.main()
