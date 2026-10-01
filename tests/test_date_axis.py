import tempfile
import unittest
from pathlib import Path

from tools.date_axis import use_date_axis


class TestDateAxis(unittest.TestCase):
    def test_date_only(self):
        with tempfile.TemporaryDirectory() as directory:
            html = Path(directory)
            (html / "graphdisplay.js").write_text(
                "var even_spacing = false;\n"
                "var date_scale = false;\n"
                "function handle_x_scale(options) {\n"
                "    if (!date_scale) {\n"
                "        options.xaxis.axisLabel = 'commits';\n"
                "    } else if (date_scale) {\n"
                "        options.xaxis.axisLabel = 'commit date';\n"
                "    }\n"
                '    text = "commit";\n'
                "}\n",
                encoding="utf-8",
            )
            (html / "index.html").write_text(
                "<html><head></head><body>"
                '<a id="even-spacing">even spacing</a>'
                '<a id="date-scale">date scale</a>'
                "</body></html>",
                encoding="utf-8",
            )

            use_date_axis(html)

            script = (html / "graphdisplay.js").read_text(encoding="utf-8")
            page = (html / "index.html").read_text(encoding="utf-8")
            self.assertIn("var date_scale = true;", script)
            self.assertIn(
                "function handle_x_scale(options) {\n"
                "        date_scale = true;\n"
                "        even_spacing = false;\n",
                script,
            )
            self.assertIn("axisLabel = 'date';", script)
            self.assertIn('text = "date";', script)
            self.assertIn(
                "#even-spacing, #date-scale { display: none !important; }", page
            )

    def test_unexpected_asv_output_is_not_modified(self):
        with tempfile.TemporaryDirectory() as directory:
            html = Path(directory)
            script = html / "graphdisplay.js"
            script.write_text("var date_scale = false;", encoding="utf-8")
            (html / "index.html").write_text("<head></head>", encoding="utf-8")

            with self.assertRaises(ValueError):
                use_date_axis(html)
            self.assertEqual(script.read_text(encoding="utf-8"), "var date_scale = false;")


if __name__ == "__main__":
    unittest.main()
