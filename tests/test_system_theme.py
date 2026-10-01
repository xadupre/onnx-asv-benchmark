import tempfile
import unittest
from pathlib import Path

from tools.system_theme import use_system_theme


class TestSystemTheme(unittest.TestCase):
    def test_system_theme(self):
        with tempfile.TemporaryDirectory() as directory:
            html = Path(directory)
            (html / "index.html").write_text(
                "<html><head></head><body></body></html>",
                encoding="utf-8",
            )
            (html / "graphdisplay.js").write_text(
                "'use strict';\n"
                "var options = {\n"
                "    xaxis: {\n"
                "        axisLabelFontSizePixels: 12\n"
                "    },\n"
                "    yaxis: {\n"
                "        axisLabelFontSizePixels: 12\n"
                "    }\n"
                "};\n"
                'tag = "color:#666;background:white;padding-left:0.25em;'
                'font-size:smaller;\';\n',
                encoding="utf-8",
            )

            use_system_theme(html)

            page = (html / "index.html").read_text(encoding="utf-8")
            script = (html / "graphdisplay.js").read_text(encoding="utf-8")
            stylesheet = (html / "system-theme.css").read_text(encoding="utf-8")
            self.assertIn('name="color-scheme" content="light dark"', page)
            self.assertIn('href="system-theme.css"', page)
            self.assertIn("prefers-color-scheme: dark", script)
            self.assertEqual(script.count("axisLabelColour"), 2)
            self.assertIn("var(--asv-background)", script)
            self.assertIn("@media (prefers-color-scheme: dark)", stylesheet)

    def test_unexpected_asv_output_is_not_modified(self):
        with tempfile.TemporaryDirectory() as directory:
            html = Path(directory)
            index = html / "index.html"
            script = html / "graphdisplay.js"
            index.write_text("<head></head>", encoding="utf-8")
            script.write_text("'use strict';", encoding="utf-8")

            with self.assertRaises(ValueError):
                use_system_theme(html)
            self.assertEqual(index.read_text(encoding="utf-8"), "<head></head>")
            self.assertEqual(script.read_text(encoding="utf-8"), "'use strict';")


if __name__ == "__main__":
    unittest.main()
