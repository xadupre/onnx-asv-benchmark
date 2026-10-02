import shutil
import tempfile
import unittest
from pathlib import Path

import asv

from tools.customize_pages import customize_pages


class TestCustomizePages(unittest.TestCase):
    def test_asv_assets(self):
        source = Path(asv.__file__).parent / "www"
        with tempfile.TemporaryDirectory() as directory:
            html = Path(directory)
            for name in (
                "index.html",
                "graphdisplay.js",
                "summarygrid.js",
                "summarylist.js",
            ):
                shutil.copy2(source / name, html / name)

            customize_pages(html)

            page = (html / "index.html").read_text(encoding="utf-8")
            graph = (html / "graphdisplay.js").read_text(encoding="utf-8")
            grid = (html / "summarygrid.js").read_text(encoding="utf-8")
            summary = (html / "summarylist.js").read_text(encoding="utf-8")
            stylesheet = (html / "system-theme.css").read_text(encoding="utf-8")
            self.assertIn('name="color-scheme" content="light dark"', page)
            self.assertIn('href="system-theme.css"', page)
            self.assertIn("#even-spacing, #date-scale", page)
            self.assertIn('class="runtime-grid"', page)
            self.assertEqual(page.count('class="runtime-card"'), 5)
            self.assertIn('id="benchmark-navigation"', page)
            self.assertIn("var date_scale = true;", graph)
            self.assertIn("axisLabel = 'date';", graph)
            self.assertIn('text = "date";', graph)
            self.assertIn("prefers-color-scheme: dark", graph)
            self.assertEqual(graph.count("axisLabelColour"), 2)
            self.assertIn("var(--asv-background)", graph)
            self.assertIn(
                "param != 'machine' && param != 'cpu'",
                graph,
            )
            self.assertIn(
                "param != 'machine' && param != 'cpu'",
                summary,
            )
            self.assertIn("param_names[axis-1] == 'backend'", graph)
            self.assertIn("parameters: graph_content[2]", graph)
            self.assertIn("new Date(item.datapoint[0]).toLocaleString()", graph)
            self.assertIn("item.series.parameters", graph)
            self.assertIn("parts.slice(0, 2).join(' / ')", grid)
            self.assertIn("bm.name.split('.').slice(2).join('.')", grid)
            self.assertIn("make_summary_navigation(groups)", grid)
            self.assertIn("data-family", grid)
            self.assertIn("data-group", grid)
            self.assertNotIn(
                '$.asv.pretty_unit(y, unit) + " @ " + commit_hash',
                graph,
            )
            self.assertIn("@media (prefers-color-scheme: dark)", stylesheet)
            self.assertIn(".runtime-grid", stylesheet)
            self.assertIn("#benchmark-navigation", stylesheet)

    def test_unexpected_asv_output_is_not_modified(self):
        with tempfile.TemporaryDirectory() as directory:
            html = Path(directory)
            index = html / "index.html"
            graph = html / "graphdisplay.js"
            grid = html / "summarygrid.js"
            summary = html / "summarylist.js"
            index.write_text("<head></head>", encoding="utf-8")
            graph.write_text("unexpected", encoding="utf-8")
            grid.write_text("unexpected", encoding="utf-8")
            summary.write_text("unexpected", encoding="utf-8")

            with self.assertRaises(ValueError):
                customize_pages(html)
            self.assertEqual(index.read_text(encoding="utf-8"), "<head></head>")
            self.assertEqual(graph.read_text(encoding="utf-8"), "unexpected")
            self.assertEqual(grid.read_text(encoding="utf-8"), "unexpected")
            self.assertEqual(summary.read_text(encoding="utf-8"), "unexpected")


if __name__ == "__main__":
    unittest.main()
