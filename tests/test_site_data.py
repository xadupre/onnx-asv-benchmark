import json
import tempfile
import unittest
from pathlib import Path

from tools.site_data import configure_remote_data, extract_site_data

ASV_SCRIPT = """'use strict';
function load_graph_data(url) {
    $.ajax({
        url: url + '?timestamp=' + $.asv.main_timestamp,
        dataType: "json"
    });
}
function init_index() {
    $.ajax({
        url: "index.json" + '?timestamp=' + $.asv.main_timestamp,
        dataType: "json"
    });
}
function init() {
    $.ajax({
        url: "info.json",
        dataType: "json"
    });
}
"""


class TestSiteData(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.html = self.root / "html"
        self.html.mkdir()
        (self.html / "asv.js").write_text(ASV_SCRIPT, encoding="utf-8")
        (self.html / "regressions.js").write_text(
            "$.ajax({url: 'regressions.json' + '?timestamp=' + timestamp});",
            encoding="utf-8",
        )
        (self.html / "index.html").write_text("<html></html>", encoding="utf-8")
        (self.html / "index.json").write_text(
            json.dumps({"project": "benchmark"}),
            encoding="utf-8",
        )
        graph = self.html / "graphs" / "math.json"
        graph.parent.mkdir()
        graph.write_text(json.dumps([[1, 2]]), encoding="utf-8")

    def test_extract_and_configure(self):
        destination = self.root / "site-data"
        stale = destination / "stale.json"
        stale.parent.mkdir()
        stale.write_text("{}", encoding="utf-8")

        self.assertEqual(extract_site_data(self.html, destination), 2)
        self.assertFalse(stale.exists())
        self.assertTrue((destination / "index.json").is_file())
        self.assertTrue((destination / "graphs" / "math.json").is_file())

        configure_remote_data(
            self.html,
            "https://raw.githubusercontent.com/example/cache/main/site-data",
        )

        script = (self.html / "asv.js").read_text(encoding="utf-8")
        self.assertIn(
            '"https://raw.githubusercontent.com/example/cache/main/site-data/"',
            script,
        )
        self.assertIn("url: asv_data_path(url) + '?timestamp='", script)
        self.assertIn('url: asv_data_path("index.json")', script)
        self.assertIn('url: asv_data_path("info.json")', script)
        self.assertIn(
            "url: asv_data_path('regressions.json')",
            (self.html / "regressions.js").read_text(encoding="utf-8"),
        )
        self.assertFalse(list(self.html.rglob("*.json")))
        self.assertTrue((self.html / "index.html").is_file())

    def test_requires_asv_json(self):
        for path in self.html.rglob("*.json"):
            path.unlink()
        with self.assertRaises(FileNotFoundError):
            extract_site_data(self.html, self.root / "site-data")

    def test_requires_https(self):
        with self.assertRaises(ValueError):
            configure_remote_data(self.html, "http://example.com/site-data")


if __name__ == "__main__":
    unittest.main()
