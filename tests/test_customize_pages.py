import json
import shutil
import subprocess
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
            self.assertIn("Qwen2 BF16 ONNX Runtime CPU is intentionally unavailable", page)
            self.assertEqual(page.count('class="runtime-card"'), 5)
            self.assertIn('class="workflow-status"', page)
            self.assertIn("genai-compatibility.yml/badge.svg?branch=main", page)
            self.assertIn('id="machine-summary"', page)
            self.assertIn('id="missing-backend-results" role="status"', page)
            self.assertIn("Logical cores", page)
            self.assertIn('id="benchmark-navigation"', page)
            self.assertIn('data-family="builder">Builder</button>', page)
            self.assertIn("Performance", page)
            self.assertIn("var date_scale = true;", graph)
            self.assertIn("axisLabel = 'date';", graph)
            self.assertIn('text = "date";', graph)
            self.assertIn("key == 'instruction_sets' ||", graph)
            self.assertNotIn("key == 'num_cpu' ||", graph)
            self.assertIn("value.replace(/^'(.*)'$/, '$1')", graph)
            self.assertNotIn("normalized_value == 'anonymous'", graph)
            self.assertIn("normalized_value == 'onnxruntime'", graph)
            self.assertIn("'ort-' + state.onnxruntime", graph)
            self.assertIn("key == 'onnxruntime'", graph)
            self.assertIn("backend: 'bck'", graph)
            self.assertIn("num_cpu: 'vCPU'", graph)
            self.assertIn(
                "key == 'env-ONNX_LIGHT_CPU_VERSION'",
                graph,
            )
            self.assertIn("key == 'env-ONNX_LIGHT_VERSION'", graph)
            self.assertNotIn("'env-olcpu'", graph)
            self.assertNotIn("'env-ol'", graph)
            self.assertNotIn('key + "-" + value', graph)
            self.assertIn("prefers-color-scheme: dark", graph)
            self.assertEqual(graph.count("axisLabelColour"), 2)
            self.assertIn("var(--asv-background)", graph)
            self.assertIn(
                "param != 'cpu' && param != 'num_cpu'",
                graph,
            )
            self.assertIn(
                "param != 'cpu' && param != 'num_cpu'",
                summary,
            )
            self.assertIn("param != 'instruction_sets'", graph)
            self.assertIn("param != 'instruction_sets'", summary)
            for script in (graph, summary):
                self.assertNotIn("Not recorded", script)
                self.assertIn(
                    "/* Generic parameter selectors */\n"
                    "        $.each(index.params, function(param, values) {\n"
                    "            values = values.filter(function(value) { return value !== null; });\n"
                    "            if (values.length > 1 && param != 'machine'",
                    script,
                )
                self.assertIn("value_display = '[default]';", script)
                self.assertIn("value_display = value;", script)
                self.assertIn("button.text(value_display);", script)
            self.assertIn("param_names[axis-1] == 'backend'", graph)
            self.assertIn("parameters: graph_content[2]", graph)
            self.assertIn("function show_missing_backends()", graph)
            self.assertIn("benchmark.param_names.indexOf('backend')", graph)
            self.assertIn("graph.data.some(function(point)", graph)
            self.assertIn("point[0] == backend", graph)
            self.assertIn("No measurements for ", graph)
            self.assertIn("if (--pending == 0 && failures == 0", graph)
            self.assertIn("$('#missing-backend-results').hide().empty()", graph)
            self.assertIn("function configuration_color(parameters)", graph)
            self.assertIn("Object.keys(parameters).sort()", graph)
            self.assertIn("key != 'commit'", graph)
            self.assertIn("Math.imul(hash, 16777619)", graph)
            self.assertIn(
                "color: configuration_color(graph_content[2])",
                graph,
            )
            self.assertIn("new Date(item.datapoint[0]).toLocaleString()", graph)
            self.assertIn("item.series.parameters", graph)
            self.assertIn("cpu: 'processor'", graph)
            self.assertIn("'onnx-ir': 'ir-py'", graph)
            self.assertIn("num_cpu: 'logical cores'", graph)
            self.assertIn("instruction_sets: 'instruction sets'", graph)
            self.assertIn("key != 'machine'", graph)
            self.assertIn("parts.slice(0, 2).join(' / ')", grid)
            self.assertIn("'Performance / ' + parts[1]", grid)
            self.assertIn("parts[0] == 'ops'", grid)
            self.assertIn("parts[0] == 'cpu_backend_cases'", grid)
            self.assertIn("replace(/^time_/, '')", grid)
            self.assertIn("replace(/^time_/, '')", graph)
            self.assertIn("parts[parts.length - 2]", grid)
            self.assertIn("parts.slice(2).join('.')", grid)
            self.assertIn("make_summary_navigation(groups)", grid)
            self.assertIn(
                "['builder', 'models', 'ops', 'cpu_backend_cases']",
                grid,
            )
            self.assertIn("make_machine_summary()", grid)
            self.assertIn("title: 'View results for ' + bm.name", grid)
            self.assertNotIn("title: bm.code", grid)
            self.assertIn("html: false,", grid)
            self.assertIn("params.instruction_sets", grid)
            self.assertIn("variant.split(',').join(', ')", grid)
            self.assertIn("params.num_cpu", grid)
            self.assertIn("Not recorded", grid)
            self.assertIn("data-family", grid)
            self.assertIn("data-group", grid)
            self.assertIn('for="benchmark-search"', page)
            self.assertIn('id="benchmark-search" class="form-control" type="search"', page)
            self.assertIn('id="benchmark-search-status" role="status"', page)
            self.assertIn("new RegExp(search.val(), 'i')", grid)
            self.assertIn("$('#benchmark-search').on('input', apply_filters)", grid)
            self.assertIn("container.attr('data-benchmark-name', bm.name)", grid)
            self.assertIn("pattern.test(card.attr('data-benchmark-name'))", grid)
            self.assertIn("pattern.test(card.find('.benchmark-text').text())", grid)
            self.assertIn("Invalid regular expression: ", grid)
            self.assertIn("search.attr('aria-invalid', 'true')", grid)
            self.assertIn("search.attr('aria-invalid', 'false')", grid)
            self.assertIn("No benchmarks match these filters.", grid)
            self.assertIn("section.toggle(matches > 0)", grid)
            self.assertEqual(grid.count("            apply_filters();"), 2)
            self.assertNotIn(
                '$.asv.pretty_unit(y, unit) + " @ " + commit_hash',
                graph,
            )
            self.assertIn("@media (prefers-color-scheme: dark)", stylesheet)
            self.assertIn(".runtime-grid", stylesheet)
            self.assertIn(".machine-table", stylesheet)
            self.assertIn("#benchmark-navigation", stylesheet)
            self.assertIn("background-color: var(--asv-hover)", stylesheet)
            self.assertIn("background-color: var(--asv-selection)", stylesheet)
            self.assertIn(".btn-default.active:hover", stylesheet)
            self.assertIn(".benchmark-container {\n    width: 220px;\n    height: 80px;", stylesheet)
            self.assertIn(".benchmark-plot {\n    width: 212px;\n    height: 52px;", stylesheet)
            self.assertIn(".benchmark-text {\n    width: 212px;", stylesheet)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required to execute graph JavaScript")
    def test_configuration_color_ignores_versions(self):
        source = Path(asv.__file__).parent / "www"
        with tempfile.TemporaryDirectory() as directory:
            html = Path(directory)
            for name in ("index.html", "graphdisplay.js", "summarygrid.js", "summarylist.js"):
                shutil.copy2(source / name, html / name)
            customize_pages(html)
            graph = (html / "graphdisplay.js").read_text(encoding="utf-8")

        start = graph.index("function configuration_color(parameters)")
        end = graph.index("\n}", start) + 2
        function = graph[start:end]
        config = json.loads(
            (Path(__file__).parents[1] / "asv.conf.json").read_text(encoding="utf-8")
        )
        versions = {
            **{key: "1.0" for key in config["matrix"]["req"]},
            **{f"env-{key}": "1.0" for key in config["matrix"]["env"]},
            "python": "3.12",
            "commit": "abc123",
        }
        parameters = {"backend": "onnxruntime", "dtype": "float32", "cpu": "processor"}
        configurations = [
            parameters,
            {**parameters, **versions},
            {**parameters, **dict.fromkeys(versions, "2.0")},
            *[{**parameters, key: "2.0"} for key in versions],
            {**parameters, **dict(reversed(list(versions.items())))},
            {**parameters, "onnx": None},
        ]
        alternatives = [
            {**parameters, key: value}
            for key, value in (
                ("backend", "onnx-light"),
                ("backend", "onnx-light-cpu"),
                ("dtype", "float64"),
                ("cpu", "another processor"),
            )
        ]
        result = subprocess.run(
            [
                "node", "-e",
                function + "\nconsole.log(JSON.stringify("
                + json.dumps(configurations + alternatives) + ".map(configuration_color)));",
            ],
            check=True, capture_output=True, text=True,
        )
        colors = json.loads(result.stdout)
        self.assertRegex(colors[0], r"^hsl\(\d+, 65%, 48%\)$")
        for configuration, color in zip(configurations, colors):
            with self.subTest(configuration=configuration):
                self.assertEqual(color, colors[0])
        for configuration, color in zip(alternatives, colors[len(configurations):]):
            with self.subTest(configuration=configuration):
                self.assertNotEqual(color, colors[0])

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
