import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import asv

from tools.customize_pages import customize_pages


class TestCustomizePages(unittest.TestCase):
    @unittest.skipUnless(
        shutil.which("node"), "Node.js is required to execute summary JavaScript"
    )
    def test_summary_family_order_preserves_all_benchmarks(self):
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
            grid = (html / "summarygrid.js").read_text(encoding="utf-8")

        start = grid.index("    function get_benchmarks_by_groups()")
        end = grid.index("    function make_machine_summary()", start)
        names = [
            "cpu_backend_cases.math.cases.Add.time_add",
            "ops.math.add.Add.time_run",
            "models.llm.tiny_llm.TinyLLM.time_decode",
            "builder.save.onnx_io.OnnxSave.time_run",
            "models.dummies.mlp.MLP.time_run",
            "builder.load.onnx_io.OnnxLoad.time_run",
            "extra.benchmark.time_run",
        ]
        script = (
            "var $ = {asv: {main_json: {benchmarks: "
            + json.dumps({name: {} for name in names})
            + "}}, each: function(values, callback) {\n"
            "    Object.keys(values).forEach(function(key) { callback(key, values[key]); });\n"
            "}};\n"
            + grid[start:end]
            + "\nconsole.log(JSON.stringify(get_benchmarks_by_groups()));"
        )
        output = subprocess.run(
            ["node", "-e", script], check=True, capture_output=True, text=True
        )
        groups = json.loads(output.stdout)
        self.assertEqual(
            list(groups),
            [
                "builder / load",
                "builder / save",
                "models / dummies",
                "models / llm",
                "ops / math",
                "Performance / math",
                "extra / benchmark",
            ],
        )
        self.assertCountEqual(
            [name for benchmarks in groups.values() for name in benchmarks], names
        )

    def test_legend_above_plot(self):
        source = Path(asv.__file__).parent / "www"
        with tempfile.TemporaryDirectory() as directory:
            html = Path(directory)
            for name in ("index.html", "graphdisplay.js", "summarygrid.js", "summarylist.js"):
                shutil.copy2(source / name, html / name)
            customize_pages(html)
            page = (html / "index.html").read_text(encoding="utf-8")
            graph = (html / "graphdisplay.js").read_text(encoding="utf-8")
            stylesheet = (html / "system-theme.css").read_text(encoding="utf-8")

        self.assertLess(page.index('id="graph-body"'), page.index('id="graph-legend"'))
        self.assertLess(page.index('id="graph-legend"'), page.index('id="main-graph"'))
        self.assertIn("container: $('#graph-legend'),", graph)
        self.assertNotIn('position: "nw",', graph)
        self.assertIn("show: show_legend,", graph)
        self.assertIn('return $("<span>").text(label).html();', graph)
        self.assertIn(
            "        graphs = [];\n        orig_graphs = [];\n"
            "        $('#graph-legend').empty();",
            graph,
        )
        self.assertIn(
            "        var plot = $.plot(graph_div, graphs, options);\n"
            "        plot.resize();\n        plot.setupGrid();\n        plot.draw();",
            graph,
        )
        self.assertIn("#graph-body {\n    display: flex;\n    flex-direction: column;", stylesheet)
        self.assertIn(
            "#graph-legend {\n    flex: 0 0 auto;\n    max-height: 30%;\n    overflow: auto;",
            stylesheet,
        )
        self.assertIn("#main-graph {\n    flex: 1 1 auto;\n    height: auto !important;", stylesheet)

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
            self.assertIn("labels[param_names[axis-1]]", graph)
            self.assertIn("param_names[axis-1] == 'backend'", graph)
            self.assertNotIn("else if (params[axis-1].length > 1)", graph)
            self.assertIn("parameters: graph_content[2]", graph)
            self.assertIn("function show_missing_backends()", graph)
            self.assertIn("benchmark.param_names.indexOf('backend')", graph)
            self.assertIn("graph.data.some(function(point)", graph)
            self.assertIn("point[0] == backend", graph)
            self.assertIn("No measurements for ", graph)
            self.assertIn("if (--pending == 0 && orig_graphs.length > 0", graph)
            self.assertIn("$('#missing-backend-results').hide().empty()", graph)
            self.assertIn("function configuration_color(parameters)", graph)
            self.assertIn("Object.keys(parameters).sort()", graph)
            self.assertIn("key != 'commit'", graph)
            self.assertIn("Math.imul(hash, 16777619)", graph)
            self.assertIn("options.hooks = {draw: [draw_configuration_lines]}", graph)
            self.assertIn("overview_options.hooks = {draw: [draw_configuration_lines]}", graph)
            self.assertIn("graph.lines = {show: false}", graph)
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
    def test_missing_backend_warning_with_partial_graph_failures(self):
        source = Path(asv.__file__).parent / "www"
        with tempfile.TemporaryDirectory() as directory:
            html = Path(directory)
            for name in ("index.html", "graphdisplay.js", "summarygrid.js", "summarylist.js"):
                shutil.copy2(source / name, html / name)
            customize_pages(html)
            graph = (html / "graphdisplay.js").read_text(encoding="utf-8")

        start = graph.index("    function show_missing_backends()")
        end = graph.index("    function replace_graphs()", start)
        warning = graph[start:end]
        start = graph.index("}).always(function () {") + len("}).always(function () {")
        end = graph.index("\n            });", start)
        completion = graph[start:end]
        script = """
const assert = require('assert');
var message = '', visible = false;
function $(selector) {
    return {
        text: function(value) { message = value; return this; },
        show: function() { visible = true; }
    };
}
$.each = (values, callback) => values.forEach((value, index) => callback(index, value));
$.asv = {
    main_json: {benchmarks: {gemm: {
        param_names: ['dtype1', 'dtype2', 'dtype3', 'input_shapes', 'backend'],
        params: [["'bfloat16'"], ["'bfloat16'"], ["'bfloat16'"],
                 ["'(256, 256) x (256, 256) x ()'"],
                 ["'onnx-light-cpu'", "'onnxruntime'"]]
    }}},
    convert_benchmark_param_value: value => value.replace(/^'(.*)'$/, '$1')
};
var current_benchmark = 'gemm', requested_benchmark = 'gemm';
var x_coordinate_axis = 0;
var benchmark_param_selection = [[null], [0], [0], [0], [0], [0, 1]];
var orig_graphs = [
    {parameters: {backend: 'onnx-light-cpu'}, data: [[129, 0.0003641265]]},
    {parameters: {backend: 'onnxruntime'}, data: [[129, null]]}
];
var pending = 2, failures = 1;
"""
        script += warning + "\nfunction complete() {\n" + completion + "\n}\n"
        script += """
complete();
assert.strictEqual(visible, false);
complete();
assert.strictEqual(visible, true);
assert.strictEqual(message,
    'No measurements for onnxruntime (benchmark failed or is unsupported).');
assert.strictEqual(orig_graphs[1].data[0][1], null);

visible = false; message = ''; pending = 1;
benchmark_param_selection[5] = [0];
complete();
assert.strictEqual(visible, false);

benchmark_param_selection[5] = [0, 1];
orig_graphs[1].data[0][1] = 0.0005;
pending = 1;
complete();
assert.strictEqual(visible, false);

orig_graphs = []; pending = 1;
complete();
assert.strictEqual(visible, false);

orig_graphs = [{parameters: {backend: 'onnx-light-cpu'}, data: [[129, 0.0003]]}];
requested_benchmark = 'another_benchmark'; pending = 1;
complete();
assert.strictEqual(visible, false);
"""
        subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required to execute graph JavaScript")
    def test_default_graph_selection_prioritizes_backends(self):
        source = Path(asv.__file__).parent / "www"
        with tempfile.TemporaryDirectory() as directory:
            html = Path(directory)
            for name in ("index.html", "graphdisplay.js", "summarygrid.js", "summarylist.js"):
                shutil.copy2(source / name, html / name)
            customize_pages(html)
            graph = (html / "graphdisplay.js").read_text(encoding="utf-8")

        start = graph.index("        /* Default plot: up to 8 lines */")
        end = graph.index("        check_x_coordinate_axis();", start)
        selection = (
            "function select(params, param_names, state_selection) {\n"
            "var benchmark_param_selection;\n"
            + graph[start:end]
            + "\nreturn benchmark_param_selection;\n}\n"
        )
        backends = ["'onnx-light-cpu'", "'onnxruntime'"]
        shapes = [str((size,)) for size in range(16)]
        cases = [
            ([["'float32'"], shapes, backends], ["dtype1", "input_shapes", "backend"], {},
             [[None], [0], [0, 1, 2, 3], [0, 1]]),
            ([backends, shapes], ["backend", "shape"], {},
             [[None], [0, 1], [0, 1, 2, 3]]),
            ([["'float16'", "'float32'", "'float64'"], shapes,
              backends + ["'onnx-reference'", "'onnx-light'"]],
             ["dtype", "shape", "backend"], {},
             [[None], [0, 1], [0], [0, 1, 2, 3]]),
            ([shapes, backends], ["shape", "backend"], {"p-backend": [backends[1]]},
             [[None], list(range(8)), [1]]),
            ([shapes, backends], ["shape", "backend"], {"p-shape": shapes},
             [[None], list(range(16)), [0, 1]]),
            ([shapes, backends], ["shape", "backend"], {"p-backend": []},
             [[None], list(range(8)), []]),
            ([shapes, shapes, backends], ["shape1", "shape2", "backend"],
             {"p-shape1": shapes},
             [[None], list(range(16)), [0], [0, 1]]),
            ([shapes], ["shape"], {}, [[None], list(range(8))]),
            ([], [], {}, [[None]]),
        ]
        result = subprocess.run(
            [
                "node", "-e", selection
                + "console.log(JSON.stringify("
                + json.dumps([case[:3] for case in cases])
                + ".map(args => select(...args))));",
            ],
            check=True, capture_output=True, text=True,
        )
        for case, actual in zip(cases, json.loads(result.stdout), strict=True):
            with self.subTest(params=case[:3]):
                self.assertEqual(actual, case[3])

    @unittest.skipUnless(shutil.which("node"), "Node.js is required to execute graph JavaScript")
    def test_configuration_color_ignores_versions(self):
        source = Path(asv.__file__).parent / "www"
        with tempfile.TemporaryDirectory() as directory:
            html = Path(directory)
            for name in ("index.html", "graphdisplay.js", "summarygrid.js", "summarylist.js"):
                shutil.copy2(source / name, html / name)
            customize_pages(html)
            graph = (html / "graphdisplay.js").read_text(encoding="utf-8")

        start = graph.index("function configuration_key(parameters, versions)")
        end = graph.index("\nfunction configuration_segments", start)
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
        parameters = {
            "machine": "machine-a", "backend": "onnxruntime",
            "dtype": "float32", "cpu": "processor",
        }
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
                ("machine", "machine-b"),
                ("num_cpu", "8"),
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

    @unittest.skipUnless(shutil.which("node"), "Node.js is required to execute graph JavaScript")
    def test_configuration_connections(self):
        source = Path(asv.__file__).parent / "www"
        with tempfile.TemporaryDirectory() as directory:
            html = Path(directory)
            for name in ("index.html", "graphdisplay.js", "summarygrid.js", "summarylist.js"):
                shutil.copy2(source / name, html / name)
            customize_pages(html)
            graph = (html / "graphdisplay.js").read_text(encoding="utf-8")

        start = graph.index("function configuration_key(parameters, versions)")
        end = graph.index("\n}", graph.index("function draw_configuration_lines", start)) + 2
        result = subprocess.run(
            ["node", "-e", graph[start:end] + """
const assert = require('node:assert/strict');
const parameters = {
    machine: 'machine-a', cpu: 'processor', backend: 'onnxruntime', dtype: 'float32',
    python: '3.12', onnxruntime: '1.0', 'env-ONNX_LIGHT_VERSION': '1.0'
};
function series(data, changes = {}) {
    return {data, parameters: {...parameters, ...changes}, color: '#123456'};
}
const graphs = [
    series([[30, 3], [40, 4]], {onnxruntime: '2.0'}),
    series([[20, 2], [10, 1]], {commit: 'abc'}),
    series([[50, 5]], {onnxruntime: '2.0', 'env-ONNX_LIGHT_VERSION': '2.0'}),
    series([[60, 6]], {onnxruntime: '2.0', 'env-ONNX_LIGHT_VERSION': '2.0',
                          python: '3.13'}),
    series([[10, 10], [60, 60]], {machine: 'machine-b'}),
    series([[10, 11]], {backend: 'onnx-light'}),
    series([[60, 12]], {dtype: 'float64'})
];
const original = JSON.stringify(graphs);
const segments = configuration_segments(graphs);
const main = segments.filter(segment => segment.data[0][1] < 10);
assert.deepEqual(main.map(segment => segment.data), [
    [[10, 1], [20, 2]], [[20, 2], [30, 3]], [[30, 3], [40, 4]],
    [[40, 4], [50, 5]], [[50, 5], [60, 6]]
]);
assert.deepEqual(main.map(segment => segment.dotted), [false, true, false, true, true]);
assert.equal(segments.length, 6); // Different configurations never connect, even with equal colors.
assert.deepEqual(configuration_segments([...graphs].reverse()), segments);
assert.equal(JSON.stringify(graphs), original);
assert.deepEqual(configuration_segments([
    series([[10, 1], [20, null], [30, 3], [40, NaN], [50, 5], [60, Infinity]])
]), []);
assert.deepEqual(configuration_segments([series([[10, null], [20, null]])]), []);
assert.deepEqual(configuration_segments([series([[null, 1], [NaN, 2]])]), []);
assert.equal(configuration_segments([
    series([[10, 1]]), series([[10, 2]]), series([[20, 3]])
]).length, 1);
assert.deepEqual(configuration_segments([
    series([[10, 1], [20, 2]]),
    series([[10, 10], [20, 20]], {onnxruntime: '2.0'})
]).map(segment => [segment.data, segment.dotted]), [
    [[[10, 1], [20, 2]], false], [[[10, 10], [20, 20]], false]
]);
assert.deepEqual(configuration_segments([
    series([[10, 1], [30, 3]]),
    series([[20, 2], [40, 4]], {onnxruntime: '2.0'})
]).map(segment => [segment.data, segment.dotted]), [
    [[[10, 1], [30, 3]], false], [[[20, 2], [40, 4]], false]
]);
assert.equal(configuration_segments([
    series([[10, 1]], {onnxruntime: null}), series([[20, 2]])
])[0].dotted, true);
assert.notEqual(configuration_key({a: 'x|b=y'}), configuration_key({a: 'x', b: 'y'}));

const strokes = [];
const calls = [];
const context = {
    save() { calls.push('save'); }, restore() { calls.push('restore'); },
    translate(x, y) { calls.push(['translate', x, y]); },
    beginPath() {}, rect(...args) { calls.push(['rect', ...args]); },
    clip() { calls.push('clip'); },
    setLineDash(dash) { this.dash = dash; },
    moveTo(x, y) { this.from = [x, y]; },
    lineTo(x, y) { this.to = [x, y]; },
    stroke() { strokes.push({from: this.from, to: this.to, dash: this.dash,
                            color: this.strokeStyle}); }
};
draw_configuration_lines({
    getData: () => graphs,
    getAxes: () => ({xaxis: {p2c: x => x * 2}, yaxis: {p2c: y => y * 3}}),
    getPlotOffset: () => ({left: 7, top: 9}),
    width: () => 100, height: () => 80
}, context);
assert.deepEqual(calls, ['save', ['translate', 7, 9], ['rect', 0, 0, 100, 80],
                         'clip', 'restore']);
assert.deepEqual(strokes[0], {from: [20, 3], to: [40, 6], dash: [], color: '#123456'});
assert.deepEqual(strokes[1].dash, [2, 3]);
assert.equal(strokes.length, segments.length);
assert.equal(JSON.stringify(graphs), original);
const logGraphs = [series([[10, 1], [20, 0], [30, 3], [40, -1], [50, 5], [60, 6]])];
strokes.length = 0;
draw_configuration_lines({
    getData: () => logGraphs,
    getAxes: () => ({
        xaxis: {p2c: x => x},
        yaxis: {options: {transform: y => y > 0 ? Math.log(y) : null},
                p2c: y => Math.log(y)}
    }),
    getPlotOffset: () => ({left: 0, top: 0}),
    width: () => 100, height: () => 80
}, context);
assert.equal(strokes.length, 1);
assert.deepEqual(strokes[0].from, [50, Math.log(5)]);
console.log('Configuration connections and canvas rendering verified');
"""],
            check=True, capture_output=True, text=True,
        )
        self.assertIn("canvas rendering verified", result.stdout)

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
