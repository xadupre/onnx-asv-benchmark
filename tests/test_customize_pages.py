import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import asv

from tools.customize_pages import customize_pages


class TestCustomizePages(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js is required to execute graph JavaScript")
    def test_speedups_require_matching_configuration_and_timestamp(self):
        source = Path(asv.__file__).parent / "www"
        with tempfile.TemporaryDirectory() as directory:
            html = Path(directory)
            for name in ("index.html", "graphdisplay.js", "summarygrid.js", "summarylist.js"):
                shutil.copy2(source / name, html / name)
            customize_pages(html)
            graph = (html / "graphdisplay.js").read_text(encoding="utf-8")
        start = graph.index("function configuration_key(")
        end = graph.index("function configuration_segments(", start)
        subprocess.run(
            ["node", "-e", "var $ = {extend: Object.assign};\n" + graph[start:end] + """
const assert = require('node:assert/strict');
function curve(backend, data, changes = {}) {
    return {
        parameters: {machine: 'a', cpu: 'processor', python: '3.12',
                     onnxruntime: '1.0', backend, dtype: 'float32', size: 32, ...changes},
        data, color: '#123456', lines: {lineWidth: backend == 'onnxruntime' ? 3 : 1}
    };
}
const baseline = curve('onnxruntime', [
    [10, 8], [20, 12], [30, null], [40, 0], [50, Infinity], [60, -3],
    [70, 1e308], [80, 1e-308], [90, 8], [100, 8], [110, NaN], [120, 8]
]);
const light = curve('onnx-light', [
    [10, 2], [20, 24], [25, 4], [30, 4], [40, 4], [50, 4], [60, 4],
    [70, 1e-308], [80, 1e308], [90, 0], [100, -2], [110, 4], [120, NaN]
]);
const graphs = [light, baseline];
const original = JSON.stringify(graphs);
const speedups = configuration_speedups(graphs);
assert.deepEqual(speedups[0].data.slice(0, 3), [[10, 4], [20, 0.5], [25, null]]);
assert.ok(speedups[0].data.slice(3).every(point => point[1] === null));
assert.equal(speedups.length, 1);
assert.deepEqual(configuration_speedups([baseline]), []);
assert.equal(JSON.stringify(graphs), original);
assert.equal(speedups[0].parameters, light.parameters);
assert.equal(speedups[0].color, light.color);
assert.equal(speedups[0].lines.lineWidth, 1);
for (const changes of [
    {machine: 'b'}, {cpu: 'other'}, {num_cpu: 8}, {python: '3.13'},
    {onnxruntime: '2.0'}, {onnxruntime: null}, {dtype: 'float64'}, {size: 64},
    {'env-ONNX_LIGHT_VERSION': '2.0'}
]) {
    assert.deepEqual(configuration_speedups([
        baseline, curve('onnx-light', [[10, 2]], changes)
    ])[0].data, [[10, null]], JSON.stringify(changes));
}
const otherMachine = curve('onnxruntime', [[10, 30]], {machine: 'b'});
assert.deepEqual(configuration_speedups([
    baseline, otherMachine, curve('onnx-light', [[10, 3]], {machine: 'b'})
])[0].data, [[10, 10]]);
assert.deepEqual(configuration_speedups([light])[0].data.map(point => point[1]),
                 light.data.map(() => null));
assert.deepEqual(configuration_speedups([
    curve('onnxruntime', [[10, 8]]), curve('onnxruntime', [[10, 9]]),
    curve('onnx-light', [[10, 2]])
])[0].data, [[10, null]]);
assert.deepEqual(configuration_speedups([
    curve('onnxruntime', [[10, 8]], {commit: 'abc'}),
    curve('onnx-light', [[10, 2]], {commit: 'last'})
])[0].data, [[10, 4]]);
assert.deepEqual(configuration_speedups([
    curve(undefined, [[10, 2]])
])[0].data, [[10, null]]);
"""],
            check=True, capture_output=True, text=True,
        )

    @unittest.skipUnless(shutil.which("node"), "Node.js is required to execute graph JavaScript")
    def test_speedup_plot_filters_and_stale_requests(self):
        source = Path(asv.__file__).parent / "www"
        with tempfile.TemporaryDirectory() as directory:
            html = Path(directory)
            for name in ("index.html", "graphdisplay.js", "summarygrid.js", "summarylist.js"):
                shutil.copy2(source / name, html / name)
            customize_pages(html)
            page = (html / "index.html").read_text(encoding="utf-8")
            graph = (html / "graphdisplay.js").read_text(encoding="utf-8")
            for name in ("graphdisplay.js", "summarygrid.js", "summarylist.js"):
                subprocess.run(
                    ["node", "--check"], input=(html / name).read_text(encoding="utf-8"),
                    check=True, capture_output=True, text=True,
                )
        self.assertIn('id="main-graph"', page)
        self.assertIn('id="overview"', page)
        self.assertIn('id="speedup-graph" aria-labelledby="speedup-title"', page)
        self.assertIn("        load_speedup_graph();", graph)
        start = graph.index("function configuration_key(")
        end = graph.index("function configuration_segments(", start)
        load_start = graph.index("    var speedup_plot = null;")
        load_end = graph.index("    function show_missing_backends()", load_start)
        subprocess.run(
            ["node", "-e", """
const assert = require('node:assert/strict');
const requests = [], plots = [];
var message = '', title = '', hover, speedupVisible;
const element = {
    empty() { message = ''; return this; },
    text(value) { message = value; return this; },
    tooltip(value) { this.tooltipState = value; return this; },
    off() { return this; },
    on(event, handler) { hover = handler; return this; },
    attr(name, value) { title = value; return this; }
};
function $(selector) {
    if (selector == '#speedup-title, #speedup-graph') {
        return {toggle(value) { speedupVisible = value; }};
    }
    assert.equal(selector, '#speedup-graph');
    return element;
}
$.extend = Object.assign;
const lifecycle = [];
$.plot = (element, series, options) => {
    plots.push({series, options});
    return {
        resize() { lifecycle.push('resize'); },
        setupGrid() { lifecycle.push('grid'); },
        draw() { lifecycle.push('draw'); },
        shutdown() { lifecycle.push('shutdown'); }
    };
};
const benchmark = {
    param_names: ['backend', 'size'],
    params: [["'onnxruntime'", "'onnx-light'"], ['32', '64']]
};
$.asv = {
    main_json: {benchmarks: {test: benchmark}},
    convert_benchmark_param_value: value => value[0] == "'" ? value.slice(1, -1) : Number(value),
    load_graph_data(url) {
        assert.equal(url, 'graphs/configurations/test.json');
        const request = {
            done(callback) { this.success = callback; return this; },
            fail(callback) { this.failure = callback; return this; }
        };
        requests.push(request);
        return request;
    }
};
var current_benchmark = 'test', state = {machine: ['a']}, x_coordinate_axis = 0;
var benchmark_param_selection = [[null], [1], [0]], system_dark_theme = false;
function draw_configuration_lines() {}
const data = {
    graph_parameters: [
        {machine: 'a', 'env-ONNX_LIGHT_VERSION': '0.2.7',
         'env-ONNX_LIGHT_CPU_VERSION': '0.2.8'},
        {machine: 'b'}
    ],
    series: [
        {graph: 0, parameters: {backend: "'onnxruntime'", size: '32'}, data: [[10, 8]]},
        {graph: 0, parameters: {backend: "'onnx-light'", size: '32'}, data: [[10, 2]]},
        {graph: 0, parameters: {backend: "'onnxruntime'", size: '64'}, data: [[10, 12]]},
        {graph: 0, parameters: {backend: "'onnx-light'", size: '64'}, data: [[10, 6]]},
        {graph: 1, parameters: {backend: "'onnxruntime'", size: '32'}, data: [[10, 30]]},
        {graph: 1, parameters: {backend: "'onnx-light'", size: '32'}, data: [[10, 3]]}
    ]
};
""" + graph[start:end] + graph[load_start:load_end] + """
load_speedup_graph();
requests[0].success(data);
assert.equal(plots.length, 1);
assert.equal(plots[0].series.length, 1); // Baseline works even when deselected.
assert.deepEqual(plots[0].series[0].data, [[10, 4]]);
assert.equal(plots[0].options.xaxis.mode, 'time');
assert.equal(plots[0].options.hooks.draw[0], draw_configuration_lines);
assert.equal(speedupVisible, true);
assert.deepEqual(plots[0].options.grid.markings[0].yaxis, {from: 1, to: 1});
assert.ok(plots[0].options.yaxis.max > 1);
resize_speedup_graph();
assert.deepEqual(lifecycle, ['resize', 'grid', 'draw']);
hover(null, null, {series: plots[0].series[0], datapoint: [10, 4]});
assert.ok(title.includes('4.000× ONNX Runtime'));
assert.ok(title.includes('<strong class="benchmark-version">onnx-light: 0.2.7</strong>'));
assert.ok(title.includes('<strong class="benchmark-version">onnx-light-cpu: 0.2.8</strong>'));
assert.ok(!title.includes('ONNX_LIGHT'));
assert.equal(element.tooltipState, 'show');
hover(null, null, null);
assert.equal(title, '');
assert.equal(element.tooltipState, 'hide');

state = {machine: ['b']};
load_speedup_graph();
assert.equal(lifecycle[lifecycle.length - 1], 'shutdown');
requests[1].success(data);
assert.deepEqual(plots[1].series[0].data, [[10, 10]]);

state = {machine: ['a']};
x_coordinate_axis = 1; // Backend on the timing plot x-axis does not discard it.
load_speedup_graph();
requests[2].success(data);
assert.deepEqual(plots[2].series.map(curve => curve.data), [[[10, 4]]]);
x_coordinate_axis = 2; // All sizes appear when size is the timing plot x-axis.
load_speedup_graph();
requests[3].success(data);
assert.deepEqual(plots[3].series.map(curve => curve.data), [[[10, 4]], [[10, 2]]]);

x_coordinate_axis = 0;
benchmark_param_selection[2] = [];
load_speedup_graph();
requests[4].success(data);
assert.ok(message.startsWith('Speedup unavailable:'));
assert.equal(plots.length, 4);
benchmark_param_selection[2] = [0];
load_speedup_graph();
load_speedup_graph();
requests[5].success(data);
requests[5].failure();
assert.equal(message, ''); // Stale responses never overwrite a new selection.
assert.equal(plots.length, 4);
requests[6].failure();
assert.ok(message.includes('could not be loaded'));
load_speedup_graph();
requests[7].success({...data, series: data.series.filter(
    curve => curve.parameters.backend != "'onnxruntime'")});
assert.ok(message.includes('no matching ONNX Runtime'));
assert.equal(plots.length, 4);
benchmark_param_selection[1] = [0]; // Selecting only the baseline never plots it.
load_speedup_graph();
requests[8].success(data);
assert.ok(message.includes('no matching ONNX Runtime'));
assert.equal(plots.length, 4);

load_speedup_graph();
current_benchmark = 'builder.load.onnx_io.OnnxLoad.time_run';
load_speedup_graph();
assert.equal(speedupVisible, false);
assert.equal(requests.length, 10); // Builder graphs do not fetch speedup data.
assert.equal(message, '');
requests[9].success(data);
requests[9].failure();
assert.equal(message, '');
assert.equal(plots.length, 4);
current_benchmark = 'test';
benchmark_param_selection[1] = [1];
load_speedup_graph();
requests[10].success(data);
assert.equal(speedupVisible, true);
assert.equal(plots.length, 5);
"""],
            check=True, capture_output=True, text=True,
        )

    @unittest.skipUnless(
        shutil.which("node"), "Node.js is required to execute summary JavaScript"
    )
    def test_preview_series_preserve_configuration_and_missing_versions(self):
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
            graph = (html / "graphdisplay.js").read_text(encoding="utf-8")
        width_start = graph.index("function configuration_line_width(")
        width_end = graph.index("\nfunction configuration_segments", width_start)
        start = grid.index("                var series = is_builder")
        end = grid.index("                var options = ", start)
        script = (source / "asv.js").read_text(encoding="utf-8")
        decode_start = script.index("    function convert_benchmark_param_value(")
        decode_end = script.index("    /* Convert loaded graph data", decode_start)
        data = {
            "graph_parameters": [
                {"machine": "slow", "onnx": None, "python": "3.12"},
                {"machine": "fast", "onnx": "1.23", "python": "3.12"},
            ],
            "series": [
                {
                    "graph": 0,
                    "parameters": {"dtype": "'float32'", "size": "32", "backend": "'onnxruntime'"},
                    "data": [[1000, 10], [2000, None]],
                },
                {
                    "graph": 1,
                    "parameters": {"dtype": "'float32'", "size": "32", "backend": "'onnx-light'"},
                    "data": [[1000, 1], [2000, 2]],
                },
            ],
        }
        output = subprocess.run(
            [
                "node",
                "-e",
                script[decode_start:decode_end]
                + "\nvar $ = {extend: Object.assign, asv: {convert_benchmark_param_value: convert_benchmark_param_value}};\n"
                + "function configuration_color(parameters) {return JSON.stringify(parameters);}\n"
                + graph[width_start:width_end]
                + "\nvar bm = {param_names: ['dtype', 'size', 'backend']};\nvar data = "
                + json.dumps(data)
                + ";\n"
                + "var series = configuration_series(data, bm);\n"
                + "\nconsole.log(JSON.stringify(series));",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        series = json.loads(output.stdout)
        self.assertEqual(len(series), 2)
        self.assertEqual(
            series[0]["parameters"],
            {
                "machine": "slow",
                "onnx": None,
                "python": "3.12",
                "dtype": "float32",
                "size": 32,
                "backend": "onnxruntime",
            },
        )
        self.assertEqual(series[1]["parameters"]["onnx"], "1.23")
        self.assertEqual(series[0]["data"], data["series"][0]["data"])
        self.assertEqual(series[1]["data"], data["series"][1]["data"])
        self.assertNotEqual(series[0]["color"], series[1]["color"])
        self.assertEqual(series[0]["lines"]["lineWidth"], 3)
        self.assertEqual(series[1]["lines"]["lineWidth"], 1)
        self.assertIn(
            "configuration_speedups(configuration_series(data, bm))", grid[start:end]
        )

    @unittest.skipUnless(shutil.which("node"), "Node.js is required to execute summary JavaScript")
    def test_speedup_preview_reference_and_builder_timings(self):
        source = Path(asv.__file__).parent / "www"
        with tempfile.TemporaryDirectory() as directory:
            html = Path(directory)
            for name in ("index.html", "graphdisplay.js", "summarygrid.js", "summarylist.js"):
                shutil.copy2(source / name, html / name)
            customize_pages(html)
            grid = (html / "summarygrid.js").read_text(encoding="utf-8")
            graph = (html / "graphdisplay.js").read_text(encoding="utf-8")
        start = graph.index("function configuration_key(")
        end = graph.index("function configuration_segments(", start)
        preview_start = grid.index("    function benchmark_container(")
        preview_end = grid.index("    function make_summary(", preview_start)
        subprocess.run(
            ["node", "-e", """
const assert = require('node:assert/strict');
const plots = [], requests = [];
let previewData = {
    graph_parameters: [{}],
    series: [
        {graph: 0, parameters: {library: "'onnx'"},
         data: [[10, 1], [20, 1]]},
        {graph: 0, parameters: {library: "'onnxlight'"},
         data: [[10, 0.002], [20, null], [30, 0.004]]},
        {graph: 0, parameters: {library: "'ir-py'"},
         data: [[10, 5], [20, 6]]}
    ]
};
var system_dark_theme = false;
function $(html) {
    return {
        html, children: [], attributes: {}, handlers: {},
        tooltip(options) { this.tooltipOptions = options; return this; },
        attr(key, value) { this.attributes[key] = value; return this; },
        on(event, handler) { this.handlers[event] = handler; return this; },
        text(value) { this.content = value; return this; },
        append(child) { this.children.push(child); return this; }
    };
}
$.extend = Object.assign;
$.asv = {
    convert_benchmark_param_value: value => value.slice(1, -1),
    load_graph_data(url) {
        requests.push(url);
        return {
            done(callback) {
                callback(previewData);
                return this;
            },
            fail() { return this; }
        };
    }
};
$.plot = (element, series, options) => { plots.push({series, options}); };
function callback_in_view(element, callback) { callback(); }
function draw_configuration_lines() {}
""" + graph[start:end] + grid[preview_start:preview_end] + """
const builder = benchmark_container({name: 'builder.save.onnx_io.OnnxSave.time_run',
                                    param_names: ['library'], unit: 'seconds'});
assert.equal(builder.children.length, 2);
assert.deepEqual(requests, ['graphs/configurations/builder.save.onnx_io.OnnxSave.time_run.json']);
assert.equal(plots.length, 1);
const timing = plots.shift();
assert.equal(timing.series.length, 1);
assert.equal(timing.series[0].parameters.library, 'onnxlight');
assert.deepEqual(timing.series[0].data, [[10, 0.002], [20, null], [30, 0.004]]);
assert.deepEqual(timing.options.grid.markings, []);
assert.equal(timing.options.yaxis.max, null);
const builderPlot = builder.children[1];
assert.match(builderPlot.tooltipOptions.title, /onnx-light timings/);
builderPlot.handlers.plothover(null, null, {
    series: timing.series[0], datapoint: [10, 0.002]
});
assert.match(builderPlot.attributes['data-original-title'], /^0.002000 seconds<br>/);
assert.ok(!builderPlot.attributes['data-original-title'].includes('ONNX Runtime'));
for (const [key, values] of [
    ['library', ['onnxlight', 'onnxlight-cpp', 'onnxlight-mmap', 'onnxlight-ifstream', 'onnxlight-nocopy']],
    ['builder', ['onnx-light']],
    ['implementation', ['onnx-light']]
]) {
    previewData.series = ['onnxscript', ...values].map(value => ({
        graph: 0, parameters: {[key]: "'" + value + "'"}, data: [[10, 0.002]]
    }));
    benchmark_container({name: 'builder.example.time_run', param_names: [key], unit: 'seconds'});
    assert.deepEqual(plots.pop().series.map(curve => curve.parameters[key]), values);
}
previewData.series = [{graph: 0, parameters: {dtype: "'float32'"}, data: [[10, 0.002]]}];
benchmark_container({name: 'builder.load.onnx_io.OnnxReferenceEvaluator.time_run',
                     param_names: ['dtype'], unit: 'seconds'});
assert.deepEqual(plots.pop().series[0].data, [[10, 0.002]]);
requests.length = 0;
previewData.series = [
    {graph: 0, parameters: {backend: "'onnxruntime'",
     'env-ONNX_LIGHT_VERSION': '0.2.7', 'env-ONNX_LIGHT_CPU_VERSION': '0.2.8'},
     data: [[10, 1], [20, 1]]},
    {graph: 0, parameters: {backend: "'onnx-light'",
     'env-ONNX_LIGHT_VERSION': '0.2.7', 'env-ONNX_LIGHT_CPU_VERSION': '0.2.8'},
     data: [[10, 2], [20, 4]]}
];
const model = benchmark_container({name: 'models.dummies.mlp.MLP.time_run',
                                   param_names: ['backend']});
assert.equal(model.children.length, 2);
assert.equal(requests.length, 1);
assert.equal(plots[0].series.length, 1);
assert.equal(plots[0].series[0].parameters.backend, 'onnx-light');
assert.deepEqual(plots[0].series[0].data, [[10, 0.5], [20, 0.25]]);
assert.deepEqual(plots[0].options.grid.markings[0].yaxis, {from: 1, to: 1});
assert.ok(plots[0].options.yaxis.max > 1); // Reference stays visible below 1x.
assert.equal(configuration_speedup_max(plots[0].series), 1.05);
assert.equal(configuration_speedup_max([{data: [[10, 4], [20, null]]}]), 4.2);
model.children[1].handlers.plothover(null, null, {
    series: plots[0].series[0], datapoint: [10, 0.5]
});
assert.match(model.children[1].attributes['data-original-title'], /^0.5000× ONNX Runtime<br>/);
assert.equal(model.children[1].tooltipOptions.html, true);
assert.ok(model.children[1].attributes['data-original-title'].includes(
    '<strong class="benchmark-version">onnx-light: 0.2.7</strong>'));
assert.ok(model.children[1].attributes['data-original-title'].includes(
    '<strong class="benchmark-version">onnx-light-cpu: 0.2.8</strong>'));
previewData.series = [
    {graph: 0, parameters: {library: "'onnx'"}, data: [[10, 1], [20, 1]]},
    {graph: 0, parameters: {library: "'onnxlight'"}, data: []}
];
for (const data of [[[10, null]], []]) {
    previewData.series[1].data = data;
    const unavailable = benchmark_container({name: 'builder.load.onnx_io.OnnxLoad.time_run',
                                           param_names: ['library'], unit: 'seconds'});
    assert.equal(unavailable.children[1].content, 'onnx-light timings unavailable');
}
previewData.series = previewData.series.slice(0, 1);
const missing = benchmark_container({name: 'builder.save.onnx_io.OnnxSave.time_run',
                                   param_names: ['library'], unit: 'seconds'});
assert.equal(missing.children[1].content, 'onnx-light timings unavailable');
assert.equal(plots.length, 1);
"""],
            check=True, capture_output=True, text=True,
        )

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
        self.assertIn(
            "#main-graph,\n#speedup-graph {\n    flex: 1 1 0;\n"
            "    min-height: 100px;\n    width: 100%;\n    height: auto !important;",
            stylesheet,
        )

    @unittest.skipUnless(shutil.which("node"), "Node.js is required to execute hover JavaScript")
    def test_hover_package_versions_are_highlighted_and_escaped(self):
        source = Path(asv.__file__).parent / "www"
        with tempfile.TemporaryDirectory() as directory:
            html = Path(directory)
            for name in ("index.html", "graphdisplay.js", "summarygrid.js", "summarylist.js"):
                shutil.copy2(source / name, html / name)
            customize_pages(html)
            graph = (html / "graphdisplay.js").read_text(encoding="utf-8")
            stylesheet = (html / "system-theme.css").read_text(encoding="utf-8")
        self.assertIn("contents.concat(configuration_details(item.series.parameters))", graph)
        self.assertIn(".benchmark-version {\n    color: #79c0ff;", stylesheet)
        start = graph.index("function configuration_escape_html(")
        end = graph.index("function configuration_series(", start)
        subprocess.run(
            ["node", "-e", """
const assert = require('node:assert/strict');
""" + graph[start:end] + """
const details = configuration_details({
    'env-ONNX_LIGHT_VERSION': '0.2.7',
    'env-ONNX_LIGHT_CPU_VERSION': '0.2.8',
    cpu: 'processor <&>', num_cpu: 4, 'onnx-ir': '1.0',
    commit: 'abc', machine: 'cpu', python: null, arch: 'anonymous',
    backend: '<img src=x onerror=alert(1)>'
});
assert.ok(details.includes('<strong class="benchmark-version">onnx-light: 0.2.7</strong>'));
assert.ok(details.includes('<strong class="benchmark-version">onnx-light-cpu: 0.2.8</strong>'));
assert.ok(details.includes('processor: processor &lt;&amp;&gt;'));
assert.ok(details.includes('logical cores: 4'));
assert.ok(details.includes('ir-py: 1.0'));
assert.ok(details.includes('backend: &lt;img src=x onerror=alert(1)&gt;'));
assert.equal(details.length, 6);
assert.deepEqual(configuration_details({'env-ONNX_LIGHT_VERSION_CPU': '1.0'}), [
    '<strong class="benchmark-version">onnx-light-cpu: 1.0</strong>'
]);
assert.deepEqual(configuration_details({'ONNX_LIGHT_VERSION': '<b>"&\\''}), [
    '<strong class="benchmark-version">onnx-light: &lt;b&gt;&quot;&amp;&#39;</strong>'
]);
"""],
            check=True, capture_output=True, text=True,
        )

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
            self.assertEqual(graph.count("axisLabelColour"), 3)
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
            self.assertIn(
                "lines: {lineWidth: configuration_line_width(graph_content[2])}",
                graph,
            )
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
            self.assertIn("title: 'View results for ' + configuration_escape_html(bm.name)", grid)
            self.assertNotIn("title: bm.code", grid)
            self.assertIn("html: true,", grid)
            self.assertIn("configuration_escape_html(bm.name)", grid)
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
assert.equal(configuration_line_width({backend: 'onnxruntime'}), 3);
for (const backend of ['onnx-light', 'onnx-light-cpu', 'reference', undefined]) {
    assert.equal(configuration_line_width({backend, onnxruntime: '1.0'}), 1);
}
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
    series([[10, 11], [60, 12]], {backend: 'onnx-light'}),
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
assert.equal(segments.length, 7); // Different configurations never connect, even with equal colors.
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
                            color: this.strokeStyle, lineWidth: this.lineWidth}); }
};
draw_configuration_lines({
    getData: () => graphs,
    getAxes: () => ({xaxis: {p2c: x => x * 2}, yaxis: {p2c: y => y * 3}}),
    getPlotOffset: () => ({left: 7, top: 9}),
    width: () => 100, height: () => 80
}, context);
assert.deepEqual(calls, ['save', ['translate', 7, 9], ['rect', 0, 0, 100, 80],
                         'clip', 'restore']);
const boldStrokes = strokes.filter(stroke => stroke.lineWidth == 3);
assert.deepEqual(boldStrokes[0], {
    from: [20, 3], to: [40, 6], dash: [], color: '#123456', lineWidth: 3
});
assert.deepEqual(boldStrokes[1].dash, [2, 3]);
assert.equal(boldStrokes.length, 6);
assert.equal(strokes.filter(stroke => stroke.lineWidth == 1).length, 1);
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
