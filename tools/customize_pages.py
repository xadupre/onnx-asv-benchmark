import argparse
import shutil
from pathlib import Path

RUNTIME_OVERVIEW = """\
      <section class="benchmark-home">
        <div class="benchmark-introduction">
          <p class="benchmark-eyebrow">ONNX runtime benchmarks</p>
          <h1>Compare execution backends over time</h1>
          <p>
            Explore model and operator performance across optimized, reference,
            and lightweight ONNX runtimes. Lower execution times are better.
          </p>
          <a
            class="workflow-status"
            href="https://github.com/xadupre/onnx-asv-benchmark/actions/workflows/genai-compatibility.yml"
          >
            <img
              src="https://github.com/xadupre/onnx-asv-benchmark/actions/workflows/genai-compatibility.yml/badge.svg?branch=main"
              alt="GenAI compatibility workflow status"
            >
          </a>
        </div>
        <div class="runtime-grid" aria-label="Runtime descriptions">
          <a class="runtime-card" href="https://onnxruntime.ai/" target="_blank">
            <strong>ONNX Runtime</strong>
            <span>Production runtime with optimized CPU execution providers.</span>
          </a>
          <a class="runtime-card" href="https://onnx.ai/onnx/api/reference.html" target="_blank">
            <strong>ONNX Reference</strong>
            <span>ONNX's Python reference evaluator, focused on specification correctness.</span>
          </a>
          <a class="runtime-card" href="https://github.com/xadupre/onnx-light" target="_blank">
            <strong>onnx-light</strong>
            <span>Lightweight ONNX execution with a compact Python kernel implementation.</span>
          </a>
          <a class="runtime-card" href="https://github.com/xadupre/onnx-light-cpu" target="_blank">
            <strong>onnx-light-cpu</strong>
            <span>onnx-light with native CPU kernels and fallback to the lightweight runtime.</span>
          </a>
          <a class="runtime-card" href="https://onnxruntime.ai/docs/genai/" target="_blank">
            <strong>ONNX Runtime GenAI</strong>
            <span>Generation API for autoregressive models, used by the Tiny-LLM benchmark.</span>
          </a>
        </div>
        <section class="machine-overview" aria-labelledby="machine-overview-title">
          <h2 id="machine-overview-title">Benchmark processors</h2>
          <p>
            Logical cores and instruction sets available to each benchmark worker.
          </p>
          <div class="table-responsive">
            <table class="table machine-table">
              <thead>
                <tr>
                  <th>Processor</th>
                  <th>Logical cores</th>
                  <th>Instruction sets</th>
                </tr>
              </thead>
              <tbody id="machine-summary"></tbody>
            </table>
          </div>
        </section>
        <nav id="benchmark-navigation" aria-label="Benchmark groups">
          <div class="benchmark-family-filter btn-group" role="group">
            <button class="btn btn-default active" type="button" data-family="all">All</button>
            <button class="btn btn-default" type="button" data-family="builder">Builder</button>
            <button class="btn btn-default" type="button" data-family="models">Models</button>
            <button class="btn btn-default" type="button" data-family="ops">Operators</button>
            <button class="btn btn-default" type="button" data-family="cpu_backend_cases">
              CPU backend cases
            </button>
          </div>
          <div id="benchmark-category-filters"></div>
        </nav>
      </section>
"""


def _replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one occurrence of {old!r} in ASV output")
    return text.replace(old, new)


def customize_pages(html_dir):
    html_dir = Path(html_dir)

    index_path = html_dir / "index.html"
    index = index_path.read_text(encoding="utf-8")
    index = _replace_once(
        index,
        "</head>",
        '<meta name="color-scheme" content="light dark">\n'
        '<link href="system-theme.css" rel="stylesheet" type="text/css"/>\n'
        "<style>#even-spacing, #date-scale { display: none !important; }</style>\n"
        "</head>",
    )
    index = _replace_once(
        index,
        '    <div id="summarygrid-display" style="position: absolute; left: 0; '
        'top: 55px; width: 100%; height: 100%">\n'
        "    </div>",
        '    <div id="summarygrid-display" style="position: absolute; left: 0; '
        'top: 55px; width: 100%; height: 100%">\n'
        f"{RUNTIME_OVERVIEW}"
        "    </div>",
    )

    graph_path = html_dir / "graphdisplay.js"
    graph = graph_path.read_text(encoding="utf-8")
    graph = _replace_once(
        graph,
        "'use strict';",
        "'use strict';\n\n"
        "var system_dark_theme = window.matchMedia("
        "'(prefers-color-scheme: dark)').matches;",
    )
    graph = _replace_once(graph, "var date_scale = false;", "var date_scale = true;")
    graph = _replace_once(
        graph,
        '                            parts.push(key + "-" + value);',
        "                            var normalized_value =\n"
        "                                typeof value == 'string'\n"
        "                                    ? value.replace(/^'(.*)'$/, '$1') : value;\n"
        "                            if (key == 'onnxruntime' ||\n"
        "                                    key == 'instruction_sets' ||\n"
        "                                    key == 'env-ONNX_LIGHT_CPU_VERSION' ||\n"
        "                                    key == 'env-ONNX_LIGHT_VERSION') {\n"
        "                                return;\n"
        "                            }\n"
        "                            if (key == 'backend' &&\n"
        "                                    normalized_value == 'onnxruntime') {\n"
        "                                parts.push(state.onnxruntime\n"
        "                                    ? 'ort-' + state.onnxruntime : 'ort');\n"
        "                                return;\n"
        "                            }\n"
        "                            var legend_names = {\n"
        "                                backend: 'bck',\n"
        "                                num_cpu: 'vCPU'\n"
        "                            };\n"
        "                            key = legend_names[key] || key;\n"
        '                            parts.push(key + "-" + normalized_value);',
    )
    graph = _replace_once(
        graph,
        "function handle_x_scale(options) {\n",
        "function handle_x_scale(options) {\n"
        "        date_scale = true;\n"
        "        even_spacing = false;\n",
    )
    graph = _replace_once(
        graph,
        "axisLabel = 'commit date';",
        "axisLabel = 'date';",
    )
    graph = _replace_once(graph, 'text = "commit";', 'text = "date";')
    graph = _replace_once(
        graph,
        "param != 'machine'",
        "param != 'machine' && param != 'cpu' && param != 'num_cpu' && "
        "param != 'instruction_sets'",
    )
    graph = _replace_once(
        graph,
        "else if (params[axis-1].length > 1) {",
        "else if (params[axis-1].length > 1 || " "param_names[axis-1] == 'backend') {",
    )
    graph = _replace_once(
        graph,
        "graph_label(labels, different)]);",
        "graph_label(labels, different), labels]);",
    )
    graph = _replace_once(
        graph,
        "label: graph_content[1],\n" "                        bars: { order: count, },",
        "label: graph_content[1],\n"
        "                        parameters: graph_content[2],\n"
        "                        bars: { order: count, },",
    )
    graph = _replace_once(
        graph,
        "                    var y = item.datapoint[1];\n"
        "                    var commit_hash = get_commit_hash(item.datapoint[0]);\n"
        "                    if (commit_hash) {\n"
        "                        var unit = $.asv.main_json.benchmarks[current_benchmark].unit;\n"
        "                        showTooltip(\n"
        "                            item.pageX, item.pageY,\n"
        '                            $.asv.pretty_unit(y, unit) + " @ " + commit_hash);\n'
        "                    }",
        "                    var y = item.datapoint[1];\n"
        "                    var unit = $.asv.main_json.benchmarks[current_benchmark].unit;\n"
        "                    var contents = [\n"
        '                        "<b>" + $.asv.pretty_unit(y, unit) + "</b>",\n'
        "                        new Date(item.datapoint[0]).toLocaleString()\n"
        "                    ];\n"
        "                    $.each(item.series.parameters, function(key, value) {\n"
        "                        if (key != 'commit' && key != 'machine' &&\n"
        "                                value !== null && value != 'anonymous') {\n"
        "                            var parameter_names = {\n"
        "                                cpu: 'processor',\n"
        "                                'onnx-ir': 'ir-py',\n"
        "                                num_cpu: 'logical cores',\n"
        "                                instruction_sets: 'instruction sets'\n"
        "                            };\n"
        "                            var name = parameter_names[key] ||\n"
        "                                key.replace(/^env-/, '');\n"
        "                            var text = name + ': ' + value;\n"
        '                            contents.push($("<span>").text(text).html());\n'
        "                        }\n"
        "                    });\n"
        "                    showTooltip(item.pageX, item.pageY, contents.join('<br>'));",
    )
    axis_font = "axisLabelFontSizePixels: 12"
    if graph.count(axis_font) != 2:
        raise ValueError(
            f"Expected exactly two occurrences of {axis_font!r} in ASV output"
        )
    graph = graph.replace(
        axis_font,
        axis_font + ",\n"
        '                axisLabelColour: system_dark_theme ? "#c9d1d9" : "#333333"',
    )
    graph = _replace_once(
        graph,
        "\"color:#666;background:white;padding-left:0.25em;font-size:smaller;'",
        '"color:var(--asv-muted);background:var(--asv-background);'
        "padding-left:0.25em;font-size:smaller;'",
    )
    summary_path = html_dir / "summarylist.js"
    summary = summary_path.read_text(encoding="utf-8")
    summary = _replace_once(
        summary,
        "param != 'machine'",
        "param != 'machine' && param != 'cpu' && param != 'num_cpu' && "
        "param != 'instruction_sets'",
    )

    grid_path = html_dir / "summarygrid.js"
    grid = grid_path.read_text(encoding="utf-8")
    grid = _replace_once(
        grid,
        "            var i = bm_name.indexOf('.');\n"
        "            var group = bm_name.slice(0, i);\n"
        "            var name = bm_name.slice(i + 1);",
        "            var parts = bm_name.split('.');\n"
        "            var group = parts.slice(0, 2).join(' / ');",
    )
    grid = _replace_once(
        grid,
        "var display_name = bm.pretty_name || "
        "bm.name.slice(bm.name.indexOf('.') + 1);",
        "var parts = bm.name.split('.');\n"
        "        var display_name = bm.pretty_name || "
        "(parts[0] == 'ops' && parts[parts.length - 1] == 'time_run'\n"
        "            ? parts[parts.length - 2]\n"
        "            : parts.slice(2).join('.'));",
    )
    grid = _replace_once(
        grid,
        "    function benchmark_container(bm) {",
        """\
    function make_machine_summary() {
        var machines = {};
        $.each($.asv.main_json.graph_param_list, function(i, params) {
            var cpu = params.cpu;
            if (!cpu || cpu == 'anonymous') {
                return;
            }
            if (machines[cpu] === undefined) {
                machines[cpu] = {cores: {}, instruction_sets: {}};
            }
            if (params.num_cpu && params.num_cpu != 'anonymous') {
                machines[cpu].cores[params.num_cpu] = true;
            }
            if (params.instruction_sets &&
                    params.instruction_sets != 'unavailable') {
                var instruction_sets = params.instruction_sets.split(';')
                    .map(function(variant) {
                        return variant.split(',').join(', ');
                    }).join('; ');
                machines[cpu].instruction_sets[instruction_sets] = true;
            }
        });

        var body = $('#machine-summary');
        $.each(Object.keys(machines).sort(), function(i, cpu) {
            var machine = machines[cpu];
            var cores = Object.keys(machine.cores).sort().join(', ') || 'Not recorded';
            var instruction_sets = Object.keys(machine.instruction_sets)
                .sort().join('; ') || 'Not recorded';
            var row = $('<tr/>');
            row.append($('<th scope="row"/>').text(cpu));
            row.append($('<td/>').text(cores));
            row.append($('<td/>').text(instruction_sets));
            body.append(row);
        });
    }

    function make_summary_navigation(groups) {
        var filters = $('#benchmark-category-filters');

        $.each(['builder', 'models', 'ops', 'cpu_backend_cases'], function(i, family) {
            var family_filters = $('<div class="benchmark-category-filter btn-group"/>');
            family_filters.attr('data-family', family);
            family_filters.hide();
            family_filters.append(
                $('<button class="btn btn-default active" type="button">All</button>')
                    .attr('data-group', 'all'));
            $.each(groups, function(group, benchmarks) {
                if (benchmarks[0].split('.')[0] == family) {
                    family_filters.append(
                        $('<button class="btn btn-default" type="button"/>')
                            .attr('data-group', group)
                            .text(group.split(' / ')[1]));
                }
            });
            filters.append(family_filters);
        });

        $('.benchmark-family-filter button').on('click', function() {
            var family = $(this).attr('data-family');
            $('.benchmark-family-filter button').removeClass('active');
            $(this).addClass('active');
            $('.benchmark-category-filter').hide();
            $('.benchmark-category-filter button').removeClass('active');
            $('.benchmark-category-filter button[data-group="all"]').addClass('active');
            if (family != 'all') {
                $('.benchmark-category-filter[data-family="' + family + '"]').show();
            }
            $('.benchmark-group').each(function() {
                $(this).toggle(family == 'all' || $(this).attr('data-family') == family);
            });
            $(window).trigger('scroll');
        });

        $('.benchmark-category-filter button').on('click', function() {
            var group = $(this).attr('data-group');
            var family = $(this).parent().attr('data-family');
            $(this).siblings().removeClass('active');
            $(this).addClass('active');
            $('.benchmark-group').each(function() {
                $(this).toggle(
                    $(this).attr('data-family') == family &&
                    (group == 'all' || $(this).attr('data-group') == group));
            });
            $(window).trigger('scroll');
        });
    }

    function benchmark_container(bm) {""",
    )
    grid = _replace_once(
        grid,
        "        $.each(get_benchmarks_by_groups(), function(group, benchmarks) {\n"
        "            var group_container = $('<div class=\"benchmark-group\"/>')\n"
        "            group_container.attr('id', 'group-' + group)\n"
        "            group_container.append($('<h1>' + group + '</h1>'));",
        "        var groups = get_benchmarks_by_groups();\n"
        "        make_machine_summary();\n"
        "        make_summary_navigation(groups);\n"
        "        $.each(groups, function(group, benchmarks) {\n"
        "            var family = benchmarks[0].split('.')[0];\n"
        "            var group_container = $('<div class=\"benchmark-group\"/>');\n"
        "            group_container.attr('id', 'group-' + group);\n"
        "            group_container.attr('data-family', family);\n"
        "            group_container.attr('data-group', group);\n"
        "            group_container.append($('<h2>' + group + '</h2>'));",
    )

    stylesheet = Path(__file__).with_name("system_theme.css")
    shutil.copy2(stylesheet, html_dir / "system-theme.css")
    index_path.write_text(index, encoding="utf-8")
    graph_path.write_text(graph, encoding="utf-8")
    summary_path.write_text(summary, encoding="utf-8")
    grid_path.write_text(grid, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Customize published ASV pages.")
    parser.add_argument("html_dir", type=Path)
    customize_pages(parser.parse_args().html_dir)
