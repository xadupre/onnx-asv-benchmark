import argparse
import shutil
from pathlib import Path


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
        "param != 'machine' && param != 'cpu'",
    )
    graph = _replace_once(
        graph,
        "else if (params[axis-1].length > 1) {",
        "else if (params[axis-1].length > 1 || "
        "param_names[axis-1] == 'backend') {",
    )
    graph = _replace_once(
        graph,
        "graph_label(labels, different)]);",
        "graph_label(labels, different), labels]);",
    )
    graph = _replace_once(
        graph,
        "label: graph_content[1],\n"
        "                        bars: { order: count, },",
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
        "                        if (key != 'commit' && key != 'cpu' &&\n"
        "                                value !== null && value != 'anonymous') {\n"
        "                            var name = key.replace(/^env-/, '');\n"
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
        '"color:#666;background:white;padding-left:0.25em;font-size:smaller;\'',
        '"color:var(--asv-muted);background:var(--asv-background);'
        'padding-left:0.25em;font-size:smaller;\'',
    )

    summary_path = html_dir / "summarylist.js"
    summary = summary_path.read_text(encoding="utf-8")
    summary = _replace_once(
        summary,
        "param != 'machine'",
        "param != 'machine' && param != 'cpu'",
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
        "var display_name = bm.pretty_name || "
        "bm.name.split('.').slice(2).join('.');",
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
