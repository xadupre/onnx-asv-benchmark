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

    stylesheet = Path(__file__).with_name("system_theme.css")
    shutil.copy2(stylesheet, html_dir / "system-theme.css")
    index_path.write_text(index, encoding="utf-8")
    graph_path.write_text(graph, encoding="utf-8")
    summary_path.write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Customize published ASV pages.")
    parser.add_argument("html_dir", type=Path)
    customize_pages(parser.parse_args().html_dir)
