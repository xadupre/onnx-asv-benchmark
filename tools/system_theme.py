import argparse
import shutil
from pathlib import Path

from tools.date_axis import replace_once


def use_system_theme(html_dir):
    html_dir = Path(html_dir)

    index = html_dir / "index.html"
    page = index.read_text(encoding="utf-8")
    page = replace_once(
        page,
        "</head>",
        '<meta name="color-scheme" content="light dark">\n'
        '<link href="system-theme.css" rel="stylesheet" type="text/css"/>\n'
        "</head>",
    )

    script = html_dir / "graphdisplay.js"
    source = script.read_text(encoding="utf-8")
    source = replace_once(
        source,
        "'use strict';",
        "'use strict';\n\n"
        "var system_dark_theme = window.matchMedia("
        "'(prefers-color-scheme: dark)').matches;",
    )
    axis_font = 'axisLabelFontSizePixels: 12'
    if source.count(axis_font) != 2:
        raise ValueError(
            f"Expected exactly two occurrences of {axis_font!r} in ASV output"
        )
    source = source.replace(
        axis_font,
        axis_font + ",\n"
        '                axisLabelColour: system_dark_theme ? "#c9d1d9" : "#333333"',
    )
    source = replace_once(
        source,
        '"color:#666;background:white;padding-left:0.25em;font-size:smaller;\'',
        '"color:var(--asv-muted);background:var(--asv-background);'
        'padding-left:0.25em;font-size:smaller;\'',
    )

    stylesheet = Path(__file__).with_suffix(".css")
    shutil.copy2(stylesheet, html_dir / "system-theme.css")
    index.write_text(page, encoding="utf-8")
    script.write_text(source, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add an operating-system-aware theme to published ASV pages."
    )
    parser.add_argument("html_dir", type=Path)
    use_system_theme(parser.parse_args().html_dir)
