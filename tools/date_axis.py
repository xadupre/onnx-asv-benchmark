import argparse
from pathlib import Path


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one occurrence of {old!r} in ASV output")
    return text.replace(old, new)


def use_date_axis(html_dir):
    script = html_dir / "graphdisplay.js"
    source = script.read_text(encoding="utf-8")
    source = replace_once(source, "var date_scale = false;", "var date_scale = true;")
    source = replace_once(
        source,
        "function handle_x_scale(options) {\n",
        "function handle_x_scale(options) {\n"
        "        date_scale = true;\n"
        "        even_spacing = false;\n",
    )
    source = replace_once(source, "axisLabel = 'commit date';", "axisLabel = 'date';")

    index = html_dir / "index.html"
    page = index.read_text(encoding="utf-8")
    page = replace_once(
        page,
        "</head>",
        "<style>#even-spacing, #date-scale { display: none !important; }</style>\n</head>",
    )
    script.write_text(source, encoding="utf-8")
    index.write_text(page, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use a date-only axis on published ASV graphs.")
    parser.add_argument("html_dir", type=Path)
    use_date_axis(parser.parse_args().html_dir)
