import argparse
from pathlib import Path

from tools.date_axis import replace_once


def show_backend_labels(html_dir):
    script = Path(html_dir) / "graphdisplay.js"
    source = script.read_text(encoding="utf-8")
    source = replace_once(
        source,
        "else if (params[axis-1].length > 1) {",
        "else if (params[axis-1].length > 1 || "
        "param_names[axis-1] == 'backend') {",
    )
    script.write_text(source, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Always identify benchmark backends in ASV graph legends."
    )
    parser.add_argument("html_dir", type=Path)
    show_backend_labels(parser.parse_args().html_dir)
