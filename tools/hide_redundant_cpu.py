import argparse
from pathlib import Path

from tools.date_axis import replace_once


def hide_redundant_cpu(html_dir):
    html_dir = Path(html_dir)
    old = "param != 'machine'"
    new = "param != 'machine' && param != 'cpu'"
    for name in ("graphdisplay.js", "summarylist.js"):
        path = html_dir / name
        source = path.read_text(encoding="utf-8")
        source = replace_once(source, old, new)
        path.write_text(source, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hide the CPU selector duplicated by processor-named machines."
    )
    parser.add_argument("html_dir", type=Path)
    hide_redundant_cpu(parser.parse_args().html_dir)
