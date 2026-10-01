import os
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    processor = subprocess.check_output(
        [sys.executable, str(root / "tools" / "setup_machine.py")],
        text=True,
    ).strip()
    env = {**os.environ, "ASV_PYTHONPATH": os.environ.get("PYTHONPATH", "")}
    command = [
        sys.executable,
        "-m",
        "asv",
        "run",
        "--environment",
        f"existing:{sys.executable}",
        "--machine",
        processor,
        *sys.argv[1:],
    ]
    raise SystemExit(subprocess.call(command, cwd=root, env=env))


if __name__ == "__main__":
    main()
