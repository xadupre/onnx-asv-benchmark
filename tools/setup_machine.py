import argparse
from pathlib import Path

from asv import util
from asv.machine import Machine, MachineCollection


def main():
    parser = argparse.ArgumentParser(
        description="Set up or check the ASV machine profile named after its processor."
    )
    parser.add_argument(
        "--check", action="store_true", help="Check the profile without changing it."
    )
    args = parser.parse_args()

    profile = Machine.get_defaults()
    processor = profile["cpu"]
    if not processor:
        raise ValueError("ASV could not detect a processor name.")
    profile["machine"] = processor

    path = Path(MachineCollection.get_machine_file_path())
    machines = util.load_json(path, MachineCollection.api_version) if path.is_file() else {}
    current = machines.get(processor)
    if not isinstance(current, dict):
        current = {}
    matches = all(current.get(key) == value for key, value in profile.items())

    if args.check:
        if not matches:
            raise SystemExit(
                f"ASV profile for {processor!r} is missing or outdated; "
                "run python tools/setup_machine.py to set it up."
            )
    elif not matches:
        MachineCollection.save(processor, {**current, **profile})

    print(processor)


if __name__ == "__main__":
    main()
