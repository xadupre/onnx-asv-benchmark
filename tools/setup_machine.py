import argparse
import sys
from pathlib import Path

from asv import util
from asv.machine import Machine, MachineCollection

MACHINE_ID = "cpu"


def main():
    parser = argparse.ArgumentParser(
        description="Set up or check the anonymous ASV machine profile."
    )
    parser.add_argument(
        "--check", action="store_true", help="Check the profile without changing it."
    )
    args = parser.parse_args()

    profile = Machine.get_defaults()
    processor = profile["cpu"]
    if not processor:
        raise ValueError("ASV could not detect a processor name.")
    profile["machine"] = MACHINE_ID

    path = Path(MachineCollection.get_machine_file_path())
    machines = (
        util.load_json(path, MachineCollection.api_version) if path.is_file() else {}
    )
    current = machines.get(MACHINE_ID)
    if not isinstance(current, dict):
        current = {}
    matches = all(current.get(key) == value for key, value in profile.items())

    if args.check:
        if not matches:
            raise SystemExit(
                f"ASV profile {MACHINE_ID!r} in {path} is missing or outdated; "
                "run python tools/setup_machine.py to set it up."
            )
        action = "Verified"
    elif not matches:
        MachineCollection.save(MACHINE_ID, {**current, **profile})
        action = "Updated" if MACHINE_ID in machines else "Created"
    else:
        action = "Already correct"

    print(f"{action} ASV profile {MACHINE_ID!r} in {path}.", file=sys.stderr)
    print(MACHINE_ID)


if __name__ == "__main__":
    main()
