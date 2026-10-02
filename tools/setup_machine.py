import argparse
import sys
from pathlib import Path

from asv import util
from asv.machine import Machine, MachineCollection


INSTRUCTION_SET_FLAGS = (
    ("SSE2", ("sse2",)),
    ("SSE3", ("sse3", "pni")),
    ("SSSE3", ("ssse3",)),
    ("SSE4.1", ("sse4_1",)),
    ("SSE4.2", ("sse4_2",)),
    ("AVX", ("avx",)),
    ("AVX2", ("avx2",)),
    ("AVX-VNNI", ("avx_vnni",)),
    ("AVX-512F", ("avx512f",)),
    ("AVX-512BW", ("avx512bw",)),
    ("AVX-512CD", ("avx512cd",)),
    ("AVX-512DQ", ("avx512dq",)),
    ("AVX-512VL", ("avx512vl",)),
    ("AVX-512VNNI", ("avx512_vnni", "avx512vnni")),
    ("AMX-BF16", ("amx_bf16",)),
    ("AMX-INT8", ("amx_int8",)),
    ("AMX-TILE", ("amx_tile",)),
    ("NEON", ("neon", "asimd")),
    ("SVE", ("sve",)),
    ("SVE2", ("sve2",)),
)


def instruction_sets(cpuinfo):
    flags = set()
    for line in cpuinfo.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip().lower() in {"flags", "features"}:
            flags.update(value.lower().split())
    return ", ".join(
        name
        for name, aliases in INSTRUCTION_SET_FLAGS
        if any(alias in flags for alias in aliases)
    )


def detect_instruction_sets():
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.is_file():
        return "unavailable"
    detected = instruction_sets(cpuinfo.read_text(encoding="utf-8"))
    return detected or "unavailable"


def main():
    parser = argparse.ArgumentParser(
        description="Set up or check the processor-named ASV machine profile."
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
    profile["instruction_sets"] = detect_instruction_sets()

    path = Path(MachineCollection.get_machine_file_path())
    machines = (
        util.load_json(path, MachineCollection.api_version) if path.is_file() else {}
    )
    current = machines.get(processor)
    if not isinstance(current, dict):
        current = {}
    matches = all(current.get(key) == value for key, value in profile.items())

    if args.check:
        if not matches:
            raise SystemExit(
                f"ASV profile {processor!r} in {path} is missing or outdated; "
                "run python tools/setup_machine.py to set it up."
            )
        action = "Verified"
    elif not matches:
        MachineCollection.save(processor, {**current, **profile})
        action = "Updated" if processor in machines else "Created"
    else:
        action = "Already correct"

    print(f"{action} ASV profile {processor!r} in {path}.", file=sys.stderr)
    print(processor)


if __name__ == "__main__":
    main()
