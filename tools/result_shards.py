import json
import shutil
import tempfile
from pathlib import Path

ANONYMOUS_MACHINE_FIELDS = {
    "os": "anonymous",
    "ram": "anonymous",
}

OPERATOR_CATEGORIES = frozenset(
    {
        "generator",
        "image",
        "logical",
        "math",
        "maths",
        "nn",
        "object_detection",
        "optional",
        "preview",
        "quantization",
        "reduction",
        "rt",
        "sequence",
        "tensor",
        "text",
        "traditionalml",
        "training",
    }
)
MODEL_GROUPS = {
    "matmul_add": "dummies",
    "mlp": "dummies",
    "tiny_llm": "llm",
}


def canonical_benchmark_name(name):
    parts = name.split(".")
    if parts[0] in OPERATOR_CATEGORIES:
        category = "math" if parts[0] == "maths" else parts[0]
        return ".".join(("ops", category, *parts[1:]))
    if parts[0] == "models" and len(parts) > 1 and parts[1] in MODEL_GROUPS:
        return ".".join(("models", MODEL_GROUPS[parts[1]], *parts[1:]))
    return name


def _benchmark_name_priority(name):
    canonical = canonical_benchmark_name(name)
    if name == canonical:
        return canonical, 2
    if name.startswith("maths."):
        return canonical, 0
    return canonical, 1


def _canonicalize_mapping(mapping, description):
    selected = {}
    for name, value in mapping.items():
        canonical, priority = _benchmark_name_priority(name)
        if canonical not in selected or priority > selected[canonical][0]:
            selected[canonical] = (priority, name, value)
        elif priority == selected[canonical][0] and value != selected[canonical][2]:
            other = selected[canonical][1]
            raise ValueError(f"Conflicting {description} for {other!r} and {name!r}.")
    return {name: value for name, (_, _, value) in selected.items()}


def canonicalize_benchmark_hierarchy(results_root):
    results_root = Path(results_root)
    benchmarks_path = results_root / "benchmarks.json"
    benchmarks = _load(benchmarks_path)
    metadata = {
        name: value for name, value in benchmarks.items() if not isinstance(value, dict)
    }
    benchmark_definitions = {
        name: value for name, value in benchmarks.items() if isinstance(value, dict)
    }
    benchmark_definitions = _canonicalize_mapping(
        benchmark_definitions,
        "benchmark metadata",
    )
    for name, definition in benchmark_definitions.items():
        if "name" in definition:
            definition["name"] = name
    _save(
        benchmarks_path,
        {
            **metadata,
            **benchmark_definitions,
        },
    )

    for machine_directory in results_root.iterdir():
        if not machine_directory.is_dir():
            continue
        for result_path in machine_directory.glob("*.json"):
            if result_path.name == "machine.json":
                continue
            result = _load(result_path)
            result["results"] = _canonicalize_mapping(
                result.get("results", {}),
                f"results in {result_path}",
            )
            result["durations"] = _canonicalize_mapping(
                result.get("durations", {}),
                f"durations in {result_path}",
            )
            _save(result_path, result)


def benchmark_shard(name):
    parts = canonical_benchmark_name(name).split(".")
    if parts[0] == "ops" and len(parts) > 1:
        return f"ops/{parts[1]}"
    if parts[0] == "models" and len(parts) > 2:
        return f"models/{parts[1]}/{parts[2]}"
    return parts[0]


def _is_legacy_track(name):
    return name.startswith(("machine.track_", "versions.track_"))


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _publishable_machine(machine):
    processor = machine.get("cpu")
    if not isinstance(processor, str) or not processor or processor == "anonymous":
        raise ValueError("ASV machine metadata does not contain a processor name.")
    if Path(processor).name != processor or processor in {".", ".."}:
        raise ValueError(f"Processor name is not a safe directory name: {processor!r}.")
    published = {
        **machine,
        **ANONYMOUS_MACHINE_FIELDS,
        "num_cpu": "anonymous",
        "machine": processor,
        "cpu": processor,
    }
    published.pop("instruction_sets", None)
    return processor, published


def _merge_mapping(target, source, description):
    overlap = target.keys() & source.keys()
    if overlap:
        names = ", ".join(sorted(overlap))
        raise ValueError(f"Duplicate {description}: {names}")
    target.update(source)


def _merge_machine_params(path, current, incoming):
    if not isinstance(current, dict) or not isinstance(incoming, dict):
        raise ValueError(f"Incompatible ASV machine parameters in {path}.")

    merged = current.copy()
    for key in ("instruction_sets", "num_cpu"):
        current_value = current.get(key)
        incoming_value = incoming.get(key)
        if current_value == incoming_value:
            continue
        if not isinstance(current_value, str) or not isinstance(incoming_value, str):
            raise ValueError(f"Incompatible ASV machine parameters in {path}.")
        variants = set(current_value.split("; "))
        variants.update(incoming_value.split("; "))
        merged[key] = "; ".join(sorted(variants))

    ignored = {"instruction_sets", "num_cpu"}
    current_stable = {key: value for key, value in current.items() if key not in ignored}
    incoming_stable = {
        key: value for key, value in incoming.items() if key not in ignored
    }
    if current_stable != incoming_stable:
        raise ValueError(f"Incompatible ASV machine parameters in {path}.")
    return merged


def _merge_result_file(path, incoming):
    if not path.is_file():
        _save(path, incoming)
        return

    current = _load(path)
    for key in (
        "commit_hash",
        "env_name",
        "python",
        "requirements",
        "env_vars",
        "result_columns",
        "version",
    ):
        if current.get(key) != incoming.get(key):
            raise ValueError(f"Incompatible ASV result field {key!r} in {path}.")
    current["params"] = _merge_machine_params(
        path,
        current.get("params"),
        incoming.get("params"),
    )

    current["date"] = max(current.get("date", 0), incoming.get("date", 0))
    current.setdefault("results", {}).update(incoming.get("results", {}))
    current.setdefault("durations", {}).update(incoming.get("durations", {}))
    _save(path, current)


def write_shards(source, shard_root, selected_shards=None):
    source = Path(source)
    shard_root = Path(shard_root)
    benchmarks = _load(source / "benchmarks.json")
    machine_directories = [
        path
        for path in source.iterdir()
        if path.is_dir() and (path / "machine.json").is_file()
    ]
    if len(machine_directories) == 1:
        machine_directory = machine_directories[0]
    else:
        raise ValueError(
            f"Expected one ASV machine directory in {source}, "
            f"found {len(machine_directories)}."
        )

    machine_id, machine = _publishable_machine(
        _load(machine_directory / "machine.json")
    )
    benchmark_metadata = {
        name: value for name, value in benchmarks.items() if not isinstance(value, dict)
    }
    selected = set(selected_shards) if selected_shards is not None else None
    written = set()

    for result_path in sorted(machine_directory.glob("*.json")):
        if result_path.name == "machine.json":
            continue
        result = _load(result_path)
        grouped = {}
        for name, value in result.get("results", {}).items():
            if name not in benchmarks:
                if _is_legacy_track(name):
                    continue
                raise KeyError(f"Missing benchmark metadata for {name!r}.")
            shard = benchmark_shard(name)
            if selected is None or shard in selected:
                grouped.setdefault(shard, {})[name] = value

        for shard, shard_results in grouped.items():
            destination = shard_root / shard
            legacy_machine = destination / "cpu"
            legacy_metadata = legacy_machine / "machine.json"
            if (
                legacy_metadata.is_file()
                and _load(legacy_metadata).get("cpu") == "anonymous"
            ):
                shutil.rmtree(legacy_machine)
            shard_benchmarks_path = destination / "benchmarks.json"
            shard_benchmarks = (
                _load(shard_benchmarks_path)
                if shard_benchmarks_path.is_file()
                else benchmark_metadata.copy()
            )
            shard_benchmarks.update(benchmark_metadata)
            for name in shard_results:
                shard_benchmarks[name] = benchmarks[name]
            _save(shard_benchmarks_path, shard_benchmarks)
            _save(destination / machine_id / "machine.json", machine)

            shard_result = {
                **result,
                "params": {
                    **result.get("params", {}),
                    **ANONYMOUS_MACHINE_FIELDS,
                    "machine": machine_id,
                    "cpu": machine_id,
                },
                "results": shard_results,
                "durations": {
                    name: duration
                    for name, duration in result.get("durations", {}).items()
                    if name in shard_results
                },
            }
            _merge_result_file(
                destination / machine_id / result_path.name,
                shard_result,
            )
            written.add(shard)

    if selected is not None:
        missing = selected - written
        if missing:
            raise ValueError(
                "No benchmark results found for shards: " + ", ".join(sorted(missing))
            )
    return written


def migrate_legacy_results(results_root):
    results_root = Path(results_root)
    benchmarks_path = results_root / "benchmarks.json"
    if not benchmarks_path.is_file():
        return set()

    written = write_shards(results_root, results_root / "shards")
    machine_directories = [
        path
        for path in results_root.iterdir()
        if path.is_dir() and (path / "machine.json").is_file()
    ]
    benchmarks_path.unlink()
    for machine_directory in machine_directories:
        shutil.rmtree(machine_directory)
    return written


def migrate_shard_hierarchy(shard_root):
    shard_root = Path(shard_root)
    if not shard_root.is_dir():
        return set()

    with tempfile.TemporaryDirectory(
        prefix=".shard-migration-",
        dir=shard_root.parent,
    ) as temporary:
        temporary = Path(temporary)
        merged = temporary / "merged"
        rebuilt = temporary / "rebuilt"
        merge_shards(shard_root, merged)

        benchmarks_path = merged / "benchmarks.json"
        machine_directories = [
            path
            for path in merged.iterdir()
            if path.is_dir() and (path / "machine.json").is_file()
        ]
        for index, machine_directory in enumerate(machine_directories):
            source = temporary / f"source-{index}"
            source.mkdir()
            shutil.copy2(benchmarks_path, source / "benchmarks.json")
            shutil.copytree(machine_directory, source / machine_directory.name)
            write_shards(source, rebuilt)

        migrated = {
            path.parent.relative_to(rebuilt).as_posix()
            for path in rebuilt.rglob("benchmarks.json")
        }
        shutil.rmtree(shard_root)
        shutil.copytree(rebuilt, shard_root)
        return migrated


def merge_shards(shard_root, destination):
    shard_root = Path(shard_root)
    destination = Path(destination)
    benchmark_files = sorted(shard_root.rglob("benchmarks.json"))
    if not benchmark_files:
        raise FileNotFoundError(f"No result shards found in {shard_root}.")

    merged_benchmarks = {}
    machines = {}
    for benchmark_path in benchmark_files:
        shard_directory = benchmark_path.parent
        shard_benchmarks = _load(benchmark_path)
        metadata = {
            name: value
            for name, value in shard_benchmarks.items()
            if not isinstance(value, dict)
        }
        for name, value in metadata.items():
            if name in merged_benchmarks and merged_benchmarks[name] != value:
                raise ValueError(
                    f"Incompatible benchmark metadata field {name!r} "
                    f"in {benchmark_path}."
                )
            merged_benchmarks[name] = value
        shard_benchmarks = {
            name: value
            for name, value in shard_benchmarks.items()
            if isinstance(value, dict)
        }
        _merge_mapping(
            merged_benchmarks,
            shard_benchmarks,
            "benchmark metadata",
        )

        machine_directories = sorted(
            path
            for path in shard_directory.iterdir()
            if path.is_dir() and (path / "machine.json").is_file()
        )
        if not machine_directories:
            raise FileNotFoundError(f"No ASV machines found in {shard_directory}.")
        for machine_directory in machine_directories:
            machine_path = machine_directory / "machine.json"
            shard_machine = _load(machine_path)
            machine_id = machine_directory.name
            if machine_id in machines and machines[machine_id] != shard_machine:
                raise ValueError(f"Incompatible machine metadata in {machine_path}.")
            machines[machine_id] = shard_machine

            for result_path in sorted(machine_directory.glob("*.json")):
                if result_path.name == "machine.json":
                    continue
                _merge_result_file(
                    destination / machine_id / result_path.name,
                    _load(result_path),
                )

    _save(destination / "benchmarks.json", merged_benchmarks)
    for machine_id, machine in machines.items():
        _save(destination / machine_id / "machine.json", machine)
    canonicalize_benchmark_hierarchy(destination)
