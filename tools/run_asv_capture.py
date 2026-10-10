"""Run ASV while retaining failed CPU backend case messages in result files."""

import json
import sys
from pathlib import Path

from asv import runner
from asv.main import main
from asv.results import Results


def _message(stderr):
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    return lines[-1][:500] if lines else ""


def install_error_capture():
    run_param = runner._run_benchmark_single_param
    add_result = Results.add_result
    save = Results.save
    failures = {}

    def capture_param(benchmark, spawner, param_idx, *args, **kwargs):
        result = run_param(benchmark, spawner, param_idx, *args, **kwargs)
        if (
            benchmark["name"].startswith("cpu_backend_cases.")
            and result.errcode != 0
            and result.stderr
        ):
            failures.setdefault(benchmark["name"], {})[param_idx] = _message(
                result.stderr
            )
        return result

    def capture_result(self, benchmark, result, *args, **kwargs):
        add_result(self, benchmark, result, *args, **kwargs)
        name = benchmark["name"]
        if not name.startswith("cpu_backend_cases."):
            return
        updates = self.__dict__.setdefault("_error_updates", {})
        captured = failures.pop(name, {})
        selected = kwargs.get("selected_idx")
        for index, value in enumerate(result.result):
            if selected is not None and index not in selected:
                continue
            error = captured.get(index)
            if value is None and not error and result.errcode != 0:
                error = _message(result.stderr)
            updates.setdefault(name, {})[str(index)] = error if value is None else ""

    def save_errors(self, result_dir):
        path = Path(result_dir) / self._filename
        previous = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        save(self, result_dir)
        errors = previous.get("benchmark_errors", {})
        for name, updates in self.__dict__.get("_error_updates", {}).items():
            values = errors.setdefault(name, {})
            for index, error in updates.items():
                if error:
                    values[index] = error
                else:
                    values.pop(index, None)
            if not values:
                errors.pop(name)
        if errors:
            data = json.loads(path.read_text(encoding="utf-8"))
            data["benchmark_errors"] = errors
            path.write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8")

    runner._run_benchmark_single_param = capture_param
    Results.add_result = capture_result
    Results.save = save_errors


def main_with_errors():
    install_error_capture()
    sys.argv = [sys.argv[0], "run", *sys.argv[1:]]
    main()


if __name__ == "__main__":
    main_with_errors()
