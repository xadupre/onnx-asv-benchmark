import argparse
import keyword
import re
from pathlib import Path

from onnx_light.onnx_py._onnxpybackend import backend_test

DISPATCH_PATH = Path("onnx_light/onnx_extensions/kernels/kernel_dispatch_table.cc")
HEADERS_PATH = Path("include/onnx_light/onnx_extensions/kernels/kernels")
ENTRY_PATTERN = re.compile(
    r'\{"([^:]+(?:\.[^:]+)*):([A-Za-z0-9_]+)",\s*'
    r"MakeKernel<onnx_kernels::kernel::([A-Za-z0-9_]+)>\(\)\}"
)
UNSUPPORTED_BACKENDS = {
    "onnxruntime": {
        "Bernoulli",
        "CastMap",
        "CausalConvWithState",
        "DictVectorizer",
        "GlobalLpPool",
        "GRU",
        "ImageDecoder",
        "LayerNormalization",
        "LinearAttention",
        "LinearClassifier",
        "LSTM",
        "MaxRoiPool",
        "Multinomial",
        "RandomNormal",
        "RandomNormalLike",
        "RandomUniform",
        "RandomUniformLike",
        "RNN",
        "SwiGLU",
        "TreeEnsembleClassifier",
    },
    "onnx-reference": {
        "CastMap",
        "CategoryMapper",
        "ConvTranspose",
        "DeformConv",
        "GRU",
        "GridSample",
        "LSTM",
        "MaxRoiPool",
        "Multinomial",
        "Optional",
        "RNN",
        "Scatter",
    },
}


def _snake_case(name):
    value = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value).lower()
    return f"{value}_op" if keyword.iskeyword(value) else value


def _category(headers_root, class_name):
    declaration = re.compile(
        rf"\b(?:class|struct)\s+(?:ONNX_LIGHT\w+\s+)?{re.escape(class_name)}\b"
    )
    matches = [
        path.parent.name
        for path in headers_root.glob("*/*.h")
        if declaration.search(path.read_text(encoding="utf-8"))
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one category for {class_name}, found {matches}.")
    return matches[0]


def _case(operator):
    if operator == "QuantizePagedCache":
        return None, None
    benchmark_cases = backend_test.collect_test_cases(
        operator,
        mode=backend_test.TestMode.BENCHMARK,
    )
    direct = [
        case
        for case in benchmark_cases
        if len(case.model.graph.node) == 1
        and case.model.graph.node[0].op_type == operator
    ]
    if direct:
        return direct[0].name, "BENCHMARK"
    if benchmark_cases:
        return benchmark_cases[0].name, "BENCHMARK"

    test_cases = backend_test.collect_test_cases(
        operator,
        mode=backend_test.TestMode.TEST,
    )
    direct = [
        case
        for case in test_cases
        if len(case.model.graph.node) == 1
        and case.model.graph.node[0].op_type == operator
    ]
    if direct:
        return direct[0].name, "TEST"
    if test_cases:
        return test_cases[0].name, "TEST"
    raise RuntimeError(f"No backend test case found for {operator}.")


def _write_module(path, class_name, case_name, case_mode, domain):
    if class_name == "QuantizePagedCache":
        path.write_text(
            "from benchmarks._operator import QuantizePagedCacheBenchmark\n\n\n"
            "class QuantizePagedCache(QuantizePagedCacheBenchmark):\n"
            "    pass\n",
            encoding="utf-8",
        )
        return
    if domain == "ai.rt":
        selected = ["onnx-light"]
    elif domain == "ai.onnx.preview":
        selected = ["onnx-reference", "onnx-light"]
    elif domain == "ai.onnx.preview.training":
        selected = ["onnx-reference", "onnx-light"]
    else:
        selected = [
            backend
            for backend in ("onnxruntime", "onnx-reference", "onnx-light")
            if class_name not in UNSUPPORTED_BACKENDS.get(backend, set())
        ]
    quoted_backends = ", ".join(f'"{backend}"' for backend in selected)
    backends = f"({quoted_backends}{',' if len(selected) == 1 else ''})"
    path.write_text(
        "from benchmarks._operator import OperatorBenchmark\n\n\n"
        f"class {class_name}(OperatorBenchmark):\n"
        f'    operator = "{class_name}"\n'
        f'    case_name = "{case_name}"\n'
        f'    case_mode = "{case_mode}"\n'
        f"    backends = {backends}\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_light", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "benchmarks",
    )
    args = parser.parse_args()

    source = args.onnx_light.resolve()
    entries = ENTRY_PATTERN.findall(
        (source / DISPATCH_PATH).read_text(encoding="utf-8")
    )
    if not entries:
        raise RuntimeError(f"No kernels found in {source / DISPATCH_PATH}.")

    categories = {
        _category(source / HEADERS_PATH, class_name) for _, _, class_name in entries
    }
    for category in categories:
        directory = args.output / category
        directory.mkdir(exist_ok=True)
        for path in directory.glob("*.py"):
            path.unlink()
        (directory / "__init__.py").touch()

    for domain, operator, class_name in sorted(entries):
        case_name, case_mode = _case(operator)
        path = (
            args.output
            / _category(source / HEADERS_PATH, class_name)
            / f"{_snake_case(operator)}.py"
        )
        _write_module(path, operator, case_name, case_mode, domain)


if __name__ == "__main__":
    main()
