import argparse
import keyword
import re
from pathlib import Path

from onnx import defs
from onnx_light.onnx_py._onnxpybackend import backend_test

DISPATCH_PATH = Path("onnx_light/onnx_extensions/kernels/kernel_dispatch_table.cc")
HEADERS_PATH = Path("include/onnx_light/onnx_extensions/kernels/kernels")
ENTRY_PATTERN = re.compile(
    r'\{"([^:]+(?:\.[^:]+)*):([A-Za-z0-9_]+)",\s*'
    r"MakeKernel<onnx_kernels::kernel::([A-Za-z0-9_]+)>\(\)\}"
)
SUPPORTED_DTYPES = {
    "tensor(float16)": "float16",
    "tensor(float)": "float32",
    "tensor(double)": "float64",
    "tensor(bfloat16)": "bfloat16",
    "tensor(uint8)": "uint8",
    "tensor(uint16)": "uint16",
    "tensor(uint32)": "uint32",
    "tensor(uint64)": "uint64",
    "tensor(int8)": "int8",
    "tensor(int16)": "int16",
    "tensor(int32)": "int32",
    "tensor(int64)": "int64",
}
TENSOR_DTYPES = {
    1: "float32",
    2: "uint8",
    3: "int8",
    4: "uint16",
    5: "int16",
    6: "int32",
    7: "int64",
    10: "float16",
    11: "float64",
    12: "uint32",
    13: "uint64",
    16: "bfloat16",
}
SCHEMA_NAMES = {(schema.domain, schema.name) for schema in defs.get_all_schemas()}


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


def _operator_dtypes(case):
    if len(case.model.graph.node) != 1:
        return ()
    node = case.model.graph.node[0]
    if (node.domain, node.op_type) not in SCHEMA_NAMES:
        return ()
    opsets = {opset.domain: opset.version for opset in case.model.opset_import}
    schema = defs.get_schema(
        node.op_type,
        max_inclusive_version=opsets.get(node.domain, opsets.get("", None)),
        domain=node.domain,
    )
    if not schema.outputs:
        return ()
    type_parameter = schema.outputs[0].type_str
    constraint = next(
        (
            constraint
            for constraint in schema.type_constraints
            if constraint.type_param_str == type_parameter
        ),
        None,
    )
    if constraint is None:
        return ()
    supported_dtypes = tuple(
        SUPPORTED_DTYPES[value]
        for value in constraint.allowed_type_strs
        if value in SUPPORTED_DTYPES
    )
    if not supported_dtypes:
        return ()

    input_types = {}
    for index, name in enumerate(node.input):
        if not name:
            continue
        if index < len(schema.inputs):
            formal = schema.inputs[index]
        elif (
            schema.inputs
            and schema.inputs[-1].option == defs.OpSchema.FormalParameterOption.Variadic
        ):
            formal = schema.inputs[-1]
        else:
            continue
        input_types[name] = formal.type_str
    source = next(
        (
            TENSOR_DTYPES.get(tensor.data_type)
            for tensor in case.data_sets[0].inputs
            if input_types.get(tensor.name) == type_parameter
        ),
        None,
    )
    if source is None:
        source = next(
            (
                TENSOR_DTYPES.get(tensor.data_type)
                for tensor in case.model.graph.initializer
                if input_types.get(tensor.name) == type_parameter
            ),
            None,
        )
    if source is None:
        return ()
    return (source, *(dtype for dtype in supported_dtypes if dtype != source))


def _case(operator):
    if operator == "QuantizePagedCache":
        return None, None, None, ()
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
        return (
            direct[0].name,
            "BENCHMARK",
            _shape_label(direct[0]),
            _operator_dtypes(direct[0]),
        )
    if benchmark_cases:
        return (
            benchmark_cases[0].name,
            "BENCHMARK",
            _shape_label(benchmark_cases[0]),
            _operator_dtypes(benchmark_cases[0]),
        )

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
        return (
            direct[0].name,
            "TEST",
            _shape_label(direct[0]),
            _operator_dtypes(direct[0]),
        )
    if test_cases:
        return (
            test_cases[0].name,
            "TEST",
            _shape_label(test_cases[0]),
            _operator_dtypes(test_cases[0]),
        )
    raise RuntimeError(f"No backend test case found for {operator}.")


def _shape_label(case):
    parts = []
    data_set = case.data_sets[0]
    for index, tensor in enumerate(data_set.inputs):
        name = tensor.name or f"input_{index}"
        if case.name == "test_cc_sequence_insert_benchmark" and index == 8:
            name = "tensor"
        elif case.name == "test_cc_split_to_sequence_1_benchmark" and index == 0:
            name = "input"
        dimensions = "x".join(map(str, tensor.shape)) if tensor.shape else "scalar"
        parts.append(f"{name}={dimensions}")
    for index, value in enumerate(data_set.maps):
        name = value.name or f"map_{index}"
        key_shape = (
            "x".join(map(str, value.keys.shape)) if value.keys.shape else "scalar"
        )
        value_shape = (
            "x".join(map(str, value.values.shape)) if value.values.shape else "scalar"
        )
        parts.append(f"{name}=map[{key_shape}->{value_shape}]")
    return ", ".join(parts) or "no inputs"


def _write_module(path, class_name, case_name, case_mode, domain, dtypes):
    if class_name == "QuantizePagedCache":
        path.write_text(
            "from benchmarks._operator import "
            "QuantizePagedCacheBenchmark as _QuantizePagedCacheBenchmark\n\n\n"
            "class QuantizePagedCache(_QuantizePagedCacheBenchmark):\n"
            "    pass\n",
            encoding="utf-8",
        )
        return
    selected = ["onnxruntime", "onnx-reference", "onnx-light"]
    quoted_backends = ", ".join(f'"{backend}"' for backend in selected)
    backends = f"({quoted_backends}{',' if len(selected) == 1 else ''})"
    dtype_line = ""
    if dtypes:
        compact = f"    dtypes = {dtypes!r}\n"
        if len(compact.rstrip()) <= 88:
            dtype_line = compact
        else:
            values = "".join(f'        "{dtype}",\n' for dtype in dtypes)
            dtype_line = f"    dtypes = (\n{values}    )\n"
    path.write_text(
        "from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark\n\n\n"
        f"class {class_name}(_OperatorBenchmark):\n"
        f'    operator = "{class_name}"\n'
        f'    case_name = "{case_name}"\n'
        f'    case_mode = "{case_mode}"\n'
        f"{dtype_line}"
        f"    backends = {backends}\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_light", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "benchmarks" / "ops",
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

    shapes = {}
    for domain, operator, class_name in sorted(entries):
        case_name, case_mode, shape, dtypes = _case(operator)
        path = (
            args.output
            / _category(source / HEADERS_PATH, class_name)
            / f"{_snake_case(operator)}.py"
        )
        _write_module(path, operator, case_name, case_mode, domain, dtypes)
        if case_name is not None:
            shapes[case_name] = shape
    shape_path = args.output.parent / "_operator_shapes.py"
    lines = ["OPERATOR_INPUT_SHAPES = {"]
    lines.extend(
        f"    {case_name!r}: {shape!r}," for case_name, shape in sorted(shapes.items())
    )
    lines.append("}")
    shape_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
