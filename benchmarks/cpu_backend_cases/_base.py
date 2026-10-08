from functools import lru_cache
from itertools import product
import os
import re

import numpy as np
import onnxruntime
from onnx_light.onnx.backend import TestMode, collect_test_cases_by_name
from onnx_light.onnx.helper import tensor_dtype_to_np_dtype
from onnx_light.onnx.reference import ReferenceEvaluator
from onnx_light_cpu import (
    clear_used_kernel_names,
    register_backend_test_cases,
    register_kernels_for_session,
    registered_kernels,
    set_kernel_usage_recording,
    used_kernel_names,
)

BACKENDS = ("onnx-light-cpu", "onnxruntime")
CPU_COUNT = (
    len(os.sched_getaffinity(0))
    if hasattr(os, "sched_getaffinity")
    else os.cpu_count() or 1
)


def _to_numpy(tensor):
    dtype = tensor_dtype_to_np_dtype(int(tensor.data_type))
    shape = tuple(int(dimension) for dimension in tensor.shape)
    return np.frombuffer(tensor.raw_data(), dtype=dtype).reshape(shape)


def _simplified_case_name(name, dtypes):
    value = name.removeprefix("test_cpu_").removesuffix("_benchmark")
    for dtype in sorted(set(dtypes), key=len, reverse=True):
        value = value.replace(dtype, "")
    value = re.sub(r"_to_(?=_|$)", "_", value)
    value = re.sub(r"_n\d+(?=_|$)", "_", value)
    value = value.replace("_swapped", "")
    value = re.sub(r"(^|_)x(?=_|$)", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


def _case_metadata(case):
    inputs = case.data_sets[0].inputs
    dtypes = tuple(
        str(tensor_dtype_to_np_dtype(int(tensor.data_type))) for tensor in inputs
    )
    shapes = " x ".join(
        str(tuple(int(dimension) for dimension in tensor.shape))
        for tensor in inputs
    )
    return _simplified_case_name(case.name, dtypes), dtypes, shapes


@lru_cache(maxsize=1)
def _all_case_records():
    register_backend_test_cases()
    cases = collect_test_cases_by_name(
        "^test_cpu_.*_benchmark$",
        mode=TestMode.BENCHMARK,
        generate_benchmark_expected_outputs=False,
    )
    records = []
    for case in cases:
        records.append((case.name, *_case_metadata(case)))
        case.unload()
    return tuple(sorted(records))


def _case_records(prefix, dtypes, shard_index=0):
    marker = f"test_cpu_{prefix}_"
    selected = tuple(
        record
        for record in _all_case_records()
        if record[0].startswith(marker) and record[2] == dtypes
    )
    if not selected:
        raise RuntimeError(
            f"No onnx-light-cpu BENCHMARK cases found for {prefix!r} and {dtypes!r}."
        )
    return selected[shard_index::4] if len(dtypes) == 2 else selected


def _load_case(name):
    if re.fullmatch(r"[A-Za-z0-9_]+", name) is None:
        raise ValueError(f"Unexpected backend case name {name!r}.")
    cases = collect_test_cases_by_name(
        f"^{name}$",
        mode=TestMode.BENCHMARK,
        generate_benchmark_expected_outputs=False,
    )
    if len(cases) != 1:
        raise RuntimeError(
            f"Expected one backend case named {name!r}, got {len(cases)}."
        )
    return cases[0]


def _feeds(case):
    input_names = [value.name for value in case.model.graph.input]
    cpu_feeds = [
        dict(zip(input_names, data_set.inputs, strict=True))
        for data_set in case.data_sets
    ]
    numpy_feeds = [
        {name: _to_numpy(tensor) for name, tensor in feed.items()} for feed in cpu_feeds
    ]
    return cpu_feeds, numpy_feeds


def _expected_kernel_names(model):
    node = model.graph.node[0]
    domain = node.domain or "ai.onnx"
    return {
        kernel.kernel_name
        for kernel in registered_kernels()
        if kernel.domain == domain and kernel.op_type == node.op_type
    }


def _prepare_case(case_name, backend):
    case = _load_case(case_name)
    cpu_feeds, numpy_feeds = _feeds(case)
    model = case.model
    if backend == "onnx-light-cpu":
        session = ReferenceEvaluator(
            model,
            cpu_execution={
                "num_threads": CPU_COUNT,
                "affinity_policy": "none",
            },
        )
        register_kernels_for_session(session)
        set_kernel_usage_recording(session, True)
        clear_used_kernel_names(session)
        for feed in cpu_feeds:
            session.run(None, feed)
        expected = _expected_kernel_names(model)
        if not expected:
            raise RuntimeError(
                f"{case_name}: no onnx-light-cpu kernel is registered for "
                f"{model.graph.node[0].op_type}."
            )
        used = set(used_kernel_names(session))
        if expected.isdisjoint(used):
            raise RuntimeError(
                f"{case_name}: expected one of {sorted(expected)!r}, "
                f"used {sorted(used)!r}."
            )
        set_kernel_usage_recording(session, False)
        feeds = cpu_feeds
    elif backend == "onnxruntime":
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = CPU_COUNT
        options.inter_op_num_threads = 1
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        session = onnxruntime.InferenceSession(
            model.SerializeToString(),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        for feed in numpy_feeds:
            session.run(None, feed)
        feeds = numpy_feeds
    else:
        raise ValueError(f"Unexpected backend {backend!r}.")
    return case, feeds, session


def _make_benchmark(records):
    input_count = len(records[0][2])
    if any(len(dtypes) != input_count for _, _, dtypes, _ in records):
        raise RuntimeError("Grouped backend cases have different input counts.")

    lookup = {}
    for name, _, dtypes, shapes in records:
        lookup.setdefault((*dtypes, shapes), name)

    axes = [
        tuple(sorted({dtypes[index] for _, _, dtypes, _ in records}))
        for index in range(input_count)
    ]
    axes.append(tuple(sorted({shapes for _, _, _, shapes in records})))
    valid = set(lookup)
    skipped = [
        (*values, backend)
        for values in product(*axes)
        if values not in valid
        for backend in BACKENDS
    ]
    state = {}

    def setup(*parameters):
        *inputs, backend = parameters
        case_name = lookup[tuple(inputs)]
        state["case"], state["feeds"], state["session"] = _prepare_case(
            case_name, backend
        )

    def time_case(self, *parameters):
        for feed in state["feeds"]:
            state["session"].run(None, feed)

    def teardown(*parameters):
        del state["session"]
        del state["feeds"]
        state["case"].unload()
        del state["case"]

    time_case.params = (*axes, BACKENDS)
    time_case.param_names = (
        *(f"dtype{index + 1}" for index in range(input_count)),
        "input_shapes",
        "backend",
    )
    time_case.skip_params = skipped
    time_case.setup = setup
    time_case.teardown = teardown
    return records[0][1], time_case


class _CpuBackendCaseBenchmark:
    number = 1
    timeout = 60
    case_prefix = None
    case_dtypes = ()
    case_shard_index = 0

    def __init_subclass__(cls):
        super().__init_subclass__()
        if cls.case_prefix is None:
            return
        grouped = {}
        for record in _case_records(cls.case_prefix, cls.case_dtypes, cls.case_shard_index):
            grouped.setdefault(record[1], []).append(record)
        for records in grouped.values():
            simplified_name, benchmark = _make_benchmark(records)
            benchmark.__name__ = f"time_{simplified_name}"
            setattr(cls, benchmark.__name__, benchmark)


def create_benchmark_classes(namespace, category):
    from benchmarks.cpu_backend_cases._manifest import CASE_SHARDS

    module_name = namespace["__name__"]
    for shard_category, class_name, prefix, dtypes, shard_index in CASE_SHARDS:
        if shard_category != category:
            continue
        namespace[class_name] = type(
            class_name,
            (_CpuBackendCaseBenchmark,),
            {
                "__module__": module_name,
                "case_prefix": prefix,
                "case_dtypes": dtypes,
                "case_shard_index": shard_index,
            },
        )
