from functools import lru_cache
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


@lru_cache(maxsize=1)
def _all_case_names():
    register_backend_test_cases()
    cases = collect_test_cases_by_name(
        "^test_cpu_.*_benchmark$",
        mode=TestMode.BENCHMARK,
        generate_benchmark_expected_outputs=False,
    )
    return tuple(sorted(case.name for case in cases))


def _case_names(prefix, start, stop):
    marker = f"test_cpu_{prefix}_"
    names = tuple(name for name in _all_case_names() if name.startswith(marker))
    selected = names[start:stop]
    if not selected:
        raise RuntimeError(
            f"No onnx-light-cpu BENCHMARK cases found for {prefix!r}[{start}:{stop}]."
        )
    return selected


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


class _CpuBackendCaseBenchmark:
    number = 1
    timeout = 60
    param_names = ("case", "backend")
    case_prefix = None
    case_start = 0
    case_stop = None

    def __init_subclass__(cls):
        super().__init_subclass__()
        if cls.case_prefix is None:
            return
        cls.params = (
            _case_names(cls.case_prefix, cls.case_start, cls.case_stop),
            BACKENDS,
        )

    def setup(self, case_name, backend):
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

        self.case = case
        self.feeds = feeds
        self.session = session

    def time_run(self, case_name, backend):
        for feed in self.feeds:
            self.session.run(None, feed)

    def teardown(self, case_name, backend):
        del self.session
        del self.feeds
        self.case.unload()
        del self.case
