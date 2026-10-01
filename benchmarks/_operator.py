from functools import lru_cache

import numpy as np
import onnx
import onnxruntime
from onnx.reference import ReferenceEvaluator as OnnxReferenceEvaluator
from onnx_light import onnx as onnx_light
import onnx_light.onnx.helper as oh
import onnx_light.onnx.numpy_helper as onh
from onnx_light.onnx.reference import ReferenceEvaluator as OnnxLightReferenceEvaluator
from onnx_light.onnx_py._onnxpybackend import backend_test
from onnx_light.onnx_py._onnxpykernels.runtime import (
    RuntimeContext,
    RuntimeSession,
    tensor_from_proto,
)

NONDETERMINISTIC_OPERATORS = {
    "Bernoulli",
    "Multinomial",
    "RandomNormal",
    "RandomNormalLike",
    "RandomUniform",
    "RandomUniformLike",
}


def _tensor_to_array(tensor):
    if tensor.data_type == 8:
        return np.asarray(tensor.string_data(), dtype=np.str_).reshape(tensor.shape)
    return np.array(np.from_dlpack(tensor), copy=True)


def _map_to_dict(value):
    keys = _tensor_to_array(value.keys).reshape(-1)
    values = _tensor_to_array(value.values).reshape(-1)
    return dict(zip(keys.tolist(), values.tolist(), strict=True))


@lru_cache(maxsize=1)
def _load_case(name, mode):
    test_mode = getattr(backend_test.TestMode, mode)
    cases = backend_test.get_test_case_by_name(
        name,
        mode=test_mode,
        generate_benchmark_expected_outputs=True,
    )
    if len(cases) != 1:
        raise RuntimeError(
            f"Expected one backend test case named {name!r}, got {len(cases)}."
        )
    case = cases[0]
    data_set = case.data_sets[0]
    if name == "test_cc_sequence_insert_benchmark":
        case.model.graph.input[8].name = "tensor"
        inputs = list(case.model.graph.node[1].input)
        inputs[1] = "tensor"
        case.model.graph.node[1].input.clear()
        case.model.graph.node[1].input.extend(inputs)
        data_set.inputs[8].name = "tensor"
    elif name == "test_cc_split_to_sequence_1_benchmark":
        case.model.graph.input[0].name = "input"
        inputs = list(case.model.graph.node[0].input)
        inputs[0] = "input"
        case.model.graph.node[0].input.clear()
        case.model.graph.node[0].input.extend(inputs)
        data_set.inputs[0].name = "input"
    feeds = {value.name: _tensor_to_array(value) for value in data_set.inputs}
    feeds.update(
        {value.name: _map_to_dict(value) for value in data_set.maps if value.name}
    )
    expected = {
        value.name: _tensor_to_array(value) for value in data_set.outputs if value.name
    }
    expected.update(
        {value.name: _map_to_dict(value) for value in data_set.maps if value.name}
    )
    return case, feeds, expected


def _assert_value(actual, expected, rtol, atol):
    if isinstance(actual, list):
        if isinstance(expected, np.ndarray):
            if len(actual) != len(expected):
                raise AssertionError(
                    f"Sequence lengths differ: {len(actual)} != {len(expected)}."
                )
            for actual_item, expected_item in zip(actual, expected, strict=True):
                _assert_value(actual_item, expected_item, rtol, atol)
            return
        if len(actual) != len(expected):
            raise AssertionError(
                f"Sequence lengths differ: {len(actual)} != {len(expected)}."
            )
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_value(actual_item, expected_item, rtol, atol)
        return
    if isinstance(actual, dict):
        if actual.keys() != expected.keys():
            raise AssertionError(
                f"Map keys differ: {actual.keys()} != {expected.keys()}."
            )
        for key in actual:
            _assert_value(actual[key], expected[key], rtol, atol)
        return
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    if actual_array.dtype.kind in "OUS" or expected_array.dtype.kind in "OUS":
        np.testing.assert_array_equal(actual_array, expected_array)
    else:
        np.testing.assert_allclose(
            actual_array,
            expected_array,
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )


class OperatorBenchmark:
    number = 1
    timeout = 60
    operator = None
    case_name = None
    case_mode = "BENCHMARK"
    backends = ("onnx-light",)
    param_names = ("backend",)

    def __init_subclass__(cls):
        super().__init_subclass__()
        cls.params = cls.backends

    def setup(self, backend):
        case, feeds, expected_by_name = _load_case(self.case_name, self.case_mode)
        model_bytes = case.model.SerializeToString()
        if backend == "onnxruntime":
            session = onnxruntime.InferenceSession(
                model_bytes,
                providers=["CPUExecutionProvider"],
            )
        elif backend == "onnx-reference":
            session = OnnxReferenceEvaluator(onnx.load_model_from_string(model_bytes))
        elif backend == "onnx-light":
            session = OnnxLightReferenceEvaluator(case.model)
        else:
            raise ValueError(f"Unexpected backend {backend!r}.")

        outputs = session.run(None, feeds)
        output_names = [output.name for output in case.model.graph.output]
        if expected_by_name and self.operator not in NONDETERMINISTIC_OPERATORS:
            for name, output in zip(output_names, outputs, strict=True):
                _assert_value(
                    output,
                    expected_by_name[name],
                    rtol=max(case.rtol, 1e-2),
                    atol=max(case.atol, 1e-4),
                )

        self.case = case
        self.feeds = feeds
        self.session = session

    def time_run(self, backend):
        self.session.run(None, self.feeds)


class QuantizePagedCacheBenchmark:
    params = ("onnx-light",)
    param_names = ("backend",)
    number = 1
    timeout = 60

    def setup(self, backend):
        cache = onnx_light.PagedCacheProto()
        for page_index in range(16):
            block = cache.blocks.add()
            block.start = page_index * 32
            block.length = 32
            values = np.linspace(
                -1,
                1,
                8 * 32 * 64,
                dtype=np.float32,
            ).reshape(1, 8, 32, 64)
            block.key.CopyFrom(onh.from_array(values))
            block.value.CopyFrom(onh.from_array(-values))

        cache_type = onnx_light.TypeProto(struct_type=onnx_light.StructTypeProto())
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "QuantizePagedCache",
                        [
                            "past",
                            "indices",
                            "key_scale",
                            "key_zero",
                            "value_scale",
                            "value_zero",
                        ],
                        ["quantized"],
                        domain="ai.rt",
                    )
                ],
                "quantize_paged_cache",
                [
                    oh.make_value_info("past", cache_type),
                    oh.make_tensor_value_info(
                        "indices",
                        onnx_light.TensorProto.INT64,
                        [16],
                    ),
                ],
                [oh.make_value_info("quantized", cache_type)],
                initializer=[
                    oh.make_tensor(
                        "key_scale",
                        onnx_light.TensorProto.FLOAT,
                        [],
                        [0.125],
                    ),
                    oh.make_tensor(
                        "key_zero",
                        onnx_light.TensorProto.INT4,
                        [],
                        [0],
                    ),
                    oh.make_tensor(
                        "value_scale",
                        onnx_light.TensorProto.FLOAT,
                        [],
                        [0.125],
                    ),
                    oh.make_tensor(
                        "value_zero",
                        onnx_light.TensorProto.UINT4,
                        [],
                        [8],
                    ),
                ],
            ),
            opset_imports=[
                oh.make_opsetid("", 23),
                oh.make_opsetid("ai.rt", 1),
            ],
            ir_version=10,
        )
        context = RuntimeContext()
        context.put_value("past", cache)
        context.set(
            "indices",
            tensor_from_proto(onh.from_array(np.arange(16, dtype=np.int64))),
        )
        session = RuntimeSession(model)
        session.run(context)

        self.context = context
        self.model = model
        self.session = session

    def time_run(self, backend):
        self.session.run(self.context)
