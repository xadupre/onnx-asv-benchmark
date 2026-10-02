import numpy as np
import onnx_light.onnx.helper as oh
import onnx_light.onnx.numpy_helper as onh

from benchmarks.common import (
    BACKENDS,
    MODEL_DTYPES,
    input_shape_label,
    run_session,
    setup_session,
    standard_normal,
)


class MatMulAdd:
    params = (("X=64x256",), MODEL_DTYPES, BACKENDS)
    param_names = ("shape", "dtype", "backend")
    number = 3
    timeout = 10

    def setup(self, shape, dtype, backend):
        numpy_dtype = np.dtype(dtype)
        tensor_dtype = oh.np_dtype_to_tensor_dtype(numpy_dtype)
        rng = np.random.default_rng(0)
        weights = standard_normal(rng, (256, 256), numpy_dtype)
        bias = standard_normal(rng, 256, numpy_dtype)
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("MatMul", ["X", "weights"], ["matmul"]),
                    oh.make_node("Add", ["matmul", "bias"], ["Y"]),
                ],
                "matmul_add",
                [oh.make_tensor_value_info("X", tensor_dtype, [64, 256])],
                [oh.make_tensor_value_info("Y", tensor_dtype, [64, 256])],
                [
                    onh.from_array(weights, "weights"),
                    onh.from_array(bias, "bias"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        feeds = {"X": standard_normal(rng, (64, 256), numpy_dtype)}
        if shape != input_shape_label(feeds):
            raise ValueError(f"Unexpected input shape parameter {shape!r}.")
        setup_session(self, backend, model, feeds)

    def time_run(self, shape, dtype, backend):
        run_session(self)
