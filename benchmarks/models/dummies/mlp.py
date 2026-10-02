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


class MLP:
    params = (("X=32x128",), MODEL_DTYPES, BACKENDS)
    param_names = ("shape", "dtype", "backend")
    number = 2
    timeout = 10

    def setup(self, shape, dtype, backend):
        numpy_dtype = np.dtype(dtype)
        tensor_dtype = oh.np_dtype_to_tensor_dtype(numpy_dtype)
        rng = np.random.default_rng(1)
        weights1 = standard_normal(rng, (128, 256), numpy_dtype)
        bias1 = standard_normal(rng, 256, numpy_dtype)
        weights2 = standard_normal(rng, (256, 64), numpy_dtype)
        bias2 = standard_normal(rng, 64, numpy_dtype)
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("MatMul", ["X", "weights1"], ["hidden_matmul"]),
                    oh.make_node("Add", ["hidden_matmul", "bias1"], ["hidden_bias"]),
                    oh.make_node("Relu", ["hidden_bias"], ["hidden"]),
                    oh.make_node("MatMul", ["hidden", "weights2"], ["output_matmul"]),
                    oh.make_node("Add", ["output_matmul", "bias2"], ["Y"]),
                ],
                "mlp",
                [oh.make_tensor_value_info("X", tensor_dtype, [32, 128])],
                [oh.make_tensor_value_info("Y", tensor_dtype, [32, 64])],
                [
                    onh.from_array(weights1, "weights1"),
                    onh.from_array(bias1, "bias1"),
                    onh.from_array(weights2, "weights2"),
                    onh.from_array(bias2, "bias2"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        feeds = {"X": standard_normal(rng, (32, 128), numpy_dtype)}
        if shape != input_shape_label(feeds):
            raise ValueError(f"Unexpected input shape parameter {shape!r}.")
        setup_session(self, backend, model, feeds)

    def time_run(self, shape, dtype, backend):
        run_session(self)
