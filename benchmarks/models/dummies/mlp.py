import numpy as np
import onnx_light.onnx.helper as oh
import onnx_light.onnx.numpy_helper as onh
from onnx_light.onnx import TensorProto

from benchmarks.common import BACKENDS, MODEL_DTYPES, run_session, setup_session


class MLP:
    params = (MODEL_DTYPES, BACKENDS)
    param_names = ("dtype", "backend")
    number = 2
    timeout = 10

    def setup(self, dtype, backend):
        numpy_dtype = np.dtype(dtype)
        rng = np.random.default_rng(1)
        weights1 = rng.standard_normal((128, 256), dtype=numpy_dtype)
        bias1 = rng.standard_normal(256, dtype=numpy_dtype)
        weights2 = rng.standard_normal((256, 64), dtype=numpy_dtype)
        bias2 = rng.standard_normal(64, dtype=numpy_dtype)
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
                [oh.make_tensor_value_info("X", TensorProto.DOUBLE, [32, 128])],
                [oh.make_tensor_value_info("Y", TensorProto.DOUBLE, [32, 64])],
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
        feeds = {"X": rng.standard_normal((32, 128), dtype=numpy_dtype)}
        setup_session(self, backend, model, feeds)

    def time_run(self, dtype, backend):
        run_session(self)
