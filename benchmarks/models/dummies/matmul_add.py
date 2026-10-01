import numpy as np
import onnx_light.onnx.helper as oh
import onnx_light.onnx.numpy_helper as onh
from onnx_light.onnx import TensorProto

from benchmarks.common import BACKENDS, MODEL_DTYPES, run_session, setup_session


class MatMulAdd:
    params = (MODEL_DTYPES, BACKENDS)
    param_names = ("dtype", "backend")
    number = 3
    timeout = 10

    def setup(self, dtype, backend):
        numpy_dtype = np.dtype(dtype)
        rng = np.random.default_rng(0)
        weights = rng.standard_normal((256, 256), dtype=numpy_dtype)
        bias = rng.standard_normal(256, dtype=numpy_dtype)
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("MatMul", ["X", "weights"], ["matmul"]),
                    oh.make_node("Add", ["matmul", "bias"], ["Y"]),
                ],
                "matmul_add",
                [oh.make_tensor_value_info("X", TensorProto.DOUBLE, [64, 256])],
                [oh.make_tensor_value_info("Y", TensorProto.DOUBLE, [64, 256])],
                [
                    onh.from_array(weights, "weights"),
                    onh.from_array(bias, "bias"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        feeds = {"X": rng.standard_normal((64, 256), dtype=numpy_dtype)}
        setup_session(self, backend, model, feeds)

    def time_run(self, dtype, backend):
        run_session(self)
