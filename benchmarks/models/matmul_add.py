import numpy as np
import onnx_light.onnx.helper as oh
from onnx_light.onnx import TensorProto, numpy_helper

from benchmarks.common import BACKENDS, run_session, setup_session


class MatMulAdd:
    params = BACKENDS
    param_names = ("backend",)
    timeout = 120

    def setup(self, backend):
        rng = np.random.default_rng(0)
        weights = rng.standard_normal((256, 256), dtype=np.float32)
        bias = rng.standard_normal(256, dtype=np.float32)
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("MatMul", ["X", "weights"], ["matmul"]),
                    oh.make_node("Add", ["matmul", "bias"], ["Y"]),
                ],
                "matmul_add",
                [oh.make_tensor_value_info("X", TensorProto.FLOAT, [64, 256])],
                [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [64, 256])],
                [
                    numpy_helper.from_array(weights, "weights"),
                    numpy_helper.from_array(bias, "bias"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        feeds = {"X": rng.standard_normal((64, 256), dtype=np.float32)}
        setup_session(self, backend, model, feeds)

    def time_run(self, backend):
        run_session(self)
