import numpy as np
from onnx import TensorProto, helper, numpy_helper

from benchmarks.common import BACKENDS, run_session, setup_session


class MatMulAdd:
    params = BACKENDS
    param_names = ("backend",)
    timeout = 120

    def setup(self, backend):
        rng = np.random.default_rng(0)
        weights = rng.standard_normal((256, 256), dtype=np.float32)
        bias = rng.standard_normal(256, dtype=np.float32)
        model = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node("MatMul", ["X", "weights"], ["matmul"]),
                    helper.make_node("Add", ["matmul", "bias"], ["Y"]),
                ],
                "matmul_add",
                [helper.make_tensor_value_info("X", TensorProto.FLOAT, [64, 256])],
                [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [64, 256])],
                [
                    numpy_helper.from_array(weights, "weights"),
                    numpy_helper.from_array(bias, "bias"),
                ],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
            ir_version=10,
        )
        feeds = {"X": rng.standard_normal((64, 256), dtype=np.float32)}
        setup_session(self, backend, model, feeds)

    def time_run(self, backend):
        run_session(self)
