import numpy as np
from onnx_light.onnx import TensorProto, helper

from benchmarks.common import BACKENDS, run_session, setup_session


class MatMul:
    params = BACKENDS
    param_names = ("backend",)

    def setup(self, backend):
        model = helper.make_model(
            helper.make_graph(
                [helper.make_node("MatMul", ["X", "Y"], ["Z"])],
                "matmul",
                [
                    helper.make_tensor_value_info("X", TensorProto.FLOAT, [256, 256]),
                    helper.make_tensor_value_info("Y", TensorProto.FLOAT, [256, 256]),
                ],
                [helper.make_tensor_value_info("Z", TensorProto.FLOAT, [256, 256])],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
            ir_version=10,
        )
        rng = np.random.default_rng(3)
        feeds = {
            "X": rng.standard_normal((256, 256), dtype=np.float32),
            "Y": rng.standard_normal((256, 256), dtype=np.float32),
        }
        setup_session(self, backend, model, feeds)

    def time_run(self, backend):
        run_session(self)
