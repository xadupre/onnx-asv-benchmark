import numpy as np
import onnx_light.onnx.helper as oh
from onnx_light.onnx import TensorProto

from benchmarks.common import BACKENDS, run_session, setup_session


class MatMul:
    params = BACKENDS
    param_names = ("backend",)
    number = 1
    timeout = 10

    def setup(self, backend):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("MatMul", ["X", "Y"], ["Z"])],
                "matmul",
                [
                    oh.make_tensor_value_info("X", TensorProto.FLOAT, [256, 256]),
                    oh.make_tensor_value_info("Y", TensorProto.FLOAT, [256, 256]),
                ],
                [oh.make_tensor_value_info("Z", TensorProto.FLOAT, [256, 256])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
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
