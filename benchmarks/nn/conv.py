import numpy as np
import onnx_light.onnx.helper as oh
from onnx_light.onnx import TensorProto

from benchmarks.common import BACKENDS, run_session, setup_session


class Conv:
    params = BACKENDS
    param_names = ("backend",)
    number = 1
    timeout = 10

    def setup(self, backend):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Conv", ["X", "W", "B"], ["Y"])],
                "conv",
                [
                    oh.make_tensor_value_info("X", TensorProto.FLOAT, [1, 8, 32, 32]),
                    oh.make_tensor_value_info("W", TensorProto.FLOAT, [16, 8, 3, 3]),
                    oh.make_tensor_value_info("B", TensorProto.FLOAT, [16]),
                ],
                [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16, 30, 30])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        rng = np.random.default_rng(6)
        feeds = {
            "X": rng.standard_normal((1, 8, 32, 32), dtype=np.float32),
            "W": rng.standard_normal((16, 8, 3, 3), dtype=np.float32),
            "B": rng.standard_normal(16, dtype=np.float32),
        }
        setup_session(self, backend, model, feeds)

    def time_run(self, backend):
        run_session(self)
