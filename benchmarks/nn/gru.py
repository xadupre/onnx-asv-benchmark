import numpy as np
import onnx_light.onnx.helper as oh
from onnx_light.onnx import TensorProto

from benchmarks.common import BACKENDS, run_session, setup_session


class Gru:
    params = BACKENDS
    param_names = ("backend",)
    number = 1
    timeout = 10

    def setup(self, backend):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("GRU", ["X", "W", "R", "B"], ["Y"], hidden_size=32)],
                "gru",
                [
                    oh.make_tensor_value_info("X", TensorProto.FLOAT, [8, 4, 16]),
                    oh.make_tensor_value_info("W", TensorProto.FLOAT, [1, 96, 16]),
                    oh.make_tensor_value_info("R", TensorProto.FLOAT, [1, 96, 32]),
                    oh.make_tensor_value_info("B", TensorProto.FLOAT, [1, 192]),
                ],
                [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [8, 1, 4, 32])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        rng = np.random.default_rng(9)
        feeds = {
            "X": rng.standard_normal((8, 4, 16), dtype=np.float32),
            "W": rng.standard_normal((1, 96, 16), dtype=np.float32) * 0.1,
            "R": rng.standard_normal((1, 96, 32), dtype=np.float32) * 0.1,
            "B": rng.standard_normal((1, 192), dtype=np.float32) * 0.1,
        }
        setup_session(self, backend, model, feeds)

    def time_run(self, backend):
        run_session(self)
