import numpy as np
import onnx_light.onnx.helper as oh
from onnx_light.onnx import TensorProto

from benchmarks.common import BACKENDS, run_session, setup_session


class ReduceSum:
    params = BACKENDS
    param_names = ("backend",)
    number = 4
    timeout = 10

    def setup(self, backend):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("ReduceSum", ["X", "axes"], ["Y"], keepdims=0)],
                "reduce_sum",
                [
                    oh.make_tensor_value_info("X", TensorProto.FLOAT, [64, 32, 64]),
                    oh.make_tensor_value_info("axes", TensorProto.INT64, [1]),
                ],
                [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [64, 64])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        rng = np.random.default_rng(11)
        feeds = {
            "X": rng.standard_normal((64, 32, 64), dtype=np.float32),
            "axes": np.array([1], dtype=np.int64),
        }
        setup_session(self, backend, model, feeds)

    def time_run(self, backend):
        run_session(self)
