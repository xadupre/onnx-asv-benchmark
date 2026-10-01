import numpy as np
import onnx_light.onnx.helper as oh
from onnx_light.onnx import TensorProto

from benchmarks.common import BACKENDS, run_session, setup_session


class Reciprocal:
    params = BACKENDS
    param_names = ("backend",)
    number = 5
    timeout = 10

    def setup(self, backend):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Reciprocal", ["X"], ["Y"])],
                "reciprocal",
                [oh.make_tensor_value_info("X", TensorProto.FLOAT, [1024, 1024])],
                [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [1024, 1024])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        rng = np.random.default_rng(7)
        feeds = {"X": rng.uniform(0.5, 2, (1024, 1024)).astype(np.float32)}
        setup_session(self, backend, model, feeds)

    def time_run(self, backend):
        run_session(self)
