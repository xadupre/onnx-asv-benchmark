import numpy as np
from onnx import TensorProto, helper

from benchmarks.common import BACKENDS, run_session, setup_session


class Relu:
    params = BACKENDS
    param_names = ("backend",)

    def setup(self, backend):
        model = helper.make_model(
            helper.make_graph(
                [helper.make_node("Relu", ["X"], ["Y"])],
                "relu",
                [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1024, 1024])],
                [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1024, 1024])],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
            ir_version=10,
        )
        rng = np.random.default_rng(4)
        feeds = {"X": rng.standard_normal((1024, 1024), dtype=np.float32)}
        setup_session(self, backend, model, feeds)

    def time_run(self, backend):
        run_session(self)
