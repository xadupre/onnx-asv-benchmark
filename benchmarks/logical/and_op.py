import numpy as np
import onnx_light.onnx.helper as oh
from onnx_light.onnx import TensorProto

from benchmarks.common import BACKENDS, run_session, setup_session


class And:
    params = BACKENDS
    param_names = ("backend",)
    number = 5
    timeout = 10

    def setup(self, backend):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("And", ["X", "Y"], ["Z"])],
                "and",
                [
                    oh.make_tensor_value_info("X", TensorProto.BOOL, [512, 512]),
                    oh.make_tensor_value_info("Y", TensorProto.BOOL, [512, 512]),
                ],
                [oh.make_tensor_value_info("Z", TensorProto.BOOL, [512, 512])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        rng = np.random.default_rng(10)
        feeds = {
            "X": rng.random((512, 512)) > 0.5,
            "Y": rng.random((512, 512)) > 0.5,
        }
        setup_session(self, backend, model, feeds)

    def time_run(self, backend):
        run_session(self)
