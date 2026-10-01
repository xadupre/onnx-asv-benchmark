import numpy as np
import onnx_light.onnx.helper as oh
from onnx_light.onnx import TensorProto

from benchmarks.common import BACKENDS, run_session, setup_session


class AffineGrid:
    params = BACKENDS
    param_names = ("backend",)
    number = 3
    timeout = 10

    def setup(self, backend):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("AffineGrid", ["theta", "size"], ["grid"])],
                "affine_grid",
                [
                    oh.make_tensor_value_info("theta", TensorProto.FLOAT, [1, 2, 3]),
                    oh.make_tensor_value_info("size", TensorProto.INT64, [4]),
                ],
                [oh.make_tensor_value_info("grid", TensorProto.FLOAT, [1, 32, 32, 2])],
            ),
            opset_imports=[oh.make_opsetid("", 20)],
            ir_version=10,
        )
        feeds = {
            "theta": np.array([[[1, 0, 0.1], [0, 1, -0.1]]], dtype=np.float32),
            "size": np.array([1, 1, 32, 32], dtype=np.int64),
        }
        setup_session(self, backend, model, feeds)

    def time_run(self, backend):
        run_session(self)
