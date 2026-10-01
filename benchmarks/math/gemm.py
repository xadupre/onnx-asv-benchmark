import numpy as np
import onnx_light.onnx.helper as oh
from onnx_light.onnx import TensorProto

from benchmarks.common import BACKENDS, run_session, setup_session


class Gemm:
    params = BACKENDS
    param_names = ("backend",)
    number = 2
    timeout = 10

    def setup(self, backend):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Gemm", ["X", "W", "B"], ["Y"])],
                "gemm",
                [
                    oh.make_tensor_value_info("X", TensorProto.FLOAT, [64, 128]),
                    oh.make_tensor_value_info("W", TensorProto.FLOAT, [128, 256]),
                    oh.make_tensor_value_info("B", TensorProto.FLOAT, [256]),
                ],
                [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [64, 256])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        rng = np.random.default_rng(5)
        feeds = {
            "X": rng.standard_normal((64, 128), dtype=np.float32),
            "W": rng.standard_normal((128, 256), dtype=np.float32),
            "B": rng.standard_normal(256, dtype=np.float32),
        }
        setup_session(self, backend, model, feeds)

    def time_run(self, backend):
        run_session(self)
