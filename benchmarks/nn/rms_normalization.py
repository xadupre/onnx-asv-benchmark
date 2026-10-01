import numpy as np
import onnx_light.onnx.helper as oh
from onnx_light.onnx import TensorProto

from benchmarks.common import BACKENDS, run_session, setup_session


class RMSNormalization:
    params = BACKENDS
    param_names = ("backend",)
    number = 3
    timeout = 10

    def setup(self, backend):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("RMSNormalization", ["X", "scale"], ["Y"], axis=-1)],
                "rms_normalization",
                [
                    oh.make_tensor_value_info("X", TensorProto.FLOAT, [32, 256]),
                    oh.make_tensor_value_info("scale", TensorProto.FLOAT, [256]),
                ],
                [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [32, 256])],
            ),
            opset_imports=[oh.make_opsetid("", 23)],
            ir_version=10,
        )
        rng = np.random.default_rng(12)
        feeds = {
            "X": rng.standard_normal((32, 256), dtype=np.float32),
            "scale": rng.uniform(0.5, 1.5, 256).astype(np.float32),
        }
        setup_session(self, backend, model, feeds)

    def time_run(self, backend):
        run_session(self)
