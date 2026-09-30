import numpy as np
import onnx_light.onnx.helper as oh
from onnx_light.onnx import TensorProto, numpy_helper

from benchmarks.common import BACKENDS, run_session, setup_session


class MLP:
    params = BACKENDS
    param_names = ("backend",)
    number = 2
    timeout = 10

    def setup(self, backend):
        rng = np.random.default_rng(1)
        weights1 = rng.standard_normal((128, 256), dtype=np.float32)
        bias1 = rng.standard_normal(256, dtype=np.float32)
        weights2 = rng.standard_normal((256, 64), dtype=np.float32)
        bias2 = rng.standard_normal(64, dtype=np.float32)
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("MatMul", ["X", "weights1"], ["hidden_matmul"]),
                    oh.make_node("Add", ["hidden_matmul", "bias1"], ["hidden_bias"]),
                    oh.make_node("Relu", ["hidden_bias"], ["hidden"]),
                    oh.make_node("MatMul", ["hidden", "weights2"], ["output_matmul"]),
                    oh.make_node("Add", ["output_matmul", "bias2"], ["Y"]),
                ],
                "mlp",
                [oh.make_tensor_value_info("X", TensorProto.FLOAT, [32, 128])],
                [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [32, 64])],
                [
                    numpy_helper.from_array(weights1, "weights1"),
                    numpy_helper.from_array(bias1, "bias1"),
                    numpy_helper.from_array(weights2, "weights2"),
                    numpy_helper.from_array(bias2, "bias2"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        feeds = {"X": rng.standard_normal((32, 128), dtype=np.float32)}
        setup_session(self, backend, model, feeds)

    def time_run(self, backend):
        run_session(self)
