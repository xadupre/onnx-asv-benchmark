import numpy as np
import onnx
import onnxruntime
import onnx_light.onnx as onnx_light
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator as OnnxReferenceEvaluator
from onnx_light.onnx.reference import ReferenceEvaluator as OnnxLightReferenceEvaluator


BACKENDS = ["onnxruntime", "onnx-reference", "onnx-light", "onnx-light-cpu"]
MODELS = ["matmul-add", "mlp"]


def _make_matmul_add():
    rng = np.random.default_rng(0)
    weights = rng.standard_normal((256, 256), dtype=np.float32)
    bias = rng.standard_normal(256, dtype=np.float32)
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("MatMul", ["X", "weights"], ["matmul"]),
                helper.make_node("Add", ["matmul", "bias"], ["Y"]),
            ],
            "matmul_add",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [64, 256])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [64, 256])],
            [
                numpy_helper.from_array(weights, "weights"),
                numpy_helper.from_array(bias, "bias"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    model.ir_version = 10
    return model, {"X": rng.standard_normal((64, 256), dtype=np.float32)}


def _make_mlp():
    rng = np.random.default_rng(1)
    weights1 = rng.standard_normal((128, 256), dtype=np.float32)
    bias1 = rng.standard_normal(256, dtype=np.float32)
    weights2 = rng.standard_normal((256, 64), dtype=np.float32)
    bias2 = rng.standard_normal(64, dtype=np.float32)
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("MatMul", ["X", "weights1"], ["hidden_matmul"]),
                helper.make_node("Add", ["hidden_matmul", "bias1"], ["hidden_bias"]),
                helper.make_node("Relu", ["hidden_bias"], ["hidden"]),
                helper.make_node("MatMul", ["hidden", "weights2"], ["output_matmul"]),
                helper.make_node("Add", ["output_matmul", "bias2"], ["Y"]),
            ],
            "mlp",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [32, 128])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [32, 64])],
            [
                numpy_helper.from_array(weights1, "weights1"),
                numpy_helper.from_array(bias1, "bias1"),
                numpy_helper.from_array(weights2, "weights2"),
                numpy_helper.from_array(bias2, "bias2"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    model.ir_version = 10
    return model, {"X": rng.standard_normal((32, 128), dtype=np.float32)}


MODEL_FACTORIES = {"matmul-add": _make_matmul_add, "mlp": _make_mlp}


class Inference:
    params = (BACKENDS, MODELS)
    param_names = ("backend", "model")
    timeout = 120

    def setup(self, backend, model):
        onnx_model, self.feeds = MODEL_FACTORIES[model]()
        onnx.checker.check_model(onnx_model)
        expected = OnnxReferenceEvaluator(onnx_model).run(None, self.feeds)

        if backend == "onnxruntime":
            self.session = onnxruntime.InferenceSession(
                onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
            )
        elif backend == "onnx-reference":
            self.session = OnnxReferenceEvaluator(onnx_model)
        else:
            light_model = onnx_light.load_model(onnx_model.SerializeToString())
            self.session = OnnxLightReferenceEvaluator(light_model)
            if backend == "onnx-light-cpu":
                from onnx_light_cpu import register_kernels_for_session

                register_kernels_for_session(self.session)

        outputs = self.session.run(None, self.feeds)
        for output, expected_output in zip(outputs, expected, strict=True):
            np.testing.assert_allclose(output, expected_output, rtol=1e-4, atol=1e-4)

    def time_run(self, backend, model):
        self.session.run(None, self.feeds)
