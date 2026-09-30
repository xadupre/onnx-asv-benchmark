import numpy as np
import onnx
import onnx_light.onnx as onnx_light
import onnxruntime
from onnx.reference import ReferenceEvaluator as OnnxReferenceEvaluator
from onnx_light.onnx.reference import ReferenceEvaluator as OnnxLightReferenceEvaluator

BACKENDS = ("onnxruntime", "onnx-reference", "onnx-light")


def setup_session(benchmark, backend, model, feeds):
    onnx.checker.check_model(model)
    expected = OnnxReferenceEvaluator(model).run(None, feeds)

    if backend == "onnxruntime":
        session = onnxruntime.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
    elif backend == "onnx-reference":
        session = OnnxReferenceEvaluator(model)
    else:
        light_model = onnx_light.load_model(model.SerializeToString())
        session = OnnxLightReferenceEvaluator(light_model)

    outputs = session.run(None, feeds)
    for output, expected_output in zip(outputs, expected, strict=True):
        np.testing.assert_allclose(output, expected_output, rtol=1e-4, atol=1e-4)

    benchmark.session = session
    benchmark.feeds = feeds


def run_session(benchmark):
    benchmark.session.run(None, benchmark.feeds)
