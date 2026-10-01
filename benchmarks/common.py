import numpy as np
import onnx
import onnx_light.onnx.checker as onnx_light_checker
import onnxruntime
from onnx.reference import ReferenceEvaluator as OnnxReferenceEvaluator
from onnx_light.onnx.reference import ReferenceEvaluator as OnnxLightReferenceEvaluator
from onnx_light_cpu import register_kernels_for_session

BACKENDS = ("onnxruntime", "onnx-reference", "onnx-light", "onnx-light-cpu")


def setup_session(benchmark, backend, model, feeds):
    onnx_light_checker.check_model(model)
    model_bytes = model.SerializeToString()
    onnx_model = onnx.load_model_from_string(model_bytes)
    onnx.checker.check_model(onnx_model)
    expected = OnnxReferenceEvaluator(onnx_model).run(None, feeds)

    if backend == "onnxruntime":
        session = onnxruntime.InferenceSession(
            model_bytes, providers=["CPUExecutionProvider"]
        )
    elif backend == "onnx-reference":
        session = OnnxReferenceEvaluator(onnx_model)
    elif backend == "onnx-light":
        session = OnnxLightReferenceEvaluator(model)
    elif backend == "onnx-light-cpu":
        session = OnnxLightReferenceEvaluator(model)
        register_kernels_for_session(session)
    else:
        raise ValueError(f"Unexpected backend {backend!r}.")

    outputs = session.run(None, feeds)
    for output, expected_output in zip(outputs, expected, strict=True):
        np.testing.assert_allclose(output, expected_output, rtol=1e-4, atol=1e-4)

    benchmark.session = session
    benchmark.feeds = feeds


def run_session(benchmark):
    benchmark.session.run(None, benchmark.feeds)
