import onnx
import onnx_light
import onnxruntime


def track_numpy():
    return np.__version__


def track_onnx():
    return onnx.__version__


def track_onnxruntime():
    return onnxruntime.__version__


def track_onnx_light():
    return onnx_light.__version__
