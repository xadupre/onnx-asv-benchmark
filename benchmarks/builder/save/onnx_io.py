from benchmarks.builder._onnx_io import (
    DTYPES,
    SAVE_CASES,
    SAVE_CPP_CASES,
    SHAPES,
    _OnnxCpp,
    _OnnxSave,
)


class OnnxSave(_OnnxSave):
    params = (SHAPES, DTYPES, SAVE_CASES)


class OnnxSaveCpp(_OnnxCpp):
    params = (SHAPES, DTYPES, SAVE_CPP_CASES)
