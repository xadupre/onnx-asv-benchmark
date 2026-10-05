from benchmarks.models._onnx_io import (
    DTYPES,
    LOAD_CASES,
    LOAD_CPP_CASES,
    SHAPES,
    _OnnxCpp,
    _OnnxLoad,
)


class OnnxLoad(_OnnxLoad):
    params = (SHAPES, DTYPES, LOAD_CASES)


class OnnxLoadCpp(_OnnxCpp):
    params = (SHAPES, DTYPES, LOAD_CPP_CASES)
