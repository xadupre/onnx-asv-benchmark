from benchmarks.builder._onnx_io import (
    DTYPES,
    LOAD_CASES,
    SHAPES,
    _OnnxLoad,
    case_parameters,
)


class OnnxLoad(_OnnxLoad):
    cases = LOAD_CASES
    params = (SHAPES, DTYPES, *case_parameters(cases))
