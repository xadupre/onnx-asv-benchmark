from benchmarks.builder._onnx_io import (
    DTYPES,
    PARSE_CASES,
    SHAPES,
    _OnnxBytes,
    case_parameters,
)


class OnnxParse(_OnnxBytes):
    cases = PARSE_CASES
    params = (SHAPES, DTYPES, *case_parameters(cases))
