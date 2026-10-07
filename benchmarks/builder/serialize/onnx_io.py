from benchmarks.builder._onnx_io import (
    DTYPES,
    SERIALIZE_CASES,
    SHAPES,
    _OnnxBytes,
    case_parameters,
)


class OnnxSerialize(_OnnxBytes):
    cases = SERIALIZE_CASES
    params = (SHAPES, DTYPES, *case_parameters(cases))
