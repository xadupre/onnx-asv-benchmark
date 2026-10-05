from benchmarks.models._onnx_io import (
    DTYPES,
    SERIALIZE_CASES,
    SHAPES,
    _OnnxBytes,
)


class OnnxSerialize(_OnnxBytes):
    params = (SHAPES, DTYPES, SERIALIZE_CASES)
