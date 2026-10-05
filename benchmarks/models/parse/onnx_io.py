from benchmarks.models._onnx_io import DTYPES, PARSE_CASES, SHAPES, _OnnxBytes


class OnnxParse(_OnnxBytes):
    params = (SHAPES, DTYPES, PARSE_CASES)
