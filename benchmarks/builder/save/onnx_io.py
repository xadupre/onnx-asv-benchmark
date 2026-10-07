from benchmarks.builder._onnx_io import (
    DTYPES,
    SAVE_CASES,
    SAVE_CPP_CASES,
    SHAPES,
    _OnnxCpp,
    _OnnxSave,
    case_parameters,
)


class OnnxSave(_OnnxSave):
    cases = SAVE_CASES
    params = (SHAPES, DTYPES, *case_parameters(cases))


class OnnxSaveCpp(_OnnxCpp):
    cases = SAVE_CPP_CASES
    params = (SHAPES, DTYPES, *case_parameters(cases))
