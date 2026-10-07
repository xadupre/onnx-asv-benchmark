from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class LessBfloat16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "less"
    case_dtypes = ('bfloat16', 'bfloat16')


class LessFloat16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "less"
    case_dtypes = ('float16', 'float16')


class LessFloat32Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "less"
    case_dtypes = ('float32', 'float32')


class LessInt16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "less"
    case_dtypes = ('int16', 'int16')


class LessInt32Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "less"
    case_dtypes = ('int32', 'int32')


class LessInt64Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "less"
    case_dtypes = ('int64', 'int64')


class LessInt8Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "less"
    case_dtypes = ('int8', 'int8')


class LessUint16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "less"
    case_dtypes = ('uint16', 'uint16')


class LessUint32Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "less"
    case_dtypes = ('uint32', 'uint32')


class LessUint64Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "less"
    case_dtypes = ('uint64', 'uint64')


class LessUint8Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "less"
    case_dtypes = ('uint8', 'uint8')
