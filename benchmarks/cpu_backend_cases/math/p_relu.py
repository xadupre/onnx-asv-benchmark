from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class PReluBfloat16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "prelu"
    case_dtypes = ('bfloat16', 'bfloat16')


class PReluFloat16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "prelu"
    case_dtypes = ('float16', 'float16')


class PReluFloat32Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "prelu"
    case_dtypes = ('float32', 'float32')


class PReluFloat64Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "prelu"
    case_dtypes = ('float64', 'float64')


class PReluInt32Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "prelu"
    case_dtypes = ('int32', 'int32')


class PReluInt64Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "prelu"
    case_dtypes = ('int64', 'int64')


class PReluUint32Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "prelu"
    case_dtypes = ('uint32', 'uint32')


class PReluUint64Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "prelu"
    case_dtypes = ('uint64', 'uint64')
