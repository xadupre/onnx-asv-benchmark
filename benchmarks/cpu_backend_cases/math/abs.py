from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class AbsBfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "abs"
    case_dtypes = ('bfloat16',)


class AbsFloat16(_CpuBackendCaseBenchmark):
    case_prefix = "abs"
    case_dtypes = ('float16',)


class AbsFloat32(_CpuBackendCaseBenchmark):
    case_prefix = "abs"
    case_dtypes = ('float32',)


class AbsFloat64(_CpuBackendCaseBenchmark):
    case_prefix = "abs"
    case_dtypes = ('float64',)


class AbsInt16(_CpuBackendCaseBenchmark):
    case_prefix = "abs"
    case_dtypes = ('int16',)


class AbsInt32(_CpuBackendCaseBenchmark):
    case_prefix = "abs"
    case_dtypes = ('int32',)


class AbsInt64(_CpuBackendCaseBenchmark):
    case_prefix = "abs"
    case_dtypes = ('int64',)


class AbsInt8(_CpuBackendCaseBenchmark):
    case_prefix = "abs"
    case_dtypes = ('int8',)
