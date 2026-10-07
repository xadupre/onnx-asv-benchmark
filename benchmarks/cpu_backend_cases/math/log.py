from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class LogBfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "log"
    case_dtypes = ('bfloat16',)


class LogFloat16(_CpuBackendCaseBenchmark):
    case_prefix = "log"
    case_dtypes = ('float16',)


class LogFloat32(_CpuBackendCaseBenchmark):
    case_prefix = "log"
    case_dtypes = ('float32',)


class LogFloat64(_CpuBackendCaseBenchmark):
    case_prefix = "log"
    case_dtypes = ('float64',)
