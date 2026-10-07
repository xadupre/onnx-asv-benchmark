from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class ExpBfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "exp"
    case_dtypes = ('bfloat16',)


class ExpFloat16(_CpuBackendCaseBenchmark):
    case_prefix = "exp"
    case_dtypes = ('float16',)


class ExpFloat32(_CpuBackendCaseBenchmark):
    case_prefix = "exp"
    case_dtypes = ('float32',)


class ExpFloat64(_CpuBackendCaseBenchmark):
    case_prefix = "exp"
    case_dtypes = ('float64',)
