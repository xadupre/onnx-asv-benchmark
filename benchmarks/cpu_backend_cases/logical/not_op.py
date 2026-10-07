from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class NotBool(_CpuBackendCaseBenchmark):
    case_prefix = "not"
    case_dtypes = ('bool',)
