from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class NonzeroBool(_CpuBackendCaseBenchmark):
    case_prefix = "nonzero"
    case_dtypes = ('bool',)
