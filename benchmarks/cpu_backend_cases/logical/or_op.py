from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class OrBoolBool(_CpuBackendCaseBenchmark):
    case_prefix = "or"
    case_dtypes = ('bool', 'bool')
