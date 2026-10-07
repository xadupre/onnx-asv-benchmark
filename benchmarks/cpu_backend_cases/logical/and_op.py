from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class AndBoolInputs2(_CpuBackendCaseBenchmark):
    case_prefix = "and"
    case_dtypes = ('bool', 'bool')
