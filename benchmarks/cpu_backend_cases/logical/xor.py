from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class XorBoolInputs2(_CpuBackendCaseBenchmark):
    case_prefix = "xor"
    case_dtypes = ('bool', 'bool')
