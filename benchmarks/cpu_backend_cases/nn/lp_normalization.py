from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class LpnormalizationBfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "lpnormalization"
    case_dtypes = ('bfloat16',)


class LpnormalizationFloat16(_CpuBackendCaseBenchmark):
    case_prefix = "lpnormalization"
    case_dtypes = ('float16',)


class LpnormalizationFloat32(_CpuBackendCaseBenchmark):
    case_prefix = "lpnormalization"
    case_dtypes = ('float32',)
