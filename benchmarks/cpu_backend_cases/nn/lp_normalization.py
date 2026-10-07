from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class LpNormalizationBfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "lpnormalization"
    case_dtypes = ('bfloat16',)


class LpNormalizationFloat16(_CpuBackendCaseBenchmark):
    case_prefix = "lpnormalization"
    case_dtypes = ('float16',)


class LpNormalizationFloat32(_CpuBackendCaseBenchmark):
    case_prefix = "lpnormalization"
    case_dtypes = ('float32',)
