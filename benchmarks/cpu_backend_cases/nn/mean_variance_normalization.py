from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class MeanVarianceNormalizationBfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "meanvariancenormalization"
    case_dtypes = ('bfloat16',)


class MeanVarianceNormalizationFloat16(_CpuBackendCaseBenchmark):
    case_prefix = "meanvariancenormalization"
    case_dtypes = ('float16',)


class MeanVarianceNormalizationFloat32(_CpuBackendCaseBenchmark):
    case_prefix = "meanvariancenormalization"
    case_dtypes = ('float32',)
