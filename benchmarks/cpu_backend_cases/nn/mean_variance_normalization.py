from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class MeanvariancenormalizationBfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "meanvariancenormalization"
    case_dtypes = ('bfloat16',)


class MeanvariancenormalizationFloat16(_CpuBackendCaseBenchmark):
    case_prefix = "meanvariancenormalization"
    case_dtypes = ('float16',)


class MeanvariancenormalizationFloat32(_CpuBackendCaseBenchmark):
    case_prefix = "meanvariancenormalization"
    case_dtypes = ('float32',)
