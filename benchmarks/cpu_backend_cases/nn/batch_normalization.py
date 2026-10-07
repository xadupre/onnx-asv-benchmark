from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class BatchNormalizationBfloat16Inputs5(_CpuBackendCaseBenchmark):
    case_prefix = "batchnormalization"
    case_dtypes = ('bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16')


class BatchNormalizationFloat16Inputs5(_CpuBackendCaseBenchmark):
    case_prefix = "batchnormalization"
    case_dtypes = ('float16', 'float16', 'float16', 'float16', 'float16')


class BatchNormalizationFloat32Inputs5(_CpuBackendCaseBenchmark):
    case_prefix = "batchnormalization"
    case_dtypes = ('float32', 'float32', 'float32', 'float32', 'float32')
