from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class BatchnormalizationBfloat16Bfloat16Bfloat16Bfloat16Bfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "batchnormalization"
    case_dtypes = ('bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16')


class BatchnormalizationFloat16Float16Float16Float16Float16(_CpuBackendCaseBenchmark):
    case_prefix = "batchnormalization"
    case_dtypes = ('float16', 'float16', 'float16', 'float16', 'float16')


class BatchnormalizationFloat32Float32Float32Float32Float32(_CpuBackendCaseBenchmark):
    case_prefix = "batchnormalization"
    case_dtypes = ('float32', 'float32', 'float32', 'float32', 'float32')
