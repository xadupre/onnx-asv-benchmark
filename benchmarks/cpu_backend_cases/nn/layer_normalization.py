from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class LayernormalizationBfloat16Bfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "layernormalization"
    case_dtypes = ('bfloat16', 'bfloat16')


class LayernormalizationFloat16Float16(_CpuBackendCaseBenchmark):
    case_prefix = "layernormalization"
    case_dtypes = ('float16', 'float16')


class LayernormalizationFloat32Float32(_CpuBackendCaseBenchmark):
    case_prefix = "layernormalization"
    case_dtypes = ('float32', 'float32')


class LayernormalizationFloat32Float32Float32(_CpuBackendCaseBenchmark):
    case_prefix = "layernormalization"
    case_dtypes = ('float32', 'float32', 'float32')
