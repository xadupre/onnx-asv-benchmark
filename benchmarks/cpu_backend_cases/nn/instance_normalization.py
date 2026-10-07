from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class InstancenormalizationBfloat16Bfloat16Bfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "instancenormalization"
    case_dtypes = ('bfloat16', 'bfloat16', 'bfloat16')


class InstancenormalizationFloat16Float16Float16(_CpuBackendCaseBenchmark):
    case_prefix = "instancenormalization"
    case_dtypes = ('float16', 'float16', 'float16')


class InstancenormalizationFloat32Float32Float32(_CpuBackendCaseBenchmark):
    case_prefix = "instancenormalization"
    case_dtypes = ('float32', 'float32', 'float32')
