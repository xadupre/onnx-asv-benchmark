from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class InstanceNormalizationBfloat16Inputs3(_CpuBackendCaseBenchmark):
    case_prefix = "instancenormalization"
    case_dtypes = ('bfloat16', 'bfloat16', 'bfloat16')


class InstanceNormalizationFloat16Inputs3(_CpuBackendCaseBenchmark):
    case_prefix = "instancenormalization"
    case_dtypes = ('float16', 'float16', 'float16')


class InstanceNormalizationFloat32Inputs3(_CpuBackendCaseBenchmark):
    case_prefix = "instancenormalization"
    case_dtypes = ('float32', 'float32', 'float32')
