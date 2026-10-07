from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class RMSNormalizationBfloat16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "rms"
    case_dtypes = ('bfloat16', 'bfloat16')


class RMSNormalizationFloat16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "rms"
    case_dtypes = ('float16', 'float16')


class RMSNormalizationFloat32Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "rms"
    case_dtypes = ('float32', 'float32')
