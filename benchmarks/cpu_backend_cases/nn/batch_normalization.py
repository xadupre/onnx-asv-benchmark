from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class BatchNormalization(_CpuBackendCaseBenchmark):
    case_prefix = "batchnormalization"
    case_start = 0
    case_stop = None
