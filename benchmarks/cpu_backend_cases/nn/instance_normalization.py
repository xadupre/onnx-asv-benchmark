from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class InstanceNormalization(_CpuBackendCaseBenchmark):
    case_prefix = "instancenormalization"
    case_start = 0
    case_stop = None
