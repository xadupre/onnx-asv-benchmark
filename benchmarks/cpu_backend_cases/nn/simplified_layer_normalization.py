from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class SimplifiedLayerNormalization(_CpuBackendCaseBenchmark):
    case_prefix = "simplified"
    case_start = 0
    case_stop = None
