from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class SwiGLU(_CpuBackendCaseBenchmark):
    case_prefix = "swiglu"
    case_start = 0
    case_stop = None
