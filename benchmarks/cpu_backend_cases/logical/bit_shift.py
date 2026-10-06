from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class BitShiftPart01(_CpuBackendCaseBenchmark):
    case_prefix = "bitshift"
    case_start = 0
    case_stop = 200


class BitShiftPart02(_CpuBackendCaseBenchmark):
    case_prefix = "bitshift"
    case_start = 200
    case_stop = None
