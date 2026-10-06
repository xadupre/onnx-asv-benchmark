from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class PowPart01(_CpuBackendCaseBenchmark):
    case_prefix = "pow"
    case_start = 0
    case_stop = 200


class PowPart02(_CpuBackendCaseBenchmark):
    case_prefix = "pow"
    case_start = 200
    case_stop = 400


class PowPart03(_CpuBackendCaseBenchmark):
    case_prefix = "pow"
    case_start = 400
    case_stop = 600


class PowPart04(_CpuBackendCaseBenchmark):
    case_prefix = "pow"
    case_start = 600
    case_stop = 800


class PowPart05(_CpuBackendCaseBenchmark):
    case_prefix = "pow"
    case_start = 800
    case_stop = 1000


class PowPart06(_CpuBackendCaseBenchmark):
    case_prefix = "pow"
    case_start = 1000
    case_stop = 1200


class PowPart07(_CpuBackendCaseBenchmark):
    case_prefix = "pow"
    case_start = 1200
    case_stop = 1400


class PowPart08(_CpuBackendCaseBenchmark):
    case_prefix = "pow"
    case_start = 1400
    case_stop = 1600


class PowPart09(_CpuBackendCaseBenchmark):
    case_prefix = "pow"
    case_start = 1600
    case_stop = None
