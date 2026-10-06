from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class DivPart01(_CpuBackendCaseBenchmark):
    case_prefix = "div"
    case_start = 0
    case_stop = 200


class DivPart02(_CpuBackendCaseBenchmark):
    case_prefix = "div"
    case_start = 200
    case_stop = 400


class DivPart03(_CpuBackendCaseBenchmark):
    case_prefix = "div"
    case_start = 400
    case_stop = 600


class DivPart04(_CpuBackendCaseBenchmark):
    case_prefix = "div"
    case_start = 600
    case_stop = None
