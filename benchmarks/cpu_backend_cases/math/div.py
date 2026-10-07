from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class DivPart01(_CpuBackendCaseBenchmark):
    case_prefix = "div"
    case_start = 0
    case_stop = 100


class DivPart02(_CpuBackendCaseBenchmark):
    case_prefix = "div"
    case_start = 100
    case_stop = 200


class DivPart03(_CpuBackendCaseBenchmark):
    case_prefix = "div"
    case_start = 200
    case_stop = 300


class DivPart04(_CpuBackendCaseBenchmark):
    case_prefix = "div"
    case_start = 300
    case_stop = 400


class DivPart05(_CpuBackendCaseBenchmark):
    case_prefix = "div"
    case_start = 400
    case_stop = 500


class DivPart06(_CpuBackendCaseBenchmark):
    case_prefix = "div"
    case_start = 500
    case_stop = 600


class DivPart07(_CpuBackendCaseBenchmark):
    case_prefix = "div"
    case_start = 600
    case_stop = None
