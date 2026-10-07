from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class QLinearMatMul(_CpuBackendCaseBenchmark):
    case_prefix = "qlinearmatmul"
    case_names = (
        'test_cpu_qlinearmatmul_square_64_int8_benchmark',
        'test_cpu_qlinearmatmul_square_64_uint8_benchmark',
    )
