from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class QLinearMatMulInt8Float32Inputs8(_CpuBackendCaseBenchmark):
    case_prefix = "qlinearmatmul"
    case_dtypes = ('int8', 'float32', 'int8', 'int8', 'float32', 'int8', 'float32', 'int8')


class QLinearMatMulUint8Float32Inputs8(_CpuBackendCaseBenchmark):
    case_prefix = "qlinearmatmul"
    case_dtypes = ('uint8', 'float32', 'uint8', 'uint8', 'float32', 'uint8', 'float32', 'uint8')
