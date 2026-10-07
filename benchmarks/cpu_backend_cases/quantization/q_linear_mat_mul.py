from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class QlinearmatmulInt8Float32Int8Int8Float32Int8Float32Int8(_CpuBackendCaseBenchmark):
    case_prefix = "qlinearmatmul"
    case_dtypes = ('int8', 'float32', 'int8', 'int8', 'float32', 'int8', 'float32', 'int8')


class QlinearmatmulUint8Float32Uint8Uint8Float32Uint8Float32Uint8(_CpuBackendCaseBenchmark):
    case_prefix = "qlinearmatmul"
    case_dtypes = ('uint8', 'float32', 'uint8', 'uint8', 'float32', 'uint8', 'float32', 'uint8')
