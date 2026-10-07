from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class Not(_CpuBackendCaseBenchmark):
    case_prefix = "not"
    case_names = (
        'test_cpu_not_n1024_bool_benchmark',
        'test_cpu_not_n1048576_bool_benchmark',
        'test_cpu_not_n131072_bool_benchmark',
        'test_cpu_not_n32768_bool_benchmark',
        'test_cpu_not_n4194304_bool_benchmark',
        'test_cpu_not_n65535_bool_benchmark',
        'test_cpu_not_n65536_bool_benchmark',
    )
