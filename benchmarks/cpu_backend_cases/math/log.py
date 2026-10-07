from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class Log(_CpuBackendCaseBenchmark):
    case_prefix = "log"
    case_names = (
        'test_cpu_log_n1024_bfloat16_benchmark',
        'test_cpu_log_n1024_float16_benchmark',
        'test_cpu_log_n1024_float32_benchmark',
        'test_cpu_log_n1024_float64_benchmark',
        'test_cpu_log_n1048576_bfloat16_benchmark',
        'test_cpu_log_n1048576_float16_benchmark',
        'test_cpu_log_n1048576_float32_benchmark',
        'test_cpu_log_n1048576_float64_benchmark',
        'test_cpu_log_n131071_bfloat16_benchmark',
        'test_cpu_log_n131071_float16_benchmark',
        'test_cpu_log_n131071_float32_benchmark',
        'test_cpu_log_n131071_float64_benchmark',
        'test_cpu_log_n131072_bfloat16_benchmark',
        'test_cpu_log_n131072_float16_benchmark',
        'test_cpu_log_n131072_float32_benchmark',
        'test_cpu_log_n131072_float64_benchmark',
        'test_cpu_log_n262144_bfloat16_benchmark',
        'test_cpu_log_n262144_float16_benchmark',
        'test_cpu_log_n262144_float32_benchmark',
        'test_cpu_log_n262144_float64_benchmark',
        'test_cpu_log_n4194304_bfloat16_benchmark',
        'test_cpu_log_n4194304_float16_benchmark',
        'test_cpu_log_n4194304_float32_benchmark',
        'test_cpu_log_n4194304_float64_benchmark',
        'test_cpu_log_n65536_bfloat16_benchmark',
        'test_cpu_log_n65536_float16_benchmark',
        'test_cpu_log_n65536_float32_benchmark',
        'test_cpu_log_n65536_float64_benchmark',
    )
