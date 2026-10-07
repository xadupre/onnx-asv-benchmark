from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class Sigmoid(_CpuBackendCaseBenchmark):
    case_prefix = "sigmoid"
    case_names = (
        'test_cpu_sigmoid_n1024_bfloat16_benchmark',
        'test_cpu_sigmoid_n1024_float16_benchmark',
        'test_cpu_sigmoid_n1024_float32_benchmark',
        'test_cpu_sigmoid_n1024_float64_benchmark',
        'test_cpu_sigmoid_n1048576_bfloat16_benchmark',
        'test_cpu_sigmoid_n1048576_float16_benchmark',
        'test_cpu_sigmoid_n1048576_float32_benchmark',
        'test_cpu_sigmoid_n1048576_float64_benchmark',
        'test_cpu_sigmoid_n131072_bfloat16_benchmark',
        'test_cpu_sigmoid_n131072_float16_benchmark',
        'test_cpu_sigmoid_n131072_float32_benchmark',
        'test_cpu_sigmoid_n131072_float64_benchmark',
        'test_cpu_sigmoid_n32768_bfloat16_benchmark',
        'test_cpu_sigmoid_n32768_float16_benchmark',
        'test_cpu_sigmoid_n32768_float32_benchmark',
        'test_cpu_sigmoid_n32768_float64_benchmark',
        'test_cpu_sigmoid_n4194304_bfloat16_benchmark',
        'test_cpu_sigmoid_n4194304_float16_benchmark',
        'test_cpu_sigmoid_n4194304_float32_benchmark',
        'test_cpu_sigmoid_n4194304_float64_benchmark',
        'test_cpu_sigmoid_n65535_bfloat16_benchmark',
        'test_cpu_sigmoid_n65535_float16_benchmark',
        'test_cpu_sigmoid_n65535_float32_benchmark',
        'test_cpu_sigmoid_n65535_float64_benchmark',
        'test_cpu_sigmoid_n65536_bfloat16_benchmark',
        'test_cpu_sigmoid_n65536_float16_benchmark',
        'test_cpu_sigmoid_n65536_float32_benchmark',
        'test_cpu_sigmoid_n65536_float64_benchmark',
    )
