from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class Tanh(_CpuBackendCaseBenchmark):
    case_prefix = "tanh"
    case_names = (
        'test_cpu_tanh_logits_1x128x202048_bfloat16_benchmark',
        'test_cpu_tanh_logits_1x128x202048_float16_benchmark',
        'test_cpu_tanh_logits_1x128x202048_float32_benchmark',
        'test_cpu_tanh_logits_1x16x202048_bfloat16_benchmark',
        'test_cpu_tanh_logits_1x16x202048_float16_benchmark',
        'test_cpu_tanh_logits_1x16x202048_float32_benchmark',
        'test_cpu_tanh_logits_1x1x202048_bfloat16_benchmark',
        'test_cpu_tanh_logits_1x1x202048_float16_benchmark',
        'test_cpu_tanh_logits_1x1x202048_float32_benchmark',
        'test_cpu_tanh_n1024_bfloat16_benchmark',
        'test_cpu_tanh_n1024_float16_benchmark',
        'test_cpu_tanh_n1024_float32_benchmark',
        'test_cpu_tanh_n65535_bfloat16_benchmark',
        'test_cpu_tanh_n65535_float16_benchmark',
        'test_cpu_tanh_n65535_float32_benchmark',
        'test_cpu_tanh_n65536_bfloat16_benchmark',
        'test_cpu_tanh_n65536_float16_benchmark',
        'test_cpu_tanh_n65536_float32_benchmark',
        'test_cpu_tanh_n8_bfloat16_benchmark',
        'test_cpu_tanh_n8_float16_benchmark',
        'test_cpu_tanh_n8_float32_benchmark',
    )
