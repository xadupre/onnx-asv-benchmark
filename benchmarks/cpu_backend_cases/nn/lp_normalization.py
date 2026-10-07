from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class LpNormalization(_CpuBackendCaseBenchmark):
    case_prefix = "lpnormalization"
    case_names = (
        'test_cpu_lpnormalization_b16_r32_w128_axis2_p2_float32_benchmark',
        'test_cpu_lpnormalization_b8_r16_w128_axis2_p2_bfloat16_benchmark',
        'test_cpu_lpnormalization_b8_r16_w128_axis2_p2_float16_benchmark',
        'test_cpu_lpnormalization_b8_r64_w32_axis1_p2_float32_benchmark',
        'test_cpu_lpnormalization_n4_c8_h16_w16_axis1_p1_float32_benchmark',
        'test_cpu_lpnormalization_r256_w128_axis1_p2_float32_benchmark',
        'test_cpu_lpnormalization_wide_r16_w4096_axis1_p1_float32_benchmark',
    )
