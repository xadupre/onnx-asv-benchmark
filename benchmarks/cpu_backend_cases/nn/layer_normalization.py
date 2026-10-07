from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class LayerNormalization(_CpuBackendCaseBenchmark):
    case_prefix = "layernormalization"
    case_names = (
        'test_cpu_layernormalization_llm_r128_w4096_axis1_float32_benchmark',
        'test_cpu_layernormalization_r64_w512_axis1_bfloat16_benchmark',
        'test_cpu_layernormalization_r64_w512_axis1_float16_benchmark',
        'test_cpu_layernormalization_rank3_n4_d8_w64_axis1_float32_benchmark',
        'test_cpu_layernormalization_rank3_n8_s16_w512_axis2_float32_benchmark',
        'test_cpu_layernormalization_small_r256_w128_axis1_float32_benchmark',
        'test_cpu_layernormalization_tall_r4096_w128_axis1_float32_benchmark',
        'test_cpu_layernormalization_wide_r1_w4096_axis1_float32_benchmark',
    )
