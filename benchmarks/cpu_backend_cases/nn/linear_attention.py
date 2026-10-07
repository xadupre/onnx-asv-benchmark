from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class LinearAttention(_CpuBackendCaseBenchmark):
    case_prefix = "linear"
    case_names = (
        'test_cpu_linear_attention_qwen3_5_decode_t1_h16_d128_gated_delta_past_bfloat16_benchmark',
        'test_cpu_linear_attention_qwen3_5_decode_t1_h16_d128_gated_delta_past_float32_benchmark',
        'test_cpu_linear_attention_qwen3_5_prefill_t128_h16_d128_gated_delta_bfloat16_benchmark',
        'test_cpu_linear_attention_qwen3_5_prefill_t128_h16_d128_gated_delta_float32_benchmark',
        'test_cpu_linear_attention_qwen3_5_prefill_t512_h16_d128_gated_delta_float32_benchmark',
    )


class LinearAttentionMicrosoft(_CpuBackendCaseBenchmark):
    case_prefix = "microsoft"
    case_names = (
        'test_cpu_microsoft_linear_attention_decode_h16_d128_float32_benchmark',
    )
