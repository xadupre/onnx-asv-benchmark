from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class RMSNormalization(_CpuBackendCaseBenchmark):
    case_prefix = "rms"
    case_names = (
        'test_cpu_rms_normalization_llm_qwen3_6_27b_hidden5120_s128_bfloat16_benchmark',
        'test_cpu_rms_normalization_llm_qwen3_6_27b_hidden5120_s1_bfloat16_benchmark',
        'test_cpu_rms_normalization_llm_qwen3_6_27b_hidden5120_s512_bfloat16_benchmark',
        'test_cpu_rms_normalization_llm_qwen3_6_27b_k_norm_hd256_s128_kvh4_bfloat16_benchmark',
        'test_cpu_rms_normalization_llm_qwen3_6_27b_k_norm_hd256_s1_kvh4_bfloat16_benchmark',
        'test_cpu_rms_normalization_llm_qwen3_6_27b_q_norm_hd256_s128_qh24_bfloat16_benchmark',
        'test_cpu_rms_normalization_llm_qwen3_6_27b_q_norm_hd256_s1_qh24_bfloat16_benchmark',
        'test_cpu_rms_normalization_llm_qwen3_6_35b_a3b_hidden2048_s128_bfloat16_benchmark',
        'test_cpu_rms_normalization_llm_qwen3_6_35b_a3b_hidden2048_s1_bfloat16_benchmark',
        'test_cpu_rms_normalization_llm_qwen3_6_35b_a3b_hidden2048_s512_bfloat16_benchmark',
        'test_cpu_rms_normalization_llm_qwen3_6_35b_a3b_k_norm_hd256_s128_kvh2_bfloat16_benchmark',
        'test_cpu_rms_normalization_llm_qwen3_6_35b_a3b_k_norm_hd256_s1_kvh2_bfloat16_benchmark',
        'test_cpu_rms_normalization_llm_qwen3_6_35b_a3b_q_norm_hd256_s128_qh16_bfloat16_benchmark',
        'test_cpu_rms_normalization_llm_qwen3_6_35b_a3b_q_norm_hd256_s1_qh16_bfloat16_benchmark',
        'test_cpu_rms_normalization_llm_qwen3_8b_hidden4096_s128_float16_benchmark',
        'test_cpu_rms_normalization_llm_qwen3_8b_hidden4096_s1_float16_benchmark',
        'test_cpu_rms_normalization_llm_qwen3_8b_k_norm_hd128_s128_kvh8_float16_benchmark',
        'test_cpu_rms_normalization_llm_qwen3_8b_k_norm_hd128_s1_kvh8_float16_benchmark',
        'test_cpu_rms_normalization_llm_qwen3_8b_q_norm_hd128_s128_qh32_float16_benchmark',
        'test_cpu_rms_normalization_llm_qwen3_8b_q_norm_hd128_s1_qh32_float16_benchmark',
        'test_cpu_rms_normalization_llm_r128_w4096_axis1_float32_benchmark',
        'test_cpu_rms_normalization_r64_w512_bfloat16_benchmark',
        'test_cpu_rms_normalization_r64_w512_float32_benchmark',
        'test_cpu_rms_normalization_rank3_n2_h8_w16_axis1_float32_benchmark',
        'test_cpu_rms_normalization_rank3_n4_r16_w512_axis2_float32_benchmark',
        'test_cpu_rms_normalization_rank4_n2_c4_h8_w16_axis2_float32_benchmark',
        'test_cpu_rms_normalization_small_r8_w32_axis1_float32_benchmark',
        'test_cpu_rms_normalization_tall_r4096_w128_axis1_float32_benchmark',
        'test_cpu_rms_normalization_wide_r1_w4096_axis1_float32_benchmark',
    )
