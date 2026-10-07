from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class GroupQueryAttention(_CpuBackendCaseBenchmark):
    case_prefix = "group"
    case_names = (
        'test_cpu_group_query_attention_gqa_b2_s64_qh8_kvh2_hd64_causal_bfloat16_benchmark',
        'test_cpu_group_query_attention_gqa_b2_s64_qh8_kvh2_hd64_causal_float16_benchmark',
        'test_cpu_group_query_attention_gqa_b2_s64_qh8_kvh2_hd64_causal_float32_benchmark',
        'test_cpu_group_query_attention_mha_b1_s16_qh4_kvh4_hd32_causal_bfloat16_benchmark',
        'test_cpu_group_query_attention_mha_b1_s16_qh4_kvh4_hd32_causal_float16_benchmark',
        'test_cpu_group_query_attention_mha_b1_s16_qh4_kvh4_hd32_causal_float32_benchmark',
        'test_cpu_group_query_attention_model_qwen3_8b_int4_mb_decode_b1_s1_qh32_kvh8_hd128_pastlen1024_causal_float32_benchmark',
        'test_cpu_group_query_attention_model_qwen3_8b_int4_mb_decode_b1_s1_qh32_kvh8_hd128_pastlen128_causal_float32_benchmark',
        'test_cpu_group_query_attention_model_qwen3_8b_int4_mb_decode_b1_s1_qh32_kvh8_hd128_pastlen16_causal_float32_benchmark',
        'test_cpu_group_query_attention_model_qwen3_8b_int4_mb_prefill_b1_s128_qh32_kvh8_hd128_pastlen0_causal_float32_benchmark',
        'test_cpu_group_query_attention_model_qwen3_8b_int4_mb_prefill_b1_s16_qh32_kvh8_hd128_pastlen0_causal_float32_benchmark',
        'test_cpu_group_query_attention_model_qwen3_8b_int4_mb_prefill_b1_s1_qh32_kvh8_hd128_pastlen0_causal_float32_benchmark',
        'test_cpu_group_query_attention_mqa_b4_s32_qh8_kvh1_hd64_causal_bfloat16_benchmark',
        'test_cpu_group_query_attention_mqa_b4_s32_qh8_kvh1_hd64_causal_float16_benchmark',
        'test_cpu_group_query_attention_mqa_b4_s32_qh8_kvh1_hd64_causal_float32_benchmark',
    )
