from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class MatMul(_CpuBackendCaseBenchmark):
    case_prefix = "matmul"
    case_names = (
        'test_cpu_matmul_large_k_bfloat16_benchmark',
        'test_cpu_matmul_large_k_float16_benchmark',
        'test_cpu_matmul_large_k_float32_benchmark',
        'test_cpu_matmul_large_k_float64_benchmark',
        'test_cpu_matmul_llm_qwen3_8b_down_m1_k12288_n4096_float16_benchmark',
        'test_cpu_matmul_llm_qwen3_8b_gate_up_m1_k4096_n12288_float16_benchmark',
        'test_cpu_matmul_llm_qwen3_8b_lm_head_m1_k4096_n151936_float16_benchmark',
        'test_cpu_matmul_llm_qwen3_8b_o_proj_m1_k4096_n4096_float16_benchmark',
        'test_cpu_matmul_llm_qwen3_8b_qkv_m1_k4096_n6144_float16_benchmark',
        'test_cpu_matmul_skinny_m_bfloat16_benchmark',
        'test_cpu_matmul_skinny_m_float16_benchmark',
        'test_cpu_matmul_skinny_m_float32_benchmark',
        'test_cpu_matmul_skinny_m_float64_benchmark',
        'test_cpu_matmul_skinny_n_bfloat16_benchmark',
        'test_cpu_matmul_skinny_n_float16_benchmark',
        'test_cpu_matmul_skinny_n_float32_benchmark',
        'test_cpu_matmul_skinny_n_float64_benchmark',
        'test_cpu_matmul_square_1024_bfloat16_benchmark',
        'test_cpu_matmul_square_1024_float16_benchmark',
        'test_cpu_matmul_square_1024_float32_benchmark',
        'test_cpu_matmul_square_1024_float64_benchmark',
        'test_cpu_matmul_square_128_bfloat16_benchmark',
        'test_cpu_matmul_square_128_float16_benchmark',
        'test_cpu_matmul_square_128_float32_benchmark',
        'test_cpu_matmul_square_128_float64_benchmark',
        'test_cpu_matmul_square_512_bfloat16_benchmark',
        'test_cpu_matmul_square_512_float16_benchmark',
        'test_cpu_matmul_square_512_float32_benchmark',
        'test_cpu_matmul_square_512_float64_benchmark',
        'test_cpu_matmul_square_64_bfloat16_benchmark',
        'test_cpu_matmul_square_64_float16_benchmark',
        'test_cpu_matmul_square_64_float32_benchmark',
        'test_cpu_matmul_square_64_float64_benchmark',
    )
