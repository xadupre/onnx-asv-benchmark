import itertools

from benchmarks.models.llm.qwen2 import Qwen2GenAI
from benchmarks.models.llm.tiny_llm import TinyLLMGenAI


BACKENDS = ("onnxruntime-genai", "onnx-light-cpu")


def run_scenario(benchmark_type, parameter_values):
    benchmark = benchmark_type()
    benchmark.setup(*parameter_values)
    benchmark.time_generate(*parameter_values)
    benchmark.teardown(*parameter_values)


def main():
    for benchmark_type in (TinyLLMGenAI, Qwen2GenAI):
        for parameter_values in itertools.product(*benchmark_type.params):
            if parameter_values[-1] not in BACKENDS:
                continue
            if not benchmark_type.is_available(*parameter_values):
                continue
            print(
                f"Running {benchmark_type.__name__}{parameter_values!r}",
                flush=True,
            )
            run_scenario(benchmark_type, parameter_values)


if __name__ == "__main__":
    main()
