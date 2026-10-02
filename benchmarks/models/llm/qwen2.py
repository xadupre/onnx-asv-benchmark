from transformers import Qwen2Config

from .tiny_llm import (
    GENERATION_SHAPES,
    INFERENCE_SHAPES,
    PRECISIONS,
    _CausalLLMBase,
)

MODEL_NAME = "Qwen/Qwen2-0.5B"


class _Qwen2Base(_CausalLLMBase):
    model_name = MODEL_NAME

    @staticmethod
    def make_config():
        return Qwen2Config(
            architectures=["Qwen2ForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=2048,
            num_attention_heads=8,
            num_hidden_layers=1,
            num_key_value_heads=4,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            tie_word_embeddings=False,
            use_sliding_window=False,
            vocab_size=32000,
        )


class Qwen2(_Qwen2Base):
    measure_inference = True
    params = (
        INFERENCE_SHAPES,
        PRECISIONS,
        ("onnxruntime", "onnx-reference", "onnx-light", "onnx-light-cpu"),
    )

    def time_prefill(self, shape, precision, backend):
        self.session.run(["logits"], self.prefill_feeds)

    def time_decode(self, shape, precision, backend):
        self.session.run(["logits"], self.decode_feeds)


class Qwen2GenAI(_Qwen2Base):
    measure_inference = False
    params = (
        GENERATION_SHAPES,
        PRECISIONS,
        ("onnxruntime-genai", "onnx-reference", "onnx-light", "onnx-light-cpu"),
    )

    def time_generate(self, shape, precision, backend):
        if backend == "onnxruntime-genai":
            self._generate_genai()
        elif backend == "onnx-reference":
            self._generate_reference()
        else:
            self._generate_onnx_light()
