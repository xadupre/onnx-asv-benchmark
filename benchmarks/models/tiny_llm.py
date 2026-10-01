import os
import tempfile

import numpy as np
import onnxruntime
import onnxruntime_genai as og
import torch
from modelbuilder.builder import create_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForCausalLM, LlamaConfig, PreTrainedTokenizerFast

MODEL_NAME = "arnir0/Tiny-LLM"
CACHE_LENGTH = 128
MAX_NEW_TOKENS = 8
PROMPT = "The future of artificial intelligence is"
PRECISIONS = ("fp32", "fp16", "bf16", "int8", "int4", "int2")


def _cache_dtype(precision):
    if precision == "fp16":
        return np.float16
    return np.float32


def _initialize_weights(model):
    generator = torch.Generator().manual_seed(42)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim >= 2:
                torch.nn.init.trunc_normal_(
                    parameter,
                    mean=0.0,
                    std=0.02,
                    a=-0.04,
                    b=0.04,
                    generator=generator,
                )


def _make_tokenizer():
    vocabulary = {
        "<unk>": 0,
        "<s>": 1,
        "</s>": 2,
        "The": 3,
        "future": 4,
        "of": 5,
        "artificial": 6,
        "intelligence": 7,
        "is": 8,
    }
    tokenizer = Tokenizer(
        WordLevel(vocab=vocabulary, unk_token="<unk>"),
    )
    tokenizer.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
    )


class _TinyLLMBase:
    param_names = ("precision", "backend")
    number = 1
    timeout = 60

    @staticmethod
    def is_available(precision, backend):
        return precision != "bf16"

    def setup(self, precision, backend):
        if not self.is_available(precision, backend):
            raise NotImplementedError(
                "ONNX Runtime does not support the required BF16 kernels on CPU."
            )

        config = LlamaConfig(
            architectures=["LlamaForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=2048,
            model_type="llama",
            num_attention_heads=8,
            num_hidden_layers=1,
            num_key_value_heads=4,
            rms_norm_eps=1e-5,
            rope_theta=10000.0,
            vocab_size=32000,
        )

        torch.manual_seed(42)
        source_model = AutoModelForCausalLM.from_config(config)
        _initialize_weights(source_model)
        source_model.eval()

        self._temporary_directory = tempfile.TemporaryDirectory()
        source_directory = os.path.join(self._temporary_directory.name, "source")
        output_directory = os.path.join(self._temporary_directory.name, "output")
        cache_directory = os.path.join(self._temporary_directory.name, "cache")
        source_model.save_pretrained(source_directory)
        tokenizer = _make_tokenizer()
        tokenizer.save_pretrained(source_directory)
        self.prompt_tokens = tokenizer.encode(PROMPT)

        create_model(
            model_name=MODEL_NAME,
            input_path=source_directory,
            output_dir=output_directory,
            precision=precision,
            execution_provider="cpu",
            cache_dir=cache_directory,
            num_hidden_layers=1,
        )

        model_path = os.path.join(output_directory, "model.onnx")
        self.session = onnxruntime.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        self.prefill_feeds = {
            "input_ids": np.arange(CACHE_LENGTH, dtype=np.int64).reshape(
                1,
                CACHE_LENGTH,
            ),
            "attention_mask": np.ones(
                (1, CACHE_LENGTH),
                dtype=np.int64,
            ),
            "past_key_values.0.key": np.empty(
                (1, 4, 0, 64),
                dtype=_cache_dtype(precision),
            ),
            "past_key_values.0.value": np.empty(
                (1, 4, 0, 64),
                dtype=_cache_dtype(precision),
            ),
        }
        output_names = [output.name for output in self.session.get_outputs()]
        self.prefill_outputs = dict(
            zip(
                output_names,
                self.session.run(output_names, self.prefill_feeds),
                strict=True,
            )
        )
        prefill_logits = self.prefill_outputs["logits"]
        if prefill_logits.shape != (
            1,
            CACHE_LENGTH,
            config.vocab_size,
        ):
            raise AssertionError(
                f"Unexpected prefill logits shape {prefill_logits.shape!r}."
            )
        self.decode_feeds = {
            "input_ids": np.argmax(
                prefill_logits[:, -1:, :],
                axis=-1,
            ).astype(np.int64),
            "attention_mask": np.ones((1, CACHE_LENGTH + 1), dtype=np.int64),
            "past_key_values.0.key": self.prefill_outputs["present.0.key"],
            "past_key_values.0.value": self.prefill_outputs["present.0.value"],
        }
        decode_logits = self.session.run(["logits"], self.decode_feeds)[0]
        if decode_logits.shape != (1, 1, config.vocab_size):
            raise AssertionError(
                f"Unexpected decode logits shape {decode_logits.shape!r}."
            )

        self.genai_model = og.Model(output_directory)
        if not self.prompt_tokens:
            raise AssertionError("The generation prompt produced no tokens.")
        self._generate()

    def teardown(self, precision, backend):
        if hasattr(self, "genai_model"):
            del self.genai_model
        if hasattr(self, "session"):
            del self.session
        if hasattr(self, "_temporary_directory"):
            self._temporary_directory.cleanup()

    def _generate(self):
        parameters = og.GeneratorParams(self.genai_model)
        parameters.set_search_options(
            do_sample=False,
            max_length=len(self.prompt_tokens) + MAX_NEW_TOKENS,
            temperature=1.0,
            top_k=1,
        )
        generator = og.Generator(self.genai_model, parameters)
        generator.append_tokens(
            np.array([self.prompt_tokens], dtype=np.int64),
        )
        generated_tokens = []
        while not generator.is_done():
            generator.generate_next_token()
            generated_tokens.append(int(generator.get_next_tokens()[0]))
        if not generated_tokens:
            raise AssertionError("ONNX Runtime GenAI produced no tokens.")
        return generated_tokens


class TinyLLM(_TinyLLMBase):
    params = (PRECISIONS, ("onnxruntime",))

    def time_prefill(self, precision, backend):
        self.session.run(["logits"], self.prefill_feeds)

    def time_decode(self, precision, backend):
        self.session.run(["logits"], self.decode_feeds)


class TinyLLMGenAI(_TinyLLMBase):
    params = (PRECISIONS, ("onnxruntime-genai",))

    def time_generate(self, precision, backend):
        self._generate()
