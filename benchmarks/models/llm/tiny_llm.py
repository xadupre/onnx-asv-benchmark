import os
import tempfile

import numpy as np
import onnx
import onnxruntime
import onnxruntime_genai as og
import torch
from onnx.reference import ReferenceEvaluator as OnnxReferenceEvaluator
from onnx_light import onnx as onnx_light
from onnx_light.onnx.reference import ReferenceEvaluator as OnnxLightReferenceEvaluator
from onnx_light_cpu import register_kernels_for_session
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
INFERENCE_SHAPES = (
    "prefill[input_ids=1x128, attention_mask=1x128, cache=1x4x0x64]; "
    "decode[input_ids=1x1, attention_mask=1x129, cache=1x4x128x64]",
)
GENERATION_SHAPES = ("input_ids=1x6, attention_mask=1x6, cache=1x4x0x64",)


def _cache_dtype(precision):
    if precision == "fp16":
        return np.float16
    if precision == "bf16":
        import ml_dtypes

        return ml_dtypes.bfloat16
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


class _CausalLLMBase:
    param_names = ("shape", "dtype", "backend")
    number = 1
    timeout = 60
    model_name = MODEL_NAME

    @staticmethod
    def make_config():
        return LlamaConfig(
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

    @staticmethod
    def is_available(shape, precision, backend):
        return _CausalLLMBase._is_backend_available(precision, backend)

    @staticmethod
    def _is_backend_available(precision, backend):
        return (
            backend not in {"onnxruntime", "onnxruntime-genai"} or precision != "bf16"
        )

    def setup(self, shape, precision, backend):
        shape_values = self.params[self.param_names.index("shape")]
        if shape != shape_values[0]:
            raise ValueError(f"Unexpected input shape parameter {shape!r}.")
        if not self._is_backend_available(precision, backend):
            raise NotImplementedError(
                "ONNX Runtime does not support the required BF16 kernels on CPU."
            )

        config = self.make_config()

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
        if not self.prompt_tokens:
            raise AssertionError("The generation prompt produced no tokens.")

        create_model(
            model_name=self.model_name,
            input_path=source_directory,
            output_dir=output_directory,
            precision=precision,
            execution_provider="cpu",
            cache_dir=cache_directory,
            num_hidden_layers=1,
        )

        model_path = os.path.join(output_directory, "model.onnx")
        if backend == "onnxruntime":
            self.session = onnxruntime.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"],
            )
            output_names = [output.name for output in self.session.get_outputs()]
            self._setup_inference(output_names, precision, config)
        elif backend == "onnx-reference":
            model = onnx.load(model_path, load_external_data=True)
            opsets = {opset.domain: opset.version for opset in model.opset_import}
            if "" not in opsets and "ai.onnx" in opsets:
                opsets[""] = opsets["ai.onnx"]
            self.session = OnnxReferenceEvaluator(model.graph, opsets=opsets)
            self.output_names = [output.name for output in model.graph.output]
            if self.measure_inference:
                self._setup_inference(self.output_names, precision, config)
            else:
                self._setup_generation_feeds(precision)
                self._generate_reference()
        elif backend == "onnxruntime-genai":
            self.genai_model = og.Model(output_directory)
            self._generate_genai()
        elif backend in {"onnx-light", "onnx-light-cpu"}:
            model = onnx_light.load(model_path, load_external_data=True)
            self.session = OnnxLightReferenceEvaluator(model)
            if backend == "onnx-light-cpu":
                register_kernels_for_session(self.session)
            if self.measure_inference:
                output_names = [output.name for output in model.graph.output]
                self._setup_inference(output_names, precision, config)
            else:
                self._setup_generation_feeds(precision)
                self._generate_onnx_light()
        else:
            raise ValueError(f"Unexpected backend {backend!r}.")

    def _setup_inference(self, output_names, precision, config):
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

    def _setup_generation_feeds(self, precision):
        self.generation_feeds = {
            "input_ids": np.array([self.prompt_tokens], dtype=np.int64),
            "attention_mask": np.ones(
                (1, len(self.prompt_tokens)),
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

    def teardown(self, shape, precision, backend):
        if hasattr(self, "genai_model"):
            del self.genai_model
        if hasattr(self, "session"):
            del self.session
        if hasattr(self, "_temporary_directory"):
            self._temporary_directory.cleanup()

    def _generate_genai(self):
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

    def _generate_onnx_light(self):
        tokens = self.session.generate(
            self.generation_feeds,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.0,
            eos_token_id=2,
            pad_token_id=2,
        )
        generated_tokens = tokens[0, len(self.prompt_tokens) :].tolist()
        if not generated_tokens:
            raise AssertionError("onnx-light produced no tokens.")
        return generated_tokens

    def _generate_reference(self):
        feeds = self.generation_feeds.copy()
        generated_tokens = []
        for _ in range(MAX_NEW_TOKENS):
            outputs = dict(
                zip(
                    self.output_names,
                    self.session.run(self.output_names, feeds),
                    strict=True,
                )
            )
            token = int(np.argmax(outputs["logits"][0, -1]))
            generated_tokens.append(token)
            if token == 2:
                break
            feeds = {
                "input_ids": np.array([[token]], dtype=np.int64),
                "attention_mask": np.ones(
                    (1, len(self.prompt_tokens) + len(generated_tokens)),
                    dtype=np.int64,
                ),
                "past_key_values.0.key": outputs["present.0.key"],
                "past_key_values.0.value": outputs["present.0.value"],
            }
        if not generated_tokens:
            raise AssertionError("ONNX Reference produced no tokens.")
        return generated_tokens


class TinyLLM(_CausalLLMBase):
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


class TinyLLMGenAI(_CausalLLMBase):
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
