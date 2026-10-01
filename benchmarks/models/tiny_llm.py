import os
import tempfile

import numpy as np
import onnxruntime
import torch
from modelbuilder.builder import create_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoModelForCausalLM, LlamaConfig, PreTrainedTokenizerFast

MODEL_NAME = "arnir0/Tiny-LLM"
SEQUENCE_LENGTH = 16


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
    tokenizer = Tokenizer(
        WordLevel(
            vocab={"<unk>": 0, "<s>": 1, "</s>": 2},
            unk_token="<unk>",
        )
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
    )


class TinyLLM:
    params = ("onnxruntime",)
    param_names = ("backend",)
    number = 1
    timeout = 60

    def setup(self, backend):
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
        _make_tokenizer().save_pretrained(source_directory)

        create_model(
            model_name=MODEL_NAME,
            input_path=source_directory,
            output_dir=output_directory,
            precision="fp32",
            execution_provider="cpu",
            cache_dir=cache_directory,
            num_hidden_layers=1,
        )

        model_path = os.path.join(output_directory, "model.onnx")
        self.session = onnxruntime.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        self.feeds = {
            "input_ids": np.arange(SEQUENCE_LENGTH, dtype=np.int64).reshape(
                1,
                SEQUENCE_LENGTH,
            ),
            "attention_mask": np.ones(
                (1, SEQUENCE_LENGTH),
                dtype=np.int64,
            ),
            "past_key_values.0.key": np.empty(
                (1, 4, 0, 64),
                dtype=np.float32,
            ),
            "past_key_values.0.value": np.empty(
                (1, 4, 0, 64),
                dtype=np.float32,
            ),
        }
        outputs = self.session.run(["logits"], self.feeds)
        if outputs[0].shape != (1, SEQUENCE_LENGTH, config.vocab_size):
            raise AssertionError(f"Unexpected logits shape {outputs[0].shape!r}.")

    def teardown(self, backend):
        del self.session
        self._temporary_directory.cleanup()

    def time_run(self, backend):
        self.session.run(["logits"], self.feeds)
