import numpy as np
import onnx_light.onnx.helper as oh
import onnx_light.onnx.numpy_helper as onh
from onnx_light.onnx import TensorProto

from benchmarks.common import BACKENDS, run_session, setup_session


class TinyLLM:
    params = BACKENDS
    param_names = ("backend",)
    number = 1
    timeout = 60

    def setup(self, backend):
        rng = np.random.default_rng(42)

        def weight(shape, name):
            values = rng.standard_normal(shape, dtype=np.float32)
            values *= np.float32(0.02)
            np.clip(values, -0.04, 0.04, out=values)
            return onh.from_array(values, name)

        positions = np.arange(2048, dtype=np.float32)[:, np.newaxis]
        frequencies = np.float32(10000) ** (
            -np.arange(0, 64, 2, dtype=np.float32) / np.float32(64)
        )
        angles = positions * frequencies[np.newaxis, :]
        initializers = [
            weight((32000, 512), "embedding"),
            onh.from_array(np.ones(512, dtype=np.float32), "input_norm"),
            weight((512, 1024), "qkv_weight"),
            onh.from_array(np.cos(angles).astype(np.float32), "cos_cache"),
            onh.from_array(np.sin(angles).astype(np.float32), "sin_cache"),
            onh.from_array(
                np.array([512, 256, 256], dtype=np.int64),
                "qkv_split",
            ),
            onh.from_array(
                np.array([1, 16, 8, 64], dtype=np.int64),
                "query_shape",
            ),
            onh.from_array(
                np.array([1, 16, 4, 64], dtype=np.int64),
                "key_value_shape",
            ),
            onh.from_array(
                np.array([1, 16, 512], dtype=np.int64),
                "hidden_shape",
            ),
            weight((512, 512), "output_weight"),
            onh.from_array(np.ones(512, dtype=np.float32), "post_attention_norm"),
            weight((512, 1376), "gate_weight"),
            weight((512, 1376), "up_weight"),
            weight((1376, 512), "down_weight"),
            onh.from_array(np.ones(512, dtype=np.float32), "final_norm"),
            weight((512, 32000), "lm_head"),
        ]
        nodes = [
            oh.make_node("Gather", ["embedding", "input_ids"], ["hidden"]),
            oh.make_node(
                "RMSNormalization",
                ["hidden", "input_norm"],
                ["normalized"],
                axis=-1,
                epsilon=1e-5,
            ),
            oh.make_node(
                "MatMul",
                ["normalized", "qkv_weight"],
                ["qkv"],
            ),
            oh.make_node(
                "Split",
                ["qkv", "qkv_split"],
                ["query_flat", "key_flat", "value_flat"],
                axis=-1,
            ),
            oh.make_node(
                "Reshape",
                ["query_flat", "query_shape"],
                ["query_reshaped"],
            ),
            oh.make_node(
                "Reshape",
                ["key_flat", "key_value_shape"],
                ["key_reshaped"],
            ),
            oh.make_node(
                "Reshape",
                ["value_flat", "key_value_shape"],
                ["value_reshaped"],
            ),
            oh.make_node(
                "Transpose",
                ["query_reshaped"],
                ["query"],
                perm=[0, 2, 1, 3],
            ),
            oh.make_node(
                "Transpose",
                ["key_reshaped"],
                ["key"],
                perm=[0, 2, 1, 3],
            ),
            oh.make_node(
                "Transpose",
                ["value_reshaped"],
                ["value"],
                perm=[0, 2, 1, 3],
            ),
            oh.make_node(
                "RotaryEmbedding",
                ["query", "cos_cache", "sin_cache", "position_ids"],
                ["rotary_query"],
            ),
            oh.make_node(
                "RotaryEmbedding",
                ["key", "cos_cache", "sin_cache", "position_ids"],
                ["rotary_key"],
            ),
            oh.make_node(
                "Attention",
                ["rotary_query", "rotary_key", "value"],
                ["attention"],
                is_causal=1,
            ),
            oh.make_node(
                "Transpose",
                ["attention"],
                ["attention_transposed"],
                perm=[0, 2, 1, 3],
            ),
            oh.make_node(
                "Reshape",
                ["attention_transposed", "hidden_shape"],
                ["attention_reshaped"],
            ),
            oh.make_node(
                "MatMul",
                ["attention_reshaped", "output_weight"],
                ["attention_output"],
            ),
            oh.make_node(
                "Add",
                ["hidden", "attention_output"],
                ["attention_residual"],
            ),
            oh.make_node(
                "RMSNormalization",
                ["attention_residual", "post_attention_norm"],
                ["mlp_input"],
                axis=-1,
                epsilon=1e-5,
            ),
            oh.make_node("MatMul", ["mlp_input", "gate_weight"], ["gate"]),
            oh.make_node("Sigmoid", ["gate"], ["gate_sigmoid"]),
            oh.make_node("Mul", ["gate", "gate_sigmoid"], ["activated_gate"]),
            oh.make_node("MatMul", ["mlp_input", "up_weight"], ["up"]),
            oh.make_node("Mul", ["activated_gate", "up"], ["gated_up"]),
            oh.make_node("MatMul", ["gated_up", "down_weight"], ["mlp_output"]),
            oh.make_node(
                "Add",
                ["attention_residual", "mlp_output"],
                ["mlp_residual"],
            ),
            oh.make_node(
                "RMSNormalization",
                ["mlp_residual", "final_norm"],
                ["final_hidden"],
                axis=-1,
                epsilon=1e-5,
            ),
            oh.make_node("MatMul", ["final_hidden", "lm_head"], ["logits"]),
        ]
        model = oh.make_model(
            oh.make_graph(
                nodes,
                "tiny_llm",
                [
                    oh.make_tensor_value_info(
                        "input_ids",
                        TensorProto.INT64,
                        [1, 16],
                    ),
                    oh.make_tensor_value_info(
                        "position_ids",
                        TensorProto.INT64,
                        [1, 16],
                    ),
                ],
                [
                    oh.make_tensor_value_info(
                        "logits",
                        TensorProto.FLOAT,
                        [1, 16, 32000],
                    )
                ],
                initializers,
            ),
            opset_imports=[oh.make_opsetid("", 23)],
            ir_version=10,
        )
        feeds = {
            "input_ids": rng.integers(
                0,
                32000,
                size=(1, 16),
                dtype=np.int64,
            ),
            "position_ids": np.arange(16, dtype=np.int64)[np.newaxis, :],
        }
        setup_session(self, backend, model, feeds)

    def time_run(self, backend):
        run_session(self)
