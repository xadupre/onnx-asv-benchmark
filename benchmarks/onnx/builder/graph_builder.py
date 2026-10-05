import numpy as np
import onnx_ir
import onnxscript
from onnx_light.onnx import TensorProto
from onnx_light.onnx_core.graph_builder import GraphBuilder

NODE_COUNTS = (100, 200, 500, 1000, 2000)
OPSET = 18
SHAPE = ["batch", "sequence", "width"]
BLOCK_SIZE = 20
PAYLOAD_BYTES = 8_000_000
LARGE_INITIALIZER_COUNT = 4


def attention_blocks(op, value, constants, node_count):
    if node_count <= 0 or node_count % BLOCK_SIZE:
        raise ValueError(f"node_count must be a positive multiple of {BLOCK_SIZE}.")
    batch_index, sequence_index, width_index, heads, half = constants
    for _ in range(node_count // BLOCK_SIZE):
        shape = op.Shape(value)
        batch = op.Gather(shape, batch_index, axis=0)
        sequence = op.Gather(shape, sequence_index, axis=0)
        width = op.Gather(shape, width_index, axis=0)
        head_width = op.Div(width, heads)
        head_shape = op.Concat(batch, sequence, heads, head_width, axis=0)
        query = op.Reshape(value, head_shape)
        query = op.Transpose(query, perm=[0, 2, 1, 3])
        key = op.Transpose(query, perm=[0, 1, 3, 2])
        scores = op.MatMul(query, key)
        scale = op.Cast(head_width, to=TensorProto.FLOAT)
        scale = op.Sqrt(scale)
        scores = op.Div(scores, scale)
        weights = op.Softmax(scores, axis=-1)
        context = op.MatMul(weights, query)
        context = op.Transpose(context, perm=[0, 2, 1, 3])
        context = op.Reshape(context, shape)
        value = op.Add(value, context)
        value = op.Mul(value, half)
        value = op.Relu(value)
    return value


def constants():
    return (
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int64),
        np.array([2], dtype=np.int64),
        np.array([2], dtype=np.int64),
        np.array(0.5, dtype=np.float32),
    )


def build_light(node_count, payload=None):
    builder = GraphBuilder("attention")
    builder.set_opset_version("", OPSET)
    value = builder.inp("X", TensorProto.FLOAT, SHAPE)
    initializers = [
        builder.init(array, name=f"c{index}") for index, array in enumerate(constants())
    ]
    value = attention_blocks(builder.op, value, initializers, node_count)
    builder.out(value, TensorProto.FLOAT, SHAPE)
    if payload is not None:
        for index in range(LARGE_INITIALIZER_COUNT):
            name = builder.init(payload, name=f"large{index}", copy=False)
            builder.out(name, TensorProto.UINT8, [len(payload)])
    return builder.to_onnx("model")


def build_onnxscript(node_count, payload=None):
    graph = onnx_ir.Graph(
        inputs=[], outputs=[], nodes=[], opset_imports={"": OPSET}, name="attention"
    )
    builder = onnxscript.GraphBuilder(graph)
    value = builder.input("X", dtype=onnx_ir.DataType.FLOAT, shape=SHAPE)
    initializers = [
        builder.initializer(onnx_ir.tensor(array), name=f"c{index}")
        for index, array in enumerate(constants())
    ]
    value = attention_blocks(builder.op, value, initializers, node_count)
    value.shape = onnx_ir.Shape(SHAPE)
    value.type = onnx_ir.TensorType(onnx_ir.DataType.FLOAT)
    builder.add_output(value, None)
    if payload is not None:
        for index in range(LARGE_INITIALIZER_COUNT):
            initializer = builder.initializer(onnx_ir.tensor(payload), name=f"large{index}")
            initializer.type = onnx_ir.TensorType(onnx_ir.DataType.UINT8)
            initializer.shape = onnx_ir.Shape([len(payload)])
            builder.add_output(initializer, None)
    return onnx_ir.to_proto(onnx_ir.Model(graph, ir_version=10))


class GraphBuilderAttention:
    params = (NODE_COUNTS, ("float32",), ("onnx-light", "onnxscript"), (False, True))
    param_names = ("nodes", "dtype", "builder", "serialize")
    number = 1
    timeout = 120

    def setup(self, nodes, dtype, builder, serialize):
        self.build = build_light if builder == "onnx-light" else build_onnxscript
        self.payload = np.zeros(PAYLOAD_BYTES, dtype=np.uint8)

    def time_build(self, nodes, dtype, builder, serialize):
        model = self.build(nodes, self.payload)
        if serialize:
            model.SerializeToString()
