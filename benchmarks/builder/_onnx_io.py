"""Shared implementation for ONNX I/O benchmarks."""

import importlib.util
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
import onnx_light.onnx.numpy_helper as onh
import onnxruntime as ort
from onnx_light.doc import find_standalone_executable, measure_cpp_with_example
from onnx_light.onnx.reference import ReferenceEvaluator

SHAPES = ("X=dynamicx2048 (40 Gemm)",)
DTYPES = ("float32",)

LOAD_CASES = (
    "load/1filex1/onnx",
    "load/1filex1/onnxlight",
    "load/1filex4/onnxlight",
    "load/1filex1/onnxlight-mmap",
    "load/1filex1/onnxlight-ifstream",
    "load/1filex1/ir-py",
    "load/1filex1/ort",
    "load/2filex1/onnx",
    "load/2filex1/onnxlight",
    "load/2filex4/onnxlight",
    "load/2filex1/onnxlight-nocopy",
    "load/2filex1/ir-py",
    "load/2filex1/ort",
)
SAVE_CASES = (
    "save/1filex1/onnx",
    "save/2filex1/onnx",
    "save/1filex1/ir-py",
    "save/2filex1/ir-py",
    "save/1filex1/onnxlight",
    "save/1filex4/onnxlight",
    "save/2filex1/onnxlight",
    "save/2filex4/onnxlight",
)
SERIALIZE_CASES = (
    "serialize/x1/onnx",
    "serialize/x1/onnxlight",
    "serialize/x4/onnxlight",
)
PARSE_CASES = (
    "parse/x1/onnx",
    "parse/x1/onnxlight",
    "parse/x4/onnxlight",
    "parse/nc/onnxlight",
    "parse/ncx4/onnxlight",
)
SAVE_CPP_CASES = (
    "save/1filex1/onnxlight-cpp",
    "save/1filex4/onnxlight-cpp",
    "save/2filex1/onnxlight-cpp",
    "save/2filex4/onnxlight-cpp",
)


def case_parameters(cases):
    case_names = []
    libraries = []
    for value in cases:
        case_name, library = value.rsplit("/", 1)
        if case_name not in case_names:
            case_names.append(case_name)
        if library not in libraries:
            libraries.append(library)
    return tuple(case_names), tuple(libraries)


def _make_model(n_init, dim):
    rng = np.random.default_rng(0)
    initializers = []
    nodes = []
    previous = "X"
    for index in range(n_init):
        weight = f"W{index}"
        output = f"Y{index}"
        initializers.append(
            onh.from_array(rng.standard_normal((dim, dim), dtype=np.float32), weight)
        )
        nodes.append(oh.make_node("Gemm", [previous, weight], [output], transB=1))
        previous = output
    return oh.make_model(
        oh.make_graph(
            nodes,
            "bench_graph",
            [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [None, dim])],
            [oh.make_tensor_value_info(previous, onnxl.TensorProto.FLOAT, [None, dim])],
            initializer=initializers,
        ),
        opset_imports=[oh.make_opsetid("", 18)],
        ir_version=9,
    )


def _flush(path):
    with open(path, "r+b") as stream:
        stream.flush()
        os.fsync(stream.fileno())


class _OnnxIOBase:
    n_init = 40
    dim = 2048
    number = 1
    timeout = 600
    param_names = ("shape", "dtype", "case", "library")
    cases = ()

    def _full_case(self, case, library):
        full_case = f"{case}/{library}"
        if full_case not in self.cases:
            raise NotImplementedError(
                f"Unsupported case/library combination {case!r}, {library!r}"
            )
        return full_case

    def setup(self, shape, dtype, case, library):
        self.tmp = None
        case = self._full_case(case, library)
        if shape != f"X=dynamicx{self.dim} ({self.n_init} Gemm)" or dtype != "float32":
            raise ValueError(f"Unexpected model parameters {shape!r}, {dtype!r}")
        if "/ir-py" in case and importlib.util.find_spec("onnx_ir") is None:
            raise NotImplementedError("onnx_ir is not installed")
        self.tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmp.name) / "bench.onnx")
        self.external_path = str(Path(self.tmp.name) / "external.onnx")
        self.external_data = self.external_path + ".data"
        model = _make_model(self.n_init, self.dim)
        onnxl.save(model, self.path)
        if case.startswith("load/2file"):
            onnxl.save(model, self.external_path, location=self.external_data)
        if case.startswith(("save/", "serialize/", "parse/")):
            library = case.rsplit("/", 1)[1]
            if library == "onnx":
                self.onnx_model = onnx.load(self.path)
            elif library == "onnxlight":
                self.light_model = model
        if "/ir-py" in case:
            import onnx_ir

            self.ir = onnx_ir
            if case.startswith("save/"):
                self.ir_model = onnx_ir.load(self.path)

    def teardown(self, shape, dtype, case, library):
        for name in (
            "data",
            "ir_model",
            "light_model",
            "loaded_model",
            "onnx_model",
        ):
            if hasattr(self, name):
                setattr(self, name, None)
        tmp = getattr(self, "tmp", None)
        if tmp is not None:
            tmp.cleanup()
            self.tmp = None


class _OnnxLoad(_OnnxIOBase):
    def setup(self, shape, dtype, case, library):
        self.loaded_model = None
        super().setup(shape, dtype, case, library)
        case = self._full_case(case, library)
        if case.endswith("/ort"):
            self.ort_options = ort.SessionOptions()
            self.ort_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            )

    def time_run(self, shape, dtype, case, library):
        case = self._full_case(case, library)
        _, files, library = case.split("/")
        path = self.external_path if files.startswith("2file") else self.path
        threads = 4 if files.endswith("x4") else 1
        if library == "onnx":
            self.loaded_model = onnx.load(path)
        elif library == "ir-py":
            model = self.ir.load(path)
            if files.startswith("2file"):
                self.ir.external_data.load_to_model(model)
            self.loaded_model = model
        elif library == "ort":
            self.loaded_model = ort.InferenceSession(
                path, sess_options=self.ort_options
            )
        else:
            options = {"num_threads": threads}
            if files.startswith("2file"):
                options["location"] = self.external_data
            if library == "onnxlight-mmap":
                options["file_load_mode"] = "MMAP"
            elif library == "onnxlight-ifstream":
                options["file_load_mode"] = "IFSTREAM"
            elif library == "onnxlight-nocopy":
                options["no_copy"] = True
                options["touch_raw_data_pages"] = True
            self.loaded_model = onnxl.load(path, **options)


class _OnnxReferenceEvaluator:
    n_init = 40
    dim = 2048
    number = 1
    repeat = 5
    timeout = 600
    param_names = ("shape", "dtype")
    params = (SHAPES, DTYPES)

    def setup(self, shape, dtype):
        if shape != f"X=dynamicx{self.dim} ({self.n_init} Gemm)" or dtype != "float32":
            raise ValueError(f"Unexpected model parameters {shape!r}, {dtype!r}")
        self.model = _make_model(self.n_init, self.dim)
        self.session = None

    def teardown(self, shape, dtype):
        self.session = None
        self.model = None

    def time_run(self, shape, dtype):
        self.session = ReferenceEvaluator(self.model)


class _OnnxSave(_OnnxIOBase):
    repeat = 1

    def setup(self, shape, dtype, case, library):
        super().setup(shape, dtype, case, library)
        self.out = str(Path(self.tmp.name) / "out.onnx")
        self.out_data = self.out + ".data"

    def time_run(self, shape, dtype, case, library):
        case = self._full_case(case, library)
        _, files, library = case.split("/")
        external = files.startswith("2file")
        threads = 4 if files.endswith("x4") else 1
        if library == "onnx":
            if external:
                onnx.save_model(
                    self.onnx_model,
                    self.out,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location=Path(self.out_data).name,
                    size_threshold=0,
                )
            else:
                onnx.save(self.onnx_model, self.out)
        elif library == "ir-py":
            if external:
                self.ir.save(
                    self.ir_model, self.out, external_data=Path(self.out_data).name
                )
            else:
                self.ir.save(self.ir_model, self.out)
        else:
            options = {"num_threads": threads}
            if external:
                options["location"] = self.out_data
            self._light_save(options)
        if external:
            _flush(self.out_data)
            _flush(self.out)

    def _light_save(self, options):
        onnxl.save(self.light_model, self.out, **options)


class _OnnxBytes(_OnnxIOBase):
    def setup(self, shape, dtype, case, library):
        super().setup(shape, dtype, case, library)
        case = self._full_case(case, library)
        if case.startswith("parse/"):
            self.data = (
                self.onnx_model.SerializeToString()
                if case.endswith("/onnx")
                else self.light_model.SerializeToString()
            )
        if case.endswith("/onnxlight") and not case.endswith("x1/onnxlight"):
            operation, mode, _ = case.split("/")
            if operation == "serialize":
                self.options = onnxl.SerializeOptions()
                self.options.num_threads = 4
            else:
                self.options = onnxl.ParseOptions()
                self.options.no_copy = mode.startswith("nc")
                self.options.num_threads = 4 if mode.endswith("x4") else 1

    def time_run(self, shape, dtype, case, library):
        case = self._full_case(case, library)
        operation, mode, library = case.split("/")
        if operation == "serialize":
            model = self.onnx_model if library == "onnx" else self.light_model
            if mode == "x4":
                model.SerializeToString(self.options)
            else:
                model.SerializeToString()
        else:
            model = onnx.ModelProto() if library == "onnx" else onnxl.ModelProto()
            if library == "onnx" or mode == "x1":
                model.ParseFromString(self.data)
            else:
                model.ParseFromString(self.data, self.options)


class _OnnxCpp(_OnnxIOBase):
    unit = "seconds"

    def setup(self, shape, dtype, case, library):
        self.out_dir = None
        self._full_case(case, library)
        executable = "save_onnx_light_time"
        paths = (
            "build/save-onnx-light-time-example/save_onnx_light_time",
            "build/examples/save_onnx_light_time/save_onnx_light_time",
            "build-save-onnx-light-time/save_onnx_light_time",
        )
        self.executable = find_standalone_executable(
            executable, list(map(Path, paths)), script_file=None
        )
        if self.executable is None:
            raise NotImplementedError(f"{executable} is unavailable")
        super().setup(shape, dtype, case, library)
        self.out_dir = tempfile.TemporaryDirectory()

    def teardown(self, shape, dtype, case, library):
        if self.out_dir is not None:
            self.out_dir.cleanup()
            self.out_dir = None
        super().teardown(shape, dtype, case, library)

    def track_run(self, shape, dtype, case, library):
        case = self._full_case(case, library)
        operation, files, _ = case.split("/")
        threads = 4 if files.endswith("x4") else 1
        args = [
            self.path,
            self.out_dir.name,
            "20",
            str(threads),
            "external" if files.startswith("2file") else "onefile",
        ]
        metric = re.compile(
            rf"^\s*(Average|Median|Min|Max|Std|Standard deviation) {operation} "
            r"\(ms\)\s*:\s*([0-9.eE+-]+)\s*$"
        )
        result = measure_cpp_with_example(
            self.executable, args, metric, case, Path(self.executable).name
        )
        if result is None:
            raise RuntimeError(f"C++ benchmark {case} failed")
        average = result["avg"]
        if not np.isfinite(average) or average < 0:
            raise RuntimeError(
                f"C++ benchmark {case} returned invalid average {average!r}"
            )
        return average
