from benchmarks.cpu_backend_cases._base import _CpuBackendCaseBenchmark
from benchmarks.cpu_backend_cases._manifest import CASE_SHARDS

for _class_name, _case_prefix, _start, _stop in CASE_SHARDS:
    globals()[_class_name] = type(
        _class_name,
        (_CpuBackendCaseBenchmark,),
        {
            "__module__": __name__,
            "case_prefix": _case_prefix,
            "case_start": _start,
            "case_stop": _stop,
        },
    )

del _class_name, _case_prefix, _start, _stop
