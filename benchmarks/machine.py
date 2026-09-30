import os
import platform
from pathlib import Path


def _processor_name():
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip() in {"model name", "Hardware", "Processor"}:
                processor = value.strip()
                if processor:
                    return processor
    return platform.processor() or platform.machine()


def track_processor():
    return _processor_name()


def track_architecture():
    return platform.machine()


def track_logical_cpu_count():
    return os.cpu_count() or 0
