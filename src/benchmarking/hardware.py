"""Standard-library hardware metadata collection."""

from __future__ import annotations

from dataclasses import dataclass
import platform as platform_module
import sys


@dataclass(frozen=True, slots=True)
class HardwareInfo:
    """Host information for a development-machine benchmark."""

    operating_system: str
    platform: str
    processor: str
    machine: str
    python_version: str


def get_hardware_info() -> HardwareInfo:
    """Return metadata for the host on which a benchmark executes."""

    return HardwareInfo(
        operating_system=platform_module.system(),
        platform=platform_module.platform(),
        processor=platform_module.processor(),
        machine=platform_module.machine(),
        python_version=sys.version.split()[0],
    )
