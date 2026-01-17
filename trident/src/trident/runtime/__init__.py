"""
Trident Runtime - Execution environment for generated code.
"""

from trident.runtime.executor import TridentRuntime
from trident.runtime.device import DeviceManager, DeviceType

__all__ = [
    "TridentRuntime",
    "DeviceManager",
    "DeviceType",
]
