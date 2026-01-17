"""
Device Manager for Trident.

Handles TPU/GPU/CPU device selection and management.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List
import os


class DeviceType(Enum):
    """Available device types."""
    CPU = auto()
    GPU = auto()
    TPU = auto()
    
    def __str__(self) -> str:
        return self.name


@dataclass
class Device:
    """Represents a compute device."""
    type: DeviceType
    id: int
    name: str
    memory_gb: float = 0.0
    
    def __str__(self) -> str:
        return f"{self.type.name}:{self.id} ({self.name})"


@dataclass
class DeviceManager:
    """
    Manages compute devices for Trident execution.
    
    Automatically detects available devices and provides
    optimal device selection.
    """
    
    _devices: list[Device] = field(default_factory=list)
    _current_device: Optional[Device] = field(default=None)
    _jax_available: bool = field(default=False, init=False)
    
    def __post_init__(self) -> None:
        """Initialize and detect available devices."""
        self._detect_devices()
        self._select_best_device()
    
    def _detect_devices(self) -> None:
        """Detect all available compute devices."""
        self._devices = []
        
        # Try to import JAX and detect devices
        try:
            import jax
            self._jax_available = True
            
            # Get JAX devices
            jax_devices = jax.devices()
            
            for i, dev in enumerate(jax_devices):
                platform = dev.platform.upper()
                
                if platform == "TPU":
                    device_type = DeviceType.TPU
                elif platform in ("GPU", "CUDA"):
                    device_type = DeviceType.GPU
                else:
                    device_type = DeviceType.CPU
                
                self._devices.append(Device(
                    type=device_type,
                    id=i,
                    name=f"{platform}:{dev.id}",
                ))
        except ImportError:
            # JAX not available, fallback to CPU
            self._devices.append(Device(
                type=DeviceType.CPU,
                id=0,
                name="cpu:0",
            ))
    
    def _select_best_device(self) -> None:
        """Select the best available device (TPU > GPU > CPU)."""
        priority = {DeviceType.TPU: 0, DeviceType.GPU: 1, DeviceType.CPU: 2}
        
        if self._devices:
            self._devices.sort(key=lambda d: priority.get(d.type, 99))
            self._current_device = self._devices[0]
    
    @property
    def current_device(self) -> Optional[Device]:
        """Get the current device."""
        return self._current_device
    
    @property
    def devices(self) -> list[Device]:
        """Get all available devices."""
        return self._devices
    
    def has_tpu(self) -> bool:
        """Check if TPU is available."""
        return any(d.type == DeviceType.TPU for d in self._devices)
    
    def has_gpu(self) -> bool:
        """Check if GPU is available."""
        return any(d.type == DeviceType.GPU for d in self._devices)
    
    def set_device(self, device_type: DeviceType, device_id: int = 0) -> bool:
        """
        Set the current device.
        
        Returns:
            True if device was set, False if not available
        """
        for device in self._devices:
            if device.type == device_type and device.id == device_id:
                self._current_device = device
                return True
        return False
    
    def get_jax_device(self):
        """Get the JAX device object for current device."""
        if not self._jax_available:
            return None
        
        import jax
        devices = jax.devices()
        
        if self._current_device and self._current_device.id < len(devices):
            return devices[self._current_device.id]
        
        return devices[0] if devices else None
    
    def info(self) -> str:
        """Get device information as a string."""
        lines = ["Trident Device Manager", "=" * 40]
        
        for device in self._devices:
            marker = " [ACTIVE]" if device == self._current_device else ""
            lines.append(f"  {device}{marker}")
        
        if not self._devices:
            lines.append("  No devices detected")
        
        return "\n".join(lines)
