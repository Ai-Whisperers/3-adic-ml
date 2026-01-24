# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Hardware monitoring for training visualization.

This module provides GPU and system memory monitoring:
- GPU memory tracking (allocated, reserved, peak)
- RAM usage and availability
- Formatted status strings for terminal output
- Warning thresholds for memory pressure

Single responsibility: Hardware resource monitoring only.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

# psutil for RAM monitoring (optional)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class HardwareMonitor:
    """Monitors GPU and system memory usage during training.

    Provides methods for tracking memory utilization and generating
    formatted status strings for terminal output.

    Attributes:
        device: PyTorch device being monitored
        warn_threshold: Fraction (0-1) at which to warn about memory usage
        is_cuda: Whether device is CUDA-enabled
    """

    def __init__(
        self,
        device: torch.device,
        warn_threshold: float = 0.9,
    ):
        """Initialize hardware monitor.

        Args:
            device: PyTorch device to monitor (cuda:X or cpu)
            warn_threshold: Fraction of GPU memory to trigger warning (0-1)
        """
        self.device = device
        self.warn_threshold = warn_threshold
        self.is_cuda = device.type == 'cuda' and torch.cuda.is_available()

        # Cache GPU total memory if available
        if self.is_cuda:
            self.gpu_total_bytes = torch.cuda.get_device_properties(device).total_memory
        else:
            self.gpu_total_bytes = 0

    def get_gpu_memory_mb(self) -> Dict[str, float]:
        """Get current GPU memory usage in MB.

        Returns:
            Dict with keys:
                - allocated: Currently allocated tensor memory
                - reserved: Total CUDA memory reserved
                - peak: Maximum allocated since last reset
                - total: Total GPU memory available
        """
        if not self.is_cuda:
            return {'allocated': 0.0, 'reserved': 0.0, 'peak': 0.0, 'total': 0.0}

        allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
        peak = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
        total = self.gpu_total_bytes / (1024 ** 2)

        return {
            'allocated': allocated,
            'reserved': reserved,
            'peak': peak,
            'total': total,
        }

    def get_gpu_memory_gb(self) -> Dict[str, float]:
        """Get current GPU memory usage in GB.

        Returns:
            Same as get_gpu_memory_mb() but in gigabytes.
        """
        mb = self.get_gpu_memory_mb()
        return {k: v / 1024 for k, v in mb.items()}

    def get_gpu_utilization_pct(self) -> float:
        """Get GPU memory utilization as percentage.

        Returns:
            Allocated memory as percentage of total (0-100)
        """
        if not self.is_cuda or self.gpu_total_bytes == 0:
            return 0.0

        allocated = torch.cuda.memory_allocated(self.device)
        return 100.0 * allocated / self.gpu_total_bytes

    def get_ram_usage_mb(self) -> Dict[str, float]:
        """Get current RAM usage in MB.

        Requires psutil to be installed.

        Returns:
            Dict with keys:
                - used: Currently used RAM
                - available: Available RAM
                - total: Total RAM
                - percent: Usage percentage (0-100)
        """
        if not PSUTIL_AVAILABLE:
            return {'used': 0.0, 'available': 0.0, 'total': 0.0, 'percent': 0.0}

        mem = psutil.virtual_memory()
        return {
            'used': mem.used / (1024 ** 2),
            'available': mem.available / (1024 ** 2),
            'total': mem.total / (1024 ** 2),
            'percent': mem.percent,
        }

    def get_ram_usage_gb(self) -> Dict[str, float]:
        """Get current RAM usage in GB.

        Returns:
            Same as get_ram_usage_mb() but in gigabytes (except percent).
        """
        mb = self.get_ram_usage_mb()
        return {
            'used': mb['used'] / 1024,
            'available': mb['available'] / 1024,
            'total': mb['total'] / 1024,
            'percent': mb['percent'],
        }

    def get_gpu_status_string(self) -> str:
        """Get formatted GPU status string.

        Returns:
            String like 'GPU: 2.1/6.0GB (35%)'
        """
        if not self.is_cuda:
            return 'GPU: N/A'

        mem = self.get_gpu_memory_gb()
        pct = self.get_gpu_utilization_pct()

        return f"GPU: {mem['allocated']:.1f}/{mem['total']:.1f}GB ({pct:.0f}%)"

    def get_ram_status_string(self) -> str:
        """Get formatted RAM status string.

        Returns:
            String like 'RAM: 8.2/32.0GB (26%)'
        """
        if not PSUTIL_AVAILABLE:
            return 'RAM: N/A (psutil not installed)'

        mem = self.get_ram_usage_gb()
        return f"RAM: {mem['used']:.1f}/{mem['total']:.1f}GB ({mem['percent']:.0f}%)"

    def get_status_string(self) -> str:
        """Get combined hardware status string.

        Returns:
            String like 'GPU: 2.1/6.0GB (35%) | RAM: 8.2/32.0GB (26%)'
        """
        parts = [self.get_gpu_status_string()]

        if PSUTIL_AVAILABLE:
            parts.append(self.get_ram_status_string())

        return ' | '.join(parts)

    def get_peak_status_string(self) -> str:
        """Get status string including peak GPU memory.

        Returns:
            String like 'GPU: 2.1/6.0GB (35%) peak=2.8GB | RAM: 8.2/32.0GB (26%)'
        """
        if not self.is_cuda:
            return self.get_status_string()

        mem = self.get_gpu_memory_gb()
        pct = self.get_gpu_utilization_pct()

        parts = [f"GPU: {mem['allocated']:.1f}/{mem['total']:.1f}GB ({pct:.0f}%) peak={mem['peak']:.1f}GB"]

        if PSUTIL_AVAILABLE:
            parts.append(self.get_ram_status_string())

        return ' | '.join(parts)

    def check_memory_warning(self) -> Optional[str]:
        """Check if memory usage exceeds warning threshold.

        Returns:
            Warning message if memory > threshold, None otherwise
        """
        if not self.is_cuda:
            return None

        utilization = self.get_gpu_utilization_pct() / 100.0

        if utilization > self.warn_threshold:
            mem = self.get_gpu_memory_gb()
            return (
                f"[WARN] GPU memory high: {mem['allocated']:.1f}/{mem['total']:.1f}GB "
                f"({utilization*100:.0f}% > {self.warn_threshold*100:.0f}% threshold)"
            )

        return None

    def reset_peak_stats(self) -> None:
        """Reset peak memory tracking.

        Resets the max_memory_allocated counter for this device.
        """
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)

    def get_oom_diagnostic(self, batch_size: int = 0) -> str:
        """Get diagnostic information for OOM debugging.

        Args:
            batch_size: Current batch size for suggestion

        Returns:
            Multi-line diagnostic string
        """
        lines = []

        if self.is_cuda:
            mem = self.get_gpu_memory_gb()
            lines.append(f"GPU Memory: allocated={mem['allocated']:.1f}GB, "
                        f"reserved={mem['reserved']:.1f}GB, peak={mem['peak']:.1f}GB")
            lines.append(f"GPU Total: {mem['total']:.1f}GB")

        if PSUTIL_AVAILABLE:
            ram = self.get_ram_usage_gb()
            lines.append(f"RAM: used={ram['used']:.1f}GB, available={ram['available']:.1f}GB "
                        f"({ram['percent']:.0f}%)")

        if batch_size > 0:
            suggested = max(32, batch_size // 2)
            lines.append(f"Suggestion: Reduce batch_size from {batch_size} to {suggested}")

        return '\n'.join(lines)


__all__ = ["HardwareMonitor", "PSUTIL_AVAILABLE"]
