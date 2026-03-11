# ARCHIVED: 2026-03-10
# REASON: Predetermined epoch-based LR schedule controller. Never used in production
#          training -- MetricBasedLR is the active controller. Could be useful for
#          ablation studies or reproducibility benchmarks with fixed schedules.
# ORIGINAL LOCATION: src/models/lr_controller.py (lines 109-181)
# DEPENDENCIES: LRController ABC, TrainingMetrics dataclass

from typing import Any, Dict, List, Tuple


class ScheduleBasedLR:
    """Predetermined schedule for LR scales.

    Simplest approach: define (epoch, lr_scale) pairs for each component.
    Interpolates linearly between points.

    Example:
        controller = ScheduleBasedLR({
            'encoder_a': [(0, 0.0), (50, 0.05), (100, 0.0)],
            'encoder_b': [(0, 0.1), (80, 0.0)],
            'projections': [(0, 1.0)],
            'decoders': [(0, 1.0)],
        })
    """

    def __init__(
        self,
        schedules: Dict[str, List[Tuple[int, float]]],
        interpolate: bool = True,
    ):
        """Initialize schedule-based controller.

        Args:
            schedules: Dict mapping component name to list of (epoch, lr_scale)
            interpolate: If True, linearly interpolate between points
        """
        self.schedules = schedules
        self.interpolate = interpolate
        self._validate_schedules()

    def _validate_schedules(self):
        """Ensure schedules are sorted by epoch."""
        for name, schedule in self.schedules.items():
            if not schedule:
                raise ValueError(f"Schedule for {name} is empty")
            sorted_schedule = sorted(schedule, key=lambda x: x[0])
            self.schedules[name] = sorted_schedule

    def _get_scale_at_epoch(
        self, schedule: List[Tuple[int, float]], epoch: int
    ) -> float:
        """Get LR scale at given epoch from schedule."""
        if epoch <= schedule[0][0]:
            return schedule[0][1]
        if epoch >= schedule[-1][0]:
            return schedule[-1][1]

        # Find surrounding points
        for i in range(len(schedule) - 1):
            e1, s1 = schedule[i]
            e2, s2 = schedule[i + 1]
            if e1 <= epoch < e2:
                if self.interpolate:
                    t = (epoch - e1) / (e2 - e1)
                    return s1 + t * (s2 - s1)
                return s1

        return schedule[-1][1]

    def get_lr_scales(self, metrics) -> Dict[str, float]:
        """Get LR scales for current epoch."""
        return {
            name: self._get_scale_at_epoch(schedule, metrics.epoch)
            for name, schedule in self.schedules.items()
        }

    def update(self, metrics) -> Dict[str, Any]:
        """Return current scales (no state to update)."""
        scales = self.get_lr_scales(metrics)
        return {
            "lr_scales": scales,
            "events": [],
            "type": "schedule",
        }
