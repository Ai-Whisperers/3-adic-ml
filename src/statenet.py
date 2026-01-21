# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Hierarchical Homeostatic Controller for V5.11.8.

Implements complementary learning systems theory with Q-gated annealing:
- Slow components (encoders): consolidate, freeze when objective met
- Fast components (projections, controller): continuously adapt
- Q-gated annealing: thresholds relax only when Q improves

Each component has its own freeze trigger:
- encoder_A: coverage-gated (freeze on drop, unfreeze on recovery + hierarchy stall)
- encoder_B: hierarchy-gated (freeze when VAE-B hierarchy plateaus)
- controller: gradient-gated (freeze when weights stabilize)
- projections: always trainable (fast adaptation layer)

V5.11.8: Q-gated annealing progressively relaxes thresholds when Q improves,
enabling exploration of higher Q values while maintaining coverage floors.
"""

from collections import deque
from typing import Any, Dict, Optional

from src.config.constants import (
    HOMEOSTATIC_ANNEALING_STEP,
    HOMEOSTATIC_CONTROLLER_GRAD_PATIENCE,
    HOMEOSTATIC_CONTROLLER_GRAD_THRESHOLD,
    HOMEOSTATIC_CONTROLLER_PATIENCE_CEILING,
    HOMEOSTATIC_COVERAGE_FLOOR,
    HOMEOSTATIC_COVERAGE_FREEZE_THRESHOLD,
    HOMEOSTATIC_COVERAGE_UNFREEZE_THRESHOLD,
    HOMEOSTATIC_HIERARCHY_PATIENCE_CEILING,
    HOMEOSTATIC_HIERARCHY_PLATEAU_PATIENCE,
    HOMEOSTATIC_HIERARCHY_PLATEAU_THRESHOLD,
    HOMEOSTATIC_HYSTERESIS_EPOCHS,
    HOMEOSTATIC_WARMUP_EPOCHS,
    HOMEOSTATIC_WINDOW_SIZE,
)


def compute_Q(dist_corr: float, hierarchy: float) -> float:
    """Compute conserved structure capacity Q.

    Q = dist_corr + 1.5 × |hierarchy|

    Higher Q indicates better structure learning.
    """
    return dist_corr + 1.5 * abs(hierarchy)


class StateNet:
    """Hierarchical homeostatic freeze/unfreeze controller with Q-gated annealing.

    Monitors training metrics and dynamically adjusts component freeze states
    to balance coverage preservation with geometric structure learning.

    V5.11.8: Thresholds anneal (relax) only when Q improves after a cycle,
    and tighten when Q decreases. This creates a reversible ratchet.
    """

    def __init__(
        self,
        # Coverage thresholds for encoder_A (from constants.py)
        coverage_freeze_threshold: float = HOMEOSTATIC_COVERAGE_FREEZE_THRESHOLD,
        coverage_unfreeze_threshold: float = HOMEOSTATIC_COVERAGE_UNFREEZE_THRESHOLD,
        # Hierarchy thresholds for encoder_B (from constants.py)
        hierarchy_plateau_threshold: float = HOMEOSTATIC_HIERARCHY_PLATEAU_THRESHOLD,
        hierarchy_plateau_patience: int = HOMEOSTATIC_HIERARCHY_PLATEAU_PATIENCE,
        # Controller gradient thresholds (from constants.py)
        controller_grad_threshold: float = HOMEOSTATIC_CONTROLLER_GRAD_THRESHOLD,
        controller_grad_patience: int = HOMEOSTATIC_CONTROLLER_GRAD_PATIENCE,
        # General settings (from constants.py)
        window_size: int = HOMEOSTATIC_WINDOW_SIZE,
        hysteresis_epochs: int = HOMEOSTATIC_HYSTERESIS_EPOCHS,
        warmup_epochs: int = HOMEOSTATIC_WARMUP_EPOCHS,
        # Q-gated annealing settings (V5.11.8, from constants.py)
        enable_annealing: bool = True,
        annealing_step: float = HOMEOSTATIC_ANNEALING_STEP,
        coverage_floor: float = HOMEOSTATIC_COVERAGE_FLOOR,
        hierarchy_patience_ceiling: int = HOMEOSTATIC_HIERARCHY_PATIENCE_CEILING,
        controller_patience_ceiling: int = HOMEOSTATIC_CONTROLLER_PATIENCE_CEILING,
    ):
        # Initial thresholds (can be annealed)
        self.coverage_freeze_threshold = coverage_freeze_threshold
        self.coverage_unfreeze_threshold = coverage_unfreeze_threshold
        self.hierarchy_plateau_threshold = hierarchy_plateau_threshold
        self.hierarchy_plateau_patience = hierarchy_plateau_patience
        self.controller_grad_threshold = controller_grad_threshold
        self.controller_grad_patience = controller_grad_patience

        # Store initial values for reference
        self._initial_coverage_freeze = coverage_freeze_threshold
        self._initial_coverage_unfreeze = coverage_unfreeze_threshold
        self._initial_hierarchy_patience = hierarchy_plateau_patience
        self._initial_controller_patience = controller_grad_patience

        self.window_size = window_size
        self.hysteresis_epochs = hysteresis_epochs
        self.warmup_epochs = warmup_epochs

        # Q-gated annealing settings
        self.enable_annealing = enable_annealing
        self.annealing_step = annealing_step
        self.coverage_floor = coverage_floor
        self.hierarchy_patience_ceiling = hierarchy_patience_ceiling
        self.controller_patience_ceiling = controller_patience_ceiling

        # Metric history (moving windows)
        self.coverage_history: deque[float] = deque(maxlen=window_size)
        self.hierarchy_A_history: deque[float] = deque(maxlen=window_size)
        self.hierarchy_B_history: deque[float] = deque(maxlen=window_size)
        self.controller_grad_history: deque[float] = deque(maxlen=window_size)
        self.Q_history: deque[float] = deque(maxlen=window_size * 2)  # Longer window for Q

        # Freeze states
        self.encoder_a_frozen = True  # Starts frozen
        self.encoder_b_frozen = False  # Starts trainable (Option C)
        self.controller_frozen = False  # Starts trainable

        # Last state change epoch (for hysteresis)
        self.encoder_a_last_change = -hysteresis_epochs
        self.encoder_b_last_change = -hysteresis_epochs
        self.controller_last_change = -hysteresis_epochs

        # Plateau counters
        self.hierarchy_b_plateau_count = 0
        self.controller_low_grad_count = 0

        # Unfreeze conditions for encoder_A
        self.hierarchy_a_stalled = False
        self.hierarchy_a_stall_count = 0
        self.hierarchy_stall_patience = 5

        # Q-gated annealing state
        # Initialize cycle start Q for components that start unfrozen
        self.Q_at_cycle_start = {
            "encoder_b": 0.0,  # Starts unfrozen in Option C
            "controller": 0.0,  # Starts unfrozen
        }
        self.cycle_count = {"encoder_a": 0, "encoder_b": 0, "controller": 0}
        self.best_Q = 0.0  # Best Q achieved so far

    def update(
        self,
        epoch: int,
        coverage: float,
        hierarchy_A: float,
        hierarchy_B: float,
        dist_corr_A: float = 0.0,
        controller_grad_norm: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Update homeostasis state based on current metrics.

        Args:
            epoch: Current training epoch
            coverage: Current coverage (0-1)
            hierarchy_A: VAE-A radial hierarchy correlation (negative is good)
            hierarchy_B: VAE-B radial hierarchy correlation (negative is good)
            dist_corr_A: VAE-A distance correlation (for Q computation)
            controller_grad_norm: Optional gradient norm of controller params

        Returns:
            Dict with freeze states and any triggered events
        """
        # Compute current Q
        current_Q = compute_Q(dist_corr_A, hierarchy_A)

        # Update histories
        self.coverage_history.append(coverage)
        self.hierarchy_A_history.append(hierarchy_A)
        self.hierarchy_B_history.append(hierarchy_B)
        self.Q_history.append(current_Q)
        if controller_grad_norm is not None:
            self.controller_grad_history.append(controller_grad_norm)

        # Track best Q
        if current_Q > self.best_Q:
            self.best_Q = current_Q

        events = []

        # Skip homeostasis during warmup
        if epoch < self.warmup_epochs:
            return {
                "encoder_a_frozen": self.encoder_a_frozen,
                "encoder_b_frozen": self.encoder_b_frozen,
                "controller_frozen": self.controller_frozen,
                "current_Q": current_Q,
                "best_Q": self.best_Q,
                "events": ["warmup"],
            }

        # === Encoder A: Coverage-gated ===
        if self._can_change_state(epoch, self.encoder_a_last_change):
            encoder_a_decision = self._decide_encoder_a(coverage)
            if encoder_a_decision is not None:
                was_frozen = self.encoder_a_frozen
                self.encoder_a_frozen = encoder_a_decision
                self.encoder_a_last_change = epoch
                events.append(f"encoder_A {'frozen' if encoder_a_decision else 'unfrozen'}")

                # Q-gated annealing: check cycle completion
                if self.enable_annealing:
                    anneal_event = self._handle_cycle("encoder_a", was_frozen, encoder_a_decision, current_Q)
                    if anneal_event:
                        events.append(anneal_event)

        # === Encoder B: Hierarchy-gated ===
        if self._can_change_state(epoch, self.encoder_b_last_change):
            encoder_b_decision = self._decide_encoder_b()
            if encoder_b_decision is not None:
                was_frozen = self.encoder_b_frozen
                self.encoder_b_frozen = encoder_b_decision
                self.encoder_b_last_change = epoch
                events.append(f"encoder_B {'frozen' if encoder_b_decision else 'unfrozen'}")

                # Q-gated annealing
                if self.enable_annealing:
                    anneal_event = self._handle_cycle("encoder_b", was_frozen, encoder_b_decision, current_Q)
                    if anneal_event:
                        events.append(anneal_event)

        # === Controller: Gradient-gated ===
        if controller_grad_norm is not None:
            if self._can_change_state(epoch, self.controller_last_change):
                controller_decision = self._decide_controller()
                if controller_decision is not None:
                    was_frozen = self.controller_frozen
                    self.controller_frozen = controller_decision
                    self.controller_last_change = epoch
                    events.append(f"controller {'frozen' if controller_decision else 'unfrozen'}")

                    # Q-gated annealing
                    if self.enable_annealing:
                        anneal_event = self._handle_cycle(
                            "controller",
                            was_frozen,
                            controller_decision,
                            current_Q,
                        )
                        if anneal_event:
                            events.append(anneal_event)

        return {
            "encoder_a_frozen": self.encoder_a_frozen,
            "encoder_b_frozen": self.encoder_b_frozen,
            "controller_frozen": self.controller_frozen,
            "current_Q": current_Q,
            "best_Q": self.best_Q,
            "events": events,
        }

    def _handle_cycle(
        self,
        component: str,
        was_frozen: bool,
        now_frozen: bool,
        current_Q: float,
    ) -> Optional[str]:
        """Handle Q-gated annealing when a cycle completes.

        A cycle is: unfrozen -> frozen (component adapted then consolidated)

        Args:
            component: 'encoder_a', 'encoder_b', or 'controller'
            was_frozen: Previous freeze state
            now_frozen: New freeze state
            current_Q: Current Q value

        Returns:
            Event string if annealing occurred, None otherwise
        """
        # Cycle start: frozen -> unfrozen
        if was_frozen and not now_frozen:
            self.Q_at_cycle_start[component] = current_Q
            return None

        # Cycle end: unfrozen -> frozen
        if not was_frozen and now_frozen:
            self.cycle_count[component] += 1
            start_Q = self.Q_at_cycle_start.get(component, current_Q)
            Q_delta = current_Q - start_Q

            if Q_delta > 0:
                # Q improved - relax thresholds
                return self._anneal_thresholds(component, relax=True, Q_delta=Q_delta)
            elif Q_delta < -0.05:  # Significant decrease
                # Q decreased - tighten thresholds
                return self._anneal_thresholds(component, relax=False, Q_delta=Q_delta)

        return None

    def _anneal_thresholds(self, component: str, relax: bool, Q_delta: float) -> str:
        """Anneal thresholds for a component based on Q change.

        Args:
            component: Which component's thresholds to adjust
            relax: True to relax (allow more adaptation), False to tighten
            Q_delta: How much Q changed

        Returns:
            Event description string
        """
        direction = "relaxed" if relax else "tightened"
        step = self.annealing_step if relax else -self.annealing_step

        if component == "encoder_a":
            # Relax coverage thresholds (lower freeze, lower unfreeze)
            new_freeze = self.coverage_freeze_threshold - step
            new_unfreeze = self.coverage_unfreeze_threshold - step

            # Enforce floor
            if new_freeze >= self.coverage_floor:
                self.coverage_freeze_threshold = new_freeze
                self.coverage_unfreeze_threshold = max(new_unfreeze, new_freeze + 0.005)
                return f"encoder_A thresholds {direction} (freeze={self.coverage_freeze_threshold:.3f}, unfreeze={self.coverage_unfreeze_threshold:.3f}, Q_delta={Q_delta:+.3f})"
            else:
                return f"encoder_A at floor (coverage_floor={self.coverage_floor})"

        elif component == "encoder_b":
            # Relax hierarchy patience (more epochs before freeze)
            new_patience = self.hierarchy_plateau_patience + (1 if relax else -1)

            if relax and new_patience <= self.hierarchy_patience_ceiling:
                self.hierarchy_plateau_patience = new_patience
                return f"encoder_B patience {direction} (patience={self.hierarchy_plateau_patience}, Q_delta={Q_delta:+.3f})"
            elif not relax and new_patience >= self._initial_hierarchy_patience:
                self.hierarchy_plateau_patience = new_patience
                return f"encoder_B patience {direction} (patience={self.hierarchy_plateau_patience}, Q_delta={Q_delta:+.3f})"
            else:
                ceiling_or_floor = "ceiling" if relax else "floor"
                return f"encoder_B at {ceiling_or_floor} (patience={self.hierarchy_plateau_patience})"

        elif component == "controller":
            # Relax controller patience
            new_patience = self.controller_grad_patience + (1 if relax else -1)

            if relax and new_patience <= self.controller_patience_ceiling:
                self.controller_grad_patience = new_patience
                return f"controller patience {direction} (patience={self.controller_grad_patience}, Q_delta={Q_delta:+.3f})"
            elif not relax and new_patience >= self._initial_controller_patience:
                self.controller_grad_patience = new_patience
                return f"controller patience {direction} (patience={self.controller_grad_patience}, Q_delta={Q_delta:+.3f})"
            else:
                ceiling_or_floor = "ceiling" if relax else "floor"
                return f"controller at {ceiling_or_floor} (patience={self.controller_grad_patience})"

        return ""

    def _can_change_state(self, epoch: int, last_change: int) -> bool:
        """Check if enough epochs have passed since last state change (hysteresis)."""
        return epoch - last_change >= self.hysteresis_epochs

    def _decide_encoder_a(self, coverage: float) -> Optional[bool]:
        """Decide encoder_A freeze state based on coverage.

        Logic:
        - If unfrozen and coverage drops below threshold -> FREEZE
        - If frozen and coverage=100% AND hierarchy stalled -> UNFREEZE

        Returns:
            True to freeze, False to unfreeze, None for no change
        """
        if not self.encoder_a_frozen:
            # Currently unfrozen - check if should freeze
            if coverage < self.coverage_freeze_threshold:
                return True  # Freeze to protect coverage
        else:
            # Currently frozen - check if should unfreeze
            if coverage >= self.coverage_unfreeze_threshold:
                # Coverage is good, check if hierarchy is stalled
                if self._is_hierarchy_a_stalled():
                    return False  # Unfreeze to escape plateau

        return None  # No change

    def _is_hierarchy_a_stalled(self) -> bool:
        """Check if VAE-A hierarchy has plateaued."""
        if len(self.hierarchy_A_history) < self.window_size:
            return False

        recent = list(self.hierarchy_A_history)
        # Hierarchy is negative, so smaller (more negative) is better
        # Stalled = no improvement in recent window
        improvement = abs(recent[-1]) - abs(recent[0])

        if improvement < self.hierarchy_plateau_threshold:
            self.hierarchy_a_stall_count += 1
        else:
            self.hierarchy_a_stall_count = 0

        return self.hierarchy_a_stall_count >= self.hierarchy_stall_patience

    def _decide_encoder_b(self) -> Optional[bool]:
        """Decide encoder_B freeze state based on VAE-B hierarchy.

        Logic:
        - If unfrozen and hierarchy plateaus for patience epochs -> FREEZE
        - If frozen and hierarchy is not at target -> UNFREEZE

        Returns:
            True to freeze, False to unfreeze, None for no change
        """
        if len(self.hierarchy_B_history) < 2:
            return None

        if not self.encoder_b_frozen:
            # Check for plateau
            recent = list(self.hierarchy_B_history)
            improvement = abs(recent[-1]) - abs(recent[0])

            if improvement < self.hierarchy_plateau_threshold:
                self.hierarchy_b_plateau_count += 1
            else:
                self.hierarchy_b_plateau_count = 0

            if self.hierarchy_b_plateau_count >= self.hierarchy_plateau_patience:
                return True  # Freeze - hierarchy plateaued
        else:
            # Currently frozen - unfreeze if hierarchy degraded
            if len(self.hierarchy_B_history) >= 2:
                if abs(self.hierarchy_B_history[-1]) < abs(self.hierarchy_B_history[-2]) - 0.01:
                    self.hierarchy_b_plateau_count = 0
                    return False  # Unfreeze - hierarchy degraded

        return None

    def _decide_controller(self) -> Optional[bool]:
        """Decide controller freeze state based on gradient norm.

        Logic:
        - If unfrozen and grad norm low for patience epochs -> FREEZE
        - If frozen and grad norm spikes -> UNFREEZE

        Returns:
            True to freeze, False to unfreeze, None for no change
        """
        if len(self.controller_grad_history) < 2:
            return None

        current_grad = self.controller_grad_history[-1]

        if not self.controller_frozen:
            if current_grad < self.controller_grad_threshold:
                self.controller_low_grad_count += 1
            else:
                self.controller_low_grad_count = 0

            if self.controller_low_grad_count >= self.controller_grad_patience:
                return True  # Freeze - controller stabilized
        else:
            # Check for gradient spike (need to adapt again)
            avg_grad = sum(self.controller_grad_history) / len(self.controller_grad_history)
            if current_grad > avg_grad * 2:  # Spike = 2x average
                self.controller_low_grad_count = 0
                return False  # Unfreeze

        return None

    def get_state_summary(self) -> str:
        """Get human-readable summary of current freeze states and Q."""
        states = []
        states.append(f"enc_A:{'F' if self.encoder_a_frozen else 'T'}")
        states.append(f"enc_B:{'F' if self.encoder_b_frozen else 'T'}")
        states.append(f"ctrl:{'F' if self.controller_frozen else 'T'}")
        if self.Q_history:
            states.append(f"Q:{self.Q_history[-1]:.2f}")
        return " ".join(states)

    def get_annealing_summary(self) -> str:
        """Get summary of current annealing state."""
        parts = []
        parts.append(f"cov_freeze={self.coverage_freeze_threshold:.3f}")
        parts.append(f"cov_unfreeze={self.coverage_unfreeze_threshold:.3f}")
        parts.append(f"hier_pat={self.hierarchy_plateau_patience}")
        parts.append(f"ctrl_pat={self.controller_grad_patience}")
        parts.append(f"cycles={sum(self.cycle_count.values())}")
        parts.append(f"best_Q={self.best_Q:.2f}")
        return " | ".join(parts)

    def reset(self):
        """Reset all state for new training run."""
        self.coverage_history.clear()
        self.hierarchy_A_history.clear()
        self.hierarchy_B_history.clear()
        self.controller_grad_history.clear()
        self.Q_history.clear()

        self.encoder_a_frozen = True
        self.encoder_b_frozen = False
        self.controller_frozen = False

        self.encoder_a_last_change = -self.hysteresis_epochs
        self.encoder_b_last_change = -self.hysteresis_epochs
        self.controller_last_change = -self.hysteresis_epochs

        self.hierarchy_b_plateau_count = 0
        self.controller_low_grad_count = 0
        self.hierarchy_a_stall_count = 0

        # Reset annealing state
        self.Q_at_cycle_start = {}
        self.cycle_count = {"encoder_a": 0, "encoder_b": 0, "controller": 0}
        self.best_Q = 0.0

        # Reset thresholds to initial values
        self.coverage_freeze_threshold = self._initial_coverage_freeze
        self.coverage_unfreeze_threshold = self._initial_coverage_unfreeze
        self.hierarchy_plateau_patience = self._initial_hierarchy_patience
        self.controller_grad_patience = self._initial_controller_patience


__all__ = ["StateNet", "compute_Q"]
