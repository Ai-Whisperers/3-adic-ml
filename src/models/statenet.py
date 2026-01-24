# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Hierarchical StateNet Controller for V5.11.8.

Implements complementary learning systems theory with Q-gated annealing:
- Slow components (encoders): consolidate, fix when objective met
- Fast components (projections, controller): continuously adapt
- Q-gated annealing: thresholds relax only when Q improves

Each component has its own trainability trigger:
- encoder_A: coverage-gated (fix on drop, make trainable on recovery + hierarchy stall)
- encoder_B: hierarchy-gated (fix when VAE-B hierarchy plateaus)
- controller: gradient-gated (fix when weights stabilize)
- projections: always trainable (fast adaptation layer)

V5.11.8: Q-gated annealing progressively relaxes thresholds when Q improves,
enabling exploration of higher Q values while maintaining coverage floors.
"""

from collections import deque
from typing import Any, Dict, Optional

from src.config.constants import (
    STATENET_ANNEALING_STEP,
    STATENET_CONTROLLER_GRAD_PATIENCE,
    STATENET_CONTROLLER_GRAD_THRESHOLD,
    STATENET_CONTROLLER_PATIENCE_CEILING,
    STATENET_COVERAGE_FLOOR,
    STATENET_COVERAGE_FIX_THRESHOLD,
    STATENET_COVERAGE_TRAIN_THRESHOLD,
    STATENET_HIERARCHY_PATIENCE_CEILING,
    STATENET_HIERARCHY_PLATEAU_PATIENCE,
    STATENET_HIERARCHY_PLATEAU_THRESHOLD,
    STATENET_HYSTERESIS_EPOCHS,
    STATENET_WARMUP_EPOCHS,
    STATENET_WINDOW_SIZE,
)


def compute_Q(dist_corr: float, hierarchy: float) -> float:
    """Compute conserved structure capacity Q.

    Q = dist_corr + 1.5 × |hierarchy|

    Higher Q indicates better structure learning.
    """
    return dist_corr + 1.5 * abs(hierarchy)


class StateNet:
    """Hierarchical controller for component trainability with Q-gated annealing.

    Monitors training metrics and dynamically adjusts which components are
    trainable vs fixed, balancing coverage preservation with geometric structure.

    V5.11.8: Thresholds anneal (relax) only when Q improves after a cycle,
    and tighten when Q decreases. This creates a reversible ratchet.
    """

    def __init__(
        self,
        # Coverage thresholds for encoder_A (from constants.py)
        coverage_fix_threshold: float = STATENET_COVERAGE_FIX_THRESHOLD,
        coverage_train_threshold: float = STATENET_COVERAGE_TRAIN_THRESHOLD,
        # Hierarchy thresholds for encoder_B (from constants.py)
        hierarchy_plateau_threshold: float = STATENET_HIERARCHY_PLATEAU_THRESHOLD,
        hierarchy_plateau_patience: int = STATENET_HIERARCHY_PLATEAU_PATIENCE,
        # Controller gradient thresholds (from constants.py)
        controller_grad_threshold: float = STATENET_CONTROLLER_GRAD_THRESHOLD,
        controller_grad_patience: int = STATENET_CONTROLLER_GRAD_PATIENCE,
        # General settings (from constants.py)
        window_size: int = STATENET_WINDOW_SIZE,
        hysteresis_epochs: int = STATENET_HYSTERESIS_EPOCHS,
        warmup_epochs: int = STATENET_WARMUP_EPOCHS,
        # Q-gated annealing settings (V5.11.8, from constants.py)
        enable_annealing: bool = True,
        annealing_step: float = STATENET_ANNEALING_STEP,
        coverage_floor: float = STATENET_COVERAGE_FLOOR,
        hierarchy_patience_ceiling: int = STATENET_HIERARCHY_PATIENCE_CEILING,
        controller_patience_ceiling: int = STATENET_CONTROLLER_PATIENCE_CEILING,
    ):
        # Initial thresholds (can be annealed)
        self.coverage_fix_threshold = coverage_fix_threshold
        self.coverage_train_threshold = coverage_train_threshold
        self.hierarchy_plateau_threshold = hierarchy_plateau_threshold
        self.hierarchy_plateau_patience = hierarchy_plateau_patience
        self.controller_grad_threshold = controller_grad_threshold
        self.controller_grad_patience = controller_grad_patience

        # Store initial values for reference
        self._initial_coverage_fix = coverage_fix_threshold
        self._initial_coverage_train = coverage_train_threshold
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

        # Trainability states (True = parameters update, False = parameters fixed)
        self.encoder_a_trainable = False  # Starts fixed (coverage anchor)
        self.encoder_b_trainable = True   # Starts trainable (hierarchy learner)
        self.controller_trainable = True  # Starts trainable

        # Last state change epoch (for hysteresis)
        self.encoder_a_last_change = -hysteresis_epochs
        self.encoder_b_last_change = -hysteresis_epochs
        self.controller_last_change = -hysteresis_epochs

        # Plateau counters
        self.hierarchy_b_plateau_count = 0
        self.controller_low_grad_count = 0

        # Conditions to make encoder_A trainable
        self.hierarchy_a_stalled = False
        self.hierarchy_a_stall_count = 0
        self.hierarchy_stall_patience = 5

        # Q-gated annealing state
        # Initialize cycle start Q for ALL components
        # - encoder_a: starts fixed, will be set when it becomes trainable
        # - encoder_b: starts trainable, needs initial Q=0 for first cycle
        # - controller: starts trainable, needs initial Q=0 for first cycle
        self.Q_at_cycle_start = {
            "encoder_a": 0.0,
            "encoder_b": 0.0,
            "controller": 0.0,
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
        """Update statenet state based on current metrics.

        Args:
            epoch: Current training epoch
            coverage: Current coverage (0-1)
            hierarchy_A: VAE-A radial hierarchy correlation (negative is good)
            hierarchy_B: VAE-B radial hierarchy correlation (negative is good)
            dist_corr_A: VAE-A distance correlation (for Q computation)
            controller_grad_norm: Optional gradient norm of controller params

        Returns:
            Dict with trainability states and any triggered events
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

        # Skip statenet during warmup
        if epoch < self.warmup_epochs:
            return {
                "encoder_a_trainable": self.encoder_a_trainable,
                "encoder_b_trainable": self.encoder_b_trainable,
                "controller_trainable": self.controller_trainable,
                "current_Q": current_Q,
                "best_Q": self.best_Q,
                "events": ["warmup"],
            }

        # === Encoder A: Coverage-gated ===
        if self._can_change_state(epoch, self.encoder_a_last_change):
            encoder_a_decision = self._decide_encoder_a_trainable(coverage)
            if encoder_a_decision is not None:
                was_trainable = self.encoder_a_trainable
                self.encoder_a_trainable = encoder_a_decision
                self.encoder_a_last_change = epoch
                events.append(f"encoder_A {'trainable' if encoder_a_decision else 'fixed'}")

                # Q-gated annealing: check cycle completion
                if self.enable_annealing:
                    anneal_event = self._handle_cycle("encoder_a", was_trainable, encoder_a_decision, current_Q)
                    if anneal_event:
                        events.append(anneal_event)

        # === Encoder B: Hierarchy-gated ===
        if self._can_change_state(epoch, self.encoder_b_last_change):
            encoder_b_decision = self._decide_encoder_b_trainable()
            if encoder_b_decision is not None:
                was_trainable = self.encoder_b_trainable
                self.encoder_b_trainable = encoder_b_decision
                self.encoder_b_last_change = epoch
                events.append(f"encoder_B {'trainable' if encoder_b_decision else 'fixed'}")

                # Q-gated annealing
                if self.enable_annealing:
                    anneal_event = self._handle_cycle("encoder_b", was_trainable, encoder_b_decision, current_Q)
                    if anneal_event:
                        events.append(anneal_event)

        # === Controller: Gradient-gated ===
        if controller_grad_norm is not None:
            if self._can_change_state(epoch, self.controller_last_change):
                controller_decision = self._decide_controller_trainable()
                if controller_decision is not None:
                    was_trainable = self.controller_trainable
                    self.controller_trainable = controller_decision
                    self.controller_last_change = epoch
                    events.append(f"controller {'trainable' if controller_decision else 'fixed'}")

                    # Q-gated annealing
                    if self.enable_annealing:
                        anneal_event = self._handle_cycle(
                            "controller",
                            was_trainable,
                            controller_decision,
                            current_Q,
                        )
                        if anneal_event:
                            events.append(anneal_event)

        return {
            "encoder_a_trainable": self.encoder_a_trainable,
            "encoder_b_trainable": self.encoder_b_trainable,
            "controller_trainable": self.controller_trainable,
            "current_Q": current_Q,
            "best_Q": self.best_Q,
            "events": events,
        }

    def _handle_cycle(
        self,
        component: str,
        was_trainable: bool,
        now_trainable: bool,
        current_Q: float,
    ) -> Optional[str]:
        """Handle Q-gated annealing when a cycle completes.

        A cycle is: trainable -> fixed (component adapted then consolidated)

        Args:
            component: 'encoder_a', 'encoder_b', or 'controller'
            was_trainable: Previous trainability state
            now_trainable: New trainability state
            current_Q: Current Q value

        Returns:
            Event string if annealing occurred, None otherwise
        """
        # Cycle start: fixed -> trainable
        if not was_trainable and now_trainable:
            self.Q_at_cycle_start[component] = current_Q
            return None

        # Cycle end: trainable -> fixed
        if was_trainable and not now_trainable:
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
            # Relax coverage thresholds (lower fix, lower train)
            new_fix = self.coverage_fix_threshold - step
            new_train = self.coverage_train_threshold - step

            # Enforce floor
            if new_fix >= self.coverage_floor:
                self.coverage_fix_threshold = new_fix
                self.coverage_train_threshold = max(new_train, new_fix + 0.005)
                return f"encoder_A thresholds {direction} (fix={self.coverage_fix_threshold:.3f}, train={self.coverage_train_threshold:.3f}, Q_delta={Q_delta:+.3f})"
            else:
                return f"encoder_A at floor (coverage_floor={self.coverage_floor})"

        elif component == "encoder_b":
            # Relax hierarchy patience (more epochs before fixing)
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

    def _decide_encoder_a_trainable(self, coverage: float) -> Optional[bool]:
        """Decide encoder_A trainability based on coverage.

        Logic:
        - If trainable and coverage drops below threshold -> make FIXED
        - If fixed and coverage=100% AND hierarchy stalled -> make TRAINABLE

        Returns:
            True to make trainable, False to make fixed, None for no change
        """
        if self.encoder_a_trainable:
            # Currently trainable - check if should fix
            if coverage < self.coverage_fix_threshold:
                return False  # Fix to protect coverage
        else:
            # Currently fixed - check if should make trainable
            if coverage >= self.coverage_train_threshold:
                # Coverage is good, check if hierarchy is stalled
                if self._is_hierarchy_a_stalled():
                    return True  # Make trainable to escape plateau

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

    def _decide_encoder_b_trainable(self) -> Optional[bool]:
        """Decide encoder_B trainability based on VAE-B hierarchy.

        Logic:
        - If trainable and hierarchy plateaus for patience epochs -> make FIXED
        - If fixed and hierarchy is degrading -> make TRAINABLE

        Returns:
            True to make trainable, False to make fixed, None for no change
        """
        if len(self.hierarchy_B_history) < 2:
            return None

        if self.encoder_b_trainable:
            # Check for plateau
            recent = list(self.hierarchy_B_history)
            improvement = abs(recent[-1]) - abs(recent[0])

            if improvement < self.hierarchy_plateau_threshold:
                self.hierarchy_b_plateau_count += 1
            else:
                self.hierarchy_b_plateau_count = 0

            if self.hierarchy_b_plateau_count >= self.hierarchy_plateau_patience:
                return False  # Fix - hierarchy plateaued
        else:
            # Currently fixed - make trainable if hierarchy degraded
            if len(self.hierarchy_B_history) >= 2:
                if abs(self.hierarchy_B_history[-1]) < abs(self.hierarchy_B_history[-2]) - 0.01:
                    self.hierarchy_b_plateau_count = 0
                    return True  # Make trainable - hierarchy degraded

        return None

    def _decide_controller_trainable(self) -> Optional[bool]:
        """Decide controller trainability based on gradient norm.

        Logic:
        - If trainable and grad norm low for patience epochs -> make FIXED
        - If fixed and grad norm spikes -> make TRAINABLE

        Returns:
            True to make trainable, False to make fixed, None for no change
        """
        if len(self.controller_grad_history) < 2:
            return None

        current_grad = self.controller_grad_history[-1]

        if self.controller_trainable:
            if current_grad < self.controller_grad_threshold:
                self.controller_low_grad_count += 1
            else:
                self.controller_low_grad_count = 0

            if self.controller_low_grad_count >= self.controller_grad_patience:
                return False  # Fix - controller stabilized
        else:
            # Check for gradient spike (need to adapt again)
            avg_grad = sum(self.controller_grad_history) / len(self.controller_grad_history)
            if current_grad > avg_grad * 2:  # Spike = 2x average
                self.controller_low_grad_count = 0
                return True  # Make trainable

        return None

    def get_state_summary(self) -> str:
        """Get human-readable summary of current trainability states and Q."""
        states = []
        states.append(f"enc_A:{'train' if self.encoder_a_trainable else 'fixed'}")
        states.append(f"enc_B:{'train' if self.encoder_b_trainable else 'fixed'}")
        states.append(f"ctrl:{'train' if self.controller_trainable else 'fixed'}")
        if self.Q_history:
            states.append(f"Q:{self.Q_history[-1]:.2f}")
        return " ".join(states)

    def get_annealing_summary(self) -> str:
        """Get summary of current annealing state."""
        parts = []
        parts.append(f"cov_fix={self.coverage_fix_threshold:.3f}")
        parts.append(f"cov_train={self.coverage_train_threshold:.3f}")
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

        self.encoder_a_trainable = False
        self.encoder_b_trainable = True
        self.controller_trainable = True

        self.encoder_a_last_change = -self.hysteresis_epochs
        self.encoder_b_last_change = -self.hysteresis_epochs
        self.controller_last_change = -self.hysteresis_epochs

        self.hierarchy_b_plateau_count = 0
        self.controller_low_grad_count = 0
        self.hierarchy_a_stall_count = 0

        # Reset annealing state (must match __init__)
        self.Q_at_cycle_start = {
            "encoder_a": 0.0,
            "encoder_b": 0.0,
            "controller": 0.0,
        }
        self.cycle_count = {"encoder_a": 0, "encoder_b": 0, "controller": 0}
        self.best_Q = 0.0

        # Reset thresholds to initial values
        self.coverage_fix_threshold = self._initial_coverage_fix
        self.coverage_train_threshold = self._initial_coverage_train
        self.hierarchy_plateau_patience = self._initial_hierarchy_patience
        self.controller_grad_patience = self._initial_controller_patience


__all__ = ["StateNet", "compute_Q"]
