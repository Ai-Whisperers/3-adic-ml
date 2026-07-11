# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Lagrangian Dual Adaptive Weighting for per-level constraint enforcement.

Phase 3B of the Valuation-Conditioned Prior plan.

Instead of manually tuning loss weights, Lagrangian dual variables adaptively
increase pressure on violated constraints and decrease on satisfied ones.

Design principles:
  - LagrangianDualState is NOT a nn.Module or nn.Parameter.
    It is outer-loop state updated by dual ascent AFTER each validation epoch,
    not by the optimizer's gradient step.

  - Three constraint types, one lambda per constraint instance:
      lambda_margin[0..8]  — monotonic radial gap for each adjacent valuation pair
      lambda_scatter[0..9] — within-level radial spread for each valuation level
      lambda_prior[0..9]   — μ-norm prior gap for each valuation level

  - Dual ascent rule: λ_v ← clip(λ_v + lr * violation_v, 0, max_lambda)
    Violation > 0 → increase pressure. Violation ≤ 0 → decrease (clamp at 0).

  - Dual weights are FLOATS, not tensors. They enter CombinedLoss as scalar
    multipliers on already-in-graph per-level violation tensors. This decoupling
    means: dual update uses float violations (no backward needed), gradient flow
    is still achieved in the forward pass by (float_lambda * tensor_violation).

Usage:
    dual_state = LagrangianDualState.from_config(config.get('lagrangian', {}))

    # After each validation epoch:
    dual_state.update(avg_violation_floats)
    dual_weights = dual_state.get_dual_weights()

    # In training forward pass:
    losses = loss_fn(..., dual_weights=dual_weights)

    # Checkpoint:
    ckpt['lagrangian_state'] = dual_state.state_dict()
    dual_state.load_state_dict(ckpt['lagrangian_state'])
"""

from typing import Any, Dict, List


class LagrangianDualState:
    """Lagrangian dual variables for per-level geometric constraint enforcement.

    NOT a nn.Module. Updated by dual ascent (not gradient descent).
    Saved to/loaded from checkpoints separately from model.state_dict().

    Required call order per epoch:
        1. dual_state.step_epoch(epoch)   # at epoch start, before training
        2. ... train + validate ...
        3. dual_state.update(violations)  # after validation, with per-level metrics
        4. dual_state.get_dual_weights()  # for next epoch's forward pass
    """

    def __init__(
        self,
        n_levels: int = 10,
        lr: float = 0.01,
        max_lambda: float = 5.0,
        warmup_epochs: int = 20,
    ):
        """Initialise with zero dual variables.

        Args:
            n_levels: Number of valuation levels (10 for p=3, 3^9 data).
            lr: Dual ascent step size.
            max_lambda: Maximum dual variable value (prevents runaway).
            warmup_epochs: Epoch before which dual variables are frozen at 0.
                Allows primal optimisation to settle before constraints activate.
        """
        self.n_levels = n_levels
        self.lr = lr
        self.max_lambda = max_lambda
        self.warmup_epochs = warmup_epochs
        self._epoch = 0

        # Dual variables: one per constraint instance
        self.lambda_margin: List[float] = [0.0] * (n_levels - 1)   # gaps v0-v1 .. v8-v9
        self.lambda_scatter: List[float] = [0.0] * n_levels          # within-level spread
        self.lambda_prior: List[float] = [0.0] * n_levels            # μ-norm prior

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "LagrangianDualState":
        """Construct from a config dict (from YAML loss.lagrangian section)."""
        return cls(
            n_levels=cfg.get('n_levels', 10),
            lr=cfg.get('lr', 0.01),
            max_lambda=cfg.get('max_lambda', 5.0),
            warmup_epochs=cfg.get('warmup_epochs', 20),
        )

    def step_epoch(self, epoch: int) -> None:
        """Advance internal epoch counter."""
        self._epoch = epoch

    def update(self, violations: Dict[str, float]) -> None:
        """Dual ascent update: increase λ where violated, clamp at 0 where satisfied.

        Args:
            violations: Dict of float violation magnitudes, keyed as:
                'gap_viol_v{v}'   — margin violation for gap v (v to v+1)
                'scatter_v{v}'    — within-level scatter at level v
                'vp_gap_v{v}'     — prior norm gap at level v
            Only keys present in the dict are updated.
        """
        if self._epoch < self.warmup_epochs:
            return  # frozen during warmup

        for v in range(self.n_levels - 1):
            key = f'gap_viol_v{v}'
            if key in violations:
                self.lambda_margin[v] = max(0.0, min(
                    self.max_lambda,
                    self.lambda_margin[v] + self.lr * violations[key]
                ))

        for v in range(self.n_levels):
            sc_key = f'scatter_v{v}'
            if sc_key in violations:
                self.lambda_scatter[v] = max(0.0, min(
                    self.max_lambda,
                    self.lambda_scatter[v] + self.lr * violations[sc_key]
                ))

            pr_key = f'vp_gap_v{v}'
            if pr_key in violations:
                self.lambda_prior[v] = max(0.0, min(
                    self.max_lambda,
                    self.lambda_prior[v] + self.lr * violations[pr_key]
                ))

    def get_dual_weights(self) -> Dict[str, List[float]]:
        """Return current dual variable vectors.

        Returns:
            Dict with keys 'lambda_margin', 'lambda_scatter', 'lambda_prior'.
            Values are Lists of floats. During warmup all values are 0.
        """
        if self._epoch < self.warmup_epochs:
            return {
                'lambda_margin': [0.0] * (self.n_levels - 1),
                'lambda_scatter': [0.0] * self.n_levels,
                'lambda_prior': [0.0] * self.n_levels,
            }
        return {
            'lambda_margin': list(self.lambda_margin),
            'lambda_scatter': list(self.lambda_scatter),
            'lambda_prior': list(self.lambda_prior),
        }

    def is_active(self) -> bool:
        """Return True if past warmup and at least one lambda is non-zero."""
        if self._epoch < self.warmup_epochs:
            return False
        return (
            any(x > 0 for x in self.lambda_margin) or
            any(x > 0 for x in self.lambda_scatter) or
            any(x > 0 for x in self.lambda_prior)
        )

    def summary(self) -> str:
        """One-line human-readable summary of current lambda values."""
        max_m = max(self.lambda_margin) if self.lambda_margin else 0.0
        max_s = max(self.lambda_scatter) if self.lambda_scatter else 0.0
        max_p = max(self.lambda_prior) if self.lambda_prior else 0.0
        n_active_m = sum(1 for x in self.lambda_margin if x > 0)
        n_active_s = sum(1 for x in self.lambda_scatter if x > 0)
        n_active_p = sum(1 for x in self.lambda_prior if x > 0)
        n_margin = len(self.lambda_margin)
        n_levels = len(self.lambda_scatter)
        return (
            f"Lagrangian ep={self._epoch} "
            f"margin({n_active_m}/{n_margin} active, max={max_m:.3f}) "
            f"scatter({n_active_s}/{n_levels} active, max={max_s:.3f}) "
            f"prior({n_active_p}/{n_levels} active, max={max_p:.3f})"
        )

    def state_dict(self) -> Dict[str, Any]:
        """Serialise for checkpoint saving."""
        return {
            'lambda_margin': list(self.lambda_margin),
            'lambda_scatter': list(self.lambda_scatter),
            'lambda_prior': list(self.lambda_prior),
            'epoch': self._epoch,
            'lr': self.lr,
            'max_lambda': self.max_lambda,
            'warmup_epochs': self.warmup_epochs,
            'n_levels': self.n_levels,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore from a saved state dict."""
        self.lambda_margin = list(state['lambda_margin'])
        self.lambda_scatter = list(state['lambda_scatter'])
        self.lambda_prior = list(state['lambda_prior'])
        self._epoch = state.get('epoch', 0)
        self.lr = state.get('lr', self.lr)
        self.max_lambda = state.get('max_lambda', self.max_lambda)
        self.warmup_epochs = state.get('warmup_epochs', self.warmup_epochs)
        self.n_levels = state.get('n_levels', self.n_levels)
