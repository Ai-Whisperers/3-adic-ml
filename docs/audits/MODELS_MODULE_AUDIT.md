# Models Module Audit: src/models/ (V6.0)

**Date**: 2025-01-24 (Updated from 2025-01-23)
**Scope**: `src/models/__init__.py`, `vae.py`, `hyperbolic_projection.py`, `statenet.py`
**Version**: V6.0 (True Hyperbolic Architecture)

---

## Executive Summary

**V6.0 Status**: The models module now implements **true hyperbolic learning** via geoopt's `expmap0`/`logmap0` operations. The critical architectural flaw from V5.11 (decoder ignoring z_hyp) has been fixed.

### V6.0 Changes Applied

| Issue | V5.11 Status | V6.0 Status |
|-------|--------------|-------------|
| Decoder uses z_euc | ❌ Critical flaw | ✅ Fixed - uses `logmap0(z_hyp)` |
| Euclidean projection | ❌ direction × radius | ✅ Fixed - uses `expmap0` |
| Class names | `TernaryVAEV5_11_PartialFreeze` | ✅ `TernaryVAEV6Controllable` |
| Freeze terminology | `encoder_a_frozen` | ✅ `encoder_a_trainable` |
| V5.5 backward compat | Key mapping, deprecated params | ✅ Removed |

**Verdict**: The module is now **architecturally correct** for true manifold learning.

**Rating**: 8/10 (was 6/10)

---

## Module Exports (__init__.py)

| Export | Type | Purpose |
|--------|------|---------|
| `StateNet` | Class | Q-gated trainability controller |
| `compute_Q` | Function | Structure capacity metric |
| `HyperbolicProjection` | Class | Single VAE projection (expmap0) |
| `DualHyperbolicProjection` | Class | Dual VAE projections |
| `TernaryVAEV6` | Class | Base dual VAE |
| `TernaryVAEV6Controllable` | Class | VAE with StateNet support |

---

## Architecture Flow (V6.0 - True Hyperbolic)

```
Input (B, 9) ternary operations
    │
    ├─→ encoder_A → fc_mu_A, fc_logvar_A → z_A_tangent (reparameterize)
    │                                          │
    │                                          ├─→ expmap0 → z_A_hyp (on manifold)
    │                                          │                │
    │                                          │                ├─→ losses (poincare_distance)
    │                                          │                │
    │                                          │                └─→ logmap0 → decoder_A → logits_A
    │
    └─→ encoder_B → fc_mu_B, fc_logvar_B → z_B_tangent (reparameterize)
                                               │
                                               └─→ expmap0 → z_B_hyp → logmap0 → decoder_B → logits_B
```

**Key insight**: Tangent space at origin T₀M IS Euclidean ℝⁿ, so standard MLPs and Gaussian sampling work there.

---

## Detailed Analysis

### 1. vae.py - TernaryVAEV6

#### 1.1 Forward Pass (V6.0)

```python
# V6.0 Implementation:
z_A_tangent = self.reparameterize(mu_A, logvar_A)
z_A_hyp = self.projections.proj_A(z_A_tangent)  # Uses expmap0

# Decoder uses logmap0 (back to tangent space)
z_A_decoded = log_map_zero(z_A_hyp, c=self.curvature)
logits_A = self.decoder_A(z_A_decoded)
```

| Aspect | V5.11 | V6.0 |
|--------|-------|------|
| z_hyp computation | ✅ | ✅ |
| z_hyp used in decoder | ❌ | ✅ via logmap0 |
| Generative model | Euclidean | Hyperbolic |

#### 1.2 Reparameterization

```python
def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std  # Samples in TANGENT SPACE (which IS Euclidean at origin)
```

| Aspect | Status | Notes |
|--------|--------|-------|
| Mathematical correctness | ✅ Correct | Standard Gaussian in tangent space |
| Gradients flow | ✅ Yes | Through mu and logvar |
| Tangent space sampling | ✅ Correct | T₀M is Euclidean, so this is valid |

#### 1.3 Removed V5.5 Compatibility

- ❌ `V5_5_TO_V5_11_KEY_MAP` - Removed
- ❌ `map_v5_5_keys()` - Removed
- ❌ `from_v5_5_checkpoint()` - Removed
- ❌ `compute_control` parameter - Removed
- ❌ Deprecated kwargs (`use_controller`, `use_dual_projection`, etc.) - Removed

---

### 2. vae.py - TernaryVAEV6Controllable

#### 2.1 Trainability Control (V6.0 Terminology)

```python
def set_encoder_a_trainable(self, trainable: bool):
    """Set encoder A trainability (True = parameters update)."""
    self._encoder_a_trainable = trainable
    for p in self.encoder_A.parameters():
        p.requires_grad = trainable  # Direct logic, no inversion
```

| Old Name | New Name | Logic |
|----------|----------|-------|
| `set_encoder_a_frozen(frozen)` | `set_encoder_a_trainable(trainable)` | Positive |
| `_encoder_a_frozen` | `_encoder_a_trainable` | Positive |
| `get_freeze_state_summary()` | `get_trainability_summary()` | Clearer |

#### 2.2 StateNet Integration

```python
def apply_statenet_state(self, state: Dict[str, Any]):
    """Apply trainability states from StateNet controller."""
    if "encoder_a_trainable" in state:
        self.set_encoder_a_trainable(state["encoder_a_trainable"])
    if "encoder_b_trainable" in state:
        self.set_encoder_b_trainable(state["encoder_b_trainable"])
```

---

### 3. hyperbolic_projection.py - HyperbolicProjection

#### 3.1 V6.0 Architecture

```python
def forward(self, z_tangent: torch.Tensor) -> torch.Tensor:
    # Transform in tangent space
    z_transformed = self.tangent_net(z_tangent)

    # Project to manifold via expmap0
    z_hyp = exp_map_zero(z_transformed, c=self.curvature)

    # Clamp to max_radius for numerical stability
    norm = z_hyp.norm(dim=-1, keepdim=True)
    z_hyp = torch.where(
        norm > self.max_radius,
        z_hyp * self.max_radius / norm,
        z_hyp
    )
    return z_hyp
```

| Aspect | V5.11 | V6.0 |
|--------|-------|------|
| Projection method | direction × radius | expmap0 |
| Manifold operations | None | geoopt expmap0 |
| Gradients | Euclidean | Riemannian-aware |

---

### 4. statenet.py - StateNet Controller

#### 4.1 V6.0 Terminology

| Old | New | Meaning |
|-----|-----|---------|
| `encoder_a_frozen` | `encoder_a_trainable` | `True` = parameters update |
| `encoder_b_frozen` | `encoder_b_trainable` | `True` = parameters update |
| `controller_frozen` | `controller_trainable` | `True` = parameters update |
| `coverage_freeze_threshold` | `coverage_fix_threshold` | Threshold to fix encoder |
| `coverage_unfreeze_threshold` | `coverage_train_threshold` | Threshold to allow training |

#### 4.2 Decision Logic (V6.0)

```python
def _decide_encoder_a_trainable(self, coverage: float) -> Optional[bool]:
    """Returns True to make trainable, False to fix, None for no change."""
    if self.encoder_a_trainable:
        if coverage < self.coverage_fix_threshold:
            return False  # Fix to protect coverage
    else:
        if coverage >= self.coverage_train_threshold:
            if self._is_hierarchy_a_stalled():
                return True  # Make trainable to escape plateau
    return None
```

#### 4.3 State Summary (V6.0)

```python
def get_state_summary(self) -> str:
    """Returns: 'enc_A:train enc_B:fixed ctrl:train Q:1.23'"""
    states = []
    states.append(f"enc_A:{'train' if self.encoder_a_trainable else 'fixed'}")
    states.append(f"enc_B:{'train' if self.encoder_b_trainable else 'fixed'}")
    states.append(f"ctrl:{'train' if self.controller_trainable else 'fixed'}")
```

---

## Issues Summary

### Resolved (V6.0)

| Issue | Location | Resolution |
|-------|----------|------------|
| ~~Decoder uses z_euc~~ | `vae.py` | ✅ Uses `logmap0(z_hyp)` |
| ~~Euclidean reparameterization~~ | `vae.py` | ✅ Tangent space IS Euclidean |
| ~~Euclidean projection~~ | `hyperbolic_projection.py` | ✅ Uses `expmap0` |
| ~~Confusing freeze terminology~~ | All files | ✅ Uses trainable |
| ~~V5.5 backward compat~~ | `vae.py` | ✅ Removed |

### Remaining (Low Priority)

| Issue | Location | Status |
|-------|----------|--------|
| Magic number 1.5 in Q formula | `statenet.py:52` | Verifying - may be intentional |
| Hardcoded -0.05 threshold | `statenet.py:295` | Verifying |

---

## Code Quality Assessment (V6.0)

| Metric | V5.11 | V6.0 | Notes |
|--------|-------|------|-------|
| Correctness | 5/10 | 9/10 | Architecture now correct |
| Architecture | 7/10 | 9/10 | True hyperbolic flow |
| Documentation | 8/10 | 8/10 | Good docstrings |
| Maintainability | 7/10 | 8/10 | Clearer naming |
| Geoopt Integration | 4/10 | 8/10 | Uses expmap0/logmap0 |
| True Hyperbolic | 2/10 | 9/10 | ✅ Now functional |

---

## File-by-File Summary (V6.0)

### vae.py
- **Purpose**: Dual VAE with true hyperbolic geometry
- **Rating**: 8/10 (was 6/10)
- **Classes**: `TernaryVAEV6`, `TernaryVAEV6Controllable`
- **Key fix**: Decoder uses `logmap0(z_hyp)`

### hyperbolic_projection.py
- **Purpose**: Tangent → Manifold projection via expmap0
- **Rating**: 8/10 (was 7/10)
- **Key fix**: Uses `expmap0` instead of direction × radius

### statenet.py
- **Purpose**: Q-gated trainability controller
- **Rating**: 8/10
- **Key fix**: Trainable terminology (positive logic)

### __init__.py
- **Purpose**: Module exports
- **Rating**: 10/10
- **Updated exports**: `TernaryVAEV6`, `TernaryVAEV6Controllable`

---

## Verdict (V6.0)

**The models module now implements true hyperbolic learning.** The critical V5.11 flaw (decoder ignoring z_hyp) has been fixed. The architecture correctly uses:

1. **Tangent space sampling** (which IS Euclidean at origin)
2. **expmap0** to project to the Poincaré manifold
3. **logmap0** to return to tangent space for decoding

The "trainable" terminology provides clear, positive logic for component control.

**Current State**: True Hyperbolic VAE with manifold-aware projections
**Rating**: 8/10 (Good)

---

**Audit updated**: 2025-01-24
**Auditor**: Claude Opus 4.5
