# Models Module Audit: src/models/

**Date**: 2025-01-23
**Scope**: `src/models/__init__.py`, `vae.py`, `hyperbolic_projection.py`, `statenet.py`
**Lines of Code**: 442 (vae.py) + 328 (hyperbolic_projection.py) + 526 (statenet.py) + 3 (__init__.py) = ~1299 total

---

## Executive Summary

The models module implements a **dual VAE architecture** with hyperbolic projections and a StateNet controller for dynamic freeze/unfreeze training. While the individual components are well-implemented, the **critical architectural flaw** is that the decoder uses Euclidean latents (`z_euc`) and completely ignores the hyperbolic projections (`z_hyp`). This makes the current system a **Euclidean VAE with hyperbolic loss supervision**, not true hyperbolic learning.

**Verdict**: The module is **functional but architecturally flawed** for true manifold learning. The hyperbolic projections are decorative—they affect only the loss, not the generative process.

---

## File Structure

```
src/models/
├── __init__.py              # Clean re-exports (4 symbols)
├── vae.py                   # Dual VAE classes (442 lines)
├── hyperbolic_projection.py # Direction/radius projection (328 lines)
└── statenet.py              # Q-gated freeze controller (526 lines)
```

---

## Module Exports (__init__.py)

| Export | Type | Purpose |
|--------|------|---------|
| `StateNet` | Class | Q-gated freeze/unfreeze controller |
| `compute_Q` | Function | Structure capacity metric |
| `HyperbolicProjection` | Class | Single VAE projection |
| `DualHyperbolicProjection` | Class | Dual VAE projections |
| `TernaryVAEV5_11` | Class | Base dual VAE |
| `TernaryVAEV5_11_PartialFreeze` | Class | VAE with StateNet support |

**Assessment**: Clean, minimal exports.

---

## Detailed Analysis

### 1. vae.py - TernaryVAEV5_11

#### 1.1 Architecture Overview

```
Input (B, 9) ternary operations
    │
    ├─→ encoder_A → fc_mu_A, fc_logvar_A → z_A_euc (reparameterize)
    │                                          │
    │                                          ├─→ proj_A → z_A_hyp (UNUSED BY DECODER!)
    │                                          │
    │                                          └─→ decoder_A → logits_A (27)
    │
    └─→ encoder_B → fc_mu_B, fc_logvar_B → z_B_euc (reparameterize)
                                               │
                                               ├─→ proj_B → z_B_hyp (UNUSED BY DECODER!)
                                               │
                                               └─→ decoder_B → logits_B (27)
```

#### 1.2 Critical Flaw: Decoder Input (Lines 246-247)

```python
# Decode from Euclidean latents
logits_A = self.decoder_A(z_A_euc)  # ← z_A_hyp is IGNORED!
logits_B = self.decoder_B(z_B_euc)  # ← z_B_hyp is IGNORED!
```

| Aspect | Status | Impact |
|--------|--------|--------|
| z_hyp computed | ✅ Yes | Projections are computed |
| z_hyp used in decoder | ❌ No | Decoders use z_euc |
| Hyperbolic loss receives z_hyp | ✅ Yes | Losses see hyperbolic projections |
| Generative model is hyperbolic | ❌ No | Generation happens in Euclidean space |

**Consequence**: The VAE learns to reconstruct from Euclidean space while the loss supervises hyperbolic structure. The hyperbolic geometry is a **loss-level regularizer**, not the learned representation.

#### 1.3 Reparameterization (Lines 220-224)

```python
def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
```

| Aspect | Status | Notes |
|--------|--------|-------|
| Mathematical correctness | ✅ Correct | Standard Gaussian reparameterization |
| Gradients flow | ✅ Yes | Through mu and logvar |
| Hyperbolic sampling | ❌ Missing | Samples in Euclidean, should use wrapped normal |

**For true hyperbolic**: Should sample in tangent space then `expmap0` to manifold.

#### 1.4 Encoder/Decoder Builders (Lines 72-131)

```python
def build_encoder(hidden_dim: int, encoder_type: str = "improved") -> nn.Sequential:
    if encoder_type == "improved":
        # SiLU + LayerNorm architecture
        return nn.Sequential(
            nn.Linear(9, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
        )
```

| Aspect | Status | Notes |
|--------|--------|-------|
| Improved architecture | ✅ Good | SiLU + LayerNorm is modern choice |
| v5.5 compatibility | ✅ Good | "standard" mode preserves old architecture |
| Output dimension | ⚠️ Inconsistent | "improved" outputs `hidden_dim`, "standard" outputs 64 |
| Decoder symmetry | ✅ Good | Mirror architecture of encoder |

#### 1.5 V5.5 Checkpoint Loading (Lines 267-302)

```python
@classmethod
def from_v5_5_checkpoint(cls, checkpoint_path, device, **model_kwargs):
    model_kwargs.setdefault("encoder_type", "standard")
    model_kwargs.setdefault("decoder_type", "standard")
    ...
    mapped_state = map_v5_5_keys(state_dict)
    model.load_state_dict(mapped_state, strict=False)
```

| Aspect | Status | Notes |
|--------|--------|-------|
| Key mapping | ✅ Correct | Properly maps nested keys to flat keys |
| strict=False | ⚠️ Silent failures | Missing keys won't error |
| Return value | ⚠️ Missing | Doesn't return missing/unexpected keys |

---

### 2. vae.py - TernaryVAEV5_11_PartialFreeze

#### 2.1 Freeze Control (Lines 341-359)

```python
def set_encoder_a_frozen(self, frozen: bool):
    self._encoder_a_frozen = frozen
    for p in self.encoder_A.parameters():
        p.requires_grad = not frozen
    for p in self.fc_mu_A.parameters():
        p.requires_grad = not frozen
    for p in self.fc_logvar_A.parameters():
        p.requires_grad = not frozen
```

| Aspect | Status | Notes |
|--------|--------|-------|
| Freeze implementation | ✅ Correct | Sets requires_grad appropriately |
| State tracking | ✅ Good | Maintains `_encoder_X_frozen` flags |
| Decoders not freezeable | ⚠️ Design choice | Only encoders can be frozen |

#### 2.2 Differential Learning Rates (Lines 374-418)

```python
def get_param_groups(self, base_lr: float) -> List[Dict[str, Any]]:
    groups = []
    # Encoder A (scaled LR)
    enc_a_params = [p for p in self.encoder_A.parameters() if p.requires_grad]
    ...
    groups.append({
        "params": enc_a_params,
        "lr": base_lr * self.encoder_a_lr_scale,
        "name": "encoder_A",
    })
```

| Aspect | Status | Notes |
|--------|--------|-------|
| Per-component LR | ✅ Good | Enables fine-grained control |
| Frozen param filtering | ✅ Good | Only includes trainable params |
| Group naming | ✅ Good | Named groups for logging |

---

### 3. hyperbolic_projection.py - HyperbolicProjection

#### 3.1 Architecture

```
z_euclidean
    │
    ├─→ direction_net → residual → z + residual → normalize → direction (unit vector)
    │
    └─→ radius_net → sigmoid → [0, 1] → × max_radius → radius

z_hyp = direction × radius  (in Poincaré ball)
```

#### 3.2 Direction Network (Lines 94-125)

```python
if n_layers == 1:
    layers = [
        nn.Linear(latent_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.SiLU(),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_dim, latent_dim))
    self.direction_net = nn.Sequential(*layers)
```

| Aspect | Status | Notes |
|--------|--------|-------|
| Residual design | ✅ Good | Output is added to input, preserving information |
| Normalization | ✅ Good | F.normalize ensures unit direction |
| Identity init | ✅ Good | Starts as identity for stability |

#### 3.3 Radius Network (Lines 128-147)

```python
radius_hidden = max(32, hidden_dim // 2)
layers = [nn.Linear(latent_dim, radius_hidden), nn.SiLU()]
...
layers.extend([nn.Linear(radius_hidden, 1), nn.Sigmoid()])
self.radius_net = nn.Sequential(*layers)
```

| Aspect | Status | Notes |
|--------|--------|-------|
| Output range | ✅ Correct | Sigmoid → [0, 1] × max_radius |
| Smaller network | ✅ Good | Radius is simpler than direction |
| Initial bias | ⚠️ Mid-range | Initialized to output ~0.5 |

#### 3.4 Forward Pass (Lines 168-196)

```python
def forward(self, z_euclidean: torch.Tensor, as_manifold: bool = False):
    direction_residual = self.direction_net(z_euclidean)
    direction = F.normalize(z_euclidean + direction_residual, dim=-1)
    radius = self.radius_net(z_euclidean) * self.max_radius
    z_hyp = direction * radius

    if as_manifold:
        z_hyp = self.manifold.projx(z_hyp)
        return ManifoldParameter(z_hyp, manifold=self.manifold)
    return z_hyp
```

| Aspect | Status | Notes |
|--------|--------|-------|
| Direction computation | ✅ Correct | Residual + normalize |
| Radius computation | ✅ Correct | Learned scalar |
| Projection | ✅ Correct | Uses geoopt projx |
| ManifoldParameter | ✅ Good | Proper manifold-aware output |

#### 3.5 Critical Issue: Not Using Manifold Operations

The projection creates points in the Poincaré ball by **Euclidean arithmetic** (direction × radius), not manifold operations (expmap). This is geometrically valid for placing points, but:

1. **No gradient flow through manifold**: Gradients are Euclidean, not Riemannian
2. **No expmap0**: Points are placed directly, not via tangent space
3. **Curvature is decorative**: The `c` parameter affects only validation, not learning

**For true hyperbolic**: Should compute `expmap0(v)` where `v = direction * arctanh(radius * sqrt(c)) / sqrt(c)`.

#### 3.6 Learnable Curvature (Lines 86-90)

```python
if learnable_curvature:
    self.manifold = geoopt.PoincareBall(c=curvature, learnable=True)
else:
    self.manifold = geoopt.PoincareBall(c=curvature)
self.curvature = self.manifold.c
```

| Aspect | Status | Notes |
|--------|--------|-------|
| Geoopt integration | ✅ Correct | Uses geoopt's learnable curvature |
| Manifold creation | ✅ Good | Single manifold per projection |
| Curvature access | ✅ Good | Exposed via get_curvature() |

---

### 4. hyperbolic_projection.py - DualHyperbolicProjection

#### 4.1 Architecture

```python
self.proj_A = HyperbolicProjection(...)  # VAE-A projection

if share_direction:
    self.proj_B_radius = nn.Sequential(...)  # B uses A's direction
else:
    self.proj_B = HyperbolicProjection(...)  # Separate projection
```

| Aspect | Status | Notes |
|--------|--------|-------|
| Independent projections | ✅ Good | Each VAE gets own mapping |
| Shared direction option | ✅ Interesting | Can share angular structure |
| Consistent interface | ✅ Good | Same forward signature |

---

### 5. statenet.py - StateNet Controller

#### 5.1 Design Philosophy

```
Complementary Learning Systems:
- Slow (encoders): consolidate knowledge, freeze when objective met
- Fast (projections, controller): continuously adapt to new patterns

Q-gated Annealing:
- Q = dist_corr + 1.5 × |hierarchy|
- Thresholds relax when Q improves
- Thresholds tighten when Q decreases
```

#### 5.2 Freeze Logic Summary

| Component | Freeze Condition | Unfreeze Condition |
|-----------|-----------------|-------------------|
| encoder_A | Coverage drops below threshold | Coverage recovered AND hierarchy stalled |
| encoder_B | Hierarchy plateaus for patience epochs | Hierarchy degrades |
| controller | Gradient norm low for patience epochs | Gradient spikes (2x average) |

#### 5.3 Q Computation (Lines 45-52)

```python
def compute_Q(dist_corr: float, hierarchy: float) -> float:
    return dist_corr + 1.5 * abs(hierarchy)
```

| Aspect | Status | Notes |
|--------|--------|-------|
| Formula | ✅ Simple | Combines coverage and structure |
| Weighting | ⚠️ Arbitrary | 1.5 weight is hardcoded magic number |
| Sign handling | ✅ Good | abs(hierarchy) handles negative correlation |

#### 5.4 Coverage-Gated Encoder A (Lines 362-383)

```python
def _decide_encoder_a(self, coverage: float) -> Optional[bool]:
    if not self.encoder_a_frozen:
        if coverage < self.coverage_freeze_threshold:
            return True  # Freeze to protect coverage
    else:
        if coverage >= self.coverage_unfreeze_threshold:
            if self._is_hierarchy_a_stalled():
                return False  # Unfreeze to escape plateau
    return None
```

| Aspect | Status | Notes |
|--------|--------|-------|
| Freeze trigger | ✅ Clear | Coverage drop → freeze |
| Unfreeze trigger | ✅ Clear | Recovery + stall → unfreeze |
| Dual condition | ✅ Good | Prevents thrashing |

#### 5.5 Q-Gated Annealing (Lines 261-299)

```python
def _handle_cycle(self, component, was_frozen, now_frozen, current_Q):
    # Cycle start: frozen -> unfrozen
    if was_frozen and not now_frozen:
        self.Q_at_cycle_start[component] = current_Q
        return None

    # Cycle end: unfrozen -> frozen
    if not was_frozen and now_frozen:
        Q_delta = current_Q - self.Q_at_cycle_start[component]
        if Q_delta > 0:
            return self._anneal_thresholds(component, relax=True, Q_delta=Q_delta)
        elif Q_delta < -0.05:
            return self._anneal_thresholds(component, relax=False, Q_delta=Q_delta)
```

| Aspect | Status | Notes |
|--------|--------|-------|
| Cycle tracking | ✅ Good | Records Q at cycle start |
| Bi-directional annealing | ✅ Good | Can relax or tighten |
| Threshold for tightening | ⚠️ Hardcoded | -0.05 is magic number |

#### 5.6 Hysteresis (Lines 358-360)

```python
def _can_change_state(self, epoch: int, last_change: int) -> bool:
    return epoch - last_change >= self.hysteresis_epochs
```

| Aspect | Status | Notes |
|--------|--------|-------|
| Purpose | ✅ Good | Prevents rapid oscillation |
| Default | ✅ Reasonable | Uses constant from config |

---

## Issues Summary

### Critical (1)

| Issue | Location | Description |
|-------|----------|-------------|
| C1 | `vae.py:246-247` | **Decoder uses z_euc, ignores z_hyp** - hyperbolic projections don't affect generation |

### High (2)

| Issue | Location | Description |
|-------|----------|-------------|
| H1 | `vae.py:220-224` | Reparameterization is Euclidean, not wrapped normal on manifold |
| H2 | `hyperbolic_projection.py:188` | Projection uses Euclidean arithmetic, not expmap0 |

### Medium (4)

| Issue | Location | Description |
|-------|----------|-------------|
| M1 | `statenet.py:52` | Magic number 1.5 in Q formula |
| M2 | `statenet.py:295` | Hardcoded -0.05 threshold for Q tightening |
| M3 | `vae.py:300` | `load_state_dict` returns missing/unexpected but discards them |
| M4 | `hyperbolic_projection.py:90` | Curvature stored but not used in projection math |

### Low (3)

| Issue | Location | Description |
|-------|----------|-------------|
| L1 | `vae.py:181` | Encoder output dim differs between "improved" (hidden_dim) and "standard" (64) |
| L2 | `hyperbolic_projection.py:166` | Radius bias init zeros mean initial radius ≈ 0.5 × max_radius |
| L3 | `statenet.py:121` | encoder_a_frozen = True initial state is hardcoded |

---

## Integration Analysis

### Geometry Module Usage

| Component | Uses src/geometry? | Notes |
|-----------|-------------------|-------|
| vae.py | ❌ No | Only imports DualHyperbolicProjection |
| hyperbolic_projection.py | ⚠️ Partial | Imports ManifoldParameter, geoopt directly |
| statenet.py | ❌ No | Pure Python logic, no geometry |

**Issue**: hyperbolic_projection.py bypasses src/geometry/ and uses geoopt directly. Should use `get_manifold()`, `exp_map_zero()`, etc. for consistency.

### Data Flow Analysis

```
Input (9 dims)
    │
    ├─→ encoder_A (Euclidean MLP)
    │       │
    │       ├─→ mu_A, logvar_A (Euclidean)
    │       │       │
    │       │       └─→ z_A_euc (Euclidean, via reparameterize)
    │       │               │
    │       │               ├─→ z_A_hyp (Poincaré, via projection)
    │       │               │       │
    │       │               │       └─→ [ONLY USED BY LOSS]
    │       │               │
    │       │               └─→ decoder_A (Euclidean MLP)
    │       │                       │
    │       │                       └─→ logits_A (27 dims)
    ...
```

The **generative path** (encoder → sample → decoder) is entirely Euclidean. The hyperbolic projection is a **dead end** for generation.

---

## Architectural Recommendations

### For True Hyperbolic Learning

1. **Use z_hyp in decoder**:
   ```python
   # Current (broken)
   logits_A = self.decoder_A(z_A_euc)

   # Option A: Decode from hyperbolic directly
   logits_A = self.decoder_A(z_A_hyp)

   # Option B: Use logmap0 bridge (tangent space)
   z_A_tangent = logmap0(z_A_hyp, c=self.curvature)
   logits_A = self.decoder_A(z_A_tangent)
   ```

2. **Hyperbolic reparameterization**:
   ```python
   def reparameterize_hyperbolic(self, mu, logvar, c=1.0):
       # Sample in tangent space at mu
       std = torch.exp(0.5 * logvar)
       eps = torch.randn_like(std)
       v = eps * std  # Tangent vector at mu

       # Transport to origin, then expmap
       # Or: use wrapped normal distribution
       return expmap(mu, v, c=c)
   ```

3. **Use src/geometry functions**:
   ```python
   from src.geometry import get_manifold, exp_map_zero, log_map_zero

   class HyperbolicProjection(nn.Module):
       def forward(self, z_euclidean):
           manifold = get_manifold(self.curvature, device=z_euclidean.device)
           # z_euclidean is in tangent space T_0 M
           return manifold.expmap0(z_euclidean)
   ```

4. **Riemannian KL divergence**:
   ```python
   # Current KL is Euclidean:
   # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))

   # Hyperbolic KL needs conformal factor:
   # KL_hyp = KL_euc * lambda(mu)^2
   ```

### For Cleaner Architecture

1. **Unify geometry access**: Use src/geometry/ consistently, don't import geoopt directly
2. **Make projection mode configurable**: "euclidean_with_supervision" vs "true_hyperbolic"
3. **Extract magic numbers**: Move 1.5, -0.05, etc. to constants.py
4. **Return checkpoint load results**: Don't silently discard missing keys

---

## Code Quality Assessment

| Metric | Score | Notes |
|--------|-------|-------|
| Correctness | 5/10 | Critical decoder flaw undermines hyperbolic claims |
| Architecture | 7/10 | Clean separation, but broken data flow |
| Documentation | 8/10 | Good docstrings, accurate version notes |
| Maintainability | 7/10 | Well-organized, but scattered magic numbers |
| Geoopt Integration | 4/10 | Minimal use, bypasses src/geometry/ |
| True Hyperbolic | 2/10 | Projections are decorative, not functional |

---

## File-by-File Summary

### vae.py (442 lines)
- **Purpose**: Dual VAE architecture with hyperbolic projections
- **Rating**: 6/10
- **Critical Issue**: Decoder ignores z_hyp, uses z_euc
- **Strengths**: Clean class hierarchy, checkpoint compatibility

### hyperbolic_projection.py (328 lines)
- **Purpose**: Direction/radius projection to Poincaré ball
- **Rating**: 7/10
- **Issue**: Uses Euclidean math, not expmap
- **Strengths**: Clever direction/radius separation, identity init

### statenet.py (526 lines)
- **Purpose**: Q-gated freeze/unfreeze controller
- **Rating**: 8/10
- **Issues**: Magic numbers, complex state logic
- **Strengths**: Well-designed annealing, clear freeze conditions

### __init__.py (3 lines)
- **Purpose**: Module exports
- **Rating**: 10/10
- **No issues**: Clean, minimal

---

## Verdict

**The models module is well-engineered at the component level but fundamentally broken for true hyperbolic learning.** The hyperbolic projections exist but don't participate in the generative process—they're loss-level supervision signals that regularize the Euclidean VAE toward hyperbolic structure.

To achieve true p-adic/hyperbolic learning, the decoder must use z_hyp (or its logmap0 to tangent space) instead of z_euc. This is a **one-line fix** in principle but requires careful handling of gradients and KL divergence.

**Current State**: Euclidean VAE with hyperbolic loss supervision
**Target State**: Hyperbolic VAE with manifold-aware sampling and decoding

**Rating**: 6/10 (Functional but architecturally misleading)

---

**Audit completed**: 2025-01-23
**Auditor**: Claude Opus 4.5
