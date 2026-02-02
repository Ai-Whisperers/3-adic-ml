# Critical Test Targets for P-Adic VAE

This document defines the testing strategy for the 3-adic ML VAE codebase, focusing on **meaningful validations** that catch real bugs rather than trivial smoke tests.

## Philosophy

- **Test mathematical invariants** - properties that MUST hold
- **Test state transitions** - behavior changes that can be verified
- **Test edge cases** - boundary conditions that could break
- **Skip trivial tests** - things Python/PyTorch already guarantees

---

## Tier 1: Mathematical Invariants (Highest Priority)

These catch foundational bugs that cascade through the entire system.

### 1.1 Core Ternary Algebra (`src/core/ternary.py`)

| Property | Test Description | Bug It Catches |
|----------|------------------|----------------|
| **Valuation formula** | `v_3(0)=9, v_3(1)=0, v_3(3)=1, v_3(9)=2, v_3(27)=3, v_3(81)=4` | Wrong hierarchy assignment |
| **Ultrametric inequality** | `d(a,c) ≤ max(d(a,b), d(b,c))` for sampled triples | Metric space violation |
| **Round-trip invertibility** | `from_ternary(to_ternary(i)) == i` for all 19,683 | Data corruption |
| **Distance symmetry** | `d(a,b) == d(b,a)` for all pairs | Asymmetric metric |
| **Level count sum** | `sum(level_count(v) for v=0..9) == 19683` | Missing/duplicate indices |
| **Target radius monotonicity** | `target_r(v) > target_r(v+1)` strictly | Inverted hierarchy targets |
| **Digit representation** | All digits in `{-1, 0, 1}` after conversion | Wrong base-3 encoding |

**Why critical**: The entire hierarchy system depends on correct 3-adic valuations. A bug here means the model learns the wrong geometry.

### 1.2 Hyperbolic Geometry (`src/geometry/poincare.py`)

| Property | Test Description | Bug It Catches |
|----------|------------------|----------------|
| **exp/log composition** | `log_map_zero(exp_map_zero(v)) ≈ v` within `atol=1e-6` | Broken manifold mapping |
| **Ball containment** | `||exp_map_zero(v)|| < 1` for all reasonable v | Points escaping manifold |
| **Distance positivity** | `poincare_distance(x, y) ≥ 0` always | Negative distances |
| **Distance identity** | `poincare_distance(x, x) == 0` | Self-distance non-zero |
| **Distance symmetry** | `poincare_distance(x, y) == poincare_distance(y, x)` | Asymmetric metric |
| **Triangle inequality** | `d(x,z) ≤ d(x,y) + d(y,z)` | Invalid metric |
| **Conformal factor positivity** | `lambda_x(z) > 0` for all valid z | Division by zero in KL |
| **Curvature effect** | Higher c produces larger distances | Curvature parameter ignored |

**Why critical**: Hyperbolic geometry is the bridge from discrete 3-adic to continuous. Numerical errors here corrupt all training.

---

## Tier 2: Loss Function Correctness (High Priority)

These verify the optimization signal is mathematically correct.

### 2.1 P-Adic Geodesic Losses (`src/losses/padic_geodesic.py`)

| Loss Class | Property | Bug It Catches |
|------------|----------|----------------|
| **PAdicGeodesicLoss** | `target_distance(v=9) < target_distance(v=0)` | Inverted distance mapping |
| **PAdicGeodesicLoss** | Correlation in `[-1, 1]`, not NaN | Numerical instability |
| **RadialHierarchyLoss** | Loss = 0 when radii exactly match targets | False positive violations |
| **RadialHierarchyLoss** | Target radii monotonically decrease with v | Inverted targets |
| **MonotonicRadialLoss** | Detects violation when `r[v] < r[v+1]` | Missed hierarchy collapse |
| **MonotonicRadialLoss** | No violation when `r[v] > r[v+1] + margin` | False violations |
| **GlobalRankLoss** | Violation rate in `[0, 1]` always | Metric overflow |
| **RichHierarchyLoss** | Each component ≥ 0 | Negative loss components |

### 2.2 Gradient Flow

| Test | What It Verifies |
|------|------------------|
| `loss.backward()` produces `grad != 0` on `z_hyp` | Optimization actually works |
| Gradients flow through `exp_map_zero` | Manifold projection is differentiable |
| Frozen encoder has `grad == None` or `grad == 0` | LR controller freeze works |
| Learnable loss weights receive gradients | Weight learning is functional |

### 2.3 Combined Loss (`src/losses/combined.py`)

| Property | Test Description | Bug It Catches |
|----------|------------------|----------------|
| **Weight formula** | `w = 0.5 * exp(-2 * log_sigma)` inverts correctly | Wrong weight computation |
| **Phase gating** | Geodesic loss = 0 before `phase_start_epoch` | Premature loss activation |
| **Total aggregation** | Sum of weighted components equals total | Missing loss terms |
| **Learnable weight bounds** | Weights stay positive and finite | exp overflow/underflow |

---

## Tier 3: State Machine Correctness (Medium-High Priority)

### 3.1 LR Controller (`src/models/lr_controller.py`)

| Transition | Trigger Condition | Bug It Catches |
|------------|-------------------|----------------|
| encoder_a: frozen → active | `coverage ≥ train_threshold AND hierarchy_stalled` | Never unfreezes |
| encoder_a: active → frozen | `coverage < fix_threshold` | Never re-freezes |
| encoder_b: active → frozen | Plateau for `patience` epochs | Doesn't detect plateau |
| encoder_b: frozen → active | Hierarchy degrades | Stays frozen forever |
| projections: active → frozen | `grad_norm < threshold` for patience | Over-aggressive freezing |
| projections: frozen → active | Gradient spike detected | Stays frozen forever |
| **Hysteresis enforcement** | Can't toggle within `hysteresis_epochs` | Rapid oscillation |
| **Warmup period** | Initial states respected during warmup | Premature transitions |

### 3.2 VAE Trainability (`src/models/vae.py`)

| Test | What It Verifies |
|------|------------------|
| `set_encoder_a_trainable(False)` → all params `requires_grad=False` | Freeze actually works |
| After `set_trainable(True)`, gradients accumulate | Unfreeze restores training |
| Param groups have correct LR scales | Optimizer integration works |
| `get_trainability_summary()` matches actual state | State reporting accurate |

---

## Tier 4: Edge Cases (Medium Priority)

These prevent crashes in production.

| Module | Edge Case | Expected Behavior |
|--------|-----------|-------------------|
| `TernarySpace` | Empty tensor `valuation(torch.tensor([]))` | Returns empty tensor |
| `TernarySpace` | Out-of-range index 99999 | Clamps silently to valid range |
| `TernarySpace` | Boundary indices 0 and 19682 | Correct valuation returned |
| `poincare_distance` | Points at origin `z = [0,0,...]` | Returns 0 distance |
| `poincare_distance` | Points near boundary `||z|| = 0.999` | Doesn't overflow, finite result |
| `exp_map_zero` | Zero tangent vector | Returns origin |
| `exp_map_zero` | Very large tangent vector | Clamps near boundary |
| `PAdicGeodesicLoss` | batch_size = 1 | Returns loss = 0, no crash |
| `RichHierarchyLoss` | All samples same valuation | Separation loss = 0 |
| `CombinedLoss` | No losses enabled | Falls back to coverage only |
| `MetricBasedLR` | First epoch (no history) | Uses initial states |
| `MetricBasedLR` | All metrics = 0 | Doesn't crash, sensible defaults |

---

## What NOT to Test (Anti-Patterns)

These tests provide zero value:

| Anti-Pattern | Why It's Useless |
|--------------|------------------|
| `assert model(x).shape == (B, 27)` | PyTorch guarantees shapes via layer definitions |
| `import src.models; assert True` | Python handles imports; catches nothing |
| `model = VAE(); assert model is not None` | Constructors can't return None in Python |
| `config = StateNetConfig()` | Dataclass instantiation is Python's job |
| `assert isinstance(loss, float)` | Type checking is static analysis, not runtime |
| `checkpoint = load(path); assert checkpoint` | Tests filesystem, not your code |
| `assert len(params) > 0` | Module construction guarantees this |

---

## Numerical Sensitivity Guidelines

These operations require tolerance-based assertions:

| Operation | Recommended Tolerance | Reason |
|-----------|----------------------|--------|
| `exp_map/log_map` composition | `atol=1e-6, rtol=1e-5` | Float64 accumulation in geoopt |
| Distance computation | `rtol=1e-5` | geoopt internal precision |
| Correlation coefficient | Check `isnan()` separately | Zero variance produces NaN |
| Learned weight formula | `atol=1e-8` | exp/log chain precision |
| Gradient magnitudes | `atol=1e-10` for "is zero" | Floating point noise |

---

## Test File Structure

### Implemented (214 tests)

```
tests/
├── conftest.py                     # Shared fixtures, deterministic seeding
│
├── test_core_ternary.py            # Tier 1: Mathematical invariants (28 tests)
│   ├── TestValuationFormula (6 tests)
│   ├── TestUltrametricInequality (2 tests)
│   ├── TestDistanceSymmetry (2 tests)
│   ├── TestRoundTripInvertibility (2 tests)
│   ├── TestLevelCounts (3 tests)
│   ├── TestTargetRadiusMonotonicity (4 tests)
│   ├── TestDistanceMatrix (3 tests)
│   ├── TestEdgeCases (4 tests)
│   └── TestDeviceConsistency (2 tests)
│
├── test_core_ternary_extended.py   # Tier 1: Formula verification (29 tests)
│   ├── TestDistanceFormulaComputation (3 tests)
│   ├── TestPropertyAccessorsAgainstDirectComputation (6 tests)
│   ├── TestTreeStructureRelationships (7 tests)
│   ├── TestLevelRankConsistency (2 tests)
│   ├── TestLevelMask (2 tests)
│   ├── TestValidationFunctions (3 tests)
│   ├── TestValuationHistogram (2 tests)
│   └── TestTernaryRepresentationConsistency (3 tests)
│
├── test_geometry_poincare.py       # Tier 1: Manifold properties (29 tests)
│   ├── TestExpLogComposition (3 tests)
│   ├── TestBallContainment (4 tests)
│   ├── TestDistancePositivity (2 tests)
│   ├── TestDistanceIdentity (2 tests)
│   ├── TestDistanceSymmetry (1 test)
│   ├── TestTriangleInequality (1 test)
│   ├── TestHyperbolicRadius (3 tests)
│   ├── TestConformalFactor (4 tests)
│   ├── TestCurvatureEffect (2 tests)
│   ├── TestManifoldCache (2 tests)
│   ├── TestNumericalStability (3 tests)
│   └── TestDeviceConsistency (2 tests)
│
├── test_geometry_poincare_extended.py  # Tier 1: Utilities & formulas (27 tests)
│   ├── TestProjectToPoincare (3 tests)
│   ├── TestMobiusAdd (3 tests)
│   ├── TestParallelTransport (2 tests)
│   ├── TestGeodesic (3 tests)
│   ├── TestPoincareDistanceMatrix (4 tests)
│   ├── TestExpMapZeroFormula (4 tests)
│   ├── TestLogMapZeroFormula (2 tests)
│   ├── TestConformalFactorFormula (2 tests)
│   ├── TestHyperbolicRadiusFormula (2 tests)
│   └── TestCurvatureScaling (2 tests)
│
├── test_losses.py                  # Tier 2: Loss correctness (64 tests)
│   ├── TestGradientFlowPAdicGeodesic (3 tests)
│   ├── TestGradientFlowRadialHierarchy (2 tests)
│   ├── TestGradientFlowGlobalRank (2 tests)
│   ├── TestGradientFlowMonotonicRadial (2 tests)
│   ├── TestGradientFlowRichHierarchy (3 tests)
│   ├── TestGradientFlowCombinedGeodesic (1 test)
│   ├── TestLossNonNegativityPAdicGeodesic (8 tests)
│   ├── TestLossNonNegativityRadialHierarchy (3 tests)
│   ├── TestLossNonNegativityGlobalRank (3 tests)
│   ├── TestLossNonNegativityMonotonicRadial (3 tests)
│   ├── TestLossNonNegativityRichHierarchy (4 tests)
│   ├── TestTargetDistanceMonotonicity (5 tests)
│   ├── TestTargetRadiusMonotonicity (4 tests)
│   ├── TestMetricBoundsPAdicGeodesic (4 tests)
│   ├── TestMetricBoundsRadialHierarchy (2 tests)
│   ├── TestMetricBoundsGlobalRank (2 tests)
│   ├── TestMetricBoundsMonotonicRadial (3 tests)
│   ├── TestEdgeCasesBatchSize (2 tests)
│   ├── TestEdgeCasesSameValuation (2 tests)
│   ├── TestEdgeCasesNearBoundary (2 tests)
│   ├── TestConsistencyHyperbolicRadius (2 tests)
│   └── TestReproducibility (2 tests)
│
└── test_losses_combined.py         # Tier 2: CombinedLoss & learnable weights (37 tests)
    ├── TestCombinedLossInstantiation (4 tests)
    ├── TestCombinedLossForward (3 tests)
    ├── TestPhaseGating (4 tests)
    ├── TestWeightToLogSigmaFormula (2 tests)
    ├── TestUncertaintyWeightFormula (2 tests)
    ├── TestWeightRoundTrip (2 tests)
    ├── TestLearnableWeightsInitialization (3 tests)
    ├── TestLearnableWeightsGradientFlow (3 tests)
    ├── TestLearnableWeightsBehavior (3 tests)
    ├── TestFixedVsLearnableWeights (5 tests)
    ├── TestCombinedLossEdgeCases (4 tests)
    └── TestCombinedLossRepr (2 tests)
```

### Not Yet Implemented

```
tests/
├── test_lr_controller.py      # Tier 3: State machine
├── test_vae_trainability.py   # Tier 3: Model integration
└── test_edge_cases.py         # Tier 4: Boundary conditions
```

---

## Priority Order for Implementation

1. ✅ **`test_core_ternary.py`** - Foundation of everything; if wrong, all else fails
2. ✅ **`test_geometry_poincare.py`** - Bridge to continuous geometry; numerical bugs hide here
3. ✅ **`test_losses.py`** - Verify training signal is mathematically correct
4. ✅ **`test_losses_combined.py`** - Verify CombinedLoss and learnable weights
5. **`test_lr_controller.py`** - Verify adaptation mechanism
6. **`test_vae_trainability.py`** - Integration verification
7. **`test_edge_cases.py`** - Production stability

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run only Tier 1 (critical)
pytest tests/test_core_ternary.py tests/test_geometry_poincare.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_core_ternary.py::test_valuation_known_values -v
```

---

## Success Criteria

A test suite is considered adequate when:

1. All Tier 1 tests pass with documented tolerances
2. Gradient flow tests confirm backprop works through all loss paths
3. State transition tests cover all controller branches
4. Edge cases don't cause crashes or NaN
5. No trivial "import worked" or "shape is correct" tests exist
