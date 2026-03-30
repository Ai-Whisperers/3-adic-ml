# Audit Index: 3-adic-ml (V6.0/V6.1)

**Last Updated**: 2026-02-26
**Auditors**: Claude Opus 4.5 (2025-01), Claude Opus 4.6 (2026-02)
**Version**: V6.1 (True Hyperbolic + Learnable Weights)
---

## Overview

This directory contains comprehensive audits of the p-adic-vaes codebase. The audits are organized into three categories:

1. **Module Audits** - Per-module code reviews
2. **Integration Audits** - Cross-cutting concerns and integration analysis
3. **Previous Audits** - Historical audits for reference

---

## V6.0 Architecture Status

### Critical Issues Resolution

| Issue | Severity | Status | Resolution |
|-------|----------|--------|------------|
| **Decoder uses z_euc** | Critical | ✅ Fixed | Decoder now uses `logmap0(z_hyp)` |
| **Euclidean reparameterization** | High | ✅ Fixed | Sample in tangent space (IS Euclidean at origin) |
| **Euclidean projection math** | High | ✅ Fixed | Uses `expmap0` via geoopt |
| **"Freeze" terminology** | Medium | ✅ Fixed | Now uses "trainable" (positive logic) |
| **V5.5 backward compat cruft** | Low | ✅ Removed | No more key mapping, deprecated params |
| **Class naming** | Low | ✅ Fixed | `TernaryVAEV6`, `TernaryVAEV6Controllable` |

### Current Class Names

| Old Name | New Name |
|----------|----------|
| `TernaryVAEV5_11` | `TernaryVAEV6` |
| `TernaryVAEV5_11_PartialFreeze` | `TernaryVAEV6Controllable` |

### StateNet Terminology

| Old | New |
|-----|-----|
| `encoder_a_frozen` | `encoder_a_trainable` |
| `coverage_freeze_threshold` | `coverage_fix_threshold` |
| `coverage_unfreeze_threshold` | `coverage_train_threshold` |

---

## Module Audits

| Module | File | Rating (2025-01) | Rating (2026-02) |
|--------|------|------------------|------------------|
| **src/core/** | [CORE_MODULE_AUDIT.md](CORE_MODULE_AUDIT.md) | 7.5/10 | **9.5/10** |
| **src/geometry/** | [GEOMETRY_MODULE_AUDIT.md](GEOMETRY_MODULE_AUDIT.md) | 8/10 | **8.5/10** |
| **src/losses/** | [LOSSES_MODULE_AUDIT.md](LOSSES_MODULE_AUDIT.md) | 8/10 | **7.5/10** ↓ |
| **src/models/** | [MODELS_MODULE_AUDIT.md](MODELS_MODULE_AUDIT.md) | 8/10 | **7/10** ↓ |
| **src/utils/** | [UTILS_MODULE_AUDIT.md](UTILS_MODULE_AUDIT.md) | 7/10 | **6.5/10** ↓ |
| **src/config/** | [CONFIG_PRESETS_MODULE_AUDIT.md](CONFIG_PRESETS_MODULE_AUDIT.md) | 8/10 | **5/10** ↓ |

### Module Ratings (2026-02-26)

```
src/core/       █████████▌  9.5/10  Exemplary
src/geometry/   ████████▌░  8.5/10  Good (improved)
src/losses/     ███████▌░░  7.5/10  Dead code, config drift
src/models/     ███████░░░  7/10    Needs tests, dead code
src/train.py    ███████░░░  7/10    Works after fixes
src/utils/      ██████▌░░░  6.5/10  Stale files, no tests
src/config/     █████░░░░░  5/10    20+ ignored YAML keys
```

---

## Integration Audits

| Topic | File | V6.0 Status |
|-------|------|-------------|
| **Geoopt Integration** | [GEOOPT_INTEGRATION_AUDIT.md](GEOOPT_INTEGRATION_AUDIT.md) | ✅ Core requirements implemented |
| **Real Non-Euclidean** | [real-non-euclidean.md](real-non-euclidean.md) | Verifying |
| **TensorBoard Logs** | [tensorboard-logs.md](tensorboard-logs.md) | Unchanged |
| **Terminal Monitoring** | [terminal-monitoring-integration.md](terminal-monitoring-integration.md) | Unchanged |

---

## Key Files (V6.1)

| File | Purpose |
|------|---------|
| `src/models/vae.py` | `TernaryVAEV6`, `TernaryVAEV6Controllable` |
| `src/models/lr_controller.py` | `MetricBasedLR`, LR scale control |
| `src/models/hyperbolic_projection.py` | expmap0/logmap0 projections |
| `src/geometry/poincare.py` | Riemannian backend (geoopt) |
| `src/losses/combined.py` | Config-driven loss composition (V6.1 learnable weights) |
| `src/losses/padic_geodesic.py` | All hierarchy/geodesic losses |
| `src/train.py` | Unified training entry point |
| `src/presets/v6.yaml` | Main V6 training config |
| `src/presets/5.12.4.yaml` | Extended grokking config |
---

## Training Readiness

**Status**: NOT TRAINING-READY (see [MASTER_AUDIT.md](MASTER_AUDIT.md))

| Severity | Count | Key Areas |
|----------|-------|-----------|
| CRITICAL | 2 | GrokkingDetector crash, 5.12.4 scheduler mismatch |
| HIGH | 4 | 20+ silent config keys, DataLoader determinism, no tests, cudnn order |
| MODERATE | 6 | Dead code (~507 lines), performance, generator checkpoint |
| LOW | 12 | Unused imports, security, cleanup |
| INFO | 4 | Design observations |
---

## Positive Findings

| Finding | Module | Impact |
|---------|--------|--------|
| True hyperbolic via expmap0/logmap0 | models | ✅ Architecture is now correct |
| Correct poincare_distance usage | losses | All losses compute hyperbolic radii correctly |
| O(1) valuation lookups | core | Efficient 3-adic computations |
| Seeded generators | losses | Reproducible pair sampling |
| Config-driven composition | losses | Flexible loss function assembly |
| Trainable terminology | models | Clear, positive logic |

---

## File Tree

```
docs/audits/
├── INDEX.md                          # This file
├── MASTER_AUDIT.md                   # ⭐ Synthesized training-readiness assessment
│
├── # Comprehensive Audits
├── COMPREHENSIVE_CODEBASE_AUDIT_2026-02-24.md  # Full audit with 10 bug fixes
│
├── # Module Audits (each updated 2026-02-26)
├── CORE_MODULE_AUDIT.md
├── LOSSES_MODULE_AUDIT.md
├── UTILS_MODULE_AUDIT.md
├── GEOMETRY_MODULE_AUDIT.md
├── CONFIG_PRESETS_MODULE_AUDIT.md
├── MODELS_MODULE_AUDIT.md
│
├── # Integration Audits
├── GEOOPT_INTEGRATION_AUDIT.md
├── real-non-euclidean.md
├── tensorboard-logs.md
├── terminal-monitoring-integration.md
│
├── # Presets Analysis
├── presets-analysis/
│   ├── YAML_COMPLETENESS_AUDIT.md
│   └── audit-yaml-gemini.md
│
└── # Previous Audits (Historical)
    └── previous-audits/
        ├── SRC_COMPREHENSIVE_AUDIT.md
        ├── FEATURE_IMPLEMENTATION_AUDIT.md
        ├── pre-audit.md
        └── presets-audit.md
```

---

**Index maintained by**: Claude Opus 4.6
