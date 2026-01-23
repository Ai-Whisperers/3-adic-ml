# Presets Index

This directory contains YAML preset files for p-adic VAE training experiments.

## Naming Convention

Presets are prefixed by category:
- `arch_` - Architecture changes/improvements
- `base_` - Foundation configurations
- `experiment_` - Ablation/exploration experiments
- `fix_` - Bugfix configurations
- `hardware_` - Hardware-specific optimizations
- `manifold_` - Manifold type configurations
- `minimal_` - Minimal/smoke test configs
- `production_` - Production-ready training configs
- `research_` - Long-running research experiments
- `test_` - Feature testing configs
- `tuning_` - Hyperparameter tuning configs
- `validation_` - Validation/audit configs

---

## Production Configs

| File | Previous Name | Description |
|------|---------------|-------------|
| `production_rich_hierarchy.yaml` | v5_12.yaml | V5.12 production config with two-phase loss strategy (RichHierarchy primary). Target: Coverage=100%, Hierarchy_B=-0.83, 200 epochs. |
| `production_hyperbolic_full.yaml` | v5_12_1.yaml | Full hyperbolic integration: decoder uses log_map_zero(z_hyp), metrics use hyperbolic distance. 200 epochs. |

## Base Configs

| File | Previous Name | Description |
|------|---------------|-------------|
| `base_frozen_encoder_geodesic.yaml` | v5_11_base.yaml | V5.11 foundation: frozen encoder (from v5.5), unified geodesic loss, curriculum blending. 100 epochs. |

## Hardware-Specific Configs

| File | Previous Name | Description |
|------|---------------|-------------|
| `hardware_rtx2060_8gb.yaml` | v5_11_11_homeostatic_rtx2060.yaml | Optimized for RTX 2060 SUPER (8GB VRAM): batch_size=512, AMP enabled, 4 workers. 150 epochs. |

## Architecture Configs

| File | Previous Name | Description |
|------|---------------|-------------|
| `arch_improved_encoder_decoder.yaml` | v5_12_4_improved_but_non_fixed.yaml | Improved encoder/decoder with SiLU activation, LayerNorm, Dropout, logvar clamping. 100 epochs. |

## Manifold Type Configs

| File | Previous Name | Description |
|------|---------------|-------------|
| `manifold_valuation_optimal.yaml` | valuation_optimal.yaml | P-adic hierarchy optimization: target_hierarchy=-0.80, frozen encoder_A, strong hierarchy loss. |
| `manifold_frequency_optimal.yaml` | frequency_optimal.yaml | Shannon information efficiency: target_hierarchy=+0.70, frequency-based volume allocation. |

## Test Configs

| File | Previous Name | Description |
|------|---------------|-------------|
| `test_adaptive_loss.yaml` | v5_12_5_adaptive_loss_test.yaml | Phase 2.2: Adaptive RichHierarchyLoss with curriculum learning. 15-epoch test. |
| `test_adaptive_lr.yaml` | v5_12_5_adaptive_lr_test.yaml | Phase 2.3: Validation-based LR scheduling with multi-metric monitoring. 18-epoch test. |
| `test_gradient_checkpointing.yaml` | v5_12_5_gradient_checkpointing_test.yaml | Phase 2.1: Gradient checkpointing for 30-40% VRAM reduction. 8-epoch test. |
| `minimal_smoke_test.yaml` | v5_12_extended.yaml | Minimal 1-epoch config for quick smoke tests. batch_size=32. |

## Validation Configs

| File | Previous Name | Description |
|------|---------------|-------------|
| `validation_hyperbolic_audit.yaml` | v5_12_3.yaml | V5.12.2 hyperbolic audit validation: all metrics use poincare_distance(). Short 50-epoch run. |

## Experiment Configs (Ablation Studies)

| File | Previous Name | Description |
|------|---------------|-------------|
| `experiment_a_encoder_unfreeze.yaml` | v5_12_5_relaxed_A_unfreeze.yaml | Ablation A: Unfreeze encoder_A at 5% LR scale. Joint encoder optimization. 15 epochs. |
| `experiment_b_aggressive_lr.yaml` | v5_12_5_relaxed_B_aggressive_lr.yaml | Ablation B: Aggressive LR (2.5x base), full LR for encoder_B. 15 epochs. |
| `experiment_c_loss_rebalance.yaml` | v5_12_5_relaxed_C_loss_rebalance.yaml | Ablation C: Loss rebalancing - hierarchy_weight=2.0, richness_weight=5.0, statenet disabled. 20 epochs. |
| `experiment_d_from_scratch.yaml` | v5_12_5_relaxed_D_from_scratch.yaml | Ablation D: From-scratch training (no frozen checkpoint), permissive statenet. 100 epochs. |

## Tuning Configs

| File | Previous Name | Description |
|------|---------------|-------------|
| `tuning_statenet_differential.yaml` | v5_12_5_homeostatic_differential.yaml | StateNet threshold tuning: relaxed coverage thresholds (0.992/0.996), longer warmup (20 epochs). 30 epochs. |

## Research Configs

| File | Previous Name | Description |
|------|---------------|-------------|
| `research_extended_grokking.yaml` | v5_12_4_extended_grokking.yaml | Extended training for grokking detection: 500 epochs, multi-phase LR, phase transition detection. |

## Bugfix Configs

| File | Previous Name | Description |
|------|---------------|-------------|
| `fix_checkpoint_loading.yaml` | v5_12_4_hotfix.yaml | Fixed null checkpoint path issue causing 0% coverage. Mixed precision + gradient checkpointing. 30 epochs. |

---

## Quick Reference: Which Config to Use

| Goal | Recommended Config |
|------|-------------------|
| Production training | `production_rich_hierarchy.yaml` |
| Quick test/CI | `minimal_smoke_test.yaml` |
| RTX 2060 / 8GB VRAM | `hardware_rtx2060_8gb.yaml` |
| Grokking research | `research_extended_grokking.yaml` |
| Feature validation | `validation_hyperbolic_audit.yaml` |
| VRAM-constrained | `test_gradient_checkpointing.yaml` |

---

## Version History

- **V5.11**: Frozen encoder architecture, unified geodesic loss
- **V5.12**: RichHierarchyLoss, two-phase training, improved metrics
- **V5.12.1**: Full hyperbolic integration (decoder input)
- **V5.12.3**: Hyperbolic audit validation
- **V5.12.4**: Improved encoder/decoder (SiLU, LayerNorm)
- **V5.12.5**: Optimization tests (gradient checkpointing, adaptive loss/LR)
