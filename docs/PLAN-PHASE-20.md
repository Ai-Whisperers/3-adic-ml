# Phase 20: Bioactive Validation Protocol

## Vision
To move beyond computational mapping and verify the therapeutic potential of the novel bioactive motifs identified by the Rosetta Manifold (Phase 17).

## Pipeline Architecture

### 1. In Silico Stability Assessment (GROMACS)
We will perform molecular dynamics simulations to test the structural stability of top-ranking candidates.
- **Goal:** Verify that motifs like `AVLGSSEGV` maintain their fold over 100ns under simulated physiological conditions.
- **Criteria:** RMSD < 2.0 Å, hydrogen-bond network stability.

### 2. Functional Activity Prediction (QSPR)
- **Tooling:** Use the latent embeddings from our trained Rosetta Manifold as input features for a Random Forest classifier trained on public antioxidant and antihypertensive databases.
- **Metric:** Predict the probability of high-activity status.

### 3. In Vitro Synthesis Protocol (Proposed)
For motifs that pass structural stability and functional prediction:
- **Phase A:** Solid-phase peptide synthesis (SPPS).
- **Phase B:** DPPH (1,1-diphenyl-2-picrylhydrazyl) free-radical scavenging assay.
- **Phase C:** Angiotensin-Converting Enzyme (ACE) inhibition assay.

## Timeline
This phase will trigger immediately upon completion of Phase 19 (Hybrid Refinement).

---
*Status: Planning Phase*
