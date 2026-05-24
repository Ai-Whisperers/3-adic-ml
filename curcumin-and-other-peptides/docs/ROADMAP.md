# Peptide Design Roadmap: Bioactive Ginger & Curcumin Variants

## 1. Data Collection & Preprocessing
- **Source sequences:** Use MALDI-TOF identified peptides (WTLTPLTPA, Cur-1, RALGWSCL).
- **Expansion:** Search UniProt for homologs in *Curcuma longa* and *Zingiber officinale*.
- **Representations:** Use ESM-2 or ProtBERT embeddings for sequence-level features.

## 2. Latent Space Mapping
- **Architecture:** Variational Autoencoder (VAE) or Latent Diffusion Model (LDM).
- **Goal:** Map discrete peptide sequences into a continuous latent space.
- **Property Guidance:** Train a surrogate regressor (Stability Predictor) on the latent vectors to guide optimization.

## 3. Stability Optimization
- **Predictors:** Integrate **DeepDigest** (cleavage site prediction) and **PeptiVerse** (half-life prediction).
- **Optimization Strategy:**
    - **Latent Bayesian Optimization (LBO):** Search for latent points that maximize stability while maintaining bioactivity embeddings.
    - **Latent Diffusion:** Reverse noise towards "stable" regions of the manifold.

## 4. Verification Pipeline
- **In Silico:**
    - Calculate **Instability Index** using `peptides.py`.
    - Perform **Molecular Dynamics (MD)** simulations (GROMACS/OpenMM) to check conformational resilience.
- **In Vitro (Future):** Synthesis and protease assay (Pepsin/Chymotrypsin).

## 5. Potential Tools
- **Generative:** PepVAE, PepGLAD, RFdiffusion.
- **Predictive:** DeepDigest, HBM Model, PeptideBERT.
- **Libraries:** `peptides.py`, `biopython`, `pyteomics`.
