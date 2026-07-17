"""Flat Euclidean VAE baseline (Condition A) for the phylogeny validation
pipeline (docs/plans/PHYLOGENY-VALIDATION-PIPELINE.md, Fase 3).

Reuses EncoderHead/_build_decoder from src.models.vae unchanged, but skips
DualHyperbolicProjection entirely: standard N(0,I) reparameterization,
decode directly from tangent space. No hyperbolic geometry, no dual-VAE
split, no StateNet/curriculum machinery -- this is the plain-VAE control
condition the p-adic/hyperbolic architecture (Condition C) is compared
against on the same cytochrome c dataset. Forcing the existing curriculum
engine (StateNet, Lagrangian dual ascent, grokking detector) onto a plain
VAE would be more code than just writing the direct loop.
"""

import torch
import torch.nn as nn

from src.core import TERNARY
from src.models.vae import EncoderHead, _build_decoder


class TernaryVAEEuclideanBaseline(nn.Module):
    """Single-VAE Euclidean baseline: encoder -> N(0,I) reparameterize -> decoder."""

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        encoder_type: str = "improved",
        decoder_type: str = "improved",
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = EncoderHead(hidden_dim, latent_dim, encoder_type, input_dim=9)
        self.decoder = _build_decoder(latent_dim, hidden_dim, decoder_type)
        self.to(torch.float64)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> dict:
        x = x.to(torch.float64)
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(z)
        return {"logits": logits, "mu": mu, "logvar": logvar, "z": z}

    def get_mu_representations(self, indices: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Deterministic latent representation for downstream evaluation
        (Fase 4: per-species aggregation), mirroring TernaryVAEV6's method
        of the same name so evaluate_phylogeny_recovery.py can treat all
        three conditions uniformly."""
        x = TERNARY.to_ternary(indices).to(device).to(torch.float64)
        mu, _ = self.encoder(x)
        return mu
