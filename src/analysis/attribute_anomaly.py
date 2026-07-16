# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Per-nucleotide attribution via perturbation analysis in Poincaré-ball space."""

from __future__ import annotations

import torch

from scripts.data.prepare_codon_data import seq_to_ternary_index

_NUCLEOTIDES = "ACGT"


def compute_attribution(sequence: str, model, detector) -> tuple[list[float], float]:
    """Importance score per nucleotide position via perturbation analysis."""

    def get_dist(seq: str) -> float:
        idx = torch.tensor([seq_to_ternary_index(seq)])
        with torch.no_grad():
            mu = model.get_mu_representations(idx, torch.device("cpu"))
            # proj_A returns (z_hyp, r) in factored mode, plain z_hyp otherwise.
            result = model.projections.proj_A(mu)
            z = result[0] if isinstance(result, tuple) else result
        res = detector.detect(z)
        return float(res["min_dist"][0])

    orig_dist = get_dist(sequence)
    attribution: list[float] = []

    for i in range(len(sequence)):
        original_nuc = sequence[i]
        pos_sens = 0.0

        for nuc in _NUCLEOTIDES:
            if nuc == original_nuc:
                continue
            perturbed_seq = sequence[:i] + nuc + sequence[i + 1:]
            new_dist = get_dist(perturbed_seq)
            pos_sens += abs(new_dist - orig_dist)  # sensitivity = magnitude of change

        attribution.append(pos_sens / (len(_NUCLEOTIDES) - 1))

    return attribution, orig_dist
