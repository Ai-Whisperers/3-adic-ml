"""Fetch NCBI ranked lineages for the species in the cytochrome c manifest and
derive a pairwise taxonomic distance matrix.

Phase 1 of docs/plans/PHYLOGENY-VALIDATION-PIPELINE.md. This is the
independent ground truth: taxon IDs come from UniProt (fetch_cytochrome_c.py
output), but the lineage/rank data used for distance comes only from NCBI
Taxonomy, never from any p-adic/hyperbolic model output.

Distance definition: for two species, walk RANK_ORDER from coarsest (domain)
to finest (species) and find the deepest rank at which their lineage names
still agree -- that is the last common ancestor's rank. Distance is the
number of rank steps below that point (0 = same species, 7 = share only
domain). This is a true ultrametric by construction (it is exactly cophenetic
distance in the rooted rank tree), matching the example in the plan doc
("mismo genero = distancia 1, mismo dominio nada mas = distancia 7").
"""

import argparse
import csv
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import requests

EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
HEADERS = {"User-Agent": "3-adic-ml/phylogeny-validation (research script)"}

# Coarsest -> finest. NCBI's dump has used both "superkingdom" (older) and
# "domain" (current) for the top rank; both map to index 0 here.
RANK_ORDER = ["domain", "kingdom", "phylum", "class", "order", "family", "genus", "species"]
RANK_ALIASES = {"superkingdom": "domain"}


def _get_xml(params: dict, timeout: float, retries: int = 3, sleep: float = 1.0) -> ET.Element:
    last_exc = None
    for attempt in range(retries):
        try:
            resp = requests.get(EFETCH_URL, params=params, headers=HEADERS, timeout=timeout)
            if resp.status_code == 200:
                return ET.fromstring(resp.text)
            last_exc = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        except (requests.RequestException, ET.ParseError) as exc:
            last_exc = exc
        time.sleep(sleep * (attempt + 1))
    raise RuntimeError(f"Failed after {retries} attempts: {last_exc}")


def fetch_ranked_lineage(taxon_id: str, timeout: float) -> dict:
    """Returns {rank: scientific_name} restricted to ranks in RANK_ORDER."""
    root = _get_xml({"db": "taxonomy", "id": taxon_id, "retmode": "xml"}, timeout=timeout)
    taxon = root.find("Taxon")
    if taxon is None:
        raise RuntimeError(f"No Taxon element for taxid={taxon_id}")

    ranked = {}
    for t in taxon.find("LineageEx").findall("Taxon"):
        rank = RANK_ALIASES.get(t.findtext("Rank"), t.findtext("Rank"))
        if rank in RANK_ORDER:
            ranked[rank] = t.findtext("ScientificName")

    leaf_rank = RANK_ALIASES.get(taxon.findtext("Rank"), taxon.findtext("Rank"))
    if leaf_rank in RANK_ORDER:
        ranked[leaf_rank] = taxon.findtext("ScientificName")
    return ranked


def rank_distance(lineage_a: dict, lineage_b: dict) -> int:
    common_depth = -1
    for depth, rank in enumerate(RANK_ORDER):
        if rank in lineage_a and rank in lineage_b and lineage_a[rank] == lineage_b[rank]:
            common_depth = depth
        elif rank in lineage_a and rank in lineage_b:
            break
    return (len(RANK_ORDER) - 1) - common_depth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/cytochrome_c/manifest.csv")
    parser.add_argument("--out-dir", default="data/cytochrome_c")
    parser.add_argument("--sleep", type=float, default=0.4, help="delay between NCBI requests")
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}. Run fetch_cytochrome_c.py first.")

    with open(manifest_path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["status"].startswith("found")]
    if not rows:
        raise SystemExit("No species with status found_reviewed/found_unreviewed in manifest.")

    lineages = {}
    for i, row in enumerate(rows):
        species, taxon_id = row["species"], row["organism_taxon_id"]
        try:
            lineages[species] = fetch_ranked_lineage(taxon_id, args.timeout)
            n_ranks = len(lineages[species])
        except RuntimeError as exc:
            print(f"[ERROR] {species} (taxid={taxon_id}): {exc}")
            lineages[species] = {}
            n_ranks = 0
        print(f"[{i+1}/{len(rows)}] {species} (taxid={taxon_id}): {n_ranks}/{len(RANK_ORDER)} ranks resolved")
        if i < len(rows) - 1:
            time.sleep(args.sleep)

    species_order = [row["species"] for row in rows if lineages[row["species"]]]
    n = len(species_order)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = rank_distance(lineages[species_order[i]], lineages[species_order[j]])
            dist[i, j] = dist[j, i] = d

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "taxonomic_distance.npy", dist)
    (out_dir / "species_order.json").write_text(json.dumps(species_order, indent=2))
    (out_dir / "taxonomy_lineage.json").write_text(json.dumps(lineages, indent=2))

    n_dropped = len(rows) - n
    print(f"\n[OK] {n} species with resolved lineage (dropped {n_dropped} with 0 ranks).")
    print(f"     Distance range: [{dist[dist > 0].min() if n > 1 else 0:.0f}, {dist.max():.0f}]")
    print(f"     Saved: {out_dir / 'taxonomic_distance.npy'}, "
          f"{out_dir / 'species_order.json'}, {out_dir / 'taxonomy_lineage.json'}")


if __name__ == "__main__":
    main()
