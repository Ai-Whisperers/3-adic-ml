"""Fetch real cytochrome c sequences from UniProt across a broad taxonomic range.

Phase 1 of docs/plans/PHYLOGENY-VALIDATION-PIPELINE.md. Species are matched by
membership in Pfam PF00034 (the class-I soluble c-type cytochrome family, the
classic Margoliash/Fitch molecular-clock protein) rather than by gene-symbol
guessing, since "CYC1" means different, non-homologous proteins in yeast
(iso-1-cytochrome c) vs. mammals (cytochrome c1, Complex III). Accessions and
taxon IDs are resolved live from UniProt, never hardcoded, so the manifest is
independently reproducible from the query alone.
"""

import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
PFAM_CYTOCHROME_C = "PF00034"
FIELDS = "accession,id,organism_name,organism_id,length,sequence,reviewed,protein_name"
HEADERS = {"User-Agent": "3-adic-ml/phylogeny-validation (research script)"}

# Canonical class-I soluble cytochrome c length range (mature chain, e.g.
# human=105, yeast=109, P. aeruginosa cytochrome c-551=104). PF00034 also
# matches unrelated multi-domain c-type proteins that share the heme-binding
# fold but are catalytic enzymes, not the single-heme electron carrier --
# length alone doesn't reliably exclude these (observed: R. sphaeroides
# "Nitric oxide reductase subunit C" is 147 aa, inside this band), so protein
# name is also checked below.
LENGTH_MIN, LENGTH_MAX = 70, 150

# General enzyme-class keywords, not species-specific: excludes catalytic
# PF00034 relatives (peroxidase, oxidase/reductase subunits, multiheme
# electron-transport complexes) so the shortest-remaining-candidate heuristic
# doesn't pick one of those over the true soluble cytochrome c.
EXCLUDE_NAME_KEYWORDS = [
    "oxidase", "peroxidase", "reductase", "synthase", "dehydrogenase",
    "diheme", "multiheme", "tetraheme", "biogenesis",
]

# Scientific names spanning bacteria -> fungi -> protists -> plants ->
# invertebrates -> vertebrates. Grouping/order here is for log readability
# only; it feeds no computation. Taxonomic distance is computed independently
# in fetch_taxonomy.py from NCBI's ranked lineage, not from this ordering.
CURATED_SPECIES = [
    # Bacteria
    "Escherichia coli",
    "Bacillus subtilis",
    "Pseudomonas aeruginosa",
    "Rhodobacter sphaeroides",
    # Fungi
    "Saccharomyces cerevisiae",
    "Neurospora crassa",
    "Aspergillus nidulans",
    "Candida albicans",
    # Protists
    "Dictyostelium discoideum",
    "Plasmodium falciparum",
    "Trypanosoma brucei",
    "Chlamydomonas reinhardtii",
    # Plants
    "Arabidopsis thaliana",
    "Oryza sativa",
    "Triticum aestivum",
    "Zea mays",
    "Glycine max",
    "Nicotiana tabacum",
    "Solanum lycopersicum",
    "Physcomitrium patens",
    # Invertebrates
    "Drosophila melanogaster",
    "Caenorhabditis elegans",
    "Anopheles gambiae",
    "Apis mellifera",
    "Strongylocentrotus purpuratus",
    "Ciona intestinalis",
    # Fish / amphibian / reptile / bird
    "Danio rerio",
    "Xenopus laevis",
    "Anolis carolinensis",
    "Gallus gallus",
    "Meleagris gallopavo",
    # Mammals
    "Mus musculus",
    "Rattus norvegicus",
    "Oryctolagus cuniculus",
    "Canis lupus familiaris",
    "Equus caballus",
    "Sus scrofa",
    "Bos taurus",
    "Macaca mulatta",
    "Pan troglodytes",
    "Homo sapiens",
]

MANIFEST_FIELDS = [
    "species", "status", "uniprot_accession", "uniprot_entry_name",
    "organism_taxon_id", "sequence_length", "reviewed", "length_plausible",
    "fetch_date_utc", "source_url",
]


def _get_json(url: str, params: dict, timeout: float, retries: int = 3, sleep: float = 1.0) -> dict:
    last_exc = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
            last_exc = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        except requests.RequestException as exc:
            last_exc = exc
        time.sleep(sleep * (attempt + 1))
    raise RuntimeError(f"Failed after {retries} attempts: {last_exc}")


def _query_uniprot(species: str, timeout: float) -> list[dict]:
    query = f'(xref:{PFAM_CYTOCHROME_C}) AND (organism_name:"{species}")'
    data = _get_json(
        UNIPROT_SEARCH_URL,
        {"query": query, "fields": FIELDS, "format": "json", "size": 25},
        timeout=timeout,
    )
    return data.get("results", [])


def _is_reviewed(entry: dict) -> bool:
    return entry["entryType"] == "UniProtKB reviewed (Swiss-Prot)"


def _protein_name(entry: dict) -> str:
    return entry.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "")


def _select_best(candidates: list[dict]) -> tuple[dict, bool]:
    """Excludes known catalytic PF00034 relatives by name (see
    EXCLUDE_NAME_KEYWORDS), then prefers hits within the canonical length
    band (LENGTH_MIN/MAX); within that pool, prefers reviewed, then shortest.
    Falls back to the unfiltered candidate pool if nothing survives, flagged
    as length_plausible=False -- reviewed status alone is not a reliable
    homology signal (observed e.g. for E. coli, where the only reviewed
    PF00034 hit is cytochrome c peroxidase, a 465 aa unrelated protein)."""
    named = [c for c in candidates
             if not any(kw in _protein_name(c).lower() for kw in EXCLUDE_NAME_KEYWORDS)]
    pool_for_length = named if named else candidates
    plausible = [c for c in pool_for_length if LENGTH_MIN <= c["sequence"]["length"] <= LENGTH_MAX]
    pool = plausible if plausible else candidates
    best = min(pool, key=lambda c: (not _is_reviewed(c), c["sequence"]["length"]))
    return best, bool(plausible)


def fetch_species(species: str, timeout: float) -> dict:
    candidates = _query_uniprot(species, timeout)
    if not candidates:
        return {
            "species": species, "status": "not_found", "uniprot_accession": "",
            "uniprot_entry_name": "", "organism_taxon_id": "", "sequence_length": "",
            "reviewed": "", "length_plausible": "",
            "fetch_date_utc": datetime.now(timezone.utc).isoformat(),
            "source_url": "", "sequence": "",
        }
    entry, length_plausible = _select_best(candidates)
    accession = entry["primaryAccession"]
    reviewed = _is_reviewed(entry)
    return {
        "species": species,
        "status": "found_reviewed" if reviewed else "found_unreviewed",
        "uniprot_accession": accession,
        "uniprot_entry_name": entry["uniProtkbId"],
        "organism_taxon_id": entry["organism"]["taxonId"],
        "sequence_length": entry["sequence"]["length"],
        "reviewed": reviewed,
        "length_plausible": length_plausible,
        "fetch_date_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": f"https://rest.uniprot.org/uniprotkb/{accession}",
        "sequence": entry["sequence"]["value"],
    }


def write_fasta(out_dir: Path, row: dict) -> None:
    safe_name = row["species"].replace(" ", "_")
    fasta_path = out_dir / "fasta" / f"{safe_name}.fasta"
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    header = f">{row['uniprot_accession']}|{row['uniprot_entry_name']}|{row['species']}"
    seq = row["sequence"]
    lines = [seq[i:i + 60] for i in range(0, len(seq), 60)]
    fasta_path.write_text(header + "\n" + "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/cytochrome_c")
    parser.add_argument("--sleep", type=float, default=0.5, help="delay between species queries")
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, species in enumerate(CURATED_SPECIES):
        try:
            row = fetch_species(species, args.timeout)
        except RuntimeError as exc:
            print(f"[ERROR] {species}: {exc}")
            row = {
                "species": species, "status": "error", "uniprot_accession": "",
                "uniprot_entry_name": "", "organism_taxon_id": "", "sequence_length": "",
                "reviewed": "", "length_plausible": "",
                "fetch_date_utc": datetime.now(timezone.utc).isoformat(),
                "source_url": "", "sequence": "",
            }
        rows.append(row)
        if row["status"].startswith("found"):
            write_fasta(out_dir, row)
        flag = "" if row.get("length_plausible", True) in (True, "") else "  [WARN: length outside canonical band, likely not cytochrome c]"
        print(f"[{i+1}/{len(CURATED_SPECIES)}] {species}: {row['status']} "
              f"({row['uniprot_accession'] or '-'}, {row['sequence_length'] or '-'} aa){flag}")
        if i < len(CURATED_SPECIES) - 1:
            time.sleep(args.sleep)

    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in MANIFEST_FIELDS})

    n_reviewed = sum(1 for r in rows if r["status"] == "found_reviewed")
    n_unreviewed = sum(1 for r in rows if r["status"] == "found_unreviewed")
    n_missing = sum(1 for r in rows if r["status"] in ("not_found", "error"))
    n_implausible = sum(1 for r in rows if r.get("length_plausible") is False)
    print(f"\n[OK] {len(rows)} species queried: {n_reviewed} reviewed, "
          f"{n_unreviewed} unreviewed, {n_missing} not found/error, "
          f"{n_implausible} length-implausible (likely non-homologous, needs review).")
    print(f"     Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
