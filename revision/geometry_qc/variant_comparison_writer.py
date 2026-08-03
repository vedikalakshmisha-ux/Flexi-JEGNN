"""
revision/geometry_qc/variant_comparison_writer.py
==================================================
Computes per-molecule RMSD between every pair of geometry-generation methods
and writes ``variant_comparison.csv``.

What is computed
----------------
For each molecule (SMILES) and each ordered pair of methods (A, B):

  * **raw RMSD** — root-mean-square deviation of heavy-atom positions after
    optimal superposition (Kabsch algorithm, no reflection).
  * **n_atoms** — number of heavy atoms (must match for RMSD to be defined).
  * **both_ok** — 1 if both methods produced a conformer, 0 otherwise.

If either method failed for a molecule the RMSD cell is left empty.

Output schema
-------------
``variant_comparison.csv`` — one row per (smiles, method_a, method_b):

    smiles, method_a, method_b, rmsd, n_atoms, both_ok,
    method_a_success, method_b_success,
    method_a_error, method_b_error

Usage
-----
    # Default: runs all available methods on every SMILES in QM9.csv
    python revision/geometry_qc/variant_comparison_writer.py

    # Custom SMILES list and output path
    python revision/geometry_qc/variant_comparison_writer.py \\
        --smiles-csv datasets/QM9.csv \\
        --smiles-col smiles \\
        --n-mols 500 \\
        --seed 42 \\
        --out-csv results/variant_comparison.csv

    # Skip OpenBabel (not installed) and the random baseline
    python revision/geometry_qc/variant_comparison_writer.py \\
        --skip-methods obabel random_dg

Out of scope (teammate handles separately)
------------------------------------------
* Protonation pipeline          -> revision/protonation/
* Conformer QC / validation     -> revision/conformer_qc/
* Baseline benchmarks           -> revision/benchmarks/reproduce_baselines.py
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from revision.geometry_qc.generate_variants import (
    ALL_METHODS,
    METHOD_OBABEL,
    ConformerResult,
    generate_all_available,
    obabel_available,
    rdkit_available,
)

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

OUTPUT_COLUMNS: List[str] = [
    "smiles",
    "method_a",
    "method_b",
    "rmsd",          # Kabsch-aligned RMSD in Angstroms; empty if either failed
    "n_atoms",       # heavy-atom count; empty if mismatch or either failed
    "both_ok",       # 1 if both methods succeeded, 0 otherwise
    "method_a_success",
    "method_b_success",
    "method_a_error",
    "method_b_error",
]


# ---------------------------------------------------------------------------
# Kabsch RMSD
# ---------------------------------------------------------------------------

def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Compute the RMSD between two sets of points after optimal superposition
    (Kabsch algorithm, translation + rotation, no reflection).

    Parameters
    ----------
    P, Q : np.ndarray, shape (n, 3)
        Two sets of n 3-D points (e.g. heavy-atom coordinates in Angstroms).

    Returns
    -------
    float
        Kabsch RMSD in the same units as P and Q.

    Raises
    ------
    ValueError
        If P and Q have different shapes or fewer than 2 rows.

    References
    ----------
    Kabsch W. (1976) Acta Crystallogr. A 32:922–923.
    """
    if P.shape != Q.shape:
        raise ValueError(
            f"Shape mismatch: P={P.shape}, Q={Q.shape}. "
            "Both must have the same number of atoms."
        )
    if P.shape[0] < 2:
        raise ValueError("Need at least 2 atoms for RMSD computation.")

    # 1. Centre both point sets
    P = P - P.mean(axis=0)
    Q = Q - Q.mean(axis=0)

    # 2. Covariance matrix H = P^T Q
    H = P.T @ Q  # (3, 3)

    # 3. SVD
    U, S, Vt = np.linalg.svd(H)

    # 4. Ensure right-handed coordinate system (no improper rotation)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])

    # 5. Optimal rotation
    R = Vt.T @ D @ U.T  # (3, 3)

    # 6. Rotate P
    P_rot = P @ R.T  # (n, 3)

    # 7. RMSD
    diff = P_rot - Q
    rmsd = float(np.sqrt((diff ** 2).sum() / len(P)))
    return rmsd


def pairwise_rmsd(
    results: Dict[str, ConformerResult],
) -> List[dict]:
    """
    Compute RMSD for every ordered pair (A, B) of methods in *results*.

    Only pairs where A < B (lexicographic) are generated to avoid duplicates.

    Parameters
    ----------
    results : dict method_name → ConformerResult

    Returns
    -------
    list of dicts, each with keys matching ``OUTPUT_COLUMNS``.
    """
    methods = sorted(results.keys())
    rows = []
    smiles = next(iter(results.values())).smiles if results else ""

    for i, method_a in enumerate(methods):
        for method_b in methods[i + 1:]:
            ra = results[method_a]
            rb = results[method_b]

            both_ok = int(ra.success and rb.success)
            rmsd_val = ""
            n_atoms_val = ""

            if both_ok:
                if ra.n_atoms != rb.n_atoms:
                    warnings.warn(
                        f"[variant_comparison] Atom count mismatch for {smiles!r}: "
                        f"{method_a}={ra.n_atoms}, {method_b}={rb.n_atoms}. "
                        "Skipping RMSD.",
                        stacklevel=2,
                    )
                    both_ok = 0
                else:
                    try:
                        rmsd_val = round(kabsch_rmsd(ra.positions, rb.positions), 6)
                        n_atoms_val = ra.n_atoms
                    except Exception as exc:
                        warnings.warn(
                            f"[variant_comparison] RMSD failed for {smiles!r} "
                            f"({method_a} vs {method_b}): {exc}",
                            stacklevel=2,
                        )
                        rmsd_val = ""

            rows.append({
                "smiles":           smiles,
                "method_a":         method_a,
                "method_b":         method_b,
                "rmsd":             rmsd_val,
                "n_atoms":          n_atoms_val,
                "both_ok":          both_ok,
                "method_a_success": int(ra.success),
                "method_b_success": int(rb.success),
                "method_a_error":   ra.error_msg,
                "method_b_error":   rb.error_msg,
            })

    return rows


# ---------------------------------------------------------------------------
# Molecule loading
# ---------------------------------------------------------------------------

def _load_smiles_list(
    csv_path: Path,
    smiles_col: str,
    n_mols: Optional[int],
) -> List[str]:
    """Read up to *n_mols* SMILES strings from *csv_path*."""
    smiles_list: List[str] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if smiles_col not in (reader.fieldnames or []):
            raise ValueError(
                f"Column {smiles_col!r} not found in {csv_path.name}. "
                f"Available: {reader.fieldnames}"
            )
        for row in reader:
            s = row[smiles_col].strip()
            if s:
                smiles_list.append(s)
            if n_mols is not None and len(smiles_list) >= n_mols:
                break
    return smiles_list


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_comparison(
    smiles_list: List[str],
    seed: int = 42,
    skip_methods: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> List[dict]:
    """
    Run all available geometry methods on every SMILES in *smiles_list*
    and return a flat list of per-pair comparison dicts.

    Parameters
    ----------
    smiles_list : list of str
    seed : int
        Random seed passed to every generator.
    skip_methods : sequence of str, optional
        Method names to skip.
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    list of dicts with keys from OUTPUT_COLUMNS
    """
    all_rows: List[dict] = []
    skip = list(skip_methods or [])

    for idx, smiles in enumerate(smiles_list):
        if verbose and idx % 100 == 0:
            print(f"  [{idx}/{len(smiles_list)}] {smiles[:60]}", flush=True)

        results = generate_all_available(smiles, seed=seed, skip_methods=skip)
        if not results:
            if verbose:
                print(f"  [SKIP] {smiles!r}: no methods available")
            continue

        rows = pairwise_rmsd(results)
        all_rows.extend(rows)

    return all_rows


def write_comparison_csv(rows: List[dict], out_path: Path) -> None:
    """Write *rows* to *out_path* in CSV format."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(rows: List[dict]) -> None:
    """Print per-pair-of-methods mean RMSD summary to stdout."""
    if not rows:
        print("No comparison rows to summarise.")
        return

    # Group by (method_a, method_b)
    from collections import defaultdict
    groups: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in rows:
        if row["rmsd"] != "" and row["both_ok"]:
            groups[(row["method_a"], row["method_b"])].append(float(row["rmsd"]))

    print(f"\n{'=' * 68}")
    print("Geometry variant comparison — mean Kabsch RMSD (Å)")
    print(f"{'=' * 68}")
    print(f"{'Method A':<14} vs {'Method B':<14}  {'N':>6}  {'Mean RMSD':>10}  {'Max RMSD':>10}")
    print("-" * 68)

    for (ma, mb), rmsds in sorted(groups.items()):
        mean_r = sum(rmsds) / len(rmsds)
        max_r = max(rmsds)
        print(f"{ma:<14}    {mb:<14}  {len(rmsds):>6}  {mean_r:>10.4f}  {max_r:>10.4f}")

    print(f"{'=' * 68}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Compute per-molecule RMSD between geometry-generation methods.\n\n"
            "Reads SMILES from a CSV, runs all available methods, writes "
            "variant_comparison.csv."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--smiles-csv",
        default=None,
        help="CSV file containing SMILES strings. Defaults to datasets/QM9.csv.",
    )
    p.add_argument(
        "--smiles-col",
        default="smiles",
        help="Column name for SMILES (default: 'smiles').",
    )
    p.add_argument(
        "--n-mols",
        type=int,
        default=None,
        help="Maximum number of molecules to process (default: all).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for conformer generation (default: 42).",
    )
    p.add_argument(
        "--skip-methods",
        nargs="*",
        default=[],
        help=(
            f"Method names to skip. Choices: {ALL_METHODS}. "
            "Default: skips unavailable methods automatically."
        ),
    )
    p.add_argument(
        "--out-csv",
        default=None,
        help="Output CSV path. Defaults to results/variant_comparison.csv.",
    )
    p.add_argument(
        "--no-summary",
        action="store_true",
        help="Suppress the per-pair RMSD summary printed to stdout.",
    )
    return p


def _resolve(cli_arg: Optional[str], default_rel: str) -> Path:
    if cli_arg:
        return Path(cli_arg).expanduser().resolve()
    here = Path(__file__).resolve().parent          # revision/geometry_qc/
    return here.parent.parent / default_rel         # repo/<default_rel>


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    smiles_csv = _resolve(args.smiles_csv, "datasets/QM9.csv")
    out_csv = _resolve(args.out_csv, "results/variant_comparison.csv")

    print(f"\nSMILES source : {smiles_csv}")
    print(f"Output        : {out_csv}")
    print(f"Seed          : {args.seed}")
    if args.skip_methods:
        print(f"Skip methods  : {args.skip_methods}")
    if not rdkit_available():
        print(
            "\n[WARNING] RDKit is NOT installed — all RDKit-based methods "
            "(ETKDGv1/v2/v3, random_dg) will be skipped.\n"
            "Install with: conda install -c conda-forge rdkit\n",
            file=sys.stderr,
        )
    if not obabel_available():
        print(
            "[INFO] OpenBabel (obabel) not found on PATH — "
            "the 'obabel' method will be skipped.",
        )
    print("-" * 60)

    # Load SMILES
    try:
        smiles_list = _load_smiles_list(smiles_csv, args.smiles_col, args.n_mols)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    if not smiles_list:
        print("[ERROR] No SMILES loaded from input CSV.", file=sys.stderr)
        return 1

    print(f"  Molecules loaded: {len(smiles_list)}")

    # Run
    rows = run_comparison(
        smiles_list,
        seed=args.seed,
        skip_methods=args.skip_methods or None,
        verbose=True,
    )

    if not rows:
        print(
            "[ERROR] No comparison rows produced. "
            "Check that at least one geometry method is available.",
            file=sys.stderr,
        )
        return 1

    # Write
    write_comparison_csv(rows, out_csv)
    print(f"\n[OK] Written {len(rows)} rows -> {out_csv}")

    if not args.no_summary:
        print_summary(rows)

    return 0


if __name__ == "__main__":
    sys.exit(main())
