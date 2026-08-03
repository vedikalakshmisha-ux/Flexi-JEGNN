"""
revision/benchmarks/qm9_geometry_comparison.py
===============================================
Compares geometry level 3 (RDKit ETKDGv3 + MMFF) vs level "3_qc"
(original DFT B3LYP/6-31G(2df,p) from the QM9 .xyz bundle) for every
model/seed combination present in both result files.

Inputs
------
* A CSV with level-3 rows — default: ``results/qm9_raw_seeds.csv``
  (same file written by ``experiments/qm9.py``).
* A CSV with level-3_qc rows — default: the same ``qm9_raw_seeds.csv``
  if level "3_qc" rows were appended there, OR a separate file produced
  by running the pipeline with ``LEVEL_ID = "3_qc"`` from
  ``revision/geometry_qc/qm9_qc_level.py``.  Pass ``--qc-csv`` to
  specify a different file.

Output
------
``qm9_geometry_comparison.csv`` — one row per (model, seed) pair with
both sets of metrics side-by-side and signed deltas (qc − rdkit):

    model, seed,
    rdkit_pearson_r, rdkit_mae, rdkit_rmse,
    qc_pearson_r,    qc_mae,    qc_rmse,
    delta_pearson_r, delta_mae, delta_rmse,
    rdkit_ms_per_mol, qc_ms_per_mol,
    rdkit_epochs_run, qc_epochs_run,
    rdkit_stopped_early, qc_stopped_early,
    rdkit_key, qc_key

Usage
-----
    # Both level-3 and 3_qc in the same file:
    python revision/benchmarks/qm9_geometry_comparison.py

    # 3_qc results in a separate file:
    python revision/benchmarks/qm9_geometry_comparison.py \\
        --qc-csv results/qm9_3qc_seeds.csv

    # Custom paths:
    python revision/benchmarks/qm9_geometry_comparison.py \\
        --results-csv results/qm9_raw_seeds.csv \\
        --qc-csv     results/qm9_3qc_seeds.csv \\
        --out-csv    results/qm9_geometry_comparison.csv

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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LEVEL_RDKIT: str = "3"
LEVEL_QC: str = "3_qc"

METRIC_COLS = ("pearson_r", "mae", "rmse")
PASS_THROUGH_COLS = ("ms_per_mol", "epochs_run", "stopped_early", "key")

OUTPUT_COLUMNS: List[str] = [
    "model",
    "seed",
    # RDKit-geometry metrics
    "rdkit_pearson_r",
    "rdkit_mae",
    "rdkit_rmse",
    # QC-geometry metrics
    "qc_pearson_r",
    "qc_mae",
    "qc_rmse",
    # Signed deltas: qc − rdkit  (positive = QC is better for r; worse for MAE/RMSE)
    "delta_pearson_r",
    "delta_mae",
    "delta_rmse",
    # Runtime / training diagnostics
    "rdkit_ms_per_mol",
    "qc_ms_per_mol",
    "rdkit_epochs_run",
    "qc_epochs_run",
    "rdkit_stopped_early",
    "qc_stopped_early",
    # Source keys for full traceability
    "rdkit_key",
    "qc_key",
]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _float_or_nan(v: str) -> float:
    """Parse a CSV value to float; return NaN on failure."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def load_rows_for_level(csv_path: Path, level_id: str) -> Dict[Tuple[str, str], dict]:
    """
    Read *csv_path* and return a dict keyed by ``(model, seed)`` for all
    rows whose ``level_id`` column matches *level_id*.

    Raises ``FileNotFoundError`` if *csv_path* does not exist.
    Raises ``ValueError`` if the file has no rows for *level_id*.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")

    rows: Dict[Tuple[str, str], dict] = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("level_id", "").strip() != level_id:
                continue
            model = row.get("model", "").strip()
            seed = row.get("seed", "").strip()
            if not model or not seed:
                continue
            key = (model, seed)
            if key in rows:
                # Duplicate — keep first occurrence, warn
                import warnings
                warnings.warn(
                    f"Duplicate (model={model}, seed={seed}) for level {level_id!r} "
                    f"in {csv_path.name}; keeping first occurrence.",
                    stacklevel=2,
                )
                continue
            rows[key] = row

    return rows


def pair_rows(
    rdkit_rows: Dict[Tuple[str, str], dict],
    qc_rows: Dict[Tuple[str, str], dict],
) -> List[dict]:
    """
    Join rdkit and qc row dicts on ``(model, seed)``.

    Returns a list of paired comparison dicts, one per matched pair.
    Unmatched keys are silently skipped (both sides logged to stderr).
    """
    common_keys = sorted(
        rdkit_rows.keys() & qc_rows.keys(),
        key=lambda k: (k[0], int(k[1]) if k[1].isdigit() else k[1]),
    )

    only_rdkit = rdkit_rows.keys() - qc_rows.keys()
    only_qc = qc_rows.keys() - rdkit_rows.keys()

    if only_rdkit:
        print(
            f"  [WARN] {len(only_rdkit)} (model,seed) pair(s) have level-3 but no "
            f"level-3_qc results — skipped:",
            file=sys.stderr,
        )
        for k in sorted(only_rdkit):
            print(f"         model={k[0]}, seed={k[1]}", file=sys.stderr)

    if only_qc:
        print(
            f"  [WARN] {len(only_qc)} (model,seed) pair(s) have level-3_qc but no "
            f"level-3 results — skipped:",
            file=sys.stderr,
        )
        for k in sorted(only_qc):
            print(f"         model={k[0]}, seed={k[1]}", file=sys.stderr)

    paired: List[dict] = []
    for model, seed in common_keys:
        r = rdkit_rows[(model, seed)]
        q = qc_rows[(model, seed)]

        # Metrics
        r_pr = _float_or_nan(r.get("pearson_r", ""))
        r_mae = _float_or_nan(r.get("mae", ""))
        r_rmse = _float_or_nan(r.get("rmse", ""))

        q_pr = _float_or_nan(q.get("pearson_r", ""))
        q_mae = _float_or_nan(q.get("mae", ""))
        q_rmse = _float_or_nan(q.get("rmse", ""))

        def _delta(a: float, b: float) -> float:
            """b − a, propagating NaN."""
            if math.isnan(a) or math.isnan(b):
                return math.nan
            return b - a

        paired.append({
            "model": model,
            "seed": seed,
            "rdkit_pearson_r": r_pr,
            "rdkit_mae": r_mae,
            "rdkit_rmse": r_rmse,
            "qc_pearson_r": q_pr,
            "qc_mae": q_mae,
            "qc_rmse": q_rmse,
            "delta_pearson_r": _delta(r_pr, q_pr),
            "delta_mae": _delta(r_mae, q_mae),
            "delta_rmse": _delta(r_rmse, q_rmse),
            "rdkit_ms_per_mol": _float_or_nan(r.get("ms_per_mol", "")),
            "qc_ms_per_mol": _float_or_nan(q.get("ms_per_mol", "")),
            "rdkit_epochs_run": r.get("epochs_run", ""),
            "qc_epochs_run": q.get("epochs_run", ""),
            "rdkit_stopped_early": r.get("stopped_early", ""),
            "qc_stopped_early": q.get("stopped_early", ""),
            "rdkit_key": r.get("key", ""),
            "qc_key": q.get("key", ""),
        })

    return paired


def write_comparison_csv(paired: List[dict], out_path: Path) -> None:
    """Write the paired comparison rows to *out_path*."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(paired)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(paired: List[dict]) -> None:
    """Print a per-model aggregate summary to stdout."""
    if not paired:
        print("No paired rows to summarise.")
        return

    # Group by model
    by_model: Dict[str, List[dict]] = {}
    for row in paired:
        by_model.setdefault(row["model"], []).append(row)

    print(f"\n{'=' * 72}")
    print("QM9 geometry comparison: RDKit (level 3) vs QC DFT (level 3_qc)")
    print(f"{'=' * 72}")
    print(f"{'Model':<16} {'N':>4}  "
          f"{'Δ Pearson_r':>12}  {'Δ MAE':>10}  {'Δ RMSE':>10}  "
          f"{'QC better (r)':>14}")
    print("-" * 72)

    for model in sorted(by_model):
        rows = by_model[model]
        deltas_pr = [r["delta_pearson_r"] for r in rows if not math.isnan(r["delta_pearson_r"])]
        deltas_mae = [r["delta_mae"] for r in rows if not math.isnan(r["delta_mae"])]
        deltas_rmse = [r["delta_rmse"] for r in rows if not math.isnan(r["delta_rmse"])]

        def _mean(lst):
            return sum(lst) / len(lst) if lst else math.nan

        mean_pr = _mean(deltas_pr)
        mean_mae = _mean(deltas_mae)
        mean_rmse = _mean(deltas_rmse)
        n_better = sum(1 for d in deltas_pr if d > 0)  # higher r = better

        print(
            f"{model:<16} {len(rows):>4}  "
            f"{mean_pr:>+12.5f}  {mean_mae:>+10.5f}  {mean_rmse:>+10.5f}  "
            f"{n_better:>6}/{len(deltas_pr):<6}"
        )

    print("-" * 72)
    all_pr = [r["delta_pearson_r"] for r in paired if not math.isnan(r["delta_pearson_r"])]
    all_mae = [r["delta_mae"] for r in paired if not math.isnan(r["delta_mae"])]
    all_rmse = [r["delta_rmse"] for r in paired if not math.isnan(r["delta_rmse"])]

    def _mean(lst):
        return sum(lst) / len(lst) if lst else math.nan

    print(
        f"{'OVERALL':<16} {len(paired):>4}  "
        f"{_mean(all_pr):>+12.5f}  {_mean(all_mae):>+10.5f}  {_mean(all_rmse):>+10.5f}  "
        f"{sum(1 for d in all_pr if d > 0):>6}/{len(all_pr):<6}"
    )
    print(f"{'=' * 72}\n")
    print("  Δ = QC − RDKit  (positive Δ Pearson_r = QC geometry helps;")
    print("                   negative Δ MAE/RMSE  = QC geometry helps)\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Compare QM9 geometry level 3 (RDKit) vs 3_qc (DFT QC) metrics.\n\n"
            "Both levels may live in the same CSV (default) or separate files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--results-csv",
        default=None,
        help=(
            "CSV containing at minimum the level-3 rows. "
            "Defaults to <repo_root>/results/qm9_raw_seeds.csv."
        ),
    )
    p.add_argument(
        "--qc-csv",
        default=None,
        help=(
            "CSV containing the level-3_qc rows. "
            "If omitted, the script looks in --results-csv for 3_qc rows."
        ),
    )
    p.add_argument(
        "--out-csv",
        default=None,
        help=(
            "Output path for the comparison CSV. "
            "Defaults to <repo_root>/results/qm9_geometry_comparison.csv."
        ),
    )
    p.add_argument(
        "--no-summary",
        action="store_true",
        help="Suppress the per-model summary table printed to stdout.",
    )
    return p


def _resolve_path(cli_arg: Optional[str], default_relative: str) -> Path:
    if cli_arg:
        return Path(cli_arg).expanduser().resolve()
    here = Path(__file__).resolve().parent           # revision/benchmarks/
    return here.parent.parent / default_relative     # repo/<default_relative>


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    results_csv = _resolve_path(args.results_csv, "results/qm9_raw_seeds.csv")
    qc_csv = _resolve_path(args.qc_csv, "results/qm9_raw_seeds.csv") if not args.qc_csv \
        else Path(args.qc_csv).expanduser().resolve()
    out_csv = _resolve_path(args.out_csv, "results/qm9_geometry_comparison.csv")

    print(f"\nLevel-3 source  : {results_csv}")
    print(f"Level-3_qc source: {qc_csv}")
    print(f"Output          : {out_csv}")
    print("-" * 60)

    # Load
    try:
        rdkit_rows = load_rows_for_level(results_csv, LEVEL_RDKIT)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    try:
        qc_rows = load_rows_for_level(qc_csv, LEVEL_QC)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(f"  Level-3 rows found   : {len(rdkit_rows)}")
    print(f"  Level-3_qc rows found: {len(qc_rows)}")

    if not rdkit_rows:
        print(
            "[ERROR] No level-3 rows found. Has the QM9 experiment been run?",
            file=sys.stderr,
        )
        return 1

    if not qc_rows:
        print(
            "[ERROR] No level-3_qc rows found.\n"
            "Run the QC geometry experiment first:\n"
            "  from revision.geometry_qc.qm9_qc_level import featurize_qc, LEVEL_ID\n"
            "Then append results to a CSV and pass it via --qc-csv.",
            file=sys.stderr,
        )
        return 1

    # Pair
    paired = pair_rows(rdkit_rows, qc_rows)
    print(f"  Matched pairs        : {len(paired)}")

    if not paired:
        print("[ERROR] No matched (model, seed) pairs — cannot produce comparison.", file=sys.stderr)
        return 1

    # Write
    write_comparison_csv(paired, out_csv)
    print(f"\n[OK] Written -> {out_csv}")

    # Summary
    if not args.no_summary:
        print_summary(paired)

    return 0


if __name__ == "__main__":
    sys.exit(main())
