"""
revision/benchmarks/qm9_geometry_comparison.py
===============================================
Post-processing tool: read ``qm9_raw_seeds.csv`` (the output of
``experiments/qm9.py``), extract rows at level 3 (ETKDGv3) and level 3_qc
(DFT geometry from ``revision/geometry_qc/qm9_qc_level.py``), pair them by
(model, seed), compute per-metric deltas, and write a paired comparison CSV.

**No training is run here.**  Training must already have been run with
``experiments/qm9.py`` at both levels to populate the input CSV before calling
this tool.

Public API  (imported by tests/test_qm9_geometry_comparison.py)
---------------------------------------------------------------
LEVEL_QC             str constant ``"3_qc"``
LEVEL_RDKIT          str constant ``"3"``
OUTPUT_COLUMNS       list[str] — column names for the comparison CSV
_float_or_nan(v)     parse string/None → float; NaN on failure
load_rows_for_level  read & index rows by (model, seed) for one level
pair_rows            intersect rdkit/qc dicts and compute deltas
write_comparison_csv write paired rows to a CSV file
print_summary        print a human-readable comparison table
main(argv)           CLI entry point; returns 0 on success, 1 on error

Usage — CLI
-----------
python -m revision.benchmarks.qm9_geometry_comparison \\
    --results-csv qm9_raw_seeds.csv \\
    --out-csv     results/qm9_geometry_comparison.csv

If level-3 and level-3_qc results live in *different* files, pass each path
separately via ``--results-csv`` (level-3) and ``--qc-csv`` (level-3_qc).

Output columns
--------------
model, seed, rdkit_key, qc_key,
rdkit_pearson_r, qc_pearson_r, delta_pearson_r,
rdkit_mae,       qc_mae,       delta_mae,
rdkit_rmse,      qc_rmse,      delta_rmse,
rdkit_ms_per_mol, qc_ms_per_mol,
rdkit_epochs_run, qc_epochs_run,
rdkit_stopped_early, qc_stopped_early
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Public level-ID constants
# ---------------------------------------------------------------------------

LEVEL_RDKIT: str = "3"
"""Level identifier for ETKDGv3 + MMFF geometry (experiments/qm9.py level 3)."""

LEVEL_QC: str = "3_qc"
"""Level identifier for DFT B3LYP/6-31G(2df,p) geometry (revision module)."""

# ---------------------------------------------------------------------------
# Output schema — must stay in sync with tests/test_qm9_geometry_comparison.py
# ---------------------------------------------------------------------------

OUTPUT_COLUMNS: List[str] = [
    "model", "seed",
    "rdkit_key", "qc_key",
    "rdkit_pearson_r", "qc_pearson_r", "delta_pearson_r",
    "rdkit_mae",       "qc_mae",       "delta_mae",
    "rdkit_rmse",      "qc_rmse",      "delta_rmse",
    "rdkit_ms_per_mol", "qc_ms_per_mol",
    "rdkit_epochs_run", "qc_epochs_run",
    "rdkit_stopped_early", "qc_stopped_early",
]


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _float_or_nan(v) -> float:
    """
    Convert *v* to ``float``; return ``NaN`` on failure, empty string, or
    ``None``.

    Parameters
    ----------
    v:
        A string (e.g. from ``csv.DictReader``), ``None``, or any numeric type.

    Returns
    -------
    float
    """
    if v is None:
        return float("nan")
    try:
        s = str(v).strip()
        if s == "":
            return float("nan")
        return float(s)
    except (ValueError, TypeError):
        return float("nan")


def _delta(a: float, b: float) -> float:
    """Return ``b - a``; propagate ``NaN`` if either operand is ``NaN``."""
    if math.isnan(a) or math.isnan(b):
        return float("nan")
    return b - a


# ---------------------------------------------------------------------------
# Core public functions
# ---------------------------------------------------------------------------

def load_rows_for_level(
    csv_path,
    level_id: str,
) -> Dict[Tuple[str, str], dict]:
    """
    Read *csv_path* and return a ``dict`` keyed by ``(model, seed)``
    containing only rows where the ``level_id`` column equals *level_id*.

    The expected CSV format is that written by ``experiments/qm9.py``::

        key, pearson_r, mae, rmse, train_time, ms_per_mol, n_params,
        epochs_run, stopped_early, dataset, model, level_id, seed

    Parameters
    ----------
    csv_path:
        Path (str or :class:`pathlib.Path`) to a ``qm9_raw_seeds.csv`` file.
    level_id:
        Geometry level to filter on, e.g. ``"3"`` or ``"3_qc"``.

    Returns
    -------
    dict
        Maps ``(model: str, seed: str)`` → ``row_dict``.  If a duplicate
        ``(model, seed)`` pair is encountered the first occurrence is kept and
        a :class:`UserWarning` is emitted.

    Raises
    ------
    FileNotFoundError
        If *csv_path* does not exist.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")

    result: Dict[Tuple[str, str], dict] = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("level_id", "").strip() != level_id:
                continue
            key = (row["model"].strip(), row["seed"].strip())
            if key in result:
                warnings.warn(
                    f"Duplicate (model={key[0]!r}, seed={key[1]!r}) at "
                    f"level={level_id!r} in {csv_path.name}; "
                    "keeping the first occurrence.",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            result[key] = row
    return result


def pair_rows(
    rdkit: Dict[Tuple[str, str], dict],
    qc:    Dict[Tuple[str, str], dict],
) -> List[dict]:
    """
    Intersect *rdkit* and *qc* dicts by ``(model, seed)``, compute deltas
    (``qc_metric − rdkit_metric``), and return a sorted list of paired rows.

    Rows present in one dict but absent from the other are silently dropped
    after printing a ``WARN`` line to ``stderr``.

    Sorting: ``(model, int(seed))`` ascending.

    Parameters
    ----------
    rdkit:
        Index returned by ``load_rows_for_level(path, LEVEL_RDKIT)``.
    qc:
        Index returned by ``load_rows_for_level(path, LEVEL_QC)``.

    Returns
    -------
    list[dict]
        Each element has all keys from :data:`OUTPUT_COLUMNS`.
        Numeric delta fields are ``float``; diagnostic pass-through fields
        (``rdkit_ms_per_mol``, etc.) are ``float`` or ``str`` respectively.
    """
    paired: List[dict] = []
    rdkit_keys = set(rdkit)
    qc_keys    = set(qc)

    # Warn about unmatched keys in either direction
    for k in sorted(rdkit_keys - qc_keys):
        print(
            f"WARN: (model={k[0]!r}, seed={k[1]!r}) has a level-3 row "
            "but no level-3_qc counterpart — skipped.",
            file=sys.stderr,
        )
    for k in sorted(qc_keys - rdkit_keys):
        print(
            f"WARN: (model={k[0]!r}, seed={k[1]!r}) has a level-3_qc row "
            "but no level-3 counterpart — skipped.",
            file=sys.stderr,
        )

    common = rdkit_keys & qc_keys
    for key in sorted(common, key=lambda k: (k[0], int(k[1]))):
        r = rdkit[key]
        q = qc[key]

        r_pr  = _float_or_nan(r.get("pearson_r"))
        q_pr  = _float_or_nan(q.get("pearson_r"))
        r_mae = _float_or_nan(r.get("mae"))
        q_mae = _float_or_nan(q.get("mae"))
        r_rms = _float_or_nan(r.get("rmse"))
        q_rms = _float_or_nan(q.get("rmse"))

        paired.append({
            "model":   key[0],
            "seed":    key[1],
            # original experiment-run keys for traceability
            "rdkit_key": r.get("key", ""),
            "qc_key":    q.get("key", ""),
            # per-metric paired values + deltas
            "rdkit_pearson_r":  r_pr,
            "qc_pearson_r":     q_pr,
            "delta_pearson_r":  _delta(r_pr, q_pr),
            "rdkit_mae":        r_mae,
            "qc_mae":           q_mae,
            "delta_mae":        _delta(r_mae, q_mae),
            "rdkit_rmse":       r_rms,
            "qc_rmse":          q_rms,
            "delta_rmse":       _delta(r_rms, q_rms),
            # diagnostic pass-through fields
            "rdkit_ms_per_mol":    _float_or_nan(r.get("ms_per_mol")),
            "qc_ms_per_mol":       _float_or_nan(q.get("ms_per_mol")),
            "rdkit_epochs_run":    r.get("epochs_run", ""),
            "qc_epochs_run":       q.get("epochs_run", ""),
            "rdkit_stopped_early": r.get("stopped_early", ""),
            "qc_stopped_early":    q.get("stopped_early", ""),
        })

    return paired


def write_comparison_csv(paired: List[dict], out_path) -> None:
    """
    Write *paired* (list of dicts from :func:`pair_rows`) to *out_path*.

    Parent directories are created automatically.  An empty *paired* list
    results in a header-only file.

    Parameters
    ----------
    paired:
        List of dicts as returned by :func:`pair_rows`.
    out_path:
        Destination path (str or :class:`pathlib.Path`).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(paired)


def print_summary(paired: List[dict]) -> None:
    """
    Print a human-readable geometry comparison summary to stdout.

    Reports mean ± std of ``delta_pearson_r``, ``delta_mae``, and
    ``delta_rmse`` overall and per model.  Handles an empty *paired* list
    gracefully.

    Parameters
    ----------
    paired:
        List of dicts as returned by :func:`pair_rows`.
    """
    if not paired:
        print("No paired rows to summarise.")
        return

    def _stats(values: list):
        vals = [v for v in values if not math.isnan(v)]
        if not vals:
            return float("nan"), float("nan")
        n    = len(vals)
        mean = sum(vals) / n
        std  = math.sqrt(sum((v - mean) ** 2 for v in vals) / max(n - 1, 1))
        return mean, std

    print("\n" + "=" * 64)
    print("QM9 Geometry Comparison: level 3 (ETKDGv3) vs 3_qc (DFT B3LYP)")
    print("  delta = qc_metric - rdkit_metric")
    print("  pearson_r: higher is better  (+delta = QC improved)")
    print("  mae/rmse:  lower  is better  (-delta = QC improved)")
    print("=" * 64)

    metrics = [
        ("pearson_r", "delta_pearson_r"),
        ("mae",       "delta_mae"),
        ("rmse",      "delta_rmse"),
    ]
    for label, col in metrics:
        mean, std = _stats([r[col] for r in paired])
        n_valid = sum(1 for r in paired if not math.isnan(r[col]))
        print(
            f"  OVERALL  delta_{label:<10s} "
            f"mean={mean:+.4f}  std={std:.4f}  (n={n_valid})"
        )

    models = sorted({r["model"] for r in paired})
    if len(models) > 1:
        print()
        for model in models:
            rows_m   = [r for r in paired if r["model"] == model]
            pr_m, pr_s = _stats([r["delta_pearson_r"] for r in rows_m])
            mae_m, _   = _stats([r["delta_mae"]       for r in rows_m])
            rms_m, _   = _stats([r["delta_rmse"]      for r in rows_m])
            print(
                f"  {model:<22s}  "
                f"d_pearson_r={pr_m:+.4f}+/-{pr_s:.4f}  "
                f"d_mae={mae_m:+.4f}  d_rmse={rms_m:+.4f}"
            )

    print("=" * 64 + "\n")


# ---------------------------------------------------------------------------
# CLI / main()
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    """
    CLI entry point.  Parse *argv* (or ``sys.argv[1:]``), run the comparison,
    and return an exit code.

    Parameters
    ----------
    argv:
        List of CLI argument strings, or ``None`` to use ``sys.argv[1:]``.

    Returns
    -------
    int
        0 on success, 1 on any error or empty data.
    """
    # Force UTF-8 on Windows consoles
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description=(
            "Pair QM9 level-3 (ETKDGv3) and level-3_qc (DFT) rows from "
            "qm9_raw_seeds.csv and write a geometry comparison CSV."
        )
    )
    parser.add_argument(
        "--results-csv",
        default="qm9_raw_seeds.csv",
        help=(
            "CSV file containing level-3 (and optionally level-3_qc) rows "
            "(default: qm9_raw_seeds.csv)"
        ),
    )
    parser.add_argument(
        "--qc-csv",
        default=None,
        help=(
            "CSV file containing level-3_qc rows.  "
            "Defaults to --results-csv when omitted."
        ),
    )
    parser.add_argument(
        "--out-csv",
        default="results/qm9_geometry_comparison.csv",
        help="Output CSV path (default: results/qm9_geometry_comparison.csv)",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip printing the summary table.",
    )

    args = parser.parse_args(argv)

    results_csv = Path(args.results_csv)
    qc_csv      = Path(args.qc_csv) if args.qc_csv else results_csv
    out_csv     = Path(args.out_csv)

    # Load both level sets
    try:
        rdkit_rows = load_rows_for_level(results_csv, LEVEL_RDKIT)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        qc_rows = load_rows_for_level(qc_csv, LEVEL_QC)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not rdkit_rows:
        print(
            f"ERROR: No level={LEVEL_RDKIT!r} rows found in {results_csv}.",
            file=sys.stderr,
        )
        return 1

    if not qc_rows:
        print(
            f"ERROR: No level={LEVEL_QC!r} rows found in {qc_csv}.",
            file=sys.stderr,
        )
        return 1

    paired = pair_rows(rdkit_rows, qc_rows)
    write_comparison_csv(paired, out_csv)
    print(f"[qm9_geometry_comparison] {len(paired)} paired rows -> {out_csv}")

    if not args.no_summary:
        print_summary(paired)

    return 0


if __name__ == "__main__":
    sys.exit(main())
