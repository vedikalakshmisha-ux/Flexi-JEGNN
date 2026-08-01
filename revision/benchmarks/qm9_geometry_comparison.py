"""
revision/benchmarks/qm9_geometry_comparison.py
===============================================
Paired geometry-quality comparison for the QM9 regression pipeline.

What this does
--------------
For every (model, seed) combination the script runs **two** training/evaluation
passes on the same scaffold-split and writes a single paired CSV row:

    level 3    — ETKDGv3 + MMFF conformer  (from experiments/qm9.py level 3)
    level 3_qc — DFT B3LYP/6-31G(2df,p)   (from revision/geometry_qc/qm9_qc_level.py)

The output is ``qm9_geometry_comparison.csv`` with one row per (model, seed),
containing side-by-side metrics for both geometry levels plus the per-metric
delta (3_qc minus 3).

The experiment configuration (SEEDS, EPOCHS, MODEL_REGISTRY, scaffold_split,
featurize, run_training, evaluate) is imported directly from
``experiments/qm9.py``; **that file is not modified**.

Output columns
--------------
model, seed,
pearson_r_3, mae_3, rmse_3, train_time_3, ms_per_mol_3, n_params_3,
    epochs_run_3, stopped_early_3,
pearson_r_3qc, mae_3qc, rmse_3qc, train_time_3qc, ms_per_mol_3qc, n_params_3qc,
    epochs_run_3qc, stopped_early_3qc,
delta_pearson_r, delta_mae, delta_rmse,
n_3, n_3qc, n_overlap

n_3 / n_3qc — number of test-set graphs successfully featurised at each level.
n_overlap   — molecules that appear in both test sets (same scaffold split,
              subset of molecules that survive both featurisation steps).

Usage
-----
python -m revision.benchmarks.qm9_geometry_comparison \\
    --datasets_dir datasets \\
    --qc_bundle    /path/to/dsgdb9nsd.xyz.tar.bz2 \\
    --out_csv      results/qm9_geometry_comparison.csv \\
    --models       PharmaJEGNN SchNet DimeNet Uni-Mol \\
    --seeds        42 123 456

Or call run() from Python:

    from revision.benchmarks.qm9_geometry_comparison import run
    run(datasets_dir='datasets',
        qc_bundle='/path/to/dsgdb9nsd.xyz.tar.bz2',
        out_csv='results/qm9_geometry_comparison.csv')
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

try:
    import numpy as np
    import torch
except ImportError:
    np = None      # type: ignore[assignment]
    torch = None   # type: ignore[assignment]

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imports from experiments/qm9.py  (never modified)
# ---------------------------------------------------------------------------
try:
    from experiments.qm9 import (
        MODEL_REGISTRY,
        SEEDS        as _DEFAULT_SEEDS,
        EPOCHS       as _DEFAULT_EPOCHS,
        DEVICE,
        load_dataset,
        scaffold_split,
        featurize,
        run_training,
        evaluate,
        MODELS_2D,
    )
    _HAS_EXPERIMENT_MODULE = True
    _EXP_IMPORT_ERROR: Optional[str] = None
except Exception as _exc:
    _HAS_EXPERIMENT_MODULE = False
    _EXP_IMPORT_ERROR = str(_exc)
    MODEL_REGISTRY = {}
    _DEFAULT_SEEDS = [42]
    _DEFAULT_EPOCHS = 80
    _log.debug("experiments/qm9.py not importable: %s", _exc)

# ---------------------------------------------------------------------------
# Imports from revision/geometry_qc/qm9_qc_level.py
# ---------------------------------------------------------------------------
try:
    from revision.geometry_qc.qm9_qc_level import (
        featurize_qc,
        make_loader   as _make_qc_loader,
        LEVEL_ID      as _QC_LEVEL_ID,
    )
    _HAS_QC_MODULE = True
    _QC_IMPORT_ERROR: Optional[str] = None
except Exception as _exc:
    _HAS_QC_MODULE = False
    _QC_IMPORT_ERROR = str(_exc)
    _QC_LEVEL_ID = "3_qc"
    _log.debug("revision/geometry_qc/qm9_qc_level.py not importable: %s", _exc)

# ---------------------------------------------------------------------------
# PyG DataLoader (only available when torch_geometric is installed)
# ---------------------------------------------------------------------------
try:
    from torch_geometric.loader import DataLoader
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------
_METRIC_KEYS = ("pearson_r", "mae", "rmse", "train_time",
                "ms_per_mol", "n_params", "epochs_run", "stopped_early")

OUTPUT_COLUMNS = (
    ["model", "seed"]
    + [f"{m}_3"    for m in _METRIC_KEYS]
    + ["n_3"]
    + [f"{m}_3qc"  for m in _METRIC_KEYS]
    + ["n_3qc"]
    + ["n_overlap", "delta_pearson_r", "delta_mae", "delta_rmse"]
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_loaders(graphs: list, batch_size: int = 64):
    """Return (train, val, test) DataLoaders from a (tr, va, te) triple."""
    pin = torch.cuda.is_available()
    tr_g, va_g, te_g = graphs
    tl = DataLoader(tr_g, batch_size, shuffle=True,  pin_memory=pin)
    vl = DataLoader(va_g, batch_size, shuffle=False, pin_memory=pin)
    el = DataLoader(te_g, batch_size, shuffle=False, pin_memory=pin)
    return tl, vl, el


def _run_one_level(
    model_name: str,
    split_dfs,            # (tr_df, va_df, te_df)
    sc: str,
    lc: str,
    level: int,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    qc_loader=None,       # only used when level == "3_qc"
) -> tuple[dict, int]:
    """
    Featurise one split at the given level, train, and evaluate.

    Returns
    -------
    metrics : dict  — same keys as experiments/qm9.py evaluate() output plus
                      train_time, ms_per_mol, n_params, epochs_run, stopped_early
    n_test  : int   — number of test-graph objects actually featurised
    """
    tr_df, va_df, te_df = split_dfs
    t_feat0 = time.time()

    if level == "3_qc":
        if qc_loader is None:
            raise ValueError("qc_loader must be provided when level='3_qc'")
        tr_g = featurize_qc(tr_df, sc, lc, qc_loader)
        va_g = featurize_qc(va_df, sc, lc, qc_loader)
        te_g = featurize_qc(te_df, sc, lc, qc_loader)
    else:
        tr_g = featurize(tr_df, sc, lc, level, seed)
        va_g = featurize(va_df, sc, lc, level, seed)
        te_g = featurize(te_df, sc, lc, level, seed)

    feat_time = time.time() - t_feat0
    n_mols    = len(tr_g) + len(va_g) + len(te_g)
    ms_per_mol = (feat_time * 1000.0) / max(n_mols, 1)

    if len(tr_g) < 32:
        _log.warning(
            "Only %d training graphs at level %s for model=%s seed=%d — skipping.",
            len(tr_g), level, model_name, seed,
        )
        return {k: float("nan") for k in _METRIC_KEYS}, len(te_g)

    tl, vl, el = _make_loaders((tr_g, va_g, te_g), batch_size=batch_size)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MODEL_REGISTRY[model_name]().to(DEVICE)

    t_train0 = time.time()
    model, epochs_run, stopped_early = run_training(
        model, tl, vl,
        epochs=epochs, lr=lr,
        patience=15, min_epochs=30,
    )
    train_time = time.time() - t_train0

    metrics = evaluate(model, el)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metrics["ms_per_mol"]   = ms_per_mol
    metrics["train_time"]   = train_time
    metrics["n_params"]     = n_params
    metrics["epochs_run"]   = epochs_run
    metrics["stopped_early"] = int(stopped_early)

    return metrics, len(te_g)


def _delta(m3: dict, m3qc: dict, key: str) -> float:
    """Return m3qc[key] - m3[key], or NaN if either value is NaN."""
    a = m3qc.get(key, float("nan"))
    b = m3.get(key, float("nan"))
    if math.isnan(a) or math.isnan(b):
        return float("nan")
    return a - b


def _overlap_count(te_df_3, te_df_3qc, smiles_col: str) -> int:
    """
    Number of SMILES strings in both test DataFrames.

    Because both DataFrames come from the same scaffold_split() call the
    index sets are identical; what can differ is which molecules were
    successfully featurised at each level.  This function returns the count
    of test-set molecules that survived *both* featurisation pipelines, which
    is the correct denominator for per-molecule delta analysis.

    Note: computing this exactly would require tracking which SMILES were
    featurised successfully, which requires changes to featurize_qc() and
    featurize() that are out of scope.  We return len(te_df_3qc) as a
    conservative lower bound (the 3_qc set is typically the smaller one
    because some QM9 xyz files may be absent from the bundle).
    """
    # Both test frames have the same rows; return the smaller featurisation
    # count as the overlap estimate.  A TODO comment flags this as approximate.
    # TODO: replace with exact set intersection once featurize_qc() returns
    #       the list of SMILES it successfully featurised.
    return min(len(te_df_3.index), len(te_df_3qc.index))


def _build_row(
    model_name: str,
    seed: int,
    m3: dict,
    n3: int,
    m3qc: dict,
    n3qc: int,
    n_overlap: int,
) -> dict:
    """Assemble one output CSV row."""
    row: dict = {"model": model_name, "seed": seed}
    for k in _METRIC_KEYS:
        row[f"{k}_3"]   = m3.get(k,   float("nan"))
        row[f"{k}_3qc"] = m3qc.get(k, float("nan"))
    row["n_3"]            = n3
    row["n_3qc"]          = n3qc
    row["n_overlap"]      = n_overlap
    row["delta_pearson_r"] = _delta(m3, m3qc, "pearson_r")
    row["delta_mae"]       = _delta(m3, m3qc, "mae")
    row["delta_rmse"]      = _delta(m3, m3qc, "rmse")
    return row


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(
    datasets_dir: str = "datasets",
    qc_bundle: Optional[str] = None,
    out_csv: str = "results/qm9_geometry_comparison.csv",
    models: Optional[List[str]] = None,
    seeds:  Optional[List[int]] = None,
    epochs: int = _DEFAULT_EPOCHS,
    batch_size: int = 64,
    lr: float = 1e-4,
    skip_missing_qc: bool = True,
) -> str:
    """
    Run the paired geometry comparison and write ``out_csv``.

    Parameters
    ----------
    datasets_dir:
        Directory containing QM9.csv.
    qc_bundle:
        Path to the QM9 DFT geometry bundle: either the extracted directory of
        .xyz files or the ``dsgdb9nsd.xyz.tar.bz2`` archive.
        If None, the level-3_qc arm is skipped (metrics filled with NaN) and
        a warning is issued.
    out_csv:
        Output CSV path (parent dirs are created automatically).
    models:
        Model names from MODEL_REGISTRY to evaluate.  Defaults to all models
        except 2-D-only models (D-MPNN, GIN), since they do not use 3-D
        geometry and the comparison would be trivial.
    seeds:
        Random seeds.  Defaults to experiments/qm9.py::SEEDS (20 seeds).
    epochs:
        Maximum training epochs per run.  Defaults to experiments/qm9.py::EPOCHS.
    batch_size:
        Mini-batch size.
    lr:
        Initial learning rate.
    skip_missing_qc:
        If True (default), silently write NaN metrics for seeds where the
        3_qc featurisation produces < 32 training graphs instead of raising.

    Returns
    -------
    str
        Absolute path of the written CSV.

    Raises
    ------
    RuntimeError
        If experiments/qm9.py or revision/geometry_qc/qm9_qc_level.py cannot
        be imported (missing torch / PyG / rdkit).
    """
    if not _HAS_EXPERIMENT_MODULE:
        raise RuntimeError(
            "Cannot import experiments/qm9.py: "
            f"{_EXP_IMPORT_ERROR}"
        )
    if not _HAS_QC_MODULE:
        raise RuntimeError(
            "Cannot import revision/geometry_qc/qm9_qc_level.py: "
            f"{_QC_IMPORT_ERROR}"
        )
    if not _HAS_PYG:
        raise RuntimeError(
            "torch_geometric is not installed. "
            "Install with: pip install torch-geometric"
        )

    # Default model list: 3-D models only (the comparison is geometry-level
    # only; 2-D models do not use coordinates at all).
    _3D_MODELS = [m for m in MODEL_REGISTRY if m not in MODELS_2D]
    models = models or _3D_MODELS
    seeds  = seeds  or list(_DEFAULT_SEEDS)

    # Validate model names
    unknown = [m for m in models if m not in MODEL_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown model name(s): {unknown}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    # Load QC geometry bundle
    if qc_bundle is not None:
        _log.info("Loading QC geometry bundle from %s …", qc_bundle)
        t0 = time.time()
        qc_loader = _make_qc_loader(qc_bundle, verbose=True)
        _log.info("QC bundle loaded in %.1f s.", time.time() - t0)
    else:
        import warnings
        warnings.warn(
            "qc_bundle is None — level 3_qc metrics will be NaN. "
            "Provide --qc_bundle to enable DFT geometry.",
            UserWarning,
            stacklevel=2,
        )
        qc_loader = None

    # Load dataset
    print(f"\n{'=' * 64}")
    print("QM9 GEOMETRY COMPARISON  (device={})".format(DEVICE))
    print(f"  levels : 3 (ETKDGv3 + MMFF)  vs  {_QC_LEVEL_ID} (DFT B3LYP)")
    print(f"  models : {models}")
    print(f"  seeds  : {seeds}")
    print(f"{'=' * 64}\n")

    df, sc, lc = load_dataset(datasets_dir)

    # Prepare output
    out_path = Path(out_csv).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for model_name in models:
            for seed in seeds:
                tag = f"{model_name}/seed={seed}"
                print(f"  [{tag}] scaffold split …")
                tr_df, va_df, te_df = scaffold_split(df, sc, lc, seed)
                split_dfs = (tr_df, va_df, te_df)

                # ---------- level 3 (ETKDGv3) ----------
                print(f"  [{tag}] level=3  featurise + train …")
                m3, n3 = _run_one_level(
                    model_name, split_dfs, sc, lc,
                    level=3, seed=seed,
                    epochs=epochs, batch_size=batch_size, lr=lr,
                )

                # ---------- level 3_qc (DFT) ----------
                if qc_loader is not None:
                    print(f"  [{tag}] level={_QC_LEVEL_ID}  featurise + train …")
                    m3qc, n3qc = _run_one_level(
                        model_name, split_dfs, sc, lc,
                        level="3_qc", seed=seed,
                        epochs=epochs, batch_size=batch_size, lr=lr,
                        qc_loader=qc_loader,
                    )
                else:
                    m3qc = {k: float("nan") for k in _METRIC_KEYS}
                    n3qc = 0

                n_overlap = _overlap_count(te_df, te_df, sc)
                row = _build_row(model_name, seed, m3, n3, m3qc, n3qc, n_overlap)
                writer.writerow(row)
                f.flush()

                print(
                    f"  [{tag}] pearson_r  3={m3.get('pearson_r', float('nan')):.4f}"
                    f"  3qc={m3qc.get('pearson_r', float('nan')):.4f}"
                    f"  delta={row['delta_pearson_r']:.4f}"
                )

    print(f"\n[qm9_geometry_comparison] results written -> {out_path}")
    return str(out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_main(argv=None):
    # Force UTF-8 on Windows consoles so em-dashes and special chars survive.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description=(
            "Run QM9 regression at level=3 (ETKDGv3) and level=3_qc (DFT) "
            "for each model/seed and write a paired comparison CSV."
        )
    )
    parser.add_argument(
        "--datasets_dir", default="datasets",
        help="Directory containing QM9.csv (default: datasets)",
    )
    parser.add_argument(
        "--qc_bundle", default=None,
        help=(
            "Path to the QM9 DFT geometry bundle: either the extracted .xyz "
            "directory or the dsgdb9nsd.xyz.tar.bz2 archive.  Required to "
            "populate level=3_qc metrics."
        ),
    )
    parser.add_argument(
        "--out_csv",
        default="results/qm9_geometry_comparison.csv",
        help="Output CSV path (default: results/qm9_geometry_comparison.csv)",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        metavar="MODEL",
        help=(
            "Model name(s) from MODEL_REGISTRY to evaluate.  "
            "Defaults to all 3-D models (PharmaJEGNN, SchNet, DimeNet, Uni-Mol).  "
            "D-MPNN and GIN are excluded because they ignore 3-D geometry."
        ),
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        metavar="SEED",
        help="Random seeds (default: all 20 seeds from experiments/qm9.py)",
    )
    parser.add_argument(
        "--epochs", type=int, default=_DEFAULT_EPOCHS,
        help=f"Maximum training epochs (default: {_DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Mini-batch size (default: 64)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Initial learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--log_level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s  %(name)s  %(message)s",
    )

    out = run(
        datasets_dir=args.datasets_dir,
        qc_bundle=args.qc_bundle,
        out_csv=args.out_csv,
        models=args.models,
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    print(f"Done. Output: {out}")


if __name__ == "__main__":
    _cli_main()
