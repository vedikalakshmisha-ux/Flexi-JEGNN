"""
revision/benchmarks/qm9_qc_runner.py
======================================
Runner that trains QM9 regression models using DFT B3LYP/6-31G(2df,p)
geometry (level ``"3_qc"``) and writes results in the **exact same CSV
format** as ``experiments/qm9.py`` (``qm9_raw_seeds.csv``).

What this does
--------------
For every (model, seed) combination the runner:

1. Calls ``scaffold_split`` from ``experiments/qm9.py``
2. Featurises each split with ``featurize_qc`` from
   ``revision/geometry_qc/qm9_qc_level.py`` (DFT atom positions from the
   official QM9 .xyz bundle instead of an RDKit-generated conformer)
3. Trains with ``run_training`` from ``experiments/qm9.py``
4. Evaluates with ``evaluate`` from ``experiments/qm9.py``
5. Writes one row per (model, seed) to ``qm9_raw_seeds_qc.csv`` in the
   identical column format used by ``qm9_raw_seeds.csv``

The output file is **separate** from the existing ``qm9_raw_seeds.csv`` and
``experiments/qm9.py`` is **never modified**.

Output columns  (identical to experiments/qm9.py OUTPUT_COLUMNS)
-----------------------------------------------------------------
key, pearson_r, mae, rmse, train_time, ms_per_mol, n_params,
epochs_run, stopped_early, dataset, model, level_id, seed

``level_id`` is always ``"3_qc"``.
``key``       follows the pattern ``QM9_{model}_{level_id}_{seed}``.

2-D models (D-MPNN, GIN)
-------------------------
These are excluded by default, consistent with how ``experiments/qm9.py``
skips them at levels 3 and 4.  Pass ``--include-2d-models`` to override.

Usage — CLI
-----------
python -m revision.benchmarks.qm9_qc_runner \\
    --datasets_dir  datasets \\
    --qc_bundle     /path/to/dsgdb9nsd.xyz.tar.bz2 \\
    --out_csv       results/qm9_raw_seeds_qc.csv

python -m revision.benchmarks.qm9_qc_runner \\
    --datasets_dir  datasets \\
    --qc_bundle     /path/to/dsgdb9nsd.xyz.tar.bz2 \\
    --models        PharmaJEGNN SchNet \\
    --seeds         42 123 456 \\
    --epochs        40

Usage — Python
--------------
from revision.benchmarks.qm9_qc_runner import run
run(
    datasets_dir="datasets",
    qc_bundle="/path/to/dsgdb9nsd.xyz.tar.bz2",
    out_csv="results/qm9_raw_seeds_qc.csv",
)
"""

from __future__ import annotations

import argparse
import copy
import csv
import logging
import math
import sys
import time
from pathlib import Path
from typing import List, Optional

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imports from experiments/qm9.py  — never modified
# ---------------------------------------------------------------------------
try:
    from experiments.qm9 import (
        MODEL_REGISTRY,
        SEEDS          as _DEFAULT_SEEDS,
        EPOCHS         as _DEFAULT_EPOCHS,
        DATASET_NAME,
        DEVICE,
        OUTPUT_COLUMNS,
        MODELS_2D,
        load_dataset,
        scaffold_split,
        run_training,
        evaluate,
    )
    _HAS_EXPERIMENT_MODULE = True
    _EXP_IMPORT_ERROR: Optional[str] = None
except Exception as _exc:
    _HAS_EXPERIMENT_MODULE = False
    _EXP_IMPORT_ERROR = str(_exc)
    MODEL_REGISTRY  = {}
    _DEFAULT_SEEDS  = [42]
    _DEFAULT_EPOCHS = 80
    DATASET_NAME    = "QM9"
    OUTPUT_COLUMNS  = [
        "key", "pearson_r", "mae", "rmse", "train_time", "ms_per_mol",
        "n_params", "epochs_run", "stopped_early",
        "dataset", "model", "level_id", "seed",
    ]
    MODELS_2D = {"D-MPNN", "GIN"}
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
# PyG DataLoader
# ---------------------------------------------------------------------------
try:
    import numpy as np
    import torch
    from torch_geometric.loader import DataLoader
    _HAS_PYG = True
except ImportError:
    np = None       # type: ignore[assignment]
    torch = None    # type: ignore[assignment]
    _HAS_PYG = False

# ---------------------------------------------------------------------------
# Level identifier written into every output row
# ---------------------------------------------------------------------------
LEVEL_ID: str = _QC_LEVEL_ID   # "3_qc"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_loaders(tr_g: list, va_g: list, te_g: list, batch_size: int):
    """Wrap three graph lists in PyG DataLoaders."""
    pin = torch.cuda.is_available()
    tl = DataLoader(tr_g, batch_size, shuffle=True,  pin_memory=pin)
    vl = DataLoader(va_g, batch_size, shuffle=False, pin_memory=pin)
    el = DataLoader(te_g, batch_size, shuffle=False, pin_memory=pin)
    return tl, vl, el


def _build_row(model_name: str, seed: int, metrics: dict) -> dict:
    """
    Assemble one output CSV row matching the ``OUTPUT_COLUMNS`` schema used
    by ``experiments/qm9.py``.

    The ``key`` field uses the same pattern:
    ``{DATASET_NAME}_{model_name}_{LEVEL_ID}_{seed}``

    Parameters
    ----------
    model_name:
        Name of the model as it appears in MODEL_REGISTRY.
    seed:
        Integer random seed used for this run.
    metrics:
        Dict returned by ``evaluate()`` and augmented with ``train_time``,
        ``ms_per_mol``, ``n_params``, ``epochs_run``, ``stopped_early``.

    Returns
    -------
    dict with exactly the keys from OUTPUT_COLUMNS.
    """
    key = f"{DATASET_NAME}_{model_name}_{LEVEL_ID}_{seed}"
    return {
        "key":           key,
        "pearson_r":     metrics.get("pearson_r"),
        "mae":           metrics.get("mae"),
        "rmse":          metrics.get("rmse"),
        "train_time":    metrics.get("train_time"),
        "ms_per_mol":    metrics.get("ms_per_mol"),
        "n_params":      metrics.get("n_params"),
        "epochs_run":    metrics.get("epochs_run"),
        "stopped_early": metrics.get("stopped_early"),
        "dataset":       DATASET_NAME,
        "model":         model_name,
        "level_id":      LEVEL_ID,
        "seed":          seed,
    }


def _run_one_seed(
    model_name: str,
    df,
    sc: str,
    lc: str,
    seed: int,
    qc_loader,
    epochs: int,
    batch_size: int,
    lr: float,
) -> Optional[dict]:
    """
    Run one (model, seed) experiment at level ``"3_qc"`` and return a metrics
    dict, or ``None`` if the featurised training set is too small.

    Parameters
    ----------
    model_name:
        Key in MODEL_REGISTRY.
    df:
        Full QM9 DataFrame (from ``load_dataset``).
    sc, lc:
        SMILES and label column names.
    seed:
        Random seed for scaffold split and torch.
    qc_loader:
        ``QM9QCGeometryLoader`` instance for DFT atom positions.
    epochs, batch_size, lr:
        Training hyper-parameters.

    Returns
    -------
    dict or None
    """
    tr_df, va_df, te_df = scaffold_split(df, sc, lc, seed)

    t_feat0 = time.time()
    tr_g = featurize_qc(tr_df, sc, lc, qc_loader)
    va_g = featurize_qc(va_df, sc, lc, qc_loader)
    te_g = featurize_qc(te_df, sc, lc, qc_loader)
    feat_time  = time.time() - t_feat0
    n_mols     = len(tr_g) + len(va_g) + len(te_g)
    ms_per_mol = (feat_time * 1000.0) / max(n_mols, 1)

    if len(tr_g) < 32:
        _log.warning(
            "Only %d training graphs for model=%s seed=%d at level=%s "
            "(molecules absent from QC bundle?). Skipping.",
            len(tr_g), model_name, seed, LEVEL_ID,
        )
        return None

    tl, vl, el = _make_loaders(tr_g, va_g, te_g, batch_size)

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

    metrics            = evaluate(model, el)
    metrics["n_params"]      = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    metrics["train_time"]    = train_time
    metrics["ms_per_mol"]    = ms_per_mol
    metrics["epochs_run"]    = epochs_run
    metrics["stopped_early"] = int(stopped_early)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(
    datasets_dir: str = "datasets",
    qc_bundle: Optional[str] = None,
    out_csv: str = "results/qm9_raw_seeds_qc.csv",
    models: Optional[List[str]] = None,
    seeds:  Optional[List[int]] = None,
    epochs: int = _DEFAULT_EPOCHS,
    batch_size: int = 64,
    lr: float = 1e-4,
    include_2d_models: bool = False,
) -> str:
    """
    Run QM9 regression at level ``"3_qc"`` for every (model, seed) and write
    results to *out_csv* in the exact column format of ``qm9_raw_seeds.csv``.

    Parameters
    ----------
    datasets_dir:
        Directory containing ``QM9.csv``.
    qc_bundle:
        Path to the QM9 DFT geometry bundle — either the extracted directory
        of ``.xyz`` files or the ``dsgdb9nsd.xyz.tar.bz2`` archive.
        **Required.**
    out_csv:
        Output CSV path (default: ``results/qm9_raw_seeds_qc.csv``).
        Parent directories are created automatically.
        **This file is separate from ``qm9_raw_seeds.csv`` and that file is
        never modified.**
    models:
        Model names from MODEL_REGISTRY to evaluate.  Defaults to all 3-D
        models (i.e. MODEL_REGISTRY minus MODELS_2D unless
        ``include_2d_models=True``).
    seeds:
        Random seeds.  Defaults to ``experiments/qm9.py::SEEDS`` (20 seeds).
    epochs:
        Maximum training epochs.  Defaults to ``experiments/qm9.py::EPOCHS``.
    batch_size:
        Mini-batch size.
    lr:
        Initial learning rate.
    include_2d_models:
        If ``False`` (default), D-MPNN and GIN are excluded — consistent with
        how ``experiments/qm9.py`` skips them at level 3.  Set ``True`` to
        include them (they will still receive 3-D-distance-based edge
        attributes; they just won't use them in their message-passing).

    Returns
    -------
    str
        Absolute path of the written CSV.

    Raises
    ------
    RuntimeError
        If ``experiments/qm9.py`` or
        ``revision/geometry_qc/qm9_qc_level.py`` cannot be imported, or if
        ``qc_bundle`` is not provided.
    """
    if not _HAS_EXPERIMENT_MODULE:
        raise RuntimeError(
            f"Cannot import experiments/qm9.py: {_EXP_IMPORT_ERROR}"
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
    if qc_bundle is None:
        raise RuntimeError(
            "--qc_bundle is required. Provide the path to the QM9 .xyz "
            "bundle (directory or dsgdb9nsd.xyz.tar.bz2)."
        )

    # Build model list
    all_models = list(MODEL_REGISTRY.keys())
    if not include_2d_models:
        all_models = [m for m in all_models if m not in MODELS_2D]
    models = models or all_models
    seeds  = seeds  or list(_DEFAULT_SEEDS)

    unknown = [m for m in models if m not in MODEL_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown model name(s): {unknown}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    # Load QC bundle
    _log.info("Loading QC geometry bundle from %s …", qc_bundle)
    t0 = time.time()
    qc_loader = _make_qc_loader(qc_bundle, verbose=True)
    _log.info("QC bundle loaded in %.1f s.", time.time() - t0)

    # Load QM9 CSV
    print(f"\n{'=' * 64}")
    print(f"QM9 QC RUNNER  (level={LEVEL_ID!r})  device={DEVICE}")
    print(f"  models : {models}")
    print(f"  seeds  : {seeds}")
    print(f"  epochs : {epochs}")
    print(f"{'=' * 64}\n")

    df, sc, lc = load_dataset(datasets_dir)

    # Prepare output — write header immediately so the file exists even if
    # every experiment is skipped.
    out_path = Path(out_csv).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for model_name in models:
            for seed in seeds:
                tag = f"{DATASET_NAME}_{model_name}_{LEVEL_ID}_{seed}"
                print(f"  {tag} …")
                metrics = _run_one_seed(
                    model_name, df, sc, lc, seed,
                    qc_loader, epochs, batch_size, lr,
                )
                if metrics is None:
                    print(f"  {tag}  SKIPPED (too few QC-featurised molecules)")
                    continue
                row = _build_row(model_name, seed, metrics)
                writer.writerow(row)
                fh.flush()
                rows_written += 1
                print(
                    f"  {tag}  "
                    f"pearson_r={metrics.get('pearson_r', float('nan')):.4f}  "
                    f"mae={metrics.get('mae', float('nan')):.4f}  "
                    f"rmse={metrics.get('rmse', float('nan')):.4f}"
                )

    print(
        f"\n[qm9_qc_runner] {rows_written} rows written -> {out_path}\n"
        f"  Feed this file to qm9_geometry_comparison.py via --qc-csv."
    )
    return str(out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_main(argv=None):
    # Force UTF-8 on Windows consoles
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description=(
            "Run QM9 regression at level=3_qc (DFT geometry) for each "
            "(model, seed) and write rows in the qm9_raw_seeds.csv format."
        )
    )
    parser.add_argument(
        "--datasets_dir", default="datasets",
        help="Directory containing QM9.csv (default: datasets)",
    )
    parser.add_argument(
        "--qc_bundle", required=True,
        help=(
            "Path to the QM9 DFT geometry bundle: either the extracted .xyz "
            "directory or the dsgdb9nsd.xyz.tar.bz2 archive."
        ),
    )
    parser.add_argument(
        "--out_csv",
        default="results/qm9_raw_seeds_qc.csv",
        help=(
            "Output CSV path (default: results/qm9_raw_seeds_qc.csv). "
            "This file is SEPARATE from qm9_raw_seeds.csv."
        ),
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        metavar="MODEL",
        help=(
            "Model name(s) from MODEL_REGISTRY. "
            "Default: all 3-D models (PharmaJEGNN, SchNet, DimeNet, Uni-Mol). "
            "D-MPNN and GIN are excluded unless --include-2d-models is set."
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
        "--include-2d-models", action="store_true",
        help=(
            "Include D-MPNN and GIN in the run (they are excluded by default, "
            "consistent with experiments/qm9.py skipping them at level 3)."
        ),
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
        include_2d_models=args.include_2d_models,
    )
    print(f"Done. Output: {out}")


if __name__ == "__main__":
    _cli_main()
