"""
revision/benchmarks/reproduce_baselines.py
==========================================
Re-run the D-MPNN and GIN baselines from experiments/classification.py and
compare the results to originally-reported literature values.

Design
------
This script is intentionally a thin wrapper around the existing
classification.py infrastructure.  It imports the model classes, atom
featuriser, scaffold-split function and training loop directly from that
module, so the **exact same PyG graph construction, atom features,
edge features, training hyper-parameters and early-stopping logic** are
used here as in the main experiment.  No reimplementation.

What this adds
--------------
  1. A structured BenchmarkResult dataclass that records the replicated
     metric alongside the originally-published value (or a citation
     placeholder where the paper value is not known with certainty).
  2. A per-model runner that sweeps n_seeds, stores per-fold / per-seed
     results, and writes a JSON artefact.
  3. A summary table printed to stdout (mean ± std ROC-AUC, delta from
     literature).
  4. A CLI so reviewers can re-run with a single command.

Working implementations
-----------------------
  D-MPNN  (Yang et al. 2019, Chemprop)   — fully implemented below.
  GIN     (Xu et al. 2019 / Hu et al. 2020) — fully implemented below.

TODO stubs (implementation deferred — see comments in each function)
--------------------------------------------------------------------
  SchNet      — requires 3-D coordinates (level-3 graph); stub below.
  DimeNet     — requires 3-D coordinates and angle features; stub below.
  Uni-Mol     — requires pre-trained transformer weights; stub below.
  AttentiveFP — Xiong et al. 2020; stub below.

Usage
-----
# Run D-MPNN on BACE with 3 seeds, scaffold split:
python -m revision.benchmarks.reproduce_baselines \\
    --model dmpnn --dataset BACE \\
    --datasets-dir datasets/ \\
    --n-seeds 3 --seed0 42 \\
    --level 0 \\
    --out revision/benchmarks/results/bace_dmpnn.json

# Run GIN on all three datasets, 5 seeds:
python -m revision.benchmarks.reproduce_baselines \\
    --model gin --dataset all \\
    --datasets-dir datasets/ \\
    --n-seeds 5 --seed0 42 \\
    --level 0 \\
    --out revision/benchmarks/results/gin_all.json

# Run both working models on all datasets:
python -m revision.benchmarks.reproduce_baselines \\
    --model all --dataset all \\
    --datasets-dir datasets/ \\
    --n-seeds 3 --seed0 42 \\
    --out revision/benchmarks/results/baselines_all.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Lazily-checked hard dependencies
# ---------------------------------------------------------------------------
def _require(pkg: str, install_hint: str = "") -> None:
    try:
        __import__(pkg)
    except ImportError:
        hint = f"  Install: {install_hint}" if install_hint else ""
        raise ImportError(
            f"Required package '{pkg}' is not installed.{hint}"
        )


# ---------------------------------------------------------------------------
# Import from the existing experiment module (no reimplementation)
# ---------------------------------------------------------------------------
# We add the project root to sys.path so the import works regardless of cwd.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from experiments.classification import (   # noqa: E402
        DATASETS,
        MODEL_REGISTRY,
        IN_DIM,
        EDGE_DIM,
        EPOCHS,
        DEVICE,
        featurize,
        scaffold_split,
        run_training,
        evaluate,
        pos_weight,
        load_dataset,
    )
    _HAS_EXPERIMENT_MODULE = True
except Exception as _exp_err:
    _HAS_EXPERIMENT_MODULE = False
    _EXP_IMPORT_ERROR = _exp_err

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
_log = logging.getLogger(__name__)

# Fallback constant so function signatures work even when classification.py
# cannot be imported (e.g. missing PyG).  Overridden at runtime if import ok.
_EPOCHS_DEFAULT = EPOCHS if _HAS_EXPERIMENT_MODULE else 80


# ---------------------------------------------------------------------------
# Published literature values
# ---------------------------------------------------------------------------
# Rules for this table:
#   - Only fill original_roc_auc when the value can be read directly from the
#     cited paper without ambiguity (correct dataset, split type, metric).
#   - Leave None + add a citation_note if there is any doubt.
#   - "scaffold" split is standard for MoleculeNet unless noted otherwise.
#
# D-MPNN: Yang et al. 2019 J. Chem. Inf. Model. 59(8) 3370-3388
#         doi: 10.1021/acs.jcim.9b00237
#         Table 3 reports scaffold-split mean ROC-AUC. We leave the exact
#         values None because the paper evaluates on several scaffold-split
#         variants and it is not certain which variant the Flexi-JEGNN paper
#         used as the reference.
#
# GIN: Hu et al. 2020 ICLR "Strategies for Pre-training GNNs"
#      The paper reports results for GIN with and without pre-training.
#      The no-pre-training row is the relevant baseline, but the exact split
#      seed may differ from ours, so we leave the values None.

LITERATURE: Dict[str, Dict[str, dict]] = {
    "dmpnn": {
        "BACE": {
            "original_roc_auc": None,          # [CITE: Yang et al. 2019, Table 3]
            "original_std":     None,
            "split":            "scaffold",
            "citation":         "Yang et al. 2019, J. Chem. Inf. Model. 59(8), doi:10.1021/acs.jcim.9b00237",
        },
        "HIV": {
            "original_roc_auc": None,          # [CITE: Yang et al. 2019, Table 3]
            "original_std":     None,
            "split":            "scaffold",
            "citation":         "Yang et al. 2019, J. Chem. Inf. Model. 59(8), doi:10.1021/acs.jcim.9b00237",
        },
        "BBBP": {
            "original_roc_auc": None,          # [CITE: Yang et al. 2019, Table 3]
            "original_std":     None,
            "split":            "scaffold",
            "citation":         "Yang et al. 2019, J. Chem. Inf. Model. 59(8), doi:10.1021/acs.jcim.9b00237",
        },
    },
    "gin": {
        "BACE": {
            "original_roc_auc": None,          # [CITE: Hu et al. 2020 ICLR, Table 1, GIN no pre-train]
            "original_std":     None,
            "split":            "scaffold",
            "citation":         "Xu et al. 2019 ICLR, 'How Powerful are Graph Neural Networks?', doi:10.48550/arXiv.1810.00826",
        },
        "HIV": {
            "original_roc_auc": None,          # [CITE: Hu et al. 2020 ICLR, Table 1, GIN no pre-train]
            "original_std":     None,
            "split":            "scaffold",
            "citation":         "Xu et al. 2019 ICLR, 'How Powerful are Graph Neural Networks?', doi:10.48550/arXiv.1810.00826",
        },
        "BBBP": {
            "original_roc_auc": None,          # [CITE: Hu et al. 2020 ICLR, Table 1, GIN no pre-train]
            "original_std":     None,
            "split":            "scaffold",
            "citation":         "Xu et al. 2019 ICLR, 'How Powerful are Graph Neural Networks?', doi:10.48550/arXiv.1810.00826",
        },
    },
    # SchNet: the manuscript cites schutt2018schnet (J. Chem. Phys. 148:241722,
    # classification numbers exist in some leaderboard entries but the exact
    # split, seed, and whether ETKDGv3 vs DFT geometries were used is not
    # confirmed for the Flexi-JEGNN comparison — leaving None.
    "schnet": {
        "BACE": {
            "original_roc_auc": None,   # [CITE: confirm table/split in Flexi-JEGNN paper]
            "original_std":     None,
            "split":            "scaffold",
            "citation":         "Schutt et al. 2018 J. Chem. Phys. 148:241722, doi:10.1063/1.5019779",
            "architecture_note": (
                "Project uses GaussianSmearing(0,5,16 bins) over ETKDGv3 "
                "distances. Original SchNet uses cosine-envelope RBF; results "
                "may differ from the published values."
            ),
        },
        "HIV": {
            "original_roc_auc": None,   # [CITE: confirm table/split in Flexi-JEGNN paper]
            "original_std":     None,
            "split":            "scaffold",
            "citation":         "Schutt et al. 2018 J. Chem. Phys. 148:241722, doi:10.1063/1.5019779",
        },
        "BBBP": {
            "original_roc_auc": None,   # [CITE: confirm table/split in Flexi-JEGNN paper]
            "original_std":     None,
            "split":            "scaffold",
            "citation":         "Schutt et al. 2018 J. Chem. Phys. 148:241722, doi:10.1063/1.5019779",
        },
    },
    # DimeNet: published on QM9 regression (Klicpera et al. 2020 ICLR).
    # MoleculeNet classification numbers exist in some works but the split
    # variant (DimeNet vs DimeNet++) and whether ETKDGv3 conformers were
    # used is not confirmed for the Flexi-JEGNN comparison.
    "dimenet": {
        "BACE": {
            "original_roc_auc": None,   # [CITE: confirm DimeNet vs DimeNet++, table/split]
            "original_std":     None,
            "split":            "scaffold",
            "citation":         "Klicpera et al. 2020 ICLR, arXiv:2003.03123",
            "architecture_note": (
                "Project DimeNetBlock uses RBF over distances only (no spherical "
                "Bessel / angle terms). Label results as 'DimeNet-simplified' "
                "when comparing to published numbers."
            ),
        },
        "HIV": {
            "original_roc_auc": None,   # [CITE: confirm DimeNet vs DimeNet++, table/split]
            "original_std":     None,
            "split":            "scaffold",
            "citation":         "Klicpera et al. 2020 ICLR, arXiv:2003.03123",
        },
        "BBBP": {
            "original_roc_auc": None,   # [CITE: confirm DimeNet vs DimeNet++, table/split]
            "original_std":     None,
            "split":            "scaffold",
            "citation":         "Klicpera et al. 2020 ICLR, arXiv:2003.03123",
        },
    },
    # Uni-Mol: published with pre-trained transformer weights (Zhou et al. 2023
    # ICLR). The UniMolLite class in classification.py has no pre-training,
    # so the architecture is similar in form but the weights are random-init.
    # Results must be labelled "Uni-Mol (no pre-train, simplified)"; comparison
    # to the published pre-trained numbers is not directly valid.
    "unimol": {
        "BACE": {
            "original_roc_auc": None,   # [CITE: Zhou et al. 2023 ICLR, Table 2 -- pre-trained variant]
            "original_std":     None,
            "split":            "scaffold",
            "citation":         "Zhou et al. 2023 ChemRxiv (preprint v4), doi:10.26434/chemrxiv-2022-jjm0j-v4, https://chemrxiv.org/doi/abs/10.26434/chemrxiv-2022-jjm0j-v4",
            "architecture_note": (
                "Published Uni-Mol uses 209M-molecule ZINC pre-training. "
                "UniMolLite here is randomly initialised; results are NOT "
                "comparable to the published pre-trained numbers."
            ),
        },
        "HIV": {
            "original_roc_auc": None,   # [CITE: Zhou et al. 2023 ICLR, Table 2 -- pre-trained variant]
            "original_std":     None,
            "split":            "scaffold",
            "citation":         "Zhou et al. 2023 ChemRxiv (preprint v4), doi:10.26434/chemrxiv-2022-jjm0j-v4, https://chemrxiv.org/doi/abs/10.26434/chemrxiv-2022-jjm0j-v4",
        },
        "BBBP": {
            "original_roc_auc": None,   # [CITE: Zhou et al. 2023 ICLR, Table 2 -- pre-trained variant]
            "original_std":     None,
            "split":            "scaffold",
            "citation":         "Zhou et al. 2023 ChemRxiv (preprint v4), doi:10.26434/chemrxiv-2022-jjm0j-v4, https://chemrxiv.org/doi/abs/10.26434/chemrxiv-2022-jjm0j-v4",
        },
    },
    # AttentiveFP: Xiong et al. 2020 J. Med. Chem. uses RANDOM splits and
    # 10-fold CV, not scaffold splits. Values are therefore not directly
    # comparable to our scaffold-split results; leave None.
    "attentivefp": {
        "BACE": {
            "original_roc_auc": None,   # [CITE: Xiong et al. 2020 Table 2 -- random split, not scaffold]
            "original_std":     None,
            "split":            "random (10-fold CV in original paper; we use scaffold)",
            "citation":         "Xiong et al. 2020 J. Med. Chem. 63(16), doi:10.1021/acs.jmedchem.9b00959",
            "architecture_note": (
                "Original paper uses random 10-fold CV. Our scaffold-split "
                "results are NOT directly comparable. Report separately."
            ),
        },
        "HIV": {
            "original_roc_auc": None,   # [CITE: not in Xiong et al. 2020; find a scaffold-split reference]
            "original_std":     None,
            "split":            "scaffold",
            "citation":         "Xiong et al. 2020 J. Med. Chem. 63(16), doi:10.1021/acs.jmedchem.9b00959",
        },
        "BBBP": {
            "original_roc_auc": None,   # [CITE: Xiong et al. 2020 Table 2 -- random split, not scaffold]
            "original_std":     None,
            "split":            "random (10-fold CV in original paper; we use scaffold)",
            "citation":         "Xiong et al. 2020 J. Med. Chem. 63(16), doi:10.1021/acs.jmedchem.9b00959",
        },
    },
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SeedResult:
    """Metrics from a single (dataset, model, seed) training run."""
    seed: int
    train_auc: float
    val_auc:   float
    test_auc:  float
    n_params:  int
    epochs_run: int
    stopped_early: bool
    wall_time_s: float


@dataclass
class BenchmarkResult:
    """Aggregated result for one (dataset, model) combination."""
    dataset:   str
    model_key: str                          # e.g. "dmpnn", "gin"
    level:     int                          # geometric fidelity level used

    # Replicated metrics (mean ± std across seeds)
    replicated_roc_auc_mean: float = float("nan")
    replicated_roc_auc_std:  float = float("nan")

    # Literature reference
    original_roc_auc: Optional[float] = None
    original_std:     Optional[float] = None
    citation:         str = ""
    delta:            Optional[float] = None   # replicated − original (None if original unknown)

    # Per-seed details
    seed_results: List[SeedResult] = field(default_factory=list)

    # Meta
    timestamp: str = ""
    git_sha:   str = ""

    def compute_aggregate(self) -> None:
        """Populate mean/std and delta from seed_results."""
        aucs = [r.test_auc for r in self.seed_results]
        if aucs:
            self.replicated_roc_auc_mean = float(np.mean(aucs))
            self.replicated_roc_auc_std  = float(np.std(aucs, ddof=1) if len(aucs) > 1 else 0.0)
        if self.original_roc_auc is not None:
            self.delta = round(self.replicated_roc_auc_mean - self.original_roc_auc, 4)

    def summary_line(self) -> str:
        lit = (f"{self.original_roc_auc:.3f}" if self.original_roc_auc is not None
               else "N/A (see citation)")
        delta_str = (f"{self.delta:+.3f}" if self.delta is not None else "—")
        return (
            f"  {self.model_key.upper():12s}  {self.dataset:6s}  L{self.level}  "
            f"replicated={self.replicated_roc_auc_mean:.3f}±{self.replicated_roc_auc_std:.3f}  "
            f"literature={lit}  delta={delta_str}"
        )


# ---------------------------------------------------------------------------
# Shared training driver
# Reuses run_training / evaluate / scaffold_split from classification.py.
# ---------------------------------------------------------------------------

def _run_one_seed(
    model_key: str,
    dataset_name: str,
    datasets_dir: Path,
    level: int,
    seed: int,
    epochs: int = _EPOCHS_DEFAULT,
    batch_size: int = 64,
    lr: float = 1e-4,
) -> SeedResult:
    """
    Train a single (model, dataset, seed) combination.

    Reuses:
      - load_dataset()    from classification.py  — CSV → DataFrame
      - scaffold_split()  from classification.py  — 80/10/10 scaffold split
      - featurize()       from classification.py  — graph tensors via _build_graph_tensors
      - run_training()    from classification.py  — Adam + CosineAnnealingLR + early stopping
      - evaluate()        from classification.py  — ROC-AUC + AUPRC + MCC + …
    """
    import torch
    from torch_geometric.loader import DataLoader as PyGLoader

    _log.info("  seed=%d  model=%s  dataset=%s  level=%d", seed, model_key, dataset_name, level)
    t0 = time.time()

    # 1. Load and split data
    df, smiles_col, label_col = load_dataset(dataset_name, datasets_dir)
    tr_df, va_df, te_df = scaffold_split(df, smiles_col, label_col, seed=seed)

    # 2. Featurize (graph construction at the requested geometric level)
    tr_g = featurize(tr_df, smiles_col, label_col, level=level, seed=seed)
    va_g = featurize(va_df, smiles_col, label_col, level=level, seed=seed)
    te_g = featurize(te_df, smiles_col, label_col, level=level, seed=seed)

    if not tr_g:
        raise RuntimeError(f"No valid graphs for {dataset_name} (level={level}, seed={seed})")

    tr_loader = PyGLoader(tr_g, batch_size=batch_size, shuffle=True,  drop_last=False)
    va_loader = PyGLoader(va_g, batch_size=batch_size, shuffle=False, drop_last=False)
    te_loader = PyGLoader(te_g, batch_size=batch_size, shuffle=False, drop_last=False)

    # 3. Build model (from the shared registry in classification.py)
    model = MODEL_REGISTRY[_model_registry_key(model_key)]().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 4. Class-imbalance weight
    tr_labels = [int(g.y.item()) for g in tr_g]
    pw = pos_weight(tr_labels).to(DEVICE)

    # 5. Train
    torch.manual_seed(seed)
    model, ep_run, stopped = run_training(
        model, tr_loader, va_loader,
        epochs=epochs, lr=lr, pw=pw,
    )

    # 6. Evaluate
    tr_m = evaluate(model, tr_loader)
    va_m = evaluate(model, va_loader)
    te_m = evaluate(model, te_loader)

    wall = round(time.time() - t0, 2)
    _log.info(
        "    → test_auc=%.4f  val_auc=%.4f  epochs=%d  time=%.1fs",
        te_m["auc"], va_m["auc"], ep_run, wall,
    )

    return SeedResult(
        seed=seed,
        train_auc=round(tr_m["auc"], 4),
        val_auc=round(va_m["auc"], 4),
        test_auc=round(te_m["auc"], 4),
        n_params=n_params,
        epochs_run=ep_run,
        stopped_early=stopped,
        wall_time_s=wall,
    )


def _model_registry_key(model_key: str) -> str:
    """Map CLI model key (lower-case) to MODEL_REGISTRY key."""
    mapping = {
        "dmpnn":       "D-MPNN",
        "gin":         "GIN",
        "schnet":      "SchNet",
        "dimenet":     "DimeNet",
        "unimol":      "Uni-Mol",
        "attentivefp": "AttentiveFP",
    }
    k = mapping.get(model_key.lower())
    if k is None:
        raise ValueError(f"Unknown model key '{model_key}'.  Valid: {list(mapping)}")
    return k


# ---------------------------------------------------------------------------
# D-MPNN runner  (working implementation)
# ---------------------------------------------------------------------------

def run_dmpnn(
    dataset_name: str,
    datasets_dir: Path,
    level: int = 0,
    seeds: Optional[List[int]] = None,
    epochs: int = _EPOCHS_DEFAULT,
    batch_size: int = 64,
    lr: float = 1e-4,
) -> BenchmarkResult:
    """
    Reproduce D-MPNN (Yang et al. 2019) using the project's own DMPNN class.

    Architecture (from experiments/classification.py, DMPNNConv + DMPNN):
      - Input projection: Linear(IN_DIM, 256)
      - 3 × DMPNNConv layers (additive aggregation, W_msg projects
        cat(x_j, bond_feat[:5]) → 256, W_upd projects 256 → 256)
      - LayerNorm residual after each conv
      - Global mean pooling
      - 2-layer MLP → 1 logit

    Atom features (IN_DIM = 18):
      - 12-d one-hot atom type (C,N,O,S,F,P,Cl,Br,Na,I,B,other)
      - degree / 6, formal_charge / 4, implicit_Hs / 4
      - is_aromatic, in_ring, has_chiral_tag

    Edge features at level 0 (2D topology only):
      - 5 bond-type one-hots  (single/double/triple/aromatic/in_ring)
      - 16 Gaussian RBF bins on hop-count × 1.4 Å proxy distances

    Training (same as main experiment):
      - Adam lr=1e-4, weight_decay=1e-5
      - CosineAnnealingLR to eta_min=1e-5
      - BCEWithLogitsLoss with pos_weight for class imbalance
      - Early stopping: patience=15, min_epochs=30

    Level note:
      D-MPNN in the paper uses only 2-D topology (level 0 or 1).
      Levels 3/4 require 3-D coordinates and will still run but are
      outside the original paper's scope.

    Parameters
    ----------
    dataset_name : str
        One of DATASETS.keys() (BACE, HIV, BBBP, ADMET).
    datasets_dir : Path
        Directory containing <dataset_name>.csv files.
    level : int
        Geometric fidelity level (0 = hop-count proxy; see classification.py).
    seeds : list of int or None
        Random seeds to sweep.  Defaults to [42].
    epochs : int
        Maximum training epochs.
    batch_size : int
        Mini-batch size for DataLoader.
    lr : float
        Initial learning rate.

    Returns
    -------
    BenchmarkResult
    """
    if not _HAS_EXPERIMENT_MODULE:
        raise RuntimeError(
            "Cannot import experiments/classification.py: "
            f"{_EXP_IMPORT_ERROR}\n"
            "Make sure you run from the project root and all dependencies "
            "are installed."
        )

    seeds = seeds or [42]
    result = BenchmarkResult(
        dataset=dataset_name,
        model_key="dmpnn",
        level=level,
        **{k: v for k, v in LITERATURE.get("dmpnn", {})
                                       .get(dataset_name, {}).items()
           if k in ("original_roc_auc", "original_std", "citation")},
    )

    for seed in seeds:
        sr = _run_one_seed(
            "dmpnn", dataset_name, datasets_dir, level, seed,
            epochs=epochs, batch_size=batch_size, lr=lr,
        )
        result.seed_results.append(sr)

    result.compute_aggregate()
    result.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return result


# ---------------------------------------------------------------------------
# GIN runner  (working implementation)
# ---------------------------------------------------------------------------

def run_gin(
    dataset_name: str,
    datasets_dir: Path,
    level: int = 0,
    seeds: Optional[List[int]] = None,
    epochs: int = _EPOCHS_DEFAULT,
    batch_size: int = 64,
    lr: float = 1e-4,
) -> BenchmarkResult:
    """
    Reproduce GIN (Xu et al. 2019 / Hu et al. 2020) using the project's
    own GINModel class.

    Architecture (from experiments/classification.py, GINModel):
      - Input projection: Linear(IN_DIM, 256)
      - 5 × GINConv layers (PyG GINConv with 2-layer MLP:
        Linear(256,512) → ReLU → Linear(512,256))
      - BatchNorm1d after each conv + ReLU
      - Global add pooling
      - 2-layer MLP → 1 logit

    The GINConv uses sum aggregation of neighbour features (Xu et al. 2019
    Eq. 4.1, epsilon=0 learnable variant).  The MLP inside each GINConv
    corresponds to f_Θ in the paper.

    Atom features and training procedure: same as run_dmpnn() above.

    Note on comparison to Hu et al. 2020:
      Hu et al. evaluate GIN *without* pre-training as a baseline.  Our
      implementation also has no pre-training, so the "no pre-train" row
      is the correct reference.  However, Hu et al. use a random 80/10/10
      scaffold split with a fixed seed, and it is not confirmed whether the
      Flexi-JEGNN paper uses the identical seed — hence original_roc_auc
      is left None in LITERATURE.

    Parameters
    ----------
    dataset_name : str
        One of DATASETS.keys() (BACE, HIV, BBBP, ADMET).
    datasets_dir : Path
        Directory containing <dataset_name>.csv files.
    level : int
        Geometric fidelity level.  GIN uses only bond topology (levels 0–2
        are meaningful; levels 3/4 are outside the original paper's scope).
    seeds : list of int or None
        Random seeds to sweep.
    epochs : int
        Maximum training epochs.
    batch_size : int
        Mini-batch size.
    lr : float
        Initial learning rate.

    Returns
    -------
    BenchmarkResult
    """
    if not _HAS_EXPERIMENT_MODULE:
        raise RuntimeError(
            "Cannot import experiments/classification.py: "
            f"{_EXP_IMPORT_ERROR}"
        )

    seeds = seeds or [42]
    result = BenchmarkResult(
        dataset=dataset_name,
        model_key="gin",
        level=level,
        **{k: v for k, v in LITERATURE.get("gin", {})
                                       .get(dataset_name, {}).items()
           if k in ("original_roc_auc", "original_std", "citation")},
    )

    for seed in seeds:
        sr = _run_one_seed(
            "gin", dataset_name, datasets_dir, level, seed,
            epochs=epochs, batch_size=batch_size, lr=lr,
        )
        result.seed_results.append(sr)

    result.compute_aggregate()
    result.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return result


# ---------------------------------------------------------------------------
# TODO stubs — implementation deferred
# ---------------------------------------------------------------------------

def run_schnet(
    dataset_name: str,
    datasets_dir: Path,
    level: int = 3,
    seeds: Optional[List[int]] = None,
    epochs: int = _EPOCHS_DEFAULT,
    batch_size: int = 64,
    lr: float = 1e-4,
) -> BenchmarkResult:
    """
    Reproduce SchNet (Schutt et al. 2017/2018) using the project's SchNet class.

    Architecture (from experiments/classification.py, SchNetLayer + SchNet):
      - Input embedding: Linear(IN_DIM, 256)
      - 6 x SchNetLayer:
          rbf_proj: Linear(16, 256)                   maps Gaussian RBF bins
          W: Linear(256,256) -> ShiftedSoftplus -> Linear(256,256)
          message: x_j * W(rbf_proj(rbf))
          upd:  Linear(256,256) -> ShiftedSoftplus -> Linear(256,256)
      - Residual connection (x = x + conv(x, ...))
      - Global add pooling
      - MLP: Linear(256,128) -> ShiftedSoftplus -> Dropout -> Linear(128,1)

    Geometry requirements:
      level=3 is required (ETKDGv3 3-D conformer distances fed as the 16-bin
      GaussianSmearing RBF in edge_attr[:,5:21]).  Levels 0-2 use proxy
      distances that are not physically meaningful for a distance-based model.

    Architectural approximation note:
      The published SchNet uses continuous-filter convolutions with cosine-
      envelope radial basis functions and a distance cutoff (typically 5 or
      10 Angstrom).  The project's SchNetLayer instead uses a fixed
      GaussianSmearing(0, 5, 16) over ETKDGv3 pairwise distances with a
      global cutoff tau=5 A applied during graph construction (level 3).
      Results should be labelled "SchNet (Gaussian RBF approx.)" when
      compared to published values in the revision.

    original_roc_auc is left None because the published SchNet numbers on
    MoleculeNet classification vary by dataset split and geometry source
    (ETKDGv3 vs DFT); the exact values used in the Flexi-JEGNN comparison
    table must be confirmed before filling in LITERATURE.

    Parameters
    ----------
    dataset_name : str
        One of DATASETS.keys() (BACE, HIV, BBBP, ADMET).
    datasets_dir : Path
        Directory containing <dataset_name>.csv files.
    level : int
        Geometric fidelity level.  Must be 3 for physically meaningful
        3-D distances.  Defaults to 3.
    seeds : list of int or None
        Random seeds to sweep.  Defaults to [42].
    epochs : int
        Maximum training epochs.
    batch_size : int
        Mini-batch size.
    lr : float
        Initial learning rate.

    Returns
    -------
    BenchmarkResult
    """
    if level not in (3, 4):
        import warnings
        warnings.warn(
            f"run_schnet called with level={level}. SchNet requires 3-D "
            "distances (level=3). Proxy distances at lower levels are not "
            "physically meaningful for this model.",
            UserWarning,
            stacklevel=2,
        )
    if not _HAS_EXPERIMENT_MODULE:
        raise RuntimeError(
            "Cannot import experiments/classification.py: "
            f"{_EXP_IMPORT_ERROR}"
        )

    seeds = seeds or [42]
    result = BenchmarkResult(
        dataset=dataset_name,
        model_key="schnet",
        level=level,
        **{k: v for k, v in LITERATURE.get("schnet", {})
                                       .get(dataset_name, {}).items()
           if k in ("original_roc_auc", "original_std", "citation")},
    )

    for seed in seeds:
        sr = _run_one_seed(
            "schnet", dataset_name, datasets_dir, level, seed,
            epochs=epochs, batch_size=batch_size, lr=lr,
        )
        result.seed_results.append(sr)

    result.compute_aggregate()
    result.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return result


def run_dimenet(
    dataset_name: str,
    datasets_dir: Path,
    level: int = 3,
    seeds: Optional[List[int]] = None,
    epochs: int = _EPOCHS_DEFAULT,
    batch_size: int = 64,
    lr: float = 1e-4,
) -> BenchmarkResult:
    """
    Reproduce DimeNet (Klicpera et al. 2020) using the project's DimeNet class.

    Architecture (from experiments/classification.py, DimeNetBlock + DimeNet):
      - Input embedding: Linear(IN_DIM, 256) -> SiLU
      - 4 x DimeNetBlock:
          rbf_proj: Linear(16, 256)                   maps Gaussian RBF bins
          msg_linear: Linear(512, 256)
          message: SiLU(msg_linear(cat(x_j * rbf_proj(rbf), x_i)))
          upd_linear: Linear(256, 256)
          update: SiLU(upd_linear(aggr))
      - LayerNorm residual after each block
      - Global mean pooling
      - MLP: Linear(256,256) -> SiLU -> Dropout -> Linear(256,1)

    Geometry requirements:
      level=3 is required (ETKDGv3 3-D conformer distances used as the
      16-bin GaussianSmearing RBF in edge_attr[:,5:21]).  Levels 0-2 use
      proxy distances that are not physically meaningful for DimeNet.

    Architectural approximation note:
      Published DimeNet uses directional message passing with spherical Bessel
      functions for distances AND Fourier series for bond angles.  The project's
      DimeNetBlock is a distance-only approximation (no angle terms, no Bessel
      basis).  Results must be labelled "DimeNet (distance-only approx.)" when
      compared to published values in the revision, and cannot be directly
      compared to DimeNet++ (Klicpera et al. 2020 NeurIPS) which adds a
      separate envelope function and scaled interaction blocks.

    original_roc_auc is left None because:
      (a) DimeNet vs DimeNet++ numbers differ;
      (b) the split variant and geometry source used in the Flexi-JEGNN paper
          must be confirmed before filling in LITERATURE.

    Parameters
    ----------
    dataset_name : str
        One of DATASETS.keys() (BACE, HIV, BBBP, ADMET).
    datasets_dir : Path
        Directory containing <dataset_name>.csv files.
    level : int
        Geometric fidelity level.  Must be 3 for 3-D distances.  Defaults to 3.
    seeds : list of int or None
        Random seeds to sweep.  Defaults to [42].
    epochs : int
        Maximum training epochs.
    batch_size : int
        Mini-batch size.
    lr : float
        Initial learning rate.

    Returns
    -------
    BenchmarkResult
    """
    if level not in (3, 4):
        import warnings
        warnings.warn(
            f"run_dimenet called with level={level}. DimeNet requires 3-D "
            "distances (level=3). Proxy distances at lower levels are not "
            "physically meaningful for this model.",
            UserWarning,
            stacklevel=2,
        )
    if not _HAS_EXPERIMENT_MODULE:
        raise RuntimeError(
            "Cannot import experiments/classification.py: "
            f"{_EXP_IMPORT_ERROR}"
        )

    seeds = seeds or [42]
    result = BenchmarkResult(
        dataset=dataset_name,
        model_key="dimenet",
        level=level,
        **{k: v for k, v in LITERATURE.get("dimenet", {})
                                       .get(dataset_name, {}).items()
           if k in ("original_roc_auc", "original_std", "citation")},
    )

    for seed in seeds:
        sr = _run_one_seed(
            "dimenet", dataset_name, datasets_dir, level, seed,
            epochs=epochs, batch_size=batch_size, lr=lr,
        )
        result.seed_results.append(sr)

    result.compute_aggregate()
    result.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return result


# ---------------------------------------------------------------------------
# AttentiveFP model wrapper + registration
# ---------------------------------------------------------------------------
# AttentiveFP is not in experiments/classification.py's MODEL_REGISTRY, so
# we define a thin wrapper here and inject it into the shared dict at import
# time if both torch_geometric and the experiment module are available.
#
# PyG's AttentiveFP (torch_geometric.nn.AttentiveFP) uses:
#   in_channels  = IN_DIM  (18 atom features)
#   hidden_channels = 256
#   out_channels = 1
#   edge_dim     = EDGE_DIM (21: 5 bond one-hots + 16 Gaussian RBF bins)
#   num_layers   = 2   (graph attention layers)
#   num_timesteps = 2  (GRU readout steps over the super-node)
#   dropout      = 0.3
#
# The wrapper converts the project's forward(data) convention to PyG's
# forward(x, edge_index, edge_attr, batch) convention.

_HAS_ATTENTIVEFP = False

try:
    import torch.nn as _nn
    from torch_geometric.nn import AttentiveFP as _PyGAttentiveFP

    class _AttentiveFPWrapper(_nn.Module):
        """
        Thin wrapper around torch_geometric.nn.AttentiveFP that follows the
        project's forward(data) -> scalar-per-graph convention.

        Registered into MODEL_REGISTRY["AttentiveFP"] at module import time
        so that _run_one_seed("attentivefp", ...) can build it the same way
        as every other model.
        """
        def __init__(self):
            super().__init__()
            # Import constants lazily so the class can be defined even when
            # classification.py is not importable.
            try:
                from experiments.classification import IN_DIM as _IN, EDGE_DIM as _ED
            except Exception:
                _IN, _ED = 18, 21      # project defaults if import fails
            self.net = _PyGAttentiveFP(
                in_channels=_IN,
                hidden_channels=256,
                out_channels=1,
                edge_dim=_ED,
                num_layers=2,
                num_timesteps=2,
                dropout=0.3,
            )

        def forward(self, data):
            # PyG AttentiveFP.forward returns (batch_size, out_channels).
            # Squeeze to (batch_size,) to match project convention.
            return self.net(
                data.x, data.edge_index, data.edge_attr, data.batch
            ).squeeze(-1)

    # Register into the shared MODEL_REGISTRY dict so _run_one_seed can find it.
    if _HAS_EXPERIMENT_MODULE:
        MODEL_REGISTRY["AttentiveFP"] = _AttentiveFPWrapper

    _HAS_ATTENTIVEFP = True
    _log.debug("torch_geometric.nn.AttentiveFP found — AttentiveFP runner enabled.")

except ImportError:
    _log.debug(
        "torch_geometric.nn.AttentiveFP not available (PyG < 2.0 or torch not "
        "installed). run_attentivefp will raise ImportError at call time."
    )


def run_unimol(
    dataset_name: str,
    datasets_dir: Path,
    level: int = 3,
    seeds: Optional[List[int]] = None,
    epochs: int = _EPOCHS_DEFAULT,
    batch_size: int = 64,
    lr: float = 1e-4,
) -> BenchmarkResult:
    """
    Reproduce Uni-Mol (Zhou et al. 2023) using the project's UniMolLite class.

    Architecture (from experiments/classification.py, UniMolLite):
      - Input projection: Linear(IN_DIM, 256)
      - Pair bias projection: Linear(16, num_heads=8)  maps Gaussian RBF bins
        to per-head attention biases (added to self-attention logits).
      - 6 x TransformerEncoderLayer (d_model=256, nhead=8, ffn=512,
        batch_first=True, norm_first=True -- pre-norm / "post-LN" style).
      - Masked mean pooling over non-padding atom tokens.
      - MLP: Linear(256,128) -> ReLU -> Dropout -> Linear(128,1)

    Geometry requirements:
      level=3 is required. The pair-bias RBF (edge_attr[:,5:21]) encodes
      pairwise distances; at lower levels these are proxy values that make
      the attention bias physically meaningless.

    Pre-training note -- IMPORTANT for the revision:
      The published Uni-Mol (Zhou et al. 2023) uses transformer weights
      pre-trained on 209 million molecules from ZINC (CCSD(T)-level
      conformers). UniMolLite is randomly initialised and uses ETKDGv3
      conformers. Results MUST be labelled "Uni-Mol (no pre-train,
      simplified)" in all tables and cannot be compared numerically to the
      published pre-trained numbers. The original_roc_auc field is therefore
      left None in LITERATURE.

    Parameters
    ----------
    dataset_name : str
        One of DATASETS.keys() (BACE, HIV, BBBP, ADMET).
    datasets_dir : Path
        Directory containing <dataset_name>.csv files.
    level : int
        Geometric fidelity level. Must be 3 for 3-D pair distances.
        Defaults to 3.
    seeds : list of int or None
        Random seeds to sweep. Defaults to [42].
    epochs : int
        Maximum training epochs.
    batch_size : int
        Mini-batch size.
    lr : float
        Initial learning rate.

    Returns
    -------
    BenchmarkResult
    """
    if level not in (3, 4):
        import warnings
        warnings.warn(
            f"run_unimol called with level={level}. Uni-Mol requires 3-D pair "
            "distances (level=3). The attention pair-bias will be meaningless "
            "with proxy distances at lower levels.",
            UserWarning,
            stacklevel=2,
        )
    if not _HAS_EXPERIMENT_MODULE:
        raise RuntimeError(
            "Cannot import experiments/classification.py: "
            f"{_EXP_IMPORT_ERROR}"
        )

    seeds = seeds or [42]
    result = BenchmarkResult(
        dataset=dataset_name,
        model_key="unimol",
        level=level,
        **{k: v for k, v in LITERATURE.get("unimol", {})
                                       .get(dataset_name, {}).items()
           if k in ("original_roc_auc", "original_std", "citation")},
    )

    for seed in seeds:
        sr = _run_one_seed(
            "unimol", dataset_name, datasets_dir, level, seed,
            epochs=epochs, batch_size=batch_size, lr=lr,
        )
        result.seed_results.append(sr)

    result.compute_aggregate()
    result.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return result


def run_attentivefp(
    dataset_name: str,
    datasets_dir: Path,
    level: int = 0,
    seeds: Optional[List[int]] = None,
    epochs: int = _EPOCHS_DEFAULT,
    batch_size: int = 64,
    lr: float = 1e-4,
) -> BenchmarkResult:
    """
    Reproduce AttentiveFP (Xiong et al. 2020) using PyG's AttentiveFP class.

    Architecture (_AttentiveFPWrapper, registered into MODEL_REGISTRY at import):
      torch_geometric.nn.AttentiveFP with:
        in_channels     = IN_DIM  (18 atom features)
        hidden_channels = 256
        out_channels    = 1
        edge_dim        = EDGE_DIM (21: 5 bond one-hots + 16 Gaussian RBF)
        num_layers      = 2   (graph-level attention layers)
        num_timesteps   = 2   (GRU super-node readout steps)
        dropout         = 0.3

    The wrapper's forward(data) passes data.x, data.edge_index,
    data.edge_attr, data.batch to PyG's AttentiveFP and squeezes the output
    to match the project's scalar-per-graph convention.

    Level note:
      AttentiveFP is a 2-D model (level 0 or 1 recommended). The edge_attr
      Gaussian RBF bins encode distances but the published AttentiveFP paper
      does not use them; level=0 (hop-count proxy) or level=1 (bond-length
      sum) are most faithful to the original. Defaults to 0.

    Comparison note -- IMPORTANT for the revision:
      Xiong et al. 2020 report results with RANDOM 10-fold CV splits, not
      scaffold splits. Our scaffold-split numbers CANNOT be directly compared
      to their Table 2. Present them in a separate column or footnote.
      The original_roc_auc field is therefore left None in LITERATURE.

    Availability:
      Requires torch_geometric >= 2.0. If PyG < 2.0 is installed,
      _HAS_ATTENTIVEFP is False and this function raises ImportError.

    Parameters
    ----------
    dataset_name : str
        One of DATASETS.keys() (BACE, HIV, BBBP, ADMET).
    datasets_dir : Path
        Directory containing <dataset_name>.csv files.
    level : int
        Geometric fidelity level. 0 or 1 recommended. Defaults to 0.
    seeds : list of int or None
        Random seeds to sweep. Defaults to [42].
    epochs : int
        Maximum training epochs.
    batch_size : int
        Mini-batch size.
    lr : float
        Initial learning rate.

    Returns
    -------
    BenchmarkResult

    Raises
    ------
    ImportError
        If torch_geometric.nn.AttentiveFP is not available (PyG < 2.0).
    RuntimeError
        If experiments/classification.py cannot be imported.
    """
    if not _HAS_ATTENTIVEFP:
        raise ImportError(
            "torch_geometric.nn.AttentiveFP is not available. "
            "Upgrade PyTorch Geometric to >= 2.0: "
            "pip install torch-geometric>=2.0"
        )
    if not _HAS_EXPERIMENT_MODULE:
        raise RuntimeError(
            "Cannot import experiments/classification.py: "
            f"{_EXP_IMPORT_ERROR}"
        )
    if "AttentiveFP" not in MODEL_REGISTRY:
        raise RuntimeError(
            "AttentiveFP was not registered into MODEL_REGISTRY at import "
            "time. This should not happen if _HAS_ATTENTIVEFP is True."
        )

    seeds = seeds or [42]
    result = BenchmarkResult(
        dataset=dataset_name,
        model_key="attentivefp",
        level=level,
        **{k: v for k, v in LITERATURE.get("attentivefp", {})
                                       .get(dataset_name, {}).items()
           if k in ("original_roc_auc", "original_std", "citation")},
    )

    for seed in seeds:
        sr = _run_one_seed(
            "attentivefp", dataset_name, datasets_dir, level, seed,
            epochs=epochs, batch_size=batch_size, lr=lr,
        )
        result.seed_results.append(sr)

    result.compute_aggregate()
    result.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return result


# ---------------------------------------------------------------------------
# Model dispatch table
# ---------------------------------------------------------------------------

_RUNNERS = {
    "dmpnn":       run_dmpnn,
    "gin":         run_gin,
    "schnet":      run_schnet,
    "dimenet":     run_dimenet,
    "unimol":      run_unimol,
    "attentivefp": run_attentivefp,
}

_WORKING_MODELS = {"dmpnn", "gin", "schnet", "dimenet", "unimol", "attentivefp"}
_TODO_MODELS    = set()   # all models implemented; no remaining stubs

_BENCHMARK_DATASETS = ["BACE", "HIV", "BBBP"]   # core MoleculeNet classification


# ---------------------------------------------------------------------------
# Top-level benchmark orchestrator
# ---------------------------------------------------------------------------

def run_benchmark(
    models:       List[str],
    datasets:     List[str],
    datasets_dir: Path,
    level:        int = 0,
    seeds:        Optional[List[int]] = None,
    epochs:       int = _EPOCHS_DEFAULT,
    batch_size:   int = 64,
    lr:           float = 1e-4,
    out_path:     Optional[Path] = None,
) -> List[BenchmarkResult]:
    """
    Run all requested (model, dataset) combinations and return results.

    Parameters
    ----------
    models : list of str
        Model keys, e.g. ["dmpnn", "gin"].  Pass ["all"] for all working models.
    datasets : list of str
        Dataset names.  Pass ["all"] for BACE, HIV, BBBP.
    datasets_dir : Path
        Directory containing the CSV files.
    level : int
        Geometric fidelity level (0 recommended for 2-D models).
    seeds : list of int
        Seeds to sweep.  Defaults to [42].
    epochs : int
        Maximum training epochs.
    batch_size : int
        DataLoader batch size.
    lr : float
        Adam initial learning rate.
    out_path : Path or None
        If given, write a JSON file with all BenchmarkResult dicts.

    Returns
    -------
    List[BenchmarkResult]
    """
    seeds = seeds or [42]

    # Expand "all" shortcuts
    if "all" in [m.lower() for m in models]:
        models = sorted(_WORKING_MODELS)
        _log.info(
            "Expanding --model all → %s  (stub models skipped)", models
        )
    if "all" in [d.upper() for d in datasets]:
        datasets = _BENCHMARK_DATASETS

    # Reject TODO models with a clear error
    for m in models:
        if m.lower() in _TODO_MODELS:
            raise ValueError(
                f"Model '{m}' is not yet implemented (stub).  "
                f"Working models: {sorted(_WORKING_MODELS)}"
            )

    results: List[BenchmarkResult] = []
    for dataset in datasets:
        for model_key in models:
            _log.info(
                "=== %s  /  %s  /  level=%d  /  seeds=%s ===",
                model_key.upper(), dataset, level, seeds,
            )
            runner = _RUNNERS[model_key.lower()]
            try:
                res = runner(
                    dataset_name=dataset,
                    datasets_dir=datasets_dir,
                    level=level,
                    seeds=seeds,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                )
                results.append(res)
                _log.info(res.summary_line())
            except Exception as exc:
                _log.error(
                    "FAILED %s / %s: %s", model_key, dataset, exc,
                    exc_info=True,
                )

    # Summary table
    _log.info("\n%s", "=" * 78)
    _log.info("BENCHMARK SUMMARY")
    _log.info("=" * 78)
    for r in results:
        _log.info(r.summary_line())
    _log.info("=" * 78)

    # Persist results
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(r) for r in results]
        out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        _log.info("Results written → %s", out_path)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m revision.benchmarks.reproduce_baselines",
        description=(
            "Reproduce D-MPNN and GIN baseline results on the Flexi-JEGNN "
            "classification benchmark datasets.\n\n"
            "Working models: dmpnn, gin\n"
            "TODO stubs:     schnet, dimenet, unimol, attentivefp\n\n"
            "Example — 3 seeds of D-MPNN on BACE:\n"
            "  python -m revision.benchmarks.reproduce_baselines \\\n"
            "      --model dmpnn --dataset BACE \\\n"
            "      --datasets-dir datasets/ \\\n"
            "      --n-seeds 3 --seed0 42 --level 0 \\\n"
            "      --out revision/benchmarks/results/bace_dmpnn.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--model",
        default="all",
        help=(
            "Model to benchmark. One of: dmpnn, gin, all. "
            "'all' runs dmpnn and gin only (stubs are skipped). "
            "Default: all."
        ),
    )
    p.add_argument(
        "--dataset",
        default="all",
        help=(
            "Dataset to run on. One of: BACE, HIV, BBBP, all. "
            "'all' runs BACE, HIV, BBBP. Default: all."
        ),
    )
    p.add_argument(
        "--datasets-dir",
        default="datasets",
        help="Directory containing the dataset CSV files. Default: datasets/",
    )
    p.add_argument(
        "--level",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4],
        help=(
            "Geometric fidelity level for graph construction (0=hop-count "
            "proxy, 3=ETKDGv3 3-D, …). D-MPNN and GIN are 2-D models; "
            "level 0 or 1 is recommended. Default: 0."
        ),
    )
    p.add_argument(
        "--n-seeds",
        type=int,
        default=3,
        help="Number of random seeds to sweep. Default: 3.",
    )
    p.add_argument(
        "--seed0",
        type=int,
        default=42,
        help="Starting seed. Seeds used are seed0, seed0+1, …, seed0+n_seeds-1. Default: 42.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=_EPOCHS_DEFAULT,
        help=f"Maximum training epochs per seed. Default: {_EPOCHS_DEFAULT} (same as main experiment).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="DataLoader batch size. Default: 64.",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Adam initial learning rate. Default: 1e-4.",
    )
    p.add_argument(
        "--out",
        default=None,
        help=(
            "Output JSON file path. If omitted, results are only printed. "
            "Example: revision/benchmarks/results/baselines.json"
        ),
    )
    p.add_argument(
        "--list-models",
        action="store_true",
        help="Print the list of models and their implementation status, then exit.",
    )
    return p


def _cli_main(argv=None) -> None:
    # Reconfigure stdout/stderr to UTF-8 so summary lines print correctly on
    # Windows consoles that default to cp1252.
    import io
    if hasattr(sys.stdout, "reconfigure"):          # Python 3.7+
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    else:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    args = _build_parser().parse_args(argv)

    if args.list_models:
        print("\nWorking implementations:")
        for m in sorted(_WORKING_MODELS):
            print(f"  {m:14s}  [OK]  implemented")
        print("\nTODO stubs (not yet runnable):")
        for m in sorted(_TODO_MODELS):
            print(f"  {m:14s}  [TODO] stub -- see run_{m}() docstring")
        print()
        return

    seeds = list(range(args.seed0, args.seed0 + args.n_seeds))
    models  = [args.model]
    datasets = [args.dataset]

    run_benchmark(
        models=models,
        datasets=datasets,
        datasets_dir=Path(args.datasets_dir),
        level=args.level,
        seeds=seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        out_path=Path(args.out) if args.out else None,
    )


if __name__ == "__main__":
    _cli_main()
