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
            "citation":         "Hu et al. 2020 ICLR, 'Strategies for Pre-training GNNs'",
        },
        "HIV": {
            "original_roc_auc": None,          # [CITE: Hu et al. 2020 ICLR, Table 1, GIN no pre-train]
            "original_std":     None,
            "split":            "scaffold",
            "citation":         "Hu et al. 2020 ICLR, 'Strategies for Pre-training GNNs'",
        },
        "BBBP": {
            "original_roc_auc": None,          # [CITE: Hu et al. 2020 ICLR, Table 1, GIN no pre-train]
            "original_std":     None,
            "split":            "scaffold",
            "citation":         "Hu et al. 2020 ICLR, 'Strategies for Pre-training GNNs'",
        },
    },
    # TODO: fill in literature values for SchNet, DimeNet, Uni-Mol, AttentiveFP
    # once the exact reference (paper, table, split) is confirmed.
    "schnet":      {},
    "dimenet":     {},
    "unimol":      {},
    "attentivefp": {},
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
    **kwargs,
) -> BenchmarkResult:
    """
    Reproduce SchNet (Schütt et al. 2017 / 2018) baseline.

    TODO: implement.

    What is needed before this can be filled in:
      1. Confirm which SchNet paper and which dataset split the Flexi-JEGNN
         paper's Table compares against (Schütt et al. 2017 JCP, Schütt et
         al. 2018 J. Chem. Theory Comput., or a MoleculeNet leaderboard
         entry).
      2. SchNet requires 3-D coordinates → level must be 3 or 4.
      3. The project's SchNetLayer in classification.py uses GaussianSmearing
         over approximate distances.  Verify that this matches the published
         architecture (continuous filter convolution with cosine envelopes in
         the original vs. Gaussian RBF here).
      4. The original SchNet uses a separate distance cutoff (5 Å, 10 Å) and
         sinusoidal RBF — confirm the Gaussian approximation is acceptable or
         add a flag to switch.

    Architecture available in MODEL_REGISTRY:
      SchNet(node_dim=IN_DIM, hidden_dim=256, num_layers=6, dropout=0.3)

    Citation placeholder:
      Schütt et al. 2017 J. Chem. Phys. 148 241722
      doi: 10.1063/1.5019779
    """
    # TODO: remove NotImplementedError and call _run_one_seed once the above
    # questions are resolved.
    raise NotImplementedError(
        "run_schnet is a stub — see docstring for what is needed before "
        "implementing.  Run with --model dmpnn or --model gin instead."
    )


def run_dimenet(
    dataset_name: str,
    datasets_dir: Path,
    level: int = 3,
    seeds: Optional[List[int]] = None,
    **kwargs,
) -> BenchmarkResult:
    """
    Reproduce DimeNet (Klicpera et al. 2020) baseline.

    TODO: implement.

    What is needed before this can be filled in:
      1. The full DimeNet architecture uses directional message passing with
         spherical Bessel functions and Fourier series for angle embeddings.
         The project's DimeNetBlock in classification.py is a simplified
         approximation (RBF over distances only, no angle terms).  Decide
         whether the simplified version or the full DimeNet is the intended
         comparison.
      2. Confirm the published MoleculeNet ROC-AUC values for DimeNet++
         (Klicpera et al. 2020, NeurIPS) vs DimeNet (ICLR 2020) — these
         differ and the correct row must be identified.
      3. level must be 3 (3-D ETKDGv3 conformer) for a meaningful comparison.

    Architecture available in MODEL_REGISTRY:
      DimeNet(node_dim=IN_DIM, hidden_dim=256, num_layers=4, dropout=0.3)

    Citation placeholder:
      Klicpera et al. 2020 ICLR "Directional Message Passing for Molecular Graphs"
      arXiv: 2003.03123
    """
    raise NotImplementedError(
        "run_dimenet is a stub — see docstring for what is needed before "
        "implementing."
    )


def run_unimol(
    dataset_name: str,
    datasets_dir: Path,
    level: int = 3,
    seeds: Optional[List[int]] = None,
    **kwargs,
) -> BenchmarkResult:
    """
    Reproduce Uni-Mol (Zhou et al. 2023) baseline.

    TODO: implement.

    What is needed before this can be filled in:
      1. The published Uni-Mol uses pre-trained transformer weights (209M
         parameter model trained on 209 M molecules from ZINC).  Reproducing
         from scratch is not feasible.  Options:
           a. Download the pre-trained weights from the official repo
              (https://github.com/dptech-corp/Uni-Mol) and fine-tune.
           b. Use the UniMolLite class in classification.py (no pre-training)
              and clearly label results as "Uni-Mol (no pre-train, simplified)".
      2. Confirm whether the Flexi-JEGNN paper compares against the pre-trained
         or scratch variant, and whether ETKDGv3 conformers are acceptable
         inputs (Uni-Mol was trained on CCSD(T)-level conformers).
      3. level must be 3 (3-D coordinates required).

    Architecture available in MODEL_REGISTRY (simplified, no pre-training):
      UniMolLite(node_dim=IN_DIM, hidden_dim=256, num_heads=8, num_layers=6)

    Citation placeholder:
      Zhou et al. 2023 ICLR "Uni-Mol: A Universal 3D Molecular Representation
      Learning Framework"
      doi: 10.26434/chemrxiv-2022-jjm0j
    """
    raise NotImplementedError(
        "run_unimol is a stub — see docstring for what is needed before "
        "implementing."
    )


def run_attentivefp(
    dataset_name: str,
    datasets_dir: Path,
    level: int = 0,
    seeds: Optional[List[int]] = None,
    **kwargs,
) -> BenchmarkResult:
    """
    Reproduce AttentiveFP (Xiong et al. 2020) baseline.

    TODO: implement.

    What is needed before this can be filled in:
      1. AttentiveFP is NOT currently in MODEL_REGISTRY in classification.py.
         The class must be added there, or implemented here, before this
         runner can be used.
      2. The published architecture uses graph attention with virtual nodes and
         a separate super-node readout.  Confirm the exact variant (with or
         without virtual atom).
      3. Confirm the published MoleculeNet ROC-AUC values for AttentiveFP and
         the split type used (Xiong et al. 2020 use random splits in the
         original paper; later work sometimes reports scaffold splits).

    Citation placeholder:
      Xiong et al. 2020 J. Med. Chem. 63(16) 8749-8760
      doi: 10.1021/acs.jmedchem.9b00959
    """
    raise NotImplementedError(
        "run_attentivefp is a stub — AttentiveFP is not in MODEL_REGISTRY yet. "
        "See docstring for what is needed before implementing."
    )


# ---------------------------------------------------------------------------
# Model dispatch table
# ---------------------------------------------------------------------------

_RUNNERS = {
    "dmpnn":       run_dmpnn,
    "gin":         run_gin,
    "schnet":      run_schnet,       # TODO stub
    "dimenet":     run_dimenet,      # TODO stub
    "unimol":      run_unimol,       # TODO stub
    "attentivefp": run_attentivefp,  # TODO stub
}

_WORKING_MODELS = {"dmpnn", "gin"}
_TODO_MODELS    = {"schnet", "dimenet", "unimol", "attentivefp"}

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
