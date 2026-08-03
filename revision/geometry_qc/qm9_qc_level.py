"""
revision/geometry_qc/qm9_qc_level.py
=====================================
Additive geometry level ``"3_qc"`` for the QM9 regression pipeline.

This module wires the DFT B3LYP/6-31G(2df,p) geometry from the official
QM9 .xyz bundle into the graph-building pipeline used by
``experiments/qm9.py``, **without modifying that file**.

How it fits in
--------------
``experiments/qm9.py::_build_graph_tensors(smiles, level, ...)`` defines
five geometry levels (0–4).  Level 3 uses RDKit ETKDGv3 + MMFF to generate
a single RDKit conformer.  Level ``"3_qc"`` replaces that conformer with the
original quantum-chemistry geometry (DFT-optimised), keeping everything else
(atom features, bond features, Gaussian smearing, edge filtering by tau)
identical.

Usage — standalone featurise call
----------------------------------
::

    from revision.geometry_qc.qm9_qc_level import build_graph_tensors_qc

    loader = make_loader('/path/to/dsgdb9nsd/')   # or .tar.bz2
    tensors = build_graph_tensors_qc(smiles, loader)
    # tensors is dict with same keys as _build_graph_tensors returns,
    # or None if the molecule is missing from the bundle / fails.

Usage — drop-in featurise replacement
--------------------------------------
::

    from revision.geometry_qc.qm9_qc_level import featurize_qc

    graphs = featurize_qc(df, smiles_col='smiles', label_col='homo',
                          loader=loader)
    # graphs is a list of torch_geometric.data.Data, same as featurize()

Out of scope (teammate handles separately)
------------------------------------------
* Protonation pipeline          -> revision/protonation/
* Conformer QC / validation     -> revision/conformer_qc/
* Baseline benchmarks           -> revision/benchmarks/reproduce_baselines.py
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np

# ---------------------------------------------------------------------------
# Lazy imports — keep this module importable even without torch/rdkit so that
# unit tests can import and inspect it without a full ML environment.
# ---------------------------------------------------------------------------
try:
    from rdkit import Chem
    from rdkit.Chem import rdmolops
    from rdkit.Chem.rdchem import BondType as BT
    _RDKIT_OK = True
except ImportError:  # pragma: no cover
    _RDKIT_OK = False

try:
    import torch
    from torch_geometric.data import Data
    _TORCH_OK = True
except ImportError:  # pragma: no cover
    _TORCH_OK = False

# Local loader (sibling module)
from revision.data.qm9_original_geometry_loader import (
    QM9QCGeometryLoader,
    make_qc_dist_fn,
)

# ---------------------------------------------------------------------------
# Re-use atom / bond feature helpers from the original experiment file
# (imported at runtime to avoid coupling at module load time).
# ---------------------------------------------------------------------------

def _get_atom_feats_fn():
    """Return the _atom_feats function from experiments/qm9.py at call time."""
    try:
        from experiments.qm9 import _atom_feats  # type: ignore[import]
        return _atom_feats
    except ImportError as exc:
        raise ImportError(
            "Cannot import _atom_feats from experiments/qm9.py. "
            "Make sure you run from the repo root."
        ) from exc


def _get_distance_expansion():
    """Return the GaussianSmearing instance from experiments/qm9.py."""
    try:
        from experiments.qm9 import distance_expansion  # type: ignore[import]
        return distance_expansion
    except ImportError as exc:
        raise ImportError(
            "Cannot import distance_expansion from experiments/qm9.py."
        ) from exc


# ---------------------------------------------------------------------------
# Loader factory
# ---------------------------------------------------------------------------

def make_loader(bundle_path: Union[str, Path], verbose: bool = True) -> QM9QCGeometryLoader:
    """
    Convenience wrapper: construct a ``QM9QCGeometryLoader`` from *bundle_path*.

    Parameters
    ----------
    bundle_path:
        Path to either the extracted directory of .xyz files, or the
        compressed ``dsgdb9nsd.xyz.tar.bz2`` archive.
    verbose:
        Print progress during the initial scan (default True).

    Returns
    -------
    QM9QCGeometryLoader
    """
    return QM9QCGeometryLoader(bundle_path=bundle_path, verbose=verbose)


# ---------------------------------------------------------------------------
# Core graph builder — level "3_qc"
# ---------------------------------------------------------------------------

def build_graph_tensors_qc(
    smiles: str,
    loader: QM9QCGeometryLoader,
    tau: float = 5.0,
    missing_ok: bool = False,
) -> Optional[dict]:
    """
    Build the same graph-tensor dict as ``_build_graph_tensors(smiles, 3)``
    in ``experiments/qm9.py``, but using DFT QC atom positions instead of
    an RDKit-generated conformer.

    The returned dict has identical keys to ``_build_graph_tensors``::

        {
            'x'          : list[list[float]]   # atom features  (n, IN_DIM)
            'edge_src'   : list[int]
            'edge_dst'   : list[int]
            'bond_feats' : list[list[float]]   # (E, 5)
            'distances'  : list[float]          # Euclidean Å
            'n'          : int                  # n heavy atoms
        }

    Parameters
    ----------
    smiles:
        SMILES string (from ``QM9.csv``).
    loader:
        A ``QM9QCGeometryLoader`` instance.
    tau:
        Distance cutoff in Angstroms (default 5.0, matching level-3).
        Edges with distance >= tau are omitted.
    missing_ok:
        If False (default), returns ``None`` when the molecule is absent
        from the bundle (mirrors the behaviour of a failed ``EmbedMolecule``
        at level 3).  If True, returns ``None`` silently without warning.

    Returns
    -------
    dict or None
    """
    if not _RDKIT_OK:
        raise ImportError("RDKit is required. Install with: conda install -c conda-forge rdkit")

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None or mol.GetNumAtoms() < 2:
        return None
    n = mol.GetNumAtoms()

    # --- atom features (identical to level 3) ---
    _atom_feats = _get_atom_feats_fn()
    x = [_atom_feats(a) for a in mol.GetAtoms()]

    # --- QC distance closure ---
    dist_fn = make_qc_dist_fn(loader, smiles, heavy_only=True)

    if not getattr(dist_fn, '_qc_found', True):
        # Molecule not in bundle
        if not missing_ok:
            warnings.warn(
                f"[qm9_qc_level] Molecule not found in QC bundle: {smiles!r}. "
                "Returning None. Set missing_ok=True to suppress.",
                stacklevel=2,
            )
        return None

    # Safety: check that QC atom count matches RDKit heavy-atom count
    qc_n = getattr(dist_fn, '_n_atoms', None)
    if qc_n is not None and qc_n != n:
        warnings.warn(
            f"[qm9_qc_level] Heavy-atom count mismatch for {smiles!r}: "
            f"RDKit={n}, QC bundle={qc_n}. Returning None.",
            stacklevel=2,
        )
        return None

    # --- edge building (identical logic to _build_graph_tensors level 3) ---
    edge_src, edge_dst, bond_feats_list, distances = [], [], [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            try:
                d = dist_fn(i, j)
            except Exception:
                d = 999.0
            if d >= tau:
                continue
            edge_src.append(i)
            edge_dst.append(j)
            distances.append(d)
            bond = mol.GetBondBetweenAtoms(i, j)
            if bond is not None:
                bt = bond.GetBondType()
                bond_feats_list.append([
                    float(bt == BT.SINGLE), float(bt == BT.DOUBLE),
                    float(bt == BT.TRIPLE), float(bt == BT.AROMATIC),
                    float(bond.IsInRing()),
                ])
            else:
                bond_feats_list.append([0.0, 0.0, 0.0, 0.0, 0.0])

    if not edge_src:
        return None

    return {
        'x': x,
        'edge_src': edge_src,
        'edge_dst': edge_dst,
        'bond_feats': bond_feats_list,
        'distances': distances,
        'n': n,
    }


# ---------------------------------------------------------------------------
# _tensors_to_data — identical to experiments/qm9.py version, kept local so
# this module can be used stand-alone without importing the experiment file.
# ---------------------------------------------------------------------------

def _tensors_to_data(t: dict, label: float):
    """Convert a graph-tensor dict to a PyG Data object."""
    if not _TORCH_OK:
        raise ImportError("PyTorch + torch_geometric required for _tensors_to_data.")
    distance_expansion = _get_distance_expansion()
    x = torch.tensor(t['x'], dtype=torch.float)
    ei = torch.tensor([t['edge_src'], t['edge_dst']], dtype=torch.long)
    dists = torch.tensor(t['distances'], dtype=torch.float)
    gauss = distance_expansion(dists)
    bond_t = torch.tensor(t['bond_feats'], dtype=torch.float)
    ea = torch.cat([bond_t, gauss], dim=-1)
    g = Data(x=x, edge_index=ei, edge_attr=ea, num_nodes=t['n'])
    g.y = torch.tensor([float(label)], dtype=torch.float)
    return g


# ---------------------------------------------------------------------------
# Drop-in featurise replacement for level "3_qc"
# ---------------------------------------------------------------------------

def featurize_qc(
    df,
    smiles_col: str,
    label_col: str,
    loader: QM9QCGeometryLoader,
    tau: float = 5.0,
    missing_ok: bool = True,
) -> list:
    """
    Featurise a DataFrame using QC geometry; returns a list of PyG Data objects.

    This is a drop-in replacement for ``experiments/qm9.py::featurize()``
    when ``level="3_qc"``.  Molecules absent from the QC bundle are silently
    skipped (``missing_ok=True``).

    Parameters
    ----------
    df:
        pandas DataFrame with at least *smiles_col* and *label_col*.
    smiles_col, label_col:
        Column names.
    loader:
        A ``QM9QCGeometryLoader`` instance.
    tau:
        Distance cutoff in Angstroms (default 5.0).
    missing_ok:
        Silently skip molecules not found in the bundle (default True).

    Returns
    -------
    list of ``torch_geometric.data.Data``
    """
    graphs = []
    for smi, label in zip(df[smiles_col].tolist(), df[label_col].tolist()):
        t = build_graph_tensors_qc(str(smi), loader, tau=tau, missing_ok=missing_ok)
        if t is None:
            continue
        graphs.append(_tensors_to_data(t, label))
    return graphs


# ---------------------------------------------------------------------------
# Level ID constant — used to tag results rows (mirrors LEVELS in qm9.py)
# ---------------------------------------------------------------------------

LEVEL_ID: str = "3_qc"
"""
String level identifier for this geometry source.
Use this when constructing experiment keys, e.g.::

    key = f"QM9_{model_name}_{LEVEL_ID}_{seed}"
"""
