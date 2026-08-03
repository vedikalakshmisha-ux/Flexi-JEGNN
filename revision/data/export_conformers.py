"""
revision/data/export_conformers.py
===================================
Export 3-D conformers for every molecule in each dataset to per-dataset,
per-method SDF files for hosting on GitHub (via Git LFS).

Methods exported
----------------
RDKit-based (from revision/geometry_qc/generate_variants.py):
  * ETKDG      — ETKDGv1
  * ETKDGv2    — ETKDGv2
  * ETKDGv3    — ETKDGv3  (matches experiments/qm9.py level 3)
  * random_dg  — random distance-geometry (negative control)

OpenBabel (subprocess):
  * obabel     — ``obabel --gen3d``  (skipped if not on PATH)

QC geometry (revision/data/qm9_original_geometry_loader.py):
  * 3_qc       — DFT B3LYP/6-31G(2df,p) from the QM9 .xyz bundle
                 **only available for QM9** (other datasets have no XYZ bundle)

Output layout
-------------
``<out_dir>/<DATASET>_<METHOD>.sdf``

Examples::

    conformers/BACE_ETKDGv3.sdf
    conformers/HIV_ETKDGv3.sdf
    conformers/QM9_ETKDGv3.sdf
    conformers/QM9_3_qc.sdf
    conformers/BACE_ETKDGv2.sdf
    ...

SDF record format (per molecule)
---------------------------------
Each SDF record contains:

  * **Mol block**: V2000 with 3D heavy-atom coordinates.
    - If RDKit is installed: proper bond block included.
    - If RDKit is NOT installed: bond block is omitted (n_bonds=0); the record
      is still valid SDF V2000 and round-trips correctly in RDKit.
  * **SD properties**:
    - ``<smiles>``        original SMILES string
    - ``<method>``        geometry method name
    - ``<dataset>``       dataset name (BACE / HIV / BBBP / QM9 / ADMET)
    - ``<n_heavy_atoms>`` heavy-atom count
    - ``<mol_name>``      molecule name if present in the source CSV (else SMILES)

Dataset registry
----------------
Hard-coded SMILES column map (matches experiments/classification.py):

    BACE  → smiles_col='mol'
    HIV   → smiles_col='smiles'
    BBBP  → smiles_col='smiles'
    QM9   → smiles_col='smiles'
    ADMET → smiles_col='smiles'

Usage
-----
    # Export all available methods for all datasets
    python revision/data/export_conformers.py

    # QM9 with QC geometry (requires the .xyz bundle)
    python revision/data/export_conformers.py \\
        --datasets QM9 \\
        --methods ETKDGv3 3_qc \\
        --qc-bundle /path/to/dsgdb9nsd.xyz.tar.bz2

    # Quick smoke-test: first 100 molecules, ETKDGv3 only
    python revision/data/export_conformers.py \\
        --n-mols 100 --methods ETKDGv3 \\
        --out-dir conformers/

Out of scope (teammate handles separately)
------------------------------------------
* Protonation pipeline          -> revision/protonation/
* Conformer QC / validation     -> revision/conformer_qc/
* Baseline benchmarks           -> revision/benchmarks/reproduce_baselines.py
"""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Geometry backends (lazy imports — importable without RDKit)
# ---------------------------------------------------------------------------
from revision.geometry_qc.generate_variants import (
    ALL_METHODS,
    METHOD_ETKDG,
    METHOD_ETKDGv2,
    METHOD_ETKDGv3,
    METHOD_OBABEL,
    METHOD_RANDOM,
    RDKIT_METHODS,
    ConformerResult,
    generate_conformer,
    obabel_available,
    rdkit_available,
)

# QC geometry loader (QM9-only)
METHOD_QC = "3_qc"

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASET_REGISTRY: Dict[str, Dict] = {
    "BACE": {
        "smiles_col": "mol",
        "name_col":   None,
        "qc_eligible": False,
    },
    "HIV": {
        "smiles_col": "smiles",
        "name_col":   None,
        "qc_eligible": False,
    },
    "BBBP": {
        "smiles_col": "smiles",
        "name_col":   "name",
        "qc_eligible": False,
    },
    "QM9": {
        "smiles_col": "smiles",
        "name_col":   None,
        "qc_eligible": True,   # only dataset with a QC XYZ bundle
    },
    "ADMET": {
        "smiles_col": "smiles",
        "name_col":   "mol_id",
        "qc_eligible": False,
    },
}

EXPORTABLE_METHODS: List[str] = RDKIT_METHODS + [METHOD_OBABEL, METHOD_QC]

# ---------------------------------------------------------------------------
# Minimal SDF V2000 writer (no-bond fallback, no RDKit required)
# ---------------------------------------------------------------------------

_SDF_ATOM_FMT = (
    "{x:10.4f}{y:10.4f}{z:10.4f} {sym:<3s} 0  0  0  0  0  0  0  0  0  0  0  0"
)

# Atomic symbols present in QM9 / MoleculeNet datasets
_RDKIT_SYMBOL_MAP: Dict[str, str] = {}  # populated lazily when RDKit available


def _atom_symbol_from_smiles_index(smiles: str, idx: int) -> str:
    """Return the element symbol for heavy atom *idx* (0-based) via RDKit."""
    if not rdkit_available():
        return "C"   # fallback — won't be called when positions are hand-rolled
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or idx >= mol.GetNumAtoms():
        return "C"
    return mol.GetAtomWithIdx(idx).GetSymbol()


def _write_sdf_record_minimal(
    fh,
    mol_name: str,
    positions: np.ndarray,
    symbols: List[str],
    smiles: str,
    method: str,
    dataset: str,
) -> None:
    """
    Write one SDF V2000 record to *fh* without bond information.

    This is the no-RDKit fallback.  The record is valid SDF V2000:
    RDKit ``SDMolSupplier`` can read it and will reconstruct connectivity
    from coordinates using ``Chem.RWMol.UpdatePropertyCache`` + DG matching,
    or the caller can pass ``removeHs=False, sanitize=False``.

    For round-trip testing, prefer ``_write_sdf_record_rdkit`` when RDKit is
    available.
    """
    n = len(symbols)
    # Header block (3 lines)
    fh.write(f"{mol_name[:80]}\n")
    fh.write("     Flexi-JEGNN revision  3D\n")
    fh.write(f"  method={method} dataset={dataset}\n")
    # Counts line: n_atoms n_bonds ...
    fh.write(f"{n:3d}  0  0  0  0  0  0  0  0  0999 V2000\n")
    # Atom block
    for i, (sym, pos) in enumerate(zip(symbols, positions)):
        fh.write(
            _SDF_ATOM_FMT.format(x=float(pos[0]), y=float(pos[1]),
                                  z=float(pos[2]), sym=sym) + "\n"
        )
    fh.write("M  END\n")
    # SD properties
    fh.write(f">  <smiles>\n{smiles}\n\n")
    fh.write(f">  <method>\n{method}\n\n")
    fh.write(f">  <dataset>\n{dataset}\n\n")
    fh.write(f">  <n_heavy_atoms>\n{n}\n\n")
    fh.write(f">  <mol_name>\n{mol_name}\n\n")
    fh.write("$$$$\n")


def _write_sdf_record_rdkit(
    writer,           # rdkit.Chem.SDWriter instance
    mol_name: str,
    positions: np.ndarray,
    smiles: str,
    method: str,
    dataset: str,
) -> bool:
    """
    Write one SDF record via RDKit's ``SDWriter``.

    Uses the molecule's full topology (bond block) from the SMILES.
    Sets the conformer from *positions* (heavy atoms).

    Returns True on success, False if the mol could not be reconstructed.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, Conformer
    from rdkit.Chem.rdchem import Conformer as RDConformer

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    n = mol.GetNumAtoms()
    if positions.shape[0] != n:
        return False

    # Attach the 3D conformer
    conf = RDConformer(n)
    for i in range(n):
        conf.SetAtomPosition(i, (
            float(positions[i, 0]),
            float(positions[i, 1]),
            float(positions[i, 2]),
        ))
    mol.AddConformer(conf, assignId=True)
    mol.SetProp("_Name", mol_name[:80])
    mol.SetProp("smiles", smiles)
    mol.SetProp("method", method)
    mol.SetProp("dataset", dataset)
    mol.SetProp("n_heavy_atoms", str(n))
    mol.SetProp("mol_name", mol_name)

    try:
        writer.write(mol)
        return True
    except Exception:
        return False


def _get_heavy_symbols_rdkit(smiles: str) -> Optional[List[str]]:
    """Return element symbols for heavy atoms via RDKit."""
    if not rdkit_available():
        return None
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return [atom.GetSymbol() for atom in mol.GetAtoms()]


def _guess_symbols_from_positions(n: int) -> List[str]:
    """
    Fallback when neither RDKit nor symbol info is available.
    Returns a list of 'C' symbols (geometrically correct, chemically naive).
    Used only for the minimal SDF writer when RDKit is absent.
    """
    return ["C"] * n


# ---------------------------------------------------------------------------
# QC conformer helper
# ---------------------------------------------------------------------------

def _get_qc_result(
    smiles: str,
    loader,          # QM9QCGeometryLoader
    method: str = METHOD_QC,
) -> ConformerResult:
    """
    Wrap QM9QCGeometryLoader output in a ConformerResult for uniform handling.
    """
    pos = loader.get_positions(smiles, heavy_only=True) if loader is not None else None
    if pos is None:
        return ConformerResult(
            method=method, smiles=smiles, positions=None,
            n_atoms=0, success=False,
            error_msg="Not found in QC bundle" if loader else "No QC bundle provided",
            seed=0,
        )
    return ConformerResult(
        method=method, smiles=smiles, positions=pos,
        n_atoms=pos.shape[0], success=True, seed=0,
    )


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def _load_smiles_from_csv(
    csv_path: Path,
    smiles_col: str,
    name_col: Optional[str],
    n_mols: Optional[int],
) -> List[Tuple[str, str]]:
    """
    Return a list of (smiles, mol_name) tuples from *csv_path*.

    Falls back to using SMILES as the name if *name_col* is None or absent.
    """
    records: List[Tuple[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        cols = reader.fieldnames or []
        if smiles_col not in cols:
            raise ValueError(
                f"SMILES column {smiles_col!r} not found in {csv_path.name}. "
                f"Available: {cols[:10]}"
            )
        use_name = name_col and name_col in cols
        for row in reader:
            smi = row[smiles_col].strip()
            if not smi:
                continue
            name = row[name_col].strip() if use_name else smi
            records.append((smi, name or smi))
            if n_mols is not None and len(records) >= n_mols:
                break
    return records


# ---------------------------------------------------------------------------
# SDF file writer (one dataset × one method)
# ---------------------------------------------------------------------------

def export_one_method(
    records: List[Tuple[str, str]],   # (smiles, mol_name)
    method: str,
    dataset: str,
    out_path: Path,
    seed: int = 42,
    qc_loader=None,                   # QM9QCGeometryLoader or None
    verbose: bool = True,
) -> Dict[str, int]:
    """
    Generate conformers for every (smiles, mol_name) in *records* using
    *method* and write them to *out_path* in SDF format.

    Returns a dict with counts: ``{'written': N, 'failed': M, 'skipped': K}``.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    use_rdkit_writer = rdkit_available()

    counts = {"written": 0, "failed": 0, "skipped": 0}
    n_total = len(records)

    if use_rdkit_writer:
        from rdkit.Chem import SDWriter
        fh_or_writer = SDWriter(str(out_path))
        # SDWriter handles the file; close at end
        _rdkit_writer = fh_or_writer
        _plain_fh = None
    else:
        _plain_fh = open(out_path, "w", encoding="utf-8")
        _rdkit_writer = None

    try:
        for idx, (smiles, mol_name) in enumerate(records):
            if verbose and idx % 500 == 0:
                print(
                    f"    [{idx}/{n_total}] {dataset}/{method} …",
                    flush=True,
                )

            # --- generate conformer ---
            if method == METHOD_QC:
                result = _get_qc_result(smiles, qc_loader, method=METHOD_QC)
            else:
                result = generate_conformer(smiles, method=method, seed=seed)

            if not result.success or result.positions is None:
                counts["failed"] += 1
                continue

            # --- get atom symbols ---
            symbols = _get_heavy_symbols_rdkit(smiles)
            if symbols is None:
                symbols = _guess_symbols_from_positions(result.n_atoms)

            if len(symbols) != result.n_atoms:
                # Symbol count mismatch (e.g. QC heavy-atom count differs from RDKit)
                warnings.warn(
                    f"[export_conformers] Symbol/position count mismatch for "
                    f"{smiles!r} ({dataset}/{method}): "
                    f"symbols={len(symbols)}, positions={result.n_atoms}. Skipping.",
                    stacklevel=2,
                )
                counts["skipped"] += 1
                continue

            # --- write SDF record ---
            if use_rdkit_writer:
                ok = _write_sdf_record_rdkit(
                    _rdkit_writer, mol_name, result.positions,
                    smiles, method, dataset,
                )
                if ok:
                    counts["written"] += 1
                else:
                    counts["failed"] += 1
            else:
                _write_sdf_record_minimal(
                    _plain_fh, mol_name, result.positions, symbols,
                    smiles, method, dataset,
                )
                counts["written"] += 1

    finally:
        if _rdkit_writer is not None:
            _rdkit_writer.close()
        if _plain_fh is not None:
            _plain_fh.close()

    return counts


# ---------------------------------------------------------------------------
# Top-level export pipeline
# ---------------------------------------------------------------------------

def export_conformers(
    datasets_dir: Path,
    out_dir: Path,
    methods: Sequence[str],
    datasets: Optional[Sequence[str]] = None,
    qc_bundle_path: Optional[Path] = None,
    n_mols: Optional[int] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Dict[str, int]]:
    """
    Export conformers for every (dataset, method) combination.

    Parameters
    ----------
    datasets_dir : Path
        Directory containing ``BACE.csv``, ``HIV.csv``, etc.
    out_dir : Path
        Output directory; SDF files are written here.
    methods : sequence of str
        Methods to run (subset of ``EXPORTABLE_METHODS``).
    datasets : sequence of str, optional
        Dataset names to process (subset of DATASET_REGISTRY keys).
        Defaults to all registered datasets whose CSV files exist.
    qc_bundle_path : Path, optional
        Path to the QM9 .xyz bundle (directory or .tar.bz2) required for
        the ``"3_qc"`` method.
    n_mols : int, optional
        Maximum molecules per dataset (useful for smoke-tests).
    seed : int
        Random seed for conformer generation (default 42).
    verbose : bool
        Print per-dataset progress (default True).

    Returns
    -------
    dict  ``{'{DATASET}_{METHOD}': {'written': N, 'failed': M, 'skipped': K}}``
    """
    # Resolve QC loader once (reused across all methods for QM9)
    qc_loader = None
    if METHOD_QC in methods:
        if qc_bundle_path is None:
            warnings.warn(
                "[export_conformers] Method '3_qc' requested but --qc-bundle "
                "was not provided. '3_qc' will be skipped for all datasets.\n"
                "Download the QM9 XYZ bundle from Figshare:\n"
                "  wget 'https://figshare.com/ndownloader/files/3195389'"
                " -O dsgdb9nsd.xyz.tar.bz2",
                stacklevel=2,
            )
        else:
            from revision.data.qm9_original_geometry_loader import QM9QCGeometryLoader
            try:
                qc_loader = QM9QCGeometryLoader(qc_bundle_path, verbose=verbose)
            except FileNotFoundError as exc:
                warnings.warn(
                    f"[export_conformers] QC bundle not found: {exc}. "
                    "Skipping '3_qc' method.",
                    stacklevel=2,
                )

    target_datasets = list(datasets) if datasets else list(DATASET_REGISTRY.keys())
    all_results: Dict[str, Dict[str, int]] = {}
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds_name in target_datasets:
        if ds_name not in DATASET_REGISTRY:
            warnings.warn(
                f"[export_conformers] Unknown dataset {ds_name!r}. "
                f"Known: {list(DATASET_REGISTRY.keys())}",
                stacklevel=2,
            )
            continue

        ds_cfg = DATASET_REGISTRY[ds_name]
        csv_path = datasets_dir / f"{ds_name}.csv"
        if not csv_path.exists():
            warnings.warn(
                f"[export_conformers] {csv_path} not found — skipping {ds_name}.",
                stacklevel=2,
            )
            continue

        if verbose:
            print(f"\n  Dataset: {ds_name}")

        try:
            records = _load_smiles_from_csv(
                csv_path,
                smiles_col=ds_cfg["smiles_col"],
                name_col=ds_cfg["name_col"],
                n_mols=n_mols,
            )
        except ValueError as exc:
            warnings.warn(f"[export_conformers] {exc}", stacklevel=2)
            continue

        if verbose:
            print(f"    Loaded {len(records)} SMILES")

        for method in methods:
            # Skip 3_qc for non-QM9 datasets
            if method == METHOD_QC and not ds_cfg["qc_eligible"]:
                if verbose:
                    print(f"    [SKIP] {method}: only supported for QM9")
                continue

            # Skip 3_qc if no loader
            if method == METHOD_QC and qc_loader is None:
                if verbose:
                    print(f"    [SKIP] {method}: no QC bundle available")
                continue

            # Skip RDKit methods if not installed
            if method in RDKIT_METHODS and not rdkit_available():
                if verbose:
                    print(f"    [SKIP] {method}: RDKit not installed")
                continue

            # Skip obabel if not on PATH
            if method == METHOD_OBABEL and not obabel_available():
                if verbose:
                    print(f"    [SKIP] {method}: obabel not on PATH")
                continue

            out_path = out_dir / f"{ds_name}_{method}.sdf"
            if verbose:
                print(f"    Exporting {method} -> {out_path.name} …")

            counts = export_one_method(
                records=records,
                method=method,
                dataset=ds_name,
                out_path=out_path,
                seed=seed,
                qc_loader=qc_loader if method == METHOD_QC else None,
                verbose=verbose,
            )
            key = f"{ds_name}_{method}"
            all_results[key] = counts
            if verbose:
                print(
                    f"      written={counts['written']}  "
                    f"failed={counts['failed']}  "
                    f"skipped={counts['skipped']}"
                )

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Export 3-D conformers to per-dataset SDF files for GitHub hosting."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--datasets-dir",
        default=None,
        help="Directory containing dataset CSVs (default: <repo_root>/datasets/).",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for SDF files (default: <repo_root>/conformers/).",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        choices=list(DATASET_REGISTRY.keys()),
        help="Datasets to process (default: all).",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=None,
        choices=EXPORTABLE_METHODS,
        help=f"Methods to run (default: all available). Choices: {EXPORTABLE_METHODS}",
    )
    p.add_argument(
        "--qc-bundle",
        default=None,
        help=(
            "Path to the QM9 .xyz bundle (directory or .tar.bz2) for the "
            "'3_qc' method. Required only when --methods includes '3_qc'."
        ),
    )
    p.add_argument(
        "--n-mols",
        type=int,
        default=None,
        help="Maximum molecules per dataset (for smoke-testing).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default 42).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-molecule progress output.",
    )
    return p


def _resolve(cli_arg: Optional[str], default_rel: str) -> Path:
    if cli_arg:
        return Path(cli_arg).expanduser().resolve()
    here = Path(__file__).resolve().parent          # revision/data/
    return here.parent.parent / default_rel         # repo/<default_rel>


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    datasets_dir = _resolve(args.datasets_dir, "datasets")
    out_dir      = _resolve(args.out_dir,      "conformers")
    qc_bundle    = Path(args.qc_bundle).expanduser().resolve() \
                   if args.qc_bundle else None

    methods = args.methods or EXPORTABLE_METHODS
    verbose = not args.quiet

    if not rdkit_available():
        print(
            "[WARNING] RDKit is NOT installed — RDKit methods will be skipped.\n"
            "Install with: conda install -c conda-forge rdkit\n",
            file=sys.stderr,
        )
    if not obabel_available():
        print(
            "[INFO] OpenBabel (obabel) not on PATH — 'obabel' method skipped.",
        )

    if not datasets_dir.exists():
        print(f"[ERROR] datasets_dir does not exist: {datasets_dir}", file=sys.stderr)
        return 1

    print(f"\nDatasets dir : {datasets_dir}")
    print(f"Output dir   : {out_dir}")
    print(f"Methods      : {methods}")
    print(f"N mols limit : {args.n_mols or 'all'}")
    print(f"Seed         : {args.seed}")
    print("-" * 60)

    results = export_conformers(
        datasets_dir=datasets_dir,
        out_dir=out_dir,
        methods=methods,
        datasets=args.datasets,
        qc_bundle_path=qc_bundle,
        n_mols=args.n_mols,
        seed=args.seed,
        verbose=verbose,
    )

    print("\n" + "=" * 60)
    print("Export summary:")
    total_written = 0
    for key, counts in sorted(results.items()):
        print(
            f"  {key:<30}  "
            f"written={counts['written']}  "
            f"failed={counts['failed']}  "
            f"skipped={counts['skipped']}"
        )
        total_written += counts["written"]
    print(f"  Total records written: {total_written}")
    print("=" * 60 + "\n")

    return 0 if total_written > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
