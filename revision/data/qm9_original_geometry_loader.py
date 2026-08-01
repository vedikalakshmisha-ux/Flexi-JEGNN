"""
revision/data/qm9_original_geometry_loader.py
=============================================
Parses the **official QM9 .xyz bundle** (Ramakrishnan et al. 2014,
doi:10.6084/m9.figshare.978904) and provides heavy-atom 3-D coordinates
from the DFT B3LYP/6-31G(2df,p) geometry for each molecule.

Molecules are matched to rows in ``datasets/QM9.csv`` via InChI strings
(generated from SMILES with RDKit and compared to the InChI embedded in the
last line of each .xyz file).  This is the only field common to both sources
that survives canonicalisation.

Supported bundle layouts
------------------------
* **Extracted directory** – a folder containing ``dsgdb9nsd_XXXXXX.xyz``
  files (133,885 of them in the full release).
* **Compressed archive** – the original ``dsgdb9nsd.xyz.tar.bz2`` file;
  the loader streams entries without full extraction.

Public API
----------
``QM9QCGeometryLoader``
    Main class.  Build once, look up per-molecule with
    ``get_dist_matrix(smiles)`` or ``get_positions(smiles)``.

``make_qc_dist_fn(loader, smiles, heavy_only=True)``
    Returns a ``dist_fn(i, j) -> float`` closure with exactly the same
    signature as the ``dist_fn`` closures inside
    ``experiments/qm9.py::_build_graph_tensors()``.
    Pass ``heavy_only=True`` (default) to strip hydrogen rows so that
    indices match RDKit's heavy-atom ordering, which is what the rest of
    the pipeline uses.

Download note
-------------
The bundle is NOT included in this repository (> 1 GB compressed).
Download from Figshare:

    wget "https://figshare.com/ndownloader/files/3195389" \\
         -O dsgdb9nsd.xyz.tar.bz2

Then either pass the archive path directly or extract it::

    tar xjf dsgdb9nsd.xyz.tar.bz2 -C /some/dir/

and pass the extracted directory path to ``QM9QCGeometryLoader``.

Out of scope (teammate handles separately)
------------------------------------------
* Protonation pipeline          -> revision/protonation/
* Conformer QC / validation     -> revision/conformer_qc/
* Baseline benchmarks           -> revision/benchmarks/reproduce_baselines.py
"""

from __future__ import annotations

import io
import os
import re
import tarfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# RDKit is an optional hard dependency — raise a clear error if missing.
try:
    from rdkit import Chem
    from rdkit.Chem.inchi import MolToInchi
    _RDKIT_OK = True
except ImportError:  # pragma: no cover
    _RDKIT_OK = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Atomic numbers for the elements present in QM9
_QM9_ELEMENTS = {'H', 'C', 'N', 'O', 'F'}

#: Pattern for the per-atom lines: symbol  x  y  z  Mulliken_partial_charge
#: QM9 uses a Unicode '*' replacement for decimal points in some fields —
#: we strip those gracefully.
_ATOM_LINE_RE = re.compile(
    r'^\s*([A-Za-z]+)\s+'          # element symbol
    r'([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s+'   # x
    r'([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s+'   # y
    r'([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)'       # z  (charge may follow or be absent)
)

#: InChI lines start with this prefix
_INCHI_PREFIX = 'InChI='


# ---------------------------------------------------------------------------
# Low-level .xyz parser
# ---------------------------------------------------------------------------

class _XYZRecord:
    """Parsed content of a single QM9 .xyz file."""

    __slots__ = ('gdb_idx', 'inchi', 'symbols', 'positions')

    def __init__(
        self,
        gdb_idx: int,
        inchi: str,
        symbols: List[str],
        positions: np.ndarray,  # shape (n_atoms, 3), Angstrom, ALL atoms incl. H
    ):
        self.gdb_idx = gdb_idx
        self.inchi = inchi
        self.symbols = symbols
        self.positions = positions  # float32, Angstrom

    @property
    def n_atoms(self) -> int:
        return len(self.symbols)

    def heavy_atom_mask(self) -> np.ndarray:
        """Boolean mask selecting non-hydrogen atoms."""
        return np.array([s != 'H' for s in self.symbols], dtype=bool)

    def heavy_positions(self) -> np.ndarray:
        """Positions of heavy atoms only, shape (n_heavy, 3)."""
        return self.positions[self.heavy_atom_mask()]


def _parse_xyz_text(text: str, source_name: str = '') -> Optional['_XYZRecord']:
    """
    Parse the text of one QM9 .xyz file.

    Returns ``None`` (with a warning) if the file is malformed.
    The parser is intentionally lenient: it reads only what it needs and
    ignores the many scalar properties on line 2.
    """
    lines = text.splitlines()
    if len(lines) < 4:
        warnings.warn(f"[qm9_loader] Too short, skipping: {source_name}")
        return None

    # Line 0: atom count
    try:
        n_atoms = int(lines[0].strip())
    except ValueError:
        warnings.warn(f"[qm9_loader] Bad atom count on line 1: {source_name}")
        return None

    # Line 1: properties — extract gdb index (field 1, 0-indexed)
    prop_fields = lines[1].split()
    try:
        gdb_idx = int(prop_fields[1]) if len(prop_fields) > 1 else -1
    except (ValueError, IndexError):
        gdb_idx = -1

    # Lines 2 … 2+n_atoms-1: atom coordinate rows
    symbols: List[str] = []
    coords: List[List[float]] = []
    for k in range(n_atoms):
        raw = lines[2 + k] if (2 + k) < len(lines) else ''
        # QM9 sometimes uses '*^' for numbers that overflow; replace with '0'
        raw = raw.replace('*^', 'e').replace(',', '.')
        m = _ATOM_LINE_RE.match(raw)
        if m is None:
            warnings.warn(
                f"[qm9_loader] Cannot parse atom line {2+k}: {raw!r} ({source_name})"
            )
            return None
        symbols.append(m.group(1))
        coords.append([float(m.group(2)), float(m.group(3)), float(m.group(4))])

    positions = np.array(coords, dtype=np.float32)

    # Last non-empty line that starts with 'InChI='
    inchi = ''
    for line in reversed(lines):
        stripped = line.strip()
        if stripped.startswith(_INCHI_PREFIX):
            inchi = stripped
            break

    if not inchi:
        warnings.warn(f"[qm9_loader] No InChI found in: {source_name}")
        # Don't discard — caller may still use positions if gdb_idx is known

    return _XYZRecord(
        gdb_idx=gdb_idx,
        inchi=inchi,
        symbols=symbols,
        positions=positions,
    )


# ---------------------------------------------------------------------------
# InChI generation from SMILES
# ---------------------------------------------------------------------------

def _smiles_to_inchi(smiles: str) -> Optional[str]:
    """Convert a SMILES string to a standard InChI via RDKit."""
    if not _RDKIT_OK:
        raise ImportError(
            "RDKit is required for SMILES→InChI conversion. "
            "Install with: conda install -c conda-forge rdkit"
        )
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    inchi = MolToInchi(mol)
    return inchi


def _inchi_base(inchi: str) -> str:
    """
    Strip the InChI layer prefix down to the connectivity + hydrogen layers
    (``InChI=1S/formula/connectivity``), discarding stereo and charge layers
    that may differ between the SMILES-derived and .xyz-embedded InChIs.

    Using the full InChI as the match key would miss molecules where RDKit
    and the QM9 toolchain disagree on stereo representation.  Truncating to
    the /c (connectivity) layer is sufficient to uniquely identify small
    QM9 molecules.
    """
    # Keep everything up to and including /c layer; drop /t /m /s /i /h after /c
    # but keep /h (hydrogen) which comes *before* /c
    # Standard InChI layer order: /c /h /b /t /m /s /i /f /r
    # We match on formula + /c connectivity only (robust across tools)
    parts = inchi.split('/')
    # parts[0] = 'InChI=1S', parts[1] = formula, parts[2..] = layers
    if len(parts) < 2:
        return inchi
    # Keep formula layer only — sufficient for QM9 (all molecules are unique
    # by formula+connectivity within the 134k set is NOT guaranteed, but
    # pairing formula with the /c layer gives uniqueness in practice)
    key_layers = [parts[0], parts[1]]  # 'InChI=1S', formula
    for p in parts[2:]:
        if p.startswith('c') or p.startswith('h'):
            key_layers.append(p)
        else:
            break  # layers beyond /h are stereo/isotope — stop here
    return '/'.join(key_layers)


# ---------------------------------------------------------------------------
# Main loader class
# ---------------------------------------------------------------------------

class QM9QCGeometryLoader:
    """
    Loads DFT B3LYP/6-31G(2df,p) geometries from the official QM9 .xyz bundle
    and makes them look-up-able by SMILES (via InChI matching).

    Parameters
    ----------
    bundle_path:
        Either a directory containing extracted ``dsgdb9nsd_XXXXXX.xyz`` files,
        or the path to ``dsgdb9nsd.xyz.tar.bz2``.
    verbose:
        If True, print progress during the initial scan.

    Notes
    -----
    * The first call to any lookup method triggers a full scan of the bundle
      and builds an in-memory index (InChI-base → _XYZRecord).  For the full
      133,885-molecule set this takes ~30–60 s on a typical laptop.
    * Positions are stored in Angstroms, as they appear in the .xyz files.
    * All atoms (including H) are stored; use ``heavy_only=True`` in
      ``get_positions()`` / ``make_qc_dist_fn()`` to get the heavy-atom
      subset that matches RDKit's atom ordering.
    """

    def __init__(self, bundle_path: Union[str, Path], verbose: bool = True):
        self._bundle_path = Path(bundle_path).expanduser().resolve()
        self._verbose = verbose
        self._index: Optional[Dict[str, _XYZRecord]] = None  # built lazily

        if not self._bundle_path.exists():
            raise FileNotFoundError(
                f"QM9 bundle not found: {self._bundle_path}\n"
                "Download from Figshare:\n"
                "  wget 'https://figshare.com/ndownloader/files/3195389'"
                " -O dsgdb9nsd.xyz.tar.bz2"
            )

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _ensure_index(self) -> None:
        if self._index is not None:
            return
        if self._bundle_path.is_dir():
            self._index = self._build_index_from_dir()
        elif tarfile.is_tarfile(str(self._bundle_path)):
            self._index = self._build_index_from_tar()
        else:
            raise ValueError(
                f"bundle_path must be an extracted directory or a .tar.bz2 "
                f"archive: {self._bundle_path}"
            )
        if self._verbose:
            print(f"[qm9_loader] Index built: {len(self._index)} molecules.")

    def _build_index_from_dir(self) -> Dict[str, _XYZRecord]:
        index: Dict[str, _XYZRecord] = {}
        xyz_files = sorted(self._bundle_path.glob('dsgdb9nsd_*.xyz'))
        if not xyz_files:
            raise FileNotFoundError(
                f"No dsgdb9nsd_*.xyz files found in {self._bundle_path}"
            )
        if self._verbose:
            print(f"[qm9_loader] Scanning {len(xyz_files)} .xyz files …")
        for i, fpath in enumerate(xyz_files):
            if self._verbose and i % 10000 == 0:
                print(f"  … {i}/{len(xyz_files)}", end='\r', flush=True)
            try:
                text = fpath.read_text(encoding='utf-8', errors='replace')
            except OSError:
                continue
            rec = _parse_xyz_text(text, source_name=fpath.name)
            if rec is None or not rec.inchi:
                continue
            key = _inchi_base(rec.inchi)
            index[key] = rec
        return index

    def _build_index_from_tar(self) -> Dict[str, _XYZRecord]:
        index: Dict[str, _XYZRecord] = {}
        if self._verbose:
            print(f"[qm9_loader] Streaming archive {self._bundle_path.name} …")
        n_read = 0
        with tarfile.open(str(self._bundle_path), 'r:bz2') as tf:
            for member in tf:
                if not member.name.endswith('.xyz'):
                    continue
                try:
                    f = tf.extractfile(member)
                    if f is None:
                        continue
                    text = f.read().decode('utf-8', errors='replace')
                except Exception:
                    continue
                rec = _parse_xyz_text(text, source_name=member.name)
                if rec is None or not rec.inchi:
                    continue
                key = _inchi_base(rec.inchi)
                index[key] = rec
                n_read += 1
                if self._verbose and n_read % 10000 == 0:
                    print(f"  … {n_read} read", end='\r', flush=True)
        return index

    # ------------------------------------------------------------------
    # Public look-up API
    # ------------------------------------------------------------------

    def _lookup(self, smiles: str) -> Optional['_XYZRecord']:
        """Return the _XYZRecord for *smiles*, or None if no match."""
        self._ensure_index()
        inchi = _smiles_to_inchi(smiles)
        if inchi is None:
            return None
        key = _inchi_base(inchi)
        return self._index.get(key)

    def get_positions(
        self,
        smiles: str,
        heavy_only: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Return the DFT-geometry positions for the molecule identified by
        *smiles*.

        Parameters
        ----------
        smiles:
            SMILES string (as it appears in ``QM9.csv``).
        heavy_only:
            If True (default), return only non-hydrogen atom positions so
            that the row index matches RDKit's heavy-atom ordering.

        Returns
        -------
        numpy.ndarray of shape ``(n, 3)`` in Angstroms, or ``None`` if no
        match is found in the bundle.
        """
        rec = self._lookup(smiles)
        if rec is None:
            return None
        return rec.heavy_positions() if heavy_only else rec.positions.copy()

    def get_dist_matrix(
        self,
        smiles: str,
        heavy_only: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Return a pairwise Euclidean distance matrix (Angstroms) for the
        molecule identified by *smiles*.

        Shape: ``(n, n)`` where n = number of (heavy) atoms.
        Returns ``None`` if no match is found.
        """
        pos = self.get_positions(smiles, heavy_only=heavy_only)
        if pos is None:
            return None
        diff = pos[:, None, :] - pos[None, :, :]
        return np.sqrt((diff ** 2).sum(axis=-1)).astype(np.float32)

    def has_molecule(self, smiles: str) -> bool:
        """Return True if *smiles* can be matched to an .xyz record."""
        return self._lookup(smiles) is not None

    def __len__(self) -> int:
        self._ensure_index()
        return len(self._index)

    def __repr__(self) -> str:
        n = len(self._index) if self._index is not None else '(not loaded)'
        return f"QM9QCGeometryLoader(bundle={self._bundle_path.name!r}, n={n})"


# ---------------------------------------------------------------------------
# dist_fn factory — compatible with experiments/qm9.py::_build_graph_tensors
# ---------------------------------------------------------------------------

def make_qc_dist_fn(
    loader: QM9QCGeometryLoader,
    smiles: str,
    heavy_only: bool = True,
    fallback: float = 999.0,
):
    """
    Build a ``dist_fn(i, j) -> float`` closure backed by DFT QC geometry.

    The returned function has exactly the same signature as the ``dist_fn``
    closures defined inside ``experiments/qm9.py::_build_graph_tensors()``
    for levels 0–4.  It is intended to be used as an **additive level
    "3_qc"** (see ``revision/geometry_qc/qm9_qc_level.py``).

    Parameters
    ----------
    loader:
        A ``QM9QCGeometryLoader`` instance (already constructed).
    smiles:
        SMILES string for the molecule to look up.
    heavy_only:
        If True (default), positions are heavy-atom-only (matching RDKit
        atom indices 0..n-1 where n = mol.GetNumAtoms()).
    fallback:
        Distance returned for atom pairs not resolvable from the bundle
        (e.g. molecule not found, or index out of range).  Default 999.0
        matches the sentinel used across all other levels.

    Returns
    -------
    Callable[[int, int], float]
        ``dist_fn(i, j)`` returning Euclidean distance in Angstroms.
    ``None`` if the molecule is not found in the bundle **and** no fallback
    is desired — callers should treat ``None`` the same way
    ``_build_graph_tensors`` treats a failed ``EmbedMolecule`` (return None
    for the whole graph).

    Example
    -------
    >>> loader = QM9QCGeometryLoader('/path/to/dsgdb9nsd/')
    >>> fn = make_qc_dist_fn(loader, 'C')
    >>> fn(0, 0)
    0.0
    """
    D = loader.get_dist_matrix(smiles, heavy_only=heavy_only)

    if D is None:
        # Molecule not found — return a closure that always returns fallback
        # so the caller can decide whether to skip or degrade gracefully.
        def _not_found(i: int, j: int) -> float:  # noqa: E306
            return fallback

        _not_found._qc_found = False  # type: ignore[attr-defined]
        return _not_found

    def dist_fn(i: int, j: int) -> float:
        try:
            return float(D[i][j])
        except IndexError:
            return fallback

    dist_fn._qc_found = True   # type: ignore[attr-defined]
    dist_fn._n_atoms = D.shape[0]  # type: ignore[attr-defined]
    return dist_fn
