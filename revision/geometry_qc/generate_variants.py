"""
revision/geometry_qc/generate_variants.py
==========================================
Multiple 3-D geometry-generation methods for small organic molecules.

Methods implemented
-------------------
All methods return a ``ConformerResult`` dataclass (see below).

Pure-RDKit methods (four variants)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``METHOD_ETKDG``      RDKit ETKDGv1   — original ETKDG (Riniker & Landrum 2015)
``METHOD_ETKDGv2``    RDKit ETKDGv2   — adds torsion terms (Riniker 2020)
``METHOD_ETKDGv3``    RDKit ETKDGv3   — current default in experiments/qm9.py
``METHOD_RANDOM``     Random distance-geometry seed — reproducible but unphysical;
                      useful as a variance baseline / negative control.

OpenBabel via subprocess
~~~~~~~~~~~~~~~~~~~~~~~~~
``METHOD_OBABEL``     ``obabel --gen3d`` — calls the ``obabel`` CLI.

**OpenBabel is not installed on this machine.**
If ``obabel`` is not on PATH the method raises ``OpenBabelNotInstalledError``
with clear installation instructions rather than silently faking output.
The rest of the module works without OpenBabel.

Availability detection
~~~~~~~~~~~~~~~~~~~~~~~
``rdkit_available()``   → bool
``obabel_available()``  → bool   (checks PATH at call time, ~10 ms)

These are used by ``generate_conformer()`` and by the test suite to skip
methods that cannot run in the current environment.

Public API
----------
``generate_conformer(smiles, method, seed, *, max_attempts)``
    Generate one conformer; returns ``ConformerResult``.

``generate_all_available(smiles, seed, *, skip_methods)``
    Try every method, collect results, skip methods whose dependency is absent.

``ConformerResult``
    Dataclass with fields: ``method``, ``smiles``, ``positions`` (numpy array,
    heavy atoms only, shape (n,3) Å), ``n_atoms``, ``success``, ``error_msg``.

Out of scope (teammate handles separately)
------------------------------------------
* Protonation pipeline          -> revision/protonation/
* Conformer QC / validation     -> revision/conformer_qc/
* Baseline benchmarks           -> revision/benchmarks/reproduce_baselines.py
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Method name constants
# ---------------------------------------------------------------------------

METHOD_ETKDG    = "ETKDG"     # RDKit ETKDGv1
METHOD_ETKDGv2  = "ETKDGv2"   # RDKit ETKDGv2
METHOD_ETKDGv3  = "ETKDGv3"   # RDKit ETKDGv3 (matches experiments/qm9.py level 3)
METHOD_RANDOM   = "random_dg"  # Random distance-geometry (negative control)
METHOD_OBABEL   = "obabel"     # OpenBabel --gen3d via subprocess

ALL_METHODS: List[str] = [
    METHOD_ETKDG,
    METHOD_ETKDGv2,
    METHOD_ETKDGv3,
    METHOD_RANDOM,
    METHOD_OBABEL,
]

RDKIT_METHODS: List[str] = [
    METHOD_ETKDG,
    METHOD_ETKDGv2,
    METHOD_ETKDGv3,
    METHOD_RANDOM,
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ConformerGenerationError(RuntimeError):
    """Raised when a conformer cannot be generated for a given molecule."""


class OpenBabelNotInstalledError(RuntimeError):
    """
    Raised when the ``obabel`` CLI is not found on PATH.

    Install OpenBabel:
      Windows : https://github.com/openbabel/openbabel/releases
                (download the .exe installer; ensure 'obabel' is added to PATH)
      Linux   : sudo apt install openbabel          # Debian/Ubuntu
              : conda install -c conda-forge openbabel
      macOS   : brew install open-babel
              : conda install -c conda-forge openbabel
    """


class RDKitNotInstalledError(RuntimeError):
    """
    Raised when RDKit is not importable.

    Install RDKit:
      conda install -c conda-forge rdkit
      # or, for pip-based envs:
      pip install rdkit
    """


# ---------------------------------------------------------------------------
# Availability detection
# ---------------------------------------------------------------------------

def rdkit_available() -> bool:
    """Return True if RDKit can be imported."""
    try:
        import rdkit  # noqa: F401
        return True
    except ImportError:
        return False


def obabel_available() -> bool:
    """Return True if the ``obabel`` CLI executable is on PATH."""
    return shutil.which("obabel") is not None


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ConformerResult:
    """
    Outcome of a single conformer-generation attempt.

    Attributes
    ----------
    method : str
        One of the ``METHOD_*`` constants.
    smiles : str
        Input SMILES string.
    positions : np.ndarray or None
        Heavy-atom 3-D coordinates, shape ``(n_heavy, 3)``, Angstroms.
        ``None`` if generation failed.
    n_atoms : int
        Number of heavy atoms (0 if unknown / failed).
    success : bool
        True if positions were successfully generated.
    error_msg : str
        Human-readable failure reason (empty on success).
    seed : int
        Random seed used.
    """
    method:    str
    smiles:    str
    positions: Optional[np.ndarray]
    n_atoms:   int
    success:   bool
    error_msg: str = ""
    seed:      int = 42

    @property
    def dist_matrix(self) -> Optional[np.ndarray]:
        """Pairwise Euclidean distance matrix, shape ``(n,n)``."""
        if self.positions is None:
            return None
        d = self.positions[:, None, :] - self.positions[None, :, :]
        return np.sqrt((d ** 2).sum(axis=-1)).astype(np.float32)

    def __repr__(self) -> str:
        status = "OK" if self.success else f"FAIL({self.error_msg[:40]})"
        return (
            f"ConformerResult(method={self.method!r}, smiles={self.smiles!r}, "
            f"n_atoms={self.n_atoms}, {status})"
        )


def _failed(method: str, smiles: str, reason: str, seed: int = 42) -> ConformerResult:
    return ConformerResult(
        method=method, smiles=smiles, positions=None,
        n_atoms=0, success=False, error_msg=reason, seed=seed,
    )


# ---------------------------------------------------------------------------
# RDKit helpers
# ---------------------------------------------------------------------------

def _require_rdkit():
    if not rdkit_available():
        raise RDKitNotInstalledError(
            "RDKit is not installed.\n"
            "Install with: conda install -c conda-forge rdkit\n"
            "              pip install rdkit"
        )


def _extract_heavy_positions(mol_with_conf, n_heavy: int) -> np.ndarray:
    """
    Return heavy-atom positions from the first conformer of *mol_with_conf*.

    *n_heavy* is the atom count of the molecule **before** AddHs, which
    matches RDKit's heavy-atom ordering (indices 0 .. n_heavy-1).
    """
    conf = mol_with_conf.GetConformer()
    pos = np.array(
        [[conf.GetAtomPosition(i).x,
          conf.GetAtomPosition(i).y,
          conf.GetAtomPosition(i).z]
         for i in range(n_heavy)],
        dtype=np.float32,
    )
    return pos


def _embed_with_params(mol_h, params, seed: int) -> bool:
    """
    Attempt ETKDGv3-style embedding with *params* and multiple fallback seeds.
    Returns True on success.
    """
    from rdkit.Chem import AllChem
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol_h, params) != -1:
        return True
    # Try a handful of additional seeds before giving up
    for extra in [seed + 1, seed + 2, seed + 7, seed + 13]:
        params.randomSeed = extra
        if AllChem.EmbedMolecule(mol_h, params) != -1:
            return True
    return False


# ---------------------------------------------------------------------------
# Individual method implementations
# ---------------------------------------------------------------------------

def _gen_etkdg(smiles: str, seed: int, version: int) -> ConformerResult:
    """
    RDKit ETKDGv1 / v2 / v3 conformer generation.

    After embedding, MMFF94 minimisation is attempted (up to 200 iterations)
    to match the behaviour of ``experiments/qm9.py`` level 3.
    """
    _require_rdkit()
    from rdkit import Chem
    from rdkit.Chem import AllChem

    method_name = {1: METHOD_ETKDG, 2: METHOD_ETKDGv2, 3: METHOD_ETKDGv3}[version]

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return _failed(method_name, smiles, "RDKit could not parse SMILES", seed)
    if mol.GetNumAtoms() < 2:
        return _failed(method_name, smiles, "Molecule has fewer than 2 atoms", seed)
    n_heavy = mol.GetNumAtoms()
    mol_h = Chem.AddHs(mol)

    # Select embedding parameters by version
    if version == 1:
        params = AllChem.EmbedParameters()
        # ETKDGv1: use ETKDG but without the v2/v3 extras
        params.useExpTorsionAnglePrefs = True
        params.useBasicKnowledge = True
        params.ETversion = 1
    elif version == 2:
        params = AllChem.ETKDGv2()
    else:  # version == 3
        params = AllChem.ETKDGv3()

    if not _embed_with_params(mol_h, params, seed):
        return _failed(
            method_name, smiles,
            f"EmbedMolecule failed (ETKDGv{version}) after multiple seeds",
            seed,
        )

    try:
        AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
    except Exception:
        pass  # optimisation is best-effort

    try:
        pos = _extract_heavy_positions(mol_h, n_heavy)
    except Exception as exc:
        return _failed(method_name, smiles, f"Position extraction failed: {exc}", seed)

    return ConformerResult(
        method=method_name, smiles=smiles, positions=pos,
        n_atoms=n_heavy, success=True, seed=seed,
    )


def _gen_etkdg_v1(smiles: str, seed: int) -> ConformerResult:
    return _gen_etkdg(smiles, seed, version=1)


def _gen_etkdg_v2(smiles: str, seed: int) -> ConformerResult:
    return _gen_etkdg(smiles, seed, version=2)


def _gen_etkdg_v3(smiles: str, seed: int) -> ConformerResult:
    return _gen_etkdg(smiles, seed, version=3)


def _gen_random(smiles: str, seed: int) -> ConformerResult:
    """
    Random distance-geometry embedding (no torsion/knowledge prefs).

    Uses ``AllChem.EmbedParameters()`` with all knowledge-based terms
    disabled.  Gives a non-physically-meaningful but reproducible 3-D
    structure — useful as a variance / noise baseline.
    """
    _require_rdkit()
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return _failed(METHOD_RANDOM, smiles, "RDKit could not parse SMILES", seed)
    if mol.GetNumAtoms() < 2:
        return _failed(METHOD_RANDOM, smiles, "Molecule has fewer than 2 atoms", seed)
    n_heavy = mol.GetNumAtoms()
    mol_h = Chem.AddHs(mol)

    params = AllChem.EmbedParameters()
    params.useExpTorsionAnglePrefs = False
    params.useBasicKnowledge = False
    params.randomSeed = seed

    if AllChem.EmbedMolecule(mol_h, params) == -1:
        return _failed(METHOD_RANDOM, smiles, "Random embedding failed", seed)

    try:
        pos = _extract_heavy_positions(mol_h, n_heavy)
    except Exception as exc:
        return _failed(METHOD_RANDOM, smiles, f"Position extraction failed: {exc}", seed)

    return ConformerResult(
        method=METHOD_RANDOM, smiles=smiles, positions=pos,
        n_atoms=n_heavy, success=True, seed=seed,
    )


def _gen_obabel(smiles: str, seed: int) -> ConformerResult:
    """
    OpenBabel ``--gen3d`` conformer generation via subprocess.

    Passes the SMILES string on stdin and reads SDF on stdout.
    Atom coordinates are parsed from the SDF V2000 block (lines 4..4+n_atoms).

    **OpenBabel is NOT installed on this machine.**
    This function raises ``OpenBabelNotInstalledError`` if ``obabel`` is not
    found on PATH.
    """
    if not obabel_available():
        raise OpenBabelNotInstalledError(
            "'obabel' CLI was not found on PATH.\n\n"
            "OpenBabel is NOT installed on this machine.\n\n"
            "To install:\n"
            "  Windows : https://github.com/openbabel/openbabel/releases\n"
            "            (download the .exe installer, add obabel to PATH)\n"
            "  Linux   : sudo apt install openbabel\n"
            "          : conda install -c conda-forge openbabel\n"
            "  macOS   : brew install open-babel\n"
            "          : conda install -c conda-forge openbabel\n\n"
            "After installation, restart the Python session."
        )

    # Build the obabel command:
    #   -ismi  : input format SMILES (from stdin)
    #   -osdf  : output format SDF (to stdout)
    #   --gen3d: generate 3D coordinates
    cmd = ["obabel", "-ismi", "-osdf", "--gen3d"]
    try:
        result = subprocess.run(
            cmd,
            input=smiles + "\n",
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return _failed(METHOD_OBABEL, smiles, "obabel timed out (>60 s)", seed)
    except Exception as exc:
        return _failed(METHOD_OBABEL, smiles, f"obabel subprocess error: {exc}", seed)

    sdf_text = result.stdout
    if not sdf_text.strip():
        err = result.stderr.strip() or "empty obabel output"
        return _failed(METHOD_OBABEL, smiles, f"obabel produced no output: {err}", seed)

    # Parse the SDF V2000 block to extract coordinates
    try:
        pos, n_heavy = _parse_sdf_positions(sdf_text)
    except Exception as exc:
        return _failed(METHOD_OBABEL, smiles, f"SDF parsing failed: {exc}", seed)

    return ConformerResult(
        method=METHOD_OBABEL, smiles=smiles, positions=pos,
        n_atoms=n_heavy, success=True, seed=seed,
    )


def _parse_sdf_positions(sdf_text: str) -> tuple[np.ndarray, int]:
    """
    Parse atom coordinates from an SDF V2000 block.

    SDF V2000 format:
      Line 0: molecule name
      Line 1: program/date info
      Line 2: comment
      Line 3: counts line  — first two fields are n_atoms and n_bonds
      Lines 4 .. 4+n_atoms-1: atom block  (x y z symbol ...)

    Returns ``(positions, n_atoms)`` where positions is float32 (n_atoms, 3).
    Raises ``ValueError`` on parse failure.
    """
    lines = sdf_text.splitlines()
    # Find counts line (line index 3 relative to the block start)
    # The block may be preceded by extra blank lines; find the first "M  END"
    # section by looking for the counts line pattern.
    block_start = 0
    for i, line in enumerate(lines):
        # Counts line: 3-char fields, positions 0-2 are n_atoms and n_bonds
        stripped = line.strip()
        if len(stripped) >= 6 and stripped[:3].strip().isdigit():
            block_start = i
            break

    counts_line = lines[block_start]
    try:
        n_atoms = int(counts_line[:3])
    except ValueError:
        raise ValueError(f"Cannot parse atom count from counts line: {counts_line!r}")

    atom_lines = lines[block_start + 1: block_start + 1 + n_atoms]
    if len(atom_lines) < n_atoms:
        raise ValueError(
            f"Expected {n_atoms} atom lines, got {len(atom_lines)}"
        )

    coords = []
    for al in atom_lines:
        parts = al.split()
        if len(parts) < 4:
            raise ValueError(f"Malformed atom line: {al!r}")
        coords.append([float(parts[0]), float(parts[1]), float(parts[2])])

    positions = np.array(coords, dtype=np.float32)
    return positions, n_atoms


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_GENERATORS = {
    METHOD_ETKDG:   _gen_etkdg_v1,
    METHOD_ETKDGv2: _gen_etkdg_v2,
    METHOD_ETKDGv3: _gen_etkdg_v3,
    METHOD_RANDOM:  _gen_random,
    METHOD_OBABEL:  _gen_obabel,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_conformer(
    smiles: str,
    method: str = METHOD_ETKDGv3,
    seed: int = 42,
    *,
    max_attempts: int = 1,
) -> ConformerResult:
    """
    Generate a single 3-D conformer for *smiles* using *method*.

    Parameters
    ----------
    smiles : str
        Input SMILES string.
    method : str
        One of the ``METHOD_*`` constants (default: ``METHOD_ETKDGv3``).
    seed : int
        Random seed (default 42).
    max_attempts : int
        For RDKit methods: number of independent embedding attempts to try
        before returning a failure result.  Has no effect for OpenBabel.

    Returns
    -------
    ConformerResult
        Always returns a result; never raises (unless *method* is unknown).

    Raises
    ------
    ValueError
        If *method* is not one of the ``ALL_METHODS`` strings.
    OpenBabelNotInstalledError
        If *method* is ``METHOD_OBABEL`` and ``obabel`` is not on PATH.
    RDKitNotInstalledError
        If a RDKit method is requested and RDKit is not importable.
    """
    if method not in _GENERATORS:
        raise ValueError(
            f"Unknown method {method!r}. Choose from: {ALL_METHODS}"
        )
    fn = _GENERATORS[method]
    for attempt in range(max(1, max_attempts)):
        result = fn(smiles, seed + attempt)
        if result.success:
            return result
    return result  # last attempt's failure


def generate_all_available(
    smiles: str,
    seed: int = 42,
    *,
    skip_methods: Optional[Sequence[str]] = None,
) -> Dict[str, ConformerResult]:
    """
    Try every method in ``ALL_METHODS`` and collect results.

    Methods whose required dependency is absent are skipped (not failed):
    * RDKit methods are skipped if ``rdkit_available()`` returns False.
    * ``METHOD_OBABEL`` is skipped if ``obabel_available()`` returns False.

    Parameters
    ----------
    smiles : str
        Input SMILES string.
    seed : int
        Random seed (default 42).
    skip_methods : sequence of str, optional
        Additional method names to skip entirely (e.g. ``[METHOD_RANDOM]``).

    Returns
    -------
    dict mapping method name → ConformerResult
        Only methods that were *attempted* (not skipped) appear in the dict.
    """
    skip = set(skip_methods or [])
    results: Dict[str, ConformerResult] = {}

    _rdkit_ok = rdkit_available()
    _obabel_ok = obabel_available()

    for method in ALL_METHODS:
        if method in skip:
            continue
        if method in RDKIT_METHODS and not _rdkit_ok:
            warnings.warn(
                f"[generate_variants] Skipping {method!r}: RDKit not installed.",
                stacklevel=2,
            )
            continue
        if method == METHOD_OBABEL and not _obabel_ok:
            warnings.warn(
                f"[generate_variants] Skipping {METHOD_OBABEL!r}: "
                "'obabel' not found on PATH. "
                "Install OpenBabel and add it to PATH to enable this method.",
                stacklevel=2,
            )
            continue
        results[method] = generate_conformer(smiles, method, seed)

    return results
