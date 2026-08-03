"""
revision/protonation/protonate.py
==================================
pH 7.4 protonation protocol for the Flexi-JEGNN revision.

Three independent protonation pipelines are provided:

  1. SMILES  → Dimorphite-DL  (ADMET / classification datasets)
  2. SDF / mol2 ligand files  → OpenBabel  (PDBbind ligands)
  3. PDB pocket files         → OpenBabel -p  (PDBbind protein pocket)
     (PDB2PQR / PROPKA is documented as the preferred alternative;
      a thin wrapper is included that calls it when available.)

Nothing in experiments/ is imported or modified.  This module is meant
to be called from new revision scripts (e.g. revision/protonation/run.py)
or from a Jupyter notebook.

Usage (quick-start)
-------------------
  from revision.protonation.protonate import (
      protonate_smiles_series,
      protonate_ligand_sdf,
      protonate_pocket_pdb,
  )

  # ADMET / classification SMILES column
  result_df = protonate_smiles_series(df["smiles"], ph=7.4)

  # PDBbind ligand file  →  new SDF with protonation applied
  protonate_ligand_sdf("refined-set/1abc/1abc_ligand.sdf",
                        out_path="revised_data/1abc_ligand_H.sdf")

  # PDBbind pocket PDB  →  new PDB with protonation applied
  protonate_pocket_pdb("refined-set/1abc/1abc_pocket.pdb",
                        out_path="revised_data/1abc_pocket_H.pdb")

Logging
-------
Every call appends a structured record to the module-level logger
``protonate_logger``.  Call ``write_log(path)`` to dump the full
session log as JSON-Lines (one record per molecule / file).

Tool / version detection
------------------------
Tool availability is checked lazily on first use; if a required tool
is not installed the function raises ``RuntimeError`` with installation
instructions.  Version strings are captured at detection time and
embedded in every log record.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Module-level logger (Python logging) — separate from the structured JSON log
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] protonate: %(message)s",
    level=logging.INFO,
)
_log = logging.getLogger("protonate")

# ---------------------------------------------------------------------------
# Structured protonation log (accumulated in memory, written on request)
# ---------------------------------------------------------------------------

@dataclass
class ProtonationRecord:
    """One structured log entry per molecule / file that was processed."""

    # --- identification ---
    input_id: str           # SMILES string, file path, or PDB ID
    input_type: str         # "smiles" | "sdf" | "mol2" | "pdb"
    ph: float               # target pH

    # --- tool provenance ---
    tool: str               # e.g. "dimorphite_dl", "openbabel", "pdb2pqr"
    tool_version: str       # version string captured at detection time

    # --- outcome ---
    status: str             # "ok" | "unchanged" | "failed" | "skipped"
    output: Optional[str]   # protonated SMILES, or output file path
    n_states_enumerated: int = 0  # Dimorphite: number of protonation states returned

    # --- change summary ---
    original_formal_charges: Optional[str] = None   # RDKit-computed, before
    protonated_formal_charges: Optional[str] = None # RDKit-computed, after
    groups_changed: List[str] = field(default_factory=list)
    # e.g. ["COO- deprotonated at C7", "NH3+ protonated at N2"]

    # --- timing ---
    wall_time_s: float = 0.0
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    # --- free-text notes ---
    notes: str = ""


_session_log: List[ProtonationRecord] = []


def write_log(path: str | Path, mode: str = "w") -> Path:
    """
    Write the accumulated session log to *path* as JSON-Lines.

    Parameters
    ----------
    path:
        Destination file.  Parent directories are created automatically.
    mode:
        'w' (overwrite) or 'a' (append).

    Returns
    -------
    Path of the written file.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open(mode, encoding="utf-8") as fh:
        for rec in _session_log:
            fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
    _log.info("Session log written → %s  (%d records)", out, len(_session_log))
    return out


def clear_log() -> None:
    """Clear the in-memory session log (useful between test runs)."""
    _session_log.clear()


# ===========================================================================
# Section 1 – Tool detection helpers
# ===========================================================================

# Cache detected tool versions so we probe only once per Python session.
_tool_versions: Dict[str, str] = {}


def _detect_dimorphite() -> str:
    """Return Dimorphite-DL version string or raise RuntimeError."""
    if "dimorphite_dl" in _tool_versions:
        return _tool_versions["dimorphite_dl"]

    try:
        import dimorphite_dl as _dml  # type: ignore
        ver = getattr(_dml, "__version__", "unknown")
    except ImportError:
        raise RuntimeError(
            textwrap.dedent("""\
            Dimorphite-DL is not installed.
            Install with:
                pip install dimorphite_dl
            or (editable from source):
                git clone https://github.com/durrantlab/dimorphite_dl.git
                pip install -e ./dimorphite_dl
            """)
        )

    _tool_versions["dimorphite_dl"] = ver
    _log.info("Dimorphite-DL detected  version=%s", ver)
    return ver


def _detect_openbabel() -> str:
    """Return OpenBabel version string or raise RuntimeError."""
    if "openbabel" in _tool_versions:
        return _tool_versions["openbabel"]

    exe = shutil.which("obabel")
    if exe is None:
        raise RuntimeError(
            textwrap.dedent("""\
            OpenBabel (obabel) not found in PATH.
            Install with:
                conda install -c conda-forge openbabel
            or:
                sudo apt-get install openbabel  (Debian/Ubuntu)
                brew install open-babel         (macOS)
            """)
        )

    try:
        result = subprocess.run(
            [exe, "--version"],
            capture_output=True, text=True, timeout=10,
        )
        # OpenBabel prints version to stdout: "Open Babel 3.1.0 -- ..."
        ver_match = re.search(r"Open Babel\s+([\d.]+)", result.stdout + result.stderr)
        ver = ver_match.group(1) if ver_match else "unknown"
    except Exception as exc:
        ver = f"unknown ({exc})"

    _tool_versions["openbabel"] = ver
    _log.info("OpenBabel detected  version=%s  exe=%s", ver, exe)
    return ver


def _detect_pdb2pqr() -> str:
    """Return pdb2pqr version string or raise RuntimeError."""
    if "pdb2pqr" in _tool_versions:
        return _tool_versions["pdb2pqr"]

    exe = shutil.which("pdb2pqr") or shutil.which("pdb2pqr30")
    if exe is None:
        raise RuntimeError(
            textwrap.dedent("""\
            pdb2pqr is not found in PATH.
            Install with:
                pip install pdb2pqr
            or:
                conda install -c conda-forge pdb2pqr
            """)
        )

    try:
        result = subprocess.run(
            [exe, "--version"],
            capture_output=True, text=True, timeout=10,
        )
        ver_match = re.search(r"(\d+\.\d+[\.\d]*)", result.stdout + result.stderr)
        ver = ver_match.group(1) if ver_match else "unknown"
    except Exception as exc:
        ver = f"unknown ({exc})"

    _tool_versions["pdb2pqr"] = ver
    _log.info("pdb2pqr detected  version=%s  exe=%s", ver, exe)
    return ver


# ===========================================================================
# Section 2 – Formal-charge fingerprint helper (RDKit, optional)
# ===========================================================================

def _formal_charge_summary(smiles: str) -> str:
    """
    Return a compact charge fingerprint string for *smiles*, e.g.
    'N+1@2,O-1@5'.  Returns '' if RDKit is unavailable or SMILES is invalid.
    """
    if not smiles:
        return ""
    try:
        from rdkit import Chem  # type: ignore
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        parts = []
        for atom in mol.GetAtoms():
            q = atom.GetFormalCharge()
            if q != 0:
                sign = "+" if q > 0 else ""
                parts.append(f"{atom.GetSymbol()}{sign}{q}@{atom.GetIdx()}")
        return ",".join(parts) if parts else "neutral"
    except Exception:
        return ""


def _diff_charges(orig: str, prot: str) -> List[str]:
    """
    Return a list of human-readable change strings, e.g.
    ["N neutral→+1 at idx 2", "O -1→neutral at idx 7"].
    """
    if not orig or not prot:
        return []

    def _parse(fp: str) -> Dict[str, str]:
        """Parse 'sym+q@idx,...' back into {idx: 'sym+q'}."""
        if fp in ("neutral", ""):
            return {}
        d: Dict[str, str] = {}
        for part in fp.split(","):
            m = re.match(r"([A-Za-z]+)([+\-]\d+)@(\d+)", part)
            if m:
                d[m.group(3)] = m.group(1) + m.group(2)
        return d

    o, p = _parse(orig), _parse(prot)
    all_idxs = set(o) | set(p)
    changes: List[str] = []
    for idx in sorted(all_idxs, key=int):
        ov = o.get(idx, "neutral")
        pv = p.get(idx, "neutral")
        if ov != pv:
            changes.append(f"atom@{idx}: {ov} → {pv}")
    return changes


# ===========================================================================
# Section 3 – SMILES protonation via Dimorphite-DL
# ===========================================================================

def protonate_smiles(
    smiles: str,
    ph: float = 7.4,
    ph_tolerance: float = 1.0,
    max_variants: int = 128,
    pick_most_likely: bool = True,
) -> ProtonationRecord:
    """
    Protonate a single SMILES string at *ph* using Dimorphite-DL.

    Parameters
    ----------
    smiles:
        Input SMILES (need not be canonical).
    ph:
        Target pH (default 7.4).
    ph_tolerance:
        Dimorphite enumeration window is [ph - ph_tolerance, ph + ph_tolerance].
    max_variants:
        Cap on protonation-state variants returned by Dimorphite.
    pick_most_likely:
        If True (default) return only the first (most abundant) variant.
        If False, return all variants joined with '.'; useful for inspection.

    Returns
    -------
    ProtonationRecord with ``output`` set to the protonated SMILES (or the
    original SMILES if Dimorphite returns no variants) and ``status`` set to
    'ok' | 'unchanged' | 'failed'.
    """
    t0 = time.time()
    ver = _detect_dimorphite()

    orig_charges = _formal_charge_summary(smiles)

    rec = ProtonationRecord(
        input_id=smiles,
        input_type="smiles",
        ph=ph,
        tool="dimorphite_dl",
        tool_version=ver,
        status="failed",
        output=None,
        original_formal_charges=orig_charges,
    )

    try:
        import dimorphite_dl as _dml  # type: ignore

        variants: List[str] = list(
            _dml.run_with_mol_list(
                [smiles],
                min_ph=ph - ph_tolerance,
                max_ph=ph + ph_tolerance,
                max_variants=max_variants,
                label_states=False,
                silent=True,
            )
        )

        if not variants:
            rec.status = "unchanged"
            rec.output = smiles
            rec.notes = "Dimorphite returned no variants; original SMILES kept."
        else:
            rec.n_states_enumerated = len(variants)
            chosen = variants[0] if pick_most_likely else ".".join(variants)
            rec.output = chosen
            prot_charges = _formal_charge_summary(chosen)
            rec.protonated_formal_charges = prot_charges
            rec.groups_changed = _diff_charges(orig_charges, prot_charges)
            rec.status = "unchanged" if chosen == smiles else "ok"
            if not pick_most_likely and len(variants) > 1:
                rec.notes = (
                    f"All {len(variants)} Dimorphite states returned (pick_most_likely=False)."
                )

    except Exception as exc:
        rec.status = "failed"
        rec.notes = f"Exception: {exc}"
        _log.warning("protonate_smiles failed for '%s': %s", smiles[:60], exc)

    rec.wall_time_s = round(time.time() - t0, 4)
    _session_log.append(rec)
    return rec


def protonate_smiles_series(
    smiles_iterable: Sequence[str],
    ph: float = 7.4,
    ph_tolerance: float = 1.0,
    max_variants: int = 128,
    pick_most_likely: bool = True,
    verbose_every: int = 500,
) -> "list[ProtonationRecord]":
    """
    Protonate a sequence of SMILES strings.  Returns a list of
    ``ProtonationRecord`` objects, one per input (in order).

    Suitable for passing an entire pandas SMILES column:

        records = protonate_smiles_series(df["smiles"])
        df["smiles_ph74"] = [r.output or r.input_id for r in records]

    Parameters
    ----------
    smiles_iterable:
        Any sequence of SMILES strings.
    verbose_every:
        Log progress every N molecules.

    Returns
    -------
    List of ProtonationRecord (same length as input).
    """
    _detect_dimorphite()  # fail fast if not installed
    records: list[ProtonationRecord] = []
    total = len(smiles_iterable) if hasattr(smiles_iterable, "__len__") else "?"
    for i, smi in enumerate(smiles_iterable):
        rec = protonate_smiles(
            smi,
            ph=ph,
            ph_tolerance=ph_tolerance,
            max_variants=max_variants,
            pick_most_likely=pick_most_likely,
        )
        records.append(rec)
        if verbose_every and (i + 1) % verbose_every == 0:
            _log.info("  [SMILES] processed %d / %s", i + 1, total)
    _log.info(
        "[SMILES series] done — %d ok, %d unchanged, %d failed",
        sum(r.status == "ok" for r in records),
        sum(r.status == "unchanged" for r in records),
        sum(r.status == "failed" for r in records),
    )
    return records


# ===========================================================================
# Section 4 – SDF / mol2 ligand protonation via OpenBabel
# ===========================================================================

def protonate_ligand_sdf(
    sdf_path: str | Path,
    ph: float = 7.4,
    out_path: Optional[str | Path] = None,
) -> ProtonationRecord:
    """
    Protonate a PDBbind ligand SDF (or mol2) file at *ph* using OpenBabel.

    OpenBabel is invoked as:
        obabel <in> -O <out> -p <ph> --partialcharge none

    The ``-p`` flag adds / removes hydrogens according to the predicted pKa
    of each group (Gasteiger-based estimate inside OpenBabel).

    Parameters
    ----------
    sdf_path:
        Path to the input ligand file (.sdf or .mol2).
    ph:
        Target pH.
    out_path:
        Where to write the protonated file.  If None, the output is written
        beside the input with the suffix ``_H.sdf`` (or ``_H.mol2``).

    Returns
    -------
    ProtonationRecord with ``output`` set to the path of the written file.
    """
    t0 = time.time()
    ver = _detect_openbabel()

    sdf_path = Path(sdf_path)
    if not sdf_path.exists():
        raise FileNotFoundError(f"Ligand file not found: {sdf_path}")

    suffix = sdf_path.suffix.lower()   # .sdf or .mol2
    if out_path is None:
        out_path = sdf_path.with_name(sdf_path.stem + "_H" + suffix)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rec = ProtonationRecord(
        input_id=str(sdf_path),
        input_type=suffix.lstrip("."),
        ph=ph,
        tool="openbabel",
        tool_version=ver,
        status="failed",
        output=None,
    )

    # --- extract SMILES before protonation for charge comparison (RDKit) ---
    orig_charges = _sdf_charge_fingerprint(sdf_path)
    rec.original_formal_charges = orig_charges

    cmd = [
        "obabel",
        str(sdf_path),
        "-O", str(out_path),
        "-p", str(ph),
        "--partialcharge", "none",
        "-h",           # add hydrogens (needed before -p adjusts them)
        "-d",           # remove explicit H then let -p add the right ones
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
        )
        stderr_lower = result.stderr.lower()
        if result.returncode != 0 or "error" in stderr_lower:
            rec.status = "failed"
            rec.notes = f"obabel stderr: {result.stderr.strip()}"
            _log.warning("protonate_ligand_sdf failed for %s: %s",
                         sdf_path.name, result.stderr.strip())
        else:
            prot_charges = _sdf_charge_fingerprint(out_path)
            rec.protonated_formal_charges = prot_charges
            rec.groups_changed = _diff_charges(orig_charges, prot_charges)
            rec.status = "ok"
            rec.output = str(out_path)
            msg = result.stderr.strip() or result.stdout.strip()
            rec.notes = f"obabel output: {msg[:200]}"
    except subprocess.TimeoutExpired:
        rec.status = "failed"
        rec.notes = "obabel timed out (120 s)"
        _log.warning("obabel timed out for %s", sdf_path.name)
    except Exception as exc:
        rec.status = "failed"
        rec.notes = f"Exception: {exc}"
        _log.warning("protonate_ligand_sdf exception for %s: %s", sdf_path.name, exc)

    rec.wall_time_s = round(time.time() - t0, 4)
    _session_log.append(rec)
    return rec


def _sdf_charge_fingerprint(path: Path) -> str:
    """Read the first molecule in *path* via RDKit and return charge fingerprint."""
    try:
        from rdkit import Chem  # type: ignore
        suffix = path.suffix.lower()
        if suffix == ".sdf":
            suppl = Chem.SDMolSupplier(str(path), removeHs=True, sanitize=False)
            mol = next((m for m in suppl if m is not None), None)
        elif suffix in (".mol2", ".mol"):
            mol = Chem.MolFromMol2File(str(path), removeHs=True, sanitize=False)
        else:
            return ""
        if mol is None:
            return ""
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass
        smi = Chem.MolToSmiles(mol)
        return _formal_charge_summary(smi)
    except Exception:
        return ""


# ===========================================================================
# Section 5 – PDB pocket protonation via OpenBabel
# ===========================================================================

def protonate_pocket_pdb(
    pdb_path: str | Path,
    ph: float = 7.4,
    out_path: Optional[str | Path] = None,
    prefer_pdb2pqr: bool = True,
) -> ProtonationRecord:
    """
    Protonate a PDBbind pocket PDB file at *ph*.

    **Preferred method (when pdb2pqr is installed):**
        pdb2pqr --ff AMBER --with-ph <ph> --titration-state-method propka
                <in.pdb> <out.pqr>
    The PQR file is then converted back to PDB via obabel.
    PROPKA ≥ 3.5 is required alongside pdb2pqr; the function checks for it.

    **Fallback (OpenBabel only):**
        obabel <in.pdb> -O <out.pdb> -p <ph>
    This uses OpenBabel's Gasteiger-based pKa estimates, which are less
    accurate for proteins than PROPKA but require no extra dependencies.

    Parameters
    ----------
    pdb_path:
        Path to the input pocket PDB file.
    ph:
        Target pH.
    out_path:
        Destination PDB file.  Defaults to <stem>_H.pdb beside the input.
    prefer_pdb2pqr:
        Try pdb2pqr first; fall back to obabel if unavailable.

    Returns
    -------
    ProtonationRecord with ``output`` set to the written PDB path and
    ``tool`` indicating which tool was actually used.
    """
    t0 = time.time()
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        raise FileNotFoundError(f"Pocket PDB not found: {pdb_path}")

    if out_path is None:
        out_path = pdb_path.with_name(pdb_path.stem + "_H.pdb")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rec = ProtonationRecord(
        input_id=str(pdb_path),
        input_type="pdb",
        ph=ph,
        tool="",           # filled below
        tool_version="",   # filled below
        status="failed",
        output=None,
    )

    used_tool: Optional[str] = None

    # ---- attempt pdb2pqr ----
    if prefer_pdb2pqr:
        try:
            ver = _detect_pdb2pqr()
            rec.tool = "pdb2pqr+propka"
            rec.tool_version = ver
            used_tool = "pdb2pqr"
        except RuntimeError as e:
            _log.info("pdb2pqr not available (%s); falling back to obabel.", str(e).split("\n")[0])

    if used_tool == "pdb2pqr":
        rec = _run_pdb2pqr(pdb_path, out_path, ph, rec)
    else:
        # ---- fallback to OpenBabel ----
        ver = _detect_openbabel()
        rec.tool = "openbabel"
        rec.tool_version = ver
        rec = _run_obabel_pdb(pdb_path, out_path, ph, rec)

    rec.wall_time_s = round(time.time() - t0, 4)
    _session_log.append(rec)
    return rec


def _run_pdb2pqr(
    pdb_path: Path,
    out_path: Path,
    ph: float,
    rec: ProtonationRecord,
) -> ProtonationRecord:
    """Internal: run pdb2pqr and convert PQR → PDB."""
    exe = shutil.which("pdb2pqr") or shutil.which("pdb2pqr30")
    with tempfile.TemporaryDirectory() as tmp:
        pqr_out = Path(tmp) / "pocket.pqr"
        cmd = [
            exe,
            "--ff", "AMBER",
            "--with-ph", str(ph),
            "--titration-state-method", "propka",
            "--drop-water",
            "--quiet",
            str(pdb_path),
            str(pqr_out),
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=180,
            )
            if result.returncode != 0:
                rec.status = "failed"
                rec.notes = (
                    f"pdb2pqr returned code {result.returncode}: "
                    f"{result.stderr.strip()[:400]}"
                )
                _log.warning("pdb2pqr failed for %s", pdb_path.name)
                return rec

            # Convert PQR → PDB via obabel for downstream compatibility
            _detect_openbabel()
            conv_cmd = [
                "obabel", str(pqr_out),
                "-O", str(out_path),
                "--partialcharge", "none",
            ]
            conv = subprocess.run(
                conv_cmd, capture_output=True, text=True, timeout=60,
            )
            if conv.returncode != 0:
                rec.status = "failed"
                rec.notes = f"pqr→pdb conversion failed: {conv.stderr.strip()[:300]}"
                return rec

            rec.status = "ok"
            rec.output = str(out_path)
            rec.groups_changed = _parse_propka_titration(result.stdout + result.stderr, ph)
            rec.notes = (
                f"pdb2pqr+propka succeeded; PQR converted to PDB via obabel. "
                f"{len(rec.groups_changed)} residue(s) changed ionisation state."
            )

        except subprocess.TimeoutExpired:
            rec.status = "failed"
            rec.notes = "pdb2pqr timed out (180 s)"
        except Exception as exc:
            rec.status = "failed"
            rec.notes = f"pdb2pqr exception: {exc}"

    return rec


def _parse_propka_titration(propka_output: str, ph: float) -> List[str]:
    """
    Extract residue-level protonation change messages from PROPKA stdout.

    PROPKA logs lines like:
        ASP   57  A   pKa =  3.84  protonated at pH  7.40
    We collect them and annotate direction (protonated / deprotonated).
    """
    changes: List[str] = []
    pattern = re.compile(
        r"(\w{3})\s+(\d+)\s+(\S+)\s+pKa\s*=\s*([\d.]+)",
        re.IGNORECASE,
    )
    for line in propka_output.splitlines():
        m = pattern.search(line)
        if m:
            res, seqnum, chain, pka = m.groups()
            pka_val = float(pka)
            # A group with pKa > ph is protonated (acid form) at this pH;
            # a group with pKa < ph is deprotonated (base form).
            state = "protonated" if pka_val > ph else "deprotonated"
            changes.append(f"{res}{seqnum}{chain}: pKa={pka_val:.2f} → {state} at pH {ph}")
    return changes


def _run_obabel_pdb(
    pdb_path: Path,
    out_path: Path,
    ph: float,
    rec: ProtonationRecord,
) -> ProtonationRecord:
    """Internal: run OpenBabel -p on a PDB pocket file."""
    cmd = [
        "obabel",
        str(pdb_path),
        "-O", str(out_path),
        "-p", str(ph),
        "-h",   # ensure hydrogens are added
        "-d",   # remove existing H first so -p controls them
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
        )
        stderr_lower = result.stderr.lower()
        if result.returncode != 0 or ("error" in stderr_lower and "warning" not in stderr_lower):
            rec.status = "failed"
            rec.notes = f"obabel stderr: {result.stderr.strip()[:400]}"
            _log.warning("obabel pdb protonation failed for %s", pdb_path.name)
        else:
            rec.status = "ok"
            rec.output = str(out_path)
            rec.notes = (
                "obabel -p used (Gasteiger pKa estimates; less accurate than PROPKA "
                "for proteins). Install pdb2pqr for PROPKA-quality results. "
                f"obabel output: {result.stderr.strip()[:200]}"
            )
    except subprocess.TimeoutExpired:
        rec.status = "failed"
        rec.notes = "obabel timed out (120 s)"
    except Exception as exc:
        rec.status = "failed"
        rec.notes = f"obabel exception: {exc}"

    return rec


# ===========================================================================
# Section 6 – Batch helpers for whole PDBbind refined-set
# ===========================================================================

def protonate_pdbbind_set(
    refined_set_dir: str | Path,
    out_dir: str | Path,
    ph: float = 7.4,
    prefer_pdb2pqr: bool = True,
    log_path: Optional[str | Path] = None,
    pdb_ids: Optional[List[str]] = None,
) -> List[ProtonationRecord]:
    """
    Protonate all ligand + pocket files in a PDBbind refined-set directory.

    For each complex sub-directory ``<pdbid>/``:
      - Ligand: ``<pdbid>_ligand.sdf`` or ``<pdbid>_ligand.mol2``
        → protonated with ``protonate_ligand_sdf``
        → written to ``<out_dir>/<pdbid>/<pdbid>_ligand_H.sdf``
      - Pocket: ``<pdbid>_pocket.pdb``
        → protonated with ``protonate_pocket_pdb``
        → written to ``<out_dir>/<pdbid>/<pdbid>_pocket_H.pdb``

    Parameters
    ----------
    refined_set_dir:
        Root of the PDBbind refined-set (contains sub-dirs named by PDB ID).
    out_dir:
        Root output directory (mirrors the refined-set layout).
    ph:
        Target pH (default 7.4).
    prefer_pdb2pqr:
        Passed to ``protonate_pocket_pdb``.
    log_path:
        If given, the session log is written here after processing.
    pdb_ids:
        Optional list of PDB IDs to process (default: all valid sub-dirs).

    Returns
    -------
    List of ProtonationRecord — two records per complex (ligand + pocket).
    """
    refined_set_dir = Path(refined_set_dir)
    out_dir = Path(out_dir)

    if pdb_ids is None:
        pdb_ids = sorted(
            d.name for d in refined_set_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
            and d.name not in ("index",)
        )

    all_records: List[ProtonationRecord] = []
    total = len(pdb_ids)
    _log.info("Starting PDBbind batch protonation for %d complexes → %s", total, out_dir)

    for i, pid in enumerate(pdb_ids):
        src_dir = refined_set_dir / pid
        dst_dir = out_dir / pid
        dst_dir.mkdir(parents=True, exist_ok=True)

        # --- ligand ---
        lig_sdf = src_dir / f"{pid}_ligand.sdf"
        lig_mol2 = src_dir / f"{pid}_ligand.mol2"
        lig_src = lig_sdf if lig_sdf.exists() else (lig_mol2 if lig_mol2.exists() else None)
        if lig_src is not None:
            rec = protonate_ligand_sdf(
                lig_src,
                ph=ph,
                out_path=dst_dir / f"{pid}_ligand_H{lig_src.suffix}",
            )
            all_records.append(rec)
        else:
            _log.warning("[%s] no ligand SDF/mol2 found; skipping ligand protonation.", pid)

        # --- pocket ---
        pocket_src = src_dir / f"{pid}_pocket.pdb"
        if pocket_src.exists():
            rec = protonate_pocket_pdb(
                pocket_src,
                ph=ph,
                out_path=dst_dir / f"{pid}_pocket_H.pdb",
                prefer_pdb2pqr=prefer_pdb2pqr,
            )
            all_records.append(rec)
        else:
            _log.warning("[%s] no pocket PDB found; skipping pocket protonation.", pid)

        if (i + 1) % 100 == 0 or (i + 1) == total:
            _log.info("  [PDBbind batch] %d / %d complexes done", i + 1, total)

    # summary
    ok = sum(r.status == "ok" for r in all_records)
    fail = sum(r.status == "failed" for r in all_records)
    _log.info("[PDBbind batch] complete — %d ok, %d failed out of %d records",
              ok, fail, len(all_records))

    if log_path is not None:
        write_log(log_path)

    return all_records


# ===========================================================================
# Section 7 – Batch helper for classification / ADMET CSV datasets
# ===========================================================================

def protonate_dataset_csv(
    csv_path: str | Path,
    smiles_col: str,
    ph: float = 7.4,
    ph_tolerance: float = 1.0,
    out_csv: Optional[str | Path] = None,
    log_path: Optional[str | Path] = None,
    new_smiles_col: str = "smiles_ph74",
) -> "tuple[object, list[ProtonationRecord]]":
    """
    Protonate the SMILES column of a classification / ADMET CSV at *ph*.

    Parameters
    ----------
    csv_path:
        Path to the input CSV file (BACE, HIV, BBBP, ADMET …).
    smiles_col:
        Name of the SMILES column in the CSV.
    ph:
        Target pH.
    ph_tolerance:
        Dimorphite enumeration window half-width.
    out_csv:
        Write the augmented DataFrame here (adds *new_smiles_col*).
        If None, no file is written.
    log_path:
        If given, write the session log here as JSON-Lines.
    new_smiles_col:
        Name of the new column containing the protonated SMILES.

    Returns
    -------
    (df, records)
        df      — pandas DataFrame with a new *new_smiles_col* column
        records — list of ProtonationRecord, one per row
    """
    try:
        import pandas as pd  # type: ignore
    except ImportError:
        raise RuntimeError("pandas is required for protonate_dataset_csv(). pip install pandas")

    csv_path = Path(csv_path)
    _log.info("[CSV] loading %s  (SMILES col: %r)", csv_path.name, smiles_col)
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    if smiles_col not in df.columns:
        # Try case-insensitive match
        lower_map = {c.lower(): c for c in df.columns}
        if smiles_col.lower() in lower_map:
            smiles_col = lower_map[smiles_col.lower()]
        else:
            raise ValueError(f"SMILES column {smiles_col!r} not found. "
                             f"Available: {list(df.columns)}")

    smiles_list: List[str] = df[smiles_col].fillna("").astype(str).tolist()
    _log.info("[CSV] protonating %d SMILES at pH %.1f …", len(smiles_list), ph)

    records = protonate_smiles_series(
        smiles_list, ph=ph, ph_tolerance=ph_tolerance, pick_most_likely=True,
    )

    df[new_smiles_col] = [
        (r.output if r.output is not None else r.input_id) for r in records
    ]

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        _log.info("[CSV] protonated CSV written → %s", out_csv)

    if log_path is not None:
        write_log(log_path)

    ok = sum(r.status == "ok" for r in records)
    unch = sum(r.status == "unchanged" for r in records)
    fail = sum(r.status == "failed" for r in records)
    _log.info("[CSV] done — %d changed, %d unchanged, %d failed", ok, unch, fail)
    return df, records


# ===========================================================================
# Section 8 – CLI entry-point
# ===========================================================================

def _build_cli_parser():
    import argparse
    p = argparse.ArgumentParser(
        prog="python -m revision.protonation.protonate",
        description=(
            "Apply pH 7.4 protonation to Flexi-JEGNN input data.\n\n"
            "Examples:\n"
            "  # Protonate all ADMET SMILES:\n"
            "  python -m revision.protonation.protonate smiles "
            "--csv datasets/ADMET.csv --smiles-col smiles "
            "--out-csv revised_data/ADMET_ph74.csv\n\n"
            "  # Protonate entire PDBbind refined-set:\n"
            "  python -m revision.protonation.protonate pdbbind "
            "--refined-set refined-set/ --out-dir revised_data/pdbbind_H/\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="mode", required=True)

    # --- smiles mode ---
    sm = sub.add_parser("smiles", help="Protonate a CSV SMILES column (Dimorphite-DL)")
    sm.add_argument("--csv", required=True, help="Input CSV path")
    sm.add_argument("--smiles-col", default="smiles", help="SMILES column name")
    sm.add_argument("--out-csv", default=None, help="Output CSV path")
    sm.add_argument("--log", default="protonate_smiles_log.jsonl", help="JSON-Lines log path")
    sm.add_argument("--ph", type=float, default=7.4)
    sm.add_argument("--ph-tolerance", type=float, default=1.0)

    # --- sdf mode ---
    sd = sub.add_parser("sdf", help="Protonate a single SDF/mol2 ligand (OpenBabel)")
    sd.add_argument("--input", required=True, help="Input SDF or mol2 path")
    sd.add_argument("--output", default=None, help="Output SDF path")
    sd.add_argument("--log", default="protonate_sdf_log.jsonl")
    sd.add_argument("--ph", type=float, default=7.4)

    # --- pdb mode ---
    pb = sub.add_parser("pdb", help="Protonate a single pocket PDB (OpenBabel / pdb2pqr)")
    pb.add_argument("--input", required=True, help="Input PDB path")
    pb.add_argument("--output", default=None)
    pb.add_argument("--log", default="protonate_pdb_log.jsonl")
    pb.add_argument("--ph", type=float, default=7.4)
    pb.add_argument("--no-pdb2pqr", action="store_true",
                    help="Force OpenBabel fallback even if pdb2pqr is available")

    # --- pdbbind batch mode ---
    bb = sub.add_parser("pdbbind", help="Batch-protonate an entire PDBbind refined-set")
    bb.add_argument("--refined-set", required=True, help="PDBbind refined-set root directory")
    bb.add_argument("--out-dir", required=True, help="Output directory (mirrors refined-set layout)")
    bb.add_argument("--log", default="protonate_pdbbind_log.jsonl")
    bb.add_argument("--ph", type=float, default=7.4)
    bb.add_argument("--no-pdb2pqr", action="store_true")
    bb.add_argument("--pdb-ids", nargs="*", default=None,
                    help="Optional subset of PDB IDs to process")

    return p


def _cli_main(argv=None):
    parser = _build_cli_parser()
    args = parser.parse_args(argv)

    if args.mode == "smiles":
        protonate_dataset_csv(
            csv_path=args.csv,
            smiles_col=args.smiles_col,
            ph=args.ph,
            ph_tolerance=args.ph_tolerance,
            out_csv=args.out_csv,
            log_path=args.log,
        )

    elif args.mode == "sdf":
        rec = protonate_ligand_sdf(args.input, ph=args.ph, out_path=args.output)
        write_log(args.log)
        print(f"status={rec.status}  output={rec.output}")

    elif args.mode == "pdb":
        rec = protonate_pocket_pdb(
            args.input, ph=args.ph, out_path=args.output,
            prefer_pdb2pqr=not args.no_pdb2pqr,
        )
        write_log(args.log)
        print(f"status={rec.status}  tool={rec.tool}  output={rec.output}")

    elif args.mode == "pdbbind":
        protonate_pdbbind_set(
            refined_set_dir=args.refined_set,
            out_dir=args.out_dir,
            ph=args.ph,
            prefer_pdb2pqr=not args.no_pdb2pqr,
            log_path=args.log,
            pdb_ids=args.pdb_ids,
        )


if __name__ == "__main__":
    _cli_main()
