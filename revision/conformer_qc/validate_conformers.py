r"""
revision/conformer_qc/validate_conformers.py
=============================================
3D conformer quality-assurance pipeline for the Flexi-JEGNN revision.

Addresses reviewer concern:
    "Provide a detailed analysis of the 3D structure generation process for
    each set to confirm the correctness of the resulting conformers (e.g.,
    absence of atomic clashes; consistency of the generated conformers'
    stereochemistry with the original structures, which can be verified
    using InChI).  Please host the datasets containing the generated 3D
    conformers on GitHub."

Two main checks are implemented for every generated conformer:

  1. Clash detection
     ---------------
     A clash occurs when any pair of non-bonded heavy atoms is closer
     than  CLASH_SCALE * (r_vdw_i + r_vdw_j).  CLASH_SCALE defaults to
     0.75, matching the tolerance used by MolProbity / CCDC CSD.
     Bonded pairs (distance <= MAX_BOND_A) are excluded from the check
     because their proximity is physically expected.

  2. Stereochemistry consistency (InChI /t layer)
     -----------------------------------------------
     The input SMILES may carry explicit stereo annotations (@, @@, /,
     \).  After ETKDGv3 embedding, the conformer is written back to SMILES
     via RDKit's AssignStereochemistryFromConformer and the InChI /t (stereo)
     layer is compared between the original molecule and the conformer.
     A mismatch means the embedding inverted or lost a stereocentre.

In addition the module:
  * Reports the embedding failure rate (EmbedMolecule returning -1).
  * Saves passing conformers as SDF files for hosting and inspection.
  * Accumulates per-molecule QC records in a JSON-Lines log.

Nothing in experiments/ is imported or modified.

Usage (quick-start)
-------------------
    from revision.conformer_qc.validate_conformers import (
        validate_smiles_dataset,
        ConformerQCRecord,
        write_qc_log,
    )

    records = validate_smiles_dataset(
        smiles_list=df["smiles"].tolist(),
        ids=df["mol_id"].tolist(),
        out_sdf="revised_data/conformers/BACE_conformers.sdf",
        log_path="revision/conformer_qc/logs/BACE_qc.jsonl",
    )

    # Summary statistics
    from revision.conformer_qc.validate_conformers import qc_summary
    print(qc_summary(records))

CLI
---
    python -m revision.conformer_qc.validate_conformers \\
        --csv datasets/BACE.csv --smiles-col smiles --id-col mol_id \\
        --out-sdf revised_data/conformers/BACE.sdf \\
        --log revision/conformer_qc/logs/BACE_qc.jsonl
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] conformer_qc: %(message)s",
    level=logging.INFO,
)
_log = logging.getLogger("conformer_qc")

# ---------------------------------------------------------------------------
# RDKit — required for all meaningful operations; checked lazily so that the
# module can be *imported* (and tests collected) even when RDKit is absent.
# Functions that need RDKit call _require_rdkit() at their start.
# ---------------------------------------------------------------------------
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import inchi as rdInchi
    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False
    Chem = None          # type: ignore
    AllChem = None       # type: ignore
    rdInchi = None       # type: ignore


def _require_rdkit():
    """Raise a clear ImportError if RDKit is not installed."""
    if not _HAS_RDKIT:
        raise ImportError(
            "RDKit is required for conformer QC.\n"
            "Install with:  conda install -c conda-forge rdkit"
        )

# ---------------------------------------------------------------------------
# Van der Waals radii (Å) — Bondi 1964 values used by most clash detectors
# Keys are element symbols; fallback is 1.70 (carbon radius).
# ---------------------------------------------------------------------------
VDW_RADII: Dict[str, float] = {
    "H":  1.20, "He": 1.40,
    "Li": 1.82, "Be": 1.53, "B":  1.92, "C":  1.70, "N":  1.55,
    "O":  1.52, "F":  1.47, "Ne": 1.54, "Na": 2.27, "Mg": 1.73,
    "Al": 1.84, "Si": 2.10, "P":  1.80, "S":  1.80, "Cl": 1.75,
    "Ar": 1.88, "K":  2.75, "Ca": 2.31, "Br": 1.85, "I":  1.98,
}
_VDW_FALLBACK = 1.70

# Pairs closer than MAX_BOND_A are treated as bonded; excluded from clash check
MAX_BOND_A = 1.70   # Å  (longest expected heavy-atom bond is C=O ~1.22 Å;
                    #     we use 1.70 to also catch coordinatively bonded atoms)

# Default clash scale: pair flagged when d < CLASH_SCALE * (r_i + r_j)
CLASH_SCALE_DEFAULT = 0.75

# ETKDGv3 parameters
ETKDG_RANDOM_SEED = 42
MMFF_MAX_ITERS = 200


# ===========================================================================
# ConformerQCRecord — one per molecule
# ===========================================================================

@dataclass
class ConformerQCRecord:
    """Quality-assurance result for a single molecule's conformer."""

    # --- identification ---
    mol_id: str                # user-supplied identifier or SMILES itself
    smiles_input: str          # original input SMILES
    n_heavy_atoms: int = 0

    # --- embedding outcome ---
    embed_status: str = "not_run"
    # "ok" | "failed" | "no_stereo_input" | "skipped"
    mmff_converged: bool = False

    # --- clash check ---
    n_clash_pairs: int = 0
    clash_pairs: List[str] = field(default_factory=list)
    # Each entry: "atom_i(sym_i)--atom_j(sym_j): d=X.XXA vdw_sum=Y.YYA"
    clash_pass: bool = False    # True when n_clash_pairs == 0

    # --- stereo check ---
    inchi_input: str = ""       # /t layer of the input SMILES InChI
    inchi_conformer: str = ""   # /t layer of the conformer InChI
    stereo_pass: bool = False
    stereo_note: str = ""
    # "match" | "mismatch" | "no_stereo_input" | "no_stereo_conformer" | "both_flat"

    # --- conformer saved? ---
    sdf_written: bool = False

    # --- timing ---
    wall_time_s: float = 0.0
    timestamp: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )

    @property
    def overall_pass(self) -> bool:
        """True when the conformer passes both clash and stereo checks."""
        return self.clash_pass and self.stereo_pass


# ===========================================================================
# Session log
# ===========================================================================

_qc_log: List[ConformerQCRecord] = []


def write_qc_log(path: str | Path, mode: str = "w") -> Path:
    """Write accumulated QC records to *path* as JSON-Lines."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open(mode, encoding="utf-8") as fh:
        for rec in _qc_log:
            fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
    _log.info("QC log written → %s  (%d records)", out, len(_qc_log))
    return out


def clear_qc_log() -> None:
    _qc_log.clear()


# ===========================================================================
# Section 1 — InChI stereo-layer helpers
# ===========================================================================

def _stereo_layer(mol: Chem.Mol) -> str:
    """
    Return the /t (tetrahedral stereo) layer of the InChI for *mol*, or ''
    if the molecule has no defined stereocentres.

    The /t layer lists stereocentre atom numbers and their parity, e.g.
    '/t2-,5+'.  Two molecules with identical /t layers have matching
    stereo configurations at every enumerated centre.
    """
    try:
        inchi_str = rdInchi.MolToInchi(mol, options="")
        if inchi_str is None:
            return ""
        for layer in inchi_str.split("/"):
            if layer.startswith("t"):
                return "/" + layer
        return ""   # no /t layer → no defined stereocentres
    except Exception:
        return ""


def _inchi_stereo_from_smiles(smiles: str) -> str:
    """Return the /t layer for a SMILES string, or '' on parse failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return _stereo_layer(mol)


def _inchi_stereo_from_conformer(mol_with_conf: Chem.Mol) -> str:
    """
    Return the /t layer extracted from the conformer's 3D coordinates.

    RDKit's AssignStereochemistryFromConformer reassigns stereo tags from
    the 3D positions; this is then encoded in the InChI /t layer.
    """
    try:
        mol_copy = Chem.RWMol(mol_with_conf)
        Chem.AssignStereochemistryFromConformer(mol_copy, mol_copy.GetConformer())
        return _stereo_layer(mol_copy)
    except Exception:
        return ""


def check_stereo_consistency(
    input_smiles: str,
    mol_with_conf: "Chem.Mol",
) -> tuple[str, str, str]:
    _require_rdkit()
    """
    Compare the /t InChI stereo layer of *input_smiles* with that derived
    from the 3D conformer in *mol_with_conf*.

    Returns
    -------
    (inchi_input, inchi_conformer, stereo_note)
    where stereo_note is one of:
      "both_flat"            — neither input nor conformer has stereocentres
      "no_stereo_input"      — input SMILES has no explicit stereo; can't verify
      "no_stereo_conformer"  — conformer lost all stereo (assignment failure)
      "match"                — /t layers identical
      "mismatch"             — /t layers differ (stereocentre inverted / lost)
    """
    inchi_in = _inchi_stereo_from_smiles(input_smiles)
    inchi_conf = _inchi_stereo_from_conformer(mol_with_conf)

    if not inchi_in and not inchi_conf:
        return inchi_in, inchi_conf, "both_flat"
    if not inchi_in:
        return inchi_in, inchi_conf, "no_stereo_input"
    if not inchi_conf:
        return inchi_in, inchi_conf, "no_stereo_conformer"
    if inchi_in == inchi_conf:
        return inchi_in, inchi_conf, "match"
    return inchi_in, inchi_conf, "mismatch"


# ===========================================================================
# Section 2 — Clash detection
# ===========================================================================

def _vdw(symbol: str) -> float:
    return VDW_RADII.get(symbol, _VDW_FALLBACK)


def check_clashes(
    mol_with_conf: "Chem.Mol",
    clash_scale: float = CLASH_SCALE_DEFAULT,
) -> tuple[int, List[str]]:
    _require_rdkit()
    """
    Detect atomic clashes in the heavy-atom skeleton of *mol_with_conf*.

    A clash is reported when:
        d(i, j) < clash_scale * (vdw_i + vdw_j)
    AND the pair is NOT bonded (d > MAX_BOND_A).

    Only heavy atoms are checked (H atoms excluded because ETKDGv3 adds
    idealised H positions that occasionally violate vdW radii for polar H).

    Returns
    -------
    (n_clash_pairs, clash_descriptions)
    where each description is a human-readable string.
    """
    conf = mol_with_conf.GetConformer()

    # Collect heavy-atom indices and their 3D positions
    heavy_idxs = [
        a.GetIdx()
        for a in mol_with_conf.GetAtoms()
        if a.GetAtomicNum() > 1
    ]
    if len(heavy_idxs) < 2:
        return 0, []

    positions = np.array([
        [conf.GetAtomPosition(i).x,
         conf.GetAtomPosition(i).y,
         conf.GetAtomPosition(i).z]
        for i in heavy_idxs
    ], dtype=np.float64)

    symbols = [
        mol_with_conf.GetAtomWithIdx(i).GetSymbol()
        for i in heavy_idxs
    ]

    n = len(heavy_idxs)
    clash_descriptions: List[str] = []

    for ii in range(n):
        for jj in range(ii + 1, n):
            diff = positions[ii] - positions[jj]
            d = float(np.linalg.norm(diff))

            # Skip bonded pairs
            if d <= MAX_BOND_A:
                continue

            vdw_sum = _vdw(symbols[ii]) + _vdw(symbols[jj])
            threshold = clash_scale * vdw_sum

            if d < threshold:
                ai = heavy_idxs[ii]
                aj = heavy_idxs[jj]
                clash_descriptions.append(
                    f"atom{ai}({symbols[ii]})--atom{aj}({symbols[jj]}): "
                    f"d={d:.3f}A  vdw_sum={vdw_sum:.2f}A  "
                    f"threshold={threshold:.2f}A"
                )

    return len(clash_descriptions), clash_descriptions


# ===========================================================================
# Section 3 — Single-molecule ETKDGv3 conformer generation + QC
# ===========================================================================

def generate_and_validate_conformer(
    smiles: str,
    mol_id: str = "",
    clash_scale: float = CLASH_SCALE_DEFAULT,
    random_seed: int = ETKDG_RANDOM_SEED,
    mmff_max_iters: int = MMFF_MAX_ITERS,
) -> tuple[Optional[Chem.Mol], ConformerQCRecord]:
    """
    Generate one ETKDGv3 conformer for *smiles* and run full QC.

    Mirrors the exact procedure used in experiments/classification.py
    (lines 157–174) and experiments/qm9.py (lines 151–168) so that the
    validation results reflect the actual experimental data.

    Parameters
    ----------
    smiles:
        Input SMILES string.
    mol_id:
        Human-readable identifier (CAS, InChIKey, row index, etc.).
    clash_scale:
        Fraction of summed vdW radii below which a pair is a clash.
    random_seed:
        Passed to ETKDGv3 for reproducibility.
    mmff_max_iters:
        Maximum MMFF94 optimisation iterations.

    Returns
    -------
    (mol_with_conformer_or_None, ConformerQCRecord)
    """
    _require_rdkit()
    t0 = time.time()
    rec = ConformerQCRecord(
        mol_id=mol_id or smiles[:80],
        smiles_input=smiles,
    )

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() < 2:
        rec.embed_status = "skipped"
        rec.wall_time_s = round(time.time() - t0, 4)
        _qc_log.append(rec)
        return None, rec

    rec.n_heavy_atoms = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)

    # --- ETKDGv3 embedding (exactly as in experiments/) ---
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    embed_result = AllChem.EmbedMolecule(mol_h, params)

    if embed_result == -1:
        rec.embed_status = "failed"
        rec.clash_pass = False
        rec.stereo_pass = False
        rec.stereo_note = "embed_failed"
        rec.wall_time_s = round(time.time() - t0, 4)
        _qc_log.append(rec)
        return None, rec

    rec.embed_status = "ok"

    # --- MMFF94 optimisation ---
    try:
        result_flag = AllChem.MMFFOptimizeMolecule(mol_h, maxIters=mmff_max_iters)
        rec.mmff_converged = (result_flag == 0)   # 0 = converged, 1 = not converged
    except Exception:
        rec.mmff_converged = False

    # --- Clash check (heavy atoms only, on the H-stripped molecule) ---
    mol_no_h = Chem.RemoveHs(mol_h)
    n_clashes, clash_descs = check_clashes(mol_no_h, clash_scale=clash_scale)
    rec.n_clash_pairs = n_clashes
    rec.clash_pairs = clash_descs
    rec.clash_pass = (n_clashes == 0)

    # --- Stereo consistency (compare input SMILES /t layer vs conformer) ---
    inchi_in, inchi_conf, stereo_note = check_stereo_consistency(smiles, mol_no_h)
    rec.inchi_input = inchi_in
    rec.inchi_conformer = inchi_conf
    rec.stereo_note = stereo_note
    # Stereo passes if: no input stereo to check, layers match, or molecule is flat
    rec.stereo_pass = stereo_note in ("match", "no_stereo_input", "both_flat")

    rec.wall_time_s = round(time.time() - t0, 4)
    _qc_log.append(rec)
    return mol_no_h, rec


# ===========================================================================
# Section 4 — Dataset-level validation
# ===========================================================================

def validate_smiles_dataset(
    smiles_list: Sequence[str],
    ids: Optional[Sequence[str]] = None,
    out_sdf: Optional[str | Path] = None,
    log_path: Optional[str | Path] = None,
    clash_scale: float = CLASH_SCALE_DEFAULT,
    random_seed: int = ETKDG_RANDOM_SEED,
    verbose_every: int = 500,
    save_failed: bool = False,
) -> List[ConformerQCRecord]:
    """
    Generate and validate ETKDGv3 conformers for an entire SMILES dataset.

    Parameters
    ----------
    smiles_list:
        Sequence of SMILES strings (one per molecule).
    ids:
        Optional sequence of molecule identifiers (same length as smiles_list).
        If None, the index is used.
    out_sdf:
        If given, all conformers that *pass* both QC checks are written to
        this SDF file for hosting / inspection.  Use ``save_failed=True``
        to also include failed / clashing conformers (tagged with a
        ``<QC_PASS>`` SD property).
    log_path:
        JSON-Lines log path; written at completion if provided.
    clash_scale:
        Passed to check_clashes().
    random_seed:
        Passed to ETKDGv3.
    verbose_every:
        Log progress every N molecules.
    save_failed:
        Include failing conformers in the SDF (tagged with QC_PASS=0).

    Returns
    -------
    List[ConformerQCRecord] — one record per input SMILES, in order.
    """
    n_total = len(smiles_list)
    if ids is None:
        ids = [str(i) for i in range(n_total)]

    _log.info("Starting conformer QC for %d molecules …", n_total)

    writer: Optional[Chem.SDWriter] = None
    if out_sdf is not None:
        out_sdf = Path(out_sdf)
        out_sdf.parent.mkdir(parents=True, exist_ok=True)
        writer = Chem.SDWriter(str(out_sdf))

    records: List[ConformerQCRecord] = []

    for i, (smi, mol_id) in enumerate(zip(smiles_list, ids)):
        mol, rec = generate_and_validate_conformer(
            smi,
            mol_id=str(mol_id),
            clash_scale=clash_scale,
            random_seed=random_seed,
        )

        # Write to SDF
        if writer is not None and mol is not None:
            if rec.overall_pass or save_failed:
                mol.SetProp("_Name", str(mol_id))
                mol.SetProp("QC_PASS", "1" if rec.overall_pass else "0")
                mol.SetProp("CLASH_PAIRS", str(rec.n_clash_pairs))
                mol.SetProp("STEREO_NOTE", rec.stereo_note)
                mol.SetProp("SMILES_INPUT", smi)
                writer.write(mol)
                rec.sdf_written = True

        records.append(rec)

        if verbose_every and (i + 1) % verbose_every == 0:
            _passed = sum(r.overall_pass for r in records)
            _log.info(
                "  [QC] %d / %d  |  pass=%d  fail=%d",
                i + 1, n_total, _passed, (i + 1) - _passed,
            )

    if writer is not None:
        writer.close()
        _log.info("Conformers written → %s", out_sdf)

    if log_path is not None:
        write_qc_log(log_path)

    _log.info("[QC] finished — %s", qc_summary(records))
    return records


# ===========================================================================
# Section 5 — Summary statistics
# ===========================================================================

def qc_summary(records: List[ConformerQCRecord]) -> str:
    """Return a compact human-readable summary string."""
    total = len(records)
    if total == 0:
        return "No records."

    embed_ok    = sum(r.embed_status == "ok"     for r in records)
    embed_fail  = sum(r.embed_status == "failed" for r in records)
    embed_skip  = sum(r.embed_status == "skipped" for r in records)

    clash_pass  = sum(r.clash_pass   for r in records if r.embed_status == "ok")
    clash_fail  = sum(not r.clash_pass for r in records if r.embed_status == "ok")

    stereo_counts: Dict[str, int] = {}
    for r in records:
        stereo_counts[r.stereo_note] = stereo_counts.get(r.stereo_note, 0) + 1

    overall_pass = sum(r.overall_pass for r in records)

    lines = [
        f"Total molecules          : {total}",
        f"  Embed OK               : {embed_ok}  ({100*embed_ok/total:.1f}%)",
        f"  Embed FAILED           : {embed_fail}  ({100*embed_fail/total:.1f}%)",
        f"  Skipped (bad SMILES)   : {embed_skip}",
        f"Clash check (embedded):",
        f"  PASS (no clashes)      : {clash_pass}",
        f"  FAIL (>=1 clash pair)  : {clash_fail}",
        f"Stereo check:",
    ]
    for note, cnt in sorted(stereo_counts.items()):
        lines.append(f"  {note:<24}: {cnt}")
    lines.append(f"Overall QC PASS          : {overall_pass}  ({100*overall_pass/total:.1f}%)")
    return "\n".join(lines)


def qc_summary_dict(records: List[ConformerQCRecord]) -> dict:
    """Return summary statistics as a plain dict (for JSON serialisation)."""
    total = len(records)
    if total == 0:
        return {"total": 0}

    stereo_counts: Dict[str, int] = {}
    for r in records:
        stereo_counts[r.stereo_note] = stereo_counts.get(r.stereo_note, 0) + 1

    return {
        "total": total,
        "embed_ok":    sum(r.embed_status == "ok"      for r in records),
        "embed_failed":sum(r.embed_status == "failed"  for r in records),
        "embed_skipped":sum(r.embed_status == "skipped" for r in records),
        "clash_pass":  sum(r.clash_pass  for r in records if r.embed_status == "ok"),
        "clash_fail":  sum(not r.clash_pass for r in records if r.embed_status == "ok"),
        "stereo_breakdown": stereo_counts,
        "overall_pass": sum(r.overall_pass for r in records),
        "overall_pass_pct": round(100 * sum(r.overall_pass for r in records) / total, 2),
    }


# ===========================================================================
# Section 6 — CLI
# ===========================================================================

def _build_parser():
    import argparse
    p = argparse.ArgumentParser(
        prog="python -m revision.conformer_qc.validate_conformers",
        description=(
            "Generate ETKDGv3 conformers and run clash + stereo QC.\n\n"
            "Example:\n"
            "  python -m revision.conformer_qc.validate_conformers \\\n"
            "    --csv datasets/BACE.csv --smiles-col smiles --id-col mol_id \\\n"
            "    --out-sdf revised_data/conformers/BACE.sdf \\\n"
            "    --log revision/conformer_qc/logs/BACE_qc.jsonl\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv",        required=True, help="Input CSV path")
    p.add_argument("--smiles-col", default="smiles", help="SMILES column name")
    p.add_argument("--id-col",     default=None,  help="Molecule ID column (optional)")
    p.add_argument("--out-sdf",    default=None,  help="Output SDF for passing conformers")
    p.add_argument("--log",        default="conformer_qc.jsonl", help="JSON-Lines log")
    p.add_argument("--clash-scale", type=float, default=CLASH_SCALE_DEFAULT,
                   help=f"Clash fraction of vdW sum (default {CLASH_SCALE_DEFAULT})")
    p.add_argument("--seed",       type=int, default=ETKDG_RANDOM_SEED,
                   help="ETKDGv3 random seed")
    p.add_argument("--save-failed", action="store_true",
                   help="Also write failed conformers to SDF (tagged QC_PASS=0)")
    p.add_argument("--summary-json", default=None,
                   help="Write summary statistics as JSON to this path")
    return p


def _cli_main(argv=None):
    try:
        import pandas as pd
    except ImportError:
        print("pandas required: pip install pandas", file=sys.stderr)
        sys.exit(1)

    args = _build_parser().parse_args(argv)

    df = pd.read_csv(args.csv)
    df.columns = [c.strip() for c in df.columns]

    smiles_col = args.smiles_col
    if smiles_col not in df.columns:
        lower_map = {c.lower(): c for c in df.columns}
        if smiles_col.lower() in lower_map:
            smiles_col = lower_map[smiles_col.lower()]
        else:
            print(f"ERROR: SMILES column {smiles_col!r} not found.", file=sys.stderr)
            sys.exit(1)

    smiles_list = df[smiles_col].fillna("").astype(str).tolist()
    ids = (
        df[args.id_col].astype(str).tolist()
        if args.id_col and args.id_col in df.columns
        else None
    )

    records = validate_smiles_dataset(
        smiles_list=smiles_list,
        ids=ids,
        out_sdf=args.out_sdf,
        log_path=args.log,
        clash_scale=args.clash_scale,
        random_seed=args.seed,
        save_failed=args.save_failed,
    )

    print(qc_summary(records))

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(qc_summary_dict(records), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Summary JSON written → {summary_path}")


if __name__ == "__main__":
    _cli_main()
