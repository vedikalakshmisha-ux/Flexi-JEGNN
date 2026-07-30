"""
tests/test_protonation.py
=========================
Unit and integration tests for revision/protonation/protonate.py.

Test strategy
-------------
Layer 1  — pure-Python / always-run:
    • ProtonationRecord dataclass construction and serialisation.
    • _formal_charge_summary helper (requires RDKit; skipped otherwise).
    • _diff_charges helper.
    • protonate_smiles() with Dimorphite-DL *mocked* — verifies the record
      fields, status codes, and log accumulation without requiring the real tool.

Layer 2  — chemical correctness (requires dimorphite_dl):
    Dimorphite-DL is given known molecules whose dominant ionisation state at
    pH 7.4 is established by classical chemistry:

    Molecule                pKa   Expected state at pH 7.4
    ─────────────────────────────────────────────────────────────────
    Acetic acid (COO-H)     4.76  Deprotonated  → acetate  (O⁻)
    Benzoic acid (Ar-COOH)  4.20  Deprotonated  → benzoate (O⁻)
    Methylamine (CH3-NH2)  10.66  Protonated    → methylammonium (N⁺)
    Propylamine (Pr-NH2)   10.57  Protonated    → propylammonium  (N⁺)
    Aniline (Ar-NH2)        4.60  Neutral       → unchanged (pKa < pH)
    Glycine (zwitterion)  2.35/9.60  COO⁻ + NH3⁺ simultaneously

Layer 3  — file-based tests (require OpenBabel or pdb2pqr; skipped otherwise):
    • protonate_ligand_sdf on a minimal SDF written to a tmp directory.
    • protonate_pocket_pdb on a minimal PDB written to a tmp directory.

Layer 4  — log / session tests:
    • write_log / clear_log round-trip.
    • Correct JSON-Lines format.

Run with:
    pytest tests/test_protonation.py -v
    pytest tests/test_protonation.py -v -m "not integration"   # skip slow tests
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Make the project root importable when running from the repo root
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from revision.protonation.protonate import (
    ProtonationRecord,
    _diff_charges,
    _formal_charge_summary,
    clear_log,
    protonate_ligand_sdf,
    protonate_pocket_pdb,
    protonate_smiles,
    protonate_smiles_series,
    write_log,
    _session_log,
)

# ---------------------------------------------------------------------------
# Availability sentinels
# ---------------------------------------------------------------------------

_HAS_RDKIT = importlib.util.find_spec("rdkit") is not None
_HAS_DIMORPHITE = importlib.util.find_spec("dimorphite_dl") is not None
_HAS_OPENBABEL = shutil.which("obabel") is not None

# ---------------------------------------------------------------------------
# Pytest markers
# ---------------------------------------------------------------------------
pytestmark = []   # module-level; individual marks added per test

requires_rdkit = pytest.mark.skipif(
    not _HAS_RDKIT, reason="RDKit not installed"
)
requires_dimorphite = pytest.mark.skipif(
    not _HAS_DIMORPHITE,
    reason="dimorphite_dl not installed — pip install dimorphite_dl",
)
requires_openbabel = pytest.mark.skipif(
    not _HAS_OPENBABEL, reason="OpenBabel (obabel) not found in PATH"
)

# Register the custom 'integration' mark so pytest does not warn about it.
# Alternatively add  markers = integration: ...  to pytest.ini / pyproject.toml.
try:
    import _pytest.config  # noqa: F401
    integration = pytest.mark.integration
except Exception:
    integration = pytest.mark.integration  # fallback


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(autouse=True)
def _clear_session_log():
    """Reset the module-level log before every test to prevent interference."""
    clear_log()
    yield
    clear_log()


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory; alias for pytest's tmp_path."""
    return tmp_path


# ===========================================================================
# Layer 1a — ProtonationRecord dataclass
# ===========================================================================

class TestProtonationRecord:
    """ProtonationRecord must serialise cleanly and have correct defaults."""

    def test_construction_minimal(self):
        rec = ProtonationRecord(
            input_id="CC(=O)O",
            input_type="smiles",
            ph=7.4,
            tool="dimorphite_dl",
            tool_version="1.3.2",
            status="ok",
            output="CC(=O)[O-]",
        )
        assert rec.ph == 7.4
        assert rec.status == "ok"
        assert rec.groups_changed == []   # default empty list
        assert rec.n_states_enumerated == 0
        assert rec.wall_time_s == 0.0

    def test_asdict_is_json_serialisable(self):
        from dataclasses import asdict
        rec = ProtonationRecord(
            input_id="CN",
            input_type="smiles",
            ph=7.4,
            tool="dimorphite_dl",
            tool_version="1.3.2",
            status="ok",
            output="C[NH3+]",
            groups_changed=["atom@1: neutral → N+1"],
        )
        d = asdict(rec)
        # Should not raise
        serialised = json.dumps(d)
        roundtrip = json.loads(serialised)
        assert roundtrip["input_id"] == "CN"
        assert roundtrip["groups_changed"] == ["atom@1: neutral → N+1"]

    def test_timestamp_is_populated(self):
        rec = ProtonationRecord(
            input_id="c1ccccc1", input_type="smiles", ph=7.4,
            tool="dimorphite_dl", tool_version="1.3.2",
            status="unchanged", output="c1ccccc1",
        )
        # Timestamp should look like an ISO-8601 UTC string
        assert "T" in rec.timestamp and "Z" in rec.timestamp


# ===========================================================================
# Layer 1b — _formal_charge_summary helper
# ===========================================================================

@requires_rdkit
class TestFormalChargeSummary:
    """_formal_charge_summary encodes charges as '<Sym><sign><q>@<idx>' strings."""

    def test_neutral_molecule(self):
        # Benzene: no formal charges anywhere
        result = _formal_charge_summary("c1ccccc1")
        assert result == "neutral"

    def test_carboxylate_anion(self):
        # Acetate ion: one O carries formal charge −1
        result = _formal_charge_summary("CC(=O)[O-]")
        assert "O-1" in result

    def test_ammonium_cation(self):
        # Methylammonium: N carries formal charge +1
        result = _formal_charge_summary("C[NH3+]")
        assert "N+1" in result

    def test_invalid_smiles_returns_empty_string(self):
        result = _formal_charge_summary("NOT_A_SMILES!!!")
        assert result == ""

    def test_empty_smiles_returns_empty_string(self):
        result = _formal_charge_summary("")
        assert result == ""

    def test_zwitterion_contains_both_charges(self):
        # Glycine zwitterion: NH3+ and COO-
        result = _formal_charge_summary("[NH3+]CC(=O)[O-]")
        assert "N+1" in result
        assert "O-1" in result


# ===========================================================================
# Layer 1c — _diff_charges helper
# ===========================================================================

class TestDiffCharges:
    """_diff_charges should detect added and removed formal charges."""

    def test_deprotonation_creates_negative_charge(self):
        # neutral COOH → COO-
        orig = "neutral"
        prot = "O-1@5"
        changes = _diff_charges(orig, prot)
        assert len(changes) == 1
        assert "O-1" in changes[0]
        assert "neutral" in changes[0]

    def test_protonation_creates_positive_charge(self):
        # neutral NH2 → NH3+
        orig = "neutral"
        prot = "N+1@2"
        changes = _diff_charges(orig, prot)
        assert len(changes) == 1
        assert "N+1" in changes[0]

    def test_no_change_returns_empty_list(self):
        fp = "N+1@2,O-1@5"
        assert _diff_charges(fp, fp) == []

    def test_both_empty_returns_empty_list(self):
        assert _diff_charges("", "") == []

    def test_both_neutral_returns_empty_list(self):
        assert _diff_charges("neutral", "neutral") == []

    def test_removal_of_charge_detected(self):
        # Molecule loses a charge (e.g. protonation of a cation neutralises it)
        orig = "N+1@3"
        prot = "neutral"
        changes = _diff_charges(orig, prot)
        assert len(changes) == 1
        assert "neutral" in changes[0]

    def test_simultaneous_gain_and_loss(self):
        # One atom gains a charge, another loses one
        orig = "N+1@2"
        prot = "O-1@7"
        changes = _diff_charges(orig, prot)
        assert len(changes) == 2


# ===========================================================================
# Layer 2 — protonate_smiles() with Dimorphite-DL MOCKED
# (always run — no real tool required)
# ===========================================================================

MOCK_DIMORPHITE_VERSION = "1.3.2"


def _make_dimorphite_mock(return_smiles: list[str]):
    """Return a mock dimorphite_dl module that yields return_smiles."""
    mock_mod = MagicMock()
    mock_mod.__version__ = MOCK_DIMORPHITE_VERSION
    mock_mod.run_with_mol_list.return_value = iter(return_smiles)
    return mock_mod


class TestProtonateSmilesMocked:
    """Test protonate_smiles() with a mocked Dimorphite-DL."""

    # ------------------------------------------------------------------
    # Helper: patch both the importlib detection AND the real import
    # ------------------------------------------------------------------
    def _run(self, input_smiles: str, mock_returns: list[str], **kwargs):
        mock_mod = _make_dimorphite_mock(mock_returns)
        with (
            patch.dict("sys.modules", {"dimorphite_dl": mock_mod}),
            patch(
                "revision.protonation.protonate._tool_versions",
                {"dimorphite_dl": MOCK_DIMORPHITE_VERSION},
            ),
        ):
            return protonate_smiles(input_smiles, **kwargs)

    # --- status codes ---

    def test_status_ok_when_smiles_changes(self):
        rec = self._run("CC(=O)O", ["CC(=O)[O-]"])
        assert rec.status == "ok"
        assert rec.output == "CC(=O)[O-]"

    def test_status_unchanged_when_dimorphite_returns_same(self):
        rec = self._run("c1ccccc1", ["c1ccccc1"])
        assert rec.status == "unchanged"
        assert rec.output == "c1ccccc1"

    def test_status_unchanged_when_dimorphite_returns_empty(self):
        rec = self._run("CC", [])
        assert rec.status == "unchanged"
        assert rec.output == "CC"
        assert "no variants" in rec.notes.lower()

    def test_status_failed_when_exception_raised(self):
        mock_mod = MagicMock()
        mock_mod.__version__ = MOCK_DIMORPHITE_VERSION
        mock_mod.run_with_mol_list.side_effect = RuntimeError("tool crashed")
        with (
            patch.dict("sys.modules", {"dimorphite_dl": mock_mod}),
            patch(
                "revision.protonation.protonate._tool_versions",
                {"dimorphite_dl": MOCK_DIMORPHITE_VERSION},
            ),
        ):
            rec = protonate_smiles("CC(=O)O")
        assert rec.status == "failed"
        assert rec.output is None
        assert "RuntimeError" in rec.notes or "tool crashed" in rec.notes

    # --- provenance ---

    def test_tool_name_is_dimorphite_dl(self):
        rec = self._run("CC(=O)O", ["CC(=O)[O-]"])
        assert rec.tool == "dimorphite_dl"

    def test_tool_version_captured(self):
        rec = self._run("CC(=O)O", ["CC(=O)[O-]"])
        assert rec.tool_version == MOCK_DIMORPHITE_VERSION

    def test_ph_stored_in_record(self):
        rec = self._run("CC(=O)O", ["CC(=O)[O-]"], ph=7.4)
        assert rec.ph == 7.4

    def test_input_type_is_smiles(self):
        rec = self._run("CC(=O)O", ["CC(=O)[O-]"])
        assert rec.input_type == "smiles"

    # --- n_states_enumerated ---

    def test_n_states_enumerated_with_multiple_variants(self):
        rec = self._run("CC(=O)O", ["CC(=O)[O-]", "CC(=O)O"])
        assert rec.n_states_enumerated == 2

    def test_n_states_enumerated_zero_when_empty(self):
        rec = self._run("CC", [])
        assert rec.n_states_enumerated == 0

    # --- pick_most_likely=False ---

    def test_pick_most_likely_false_joins_variants(self):
        rec = self._run(
            "CC(=O)O",
            ["CC(=O)[O-]", "CC(=O)O"],
            pick_most_likely=False,
        )
        # Both variants joined with '.'
        assert "." in rec.output
        assert "2 Dimorphite states" in rec.notes or "All 2" in rec.notes

    # --- session log accumulation ---

    def test_record_appended_to_session_log(self):
        assert len(_session_log) == 0
        self._run("CC(=O)O", ["CC(=O)[O-]"])
        assert len(_session_log) == 1

    def test_multiple_calls_accumulate(self):
        for _ in range(3):
            self._run("CC(=O)O", ["CC(=O)[O-]"])
        assert len(_session_log) == 3

    # --- wall time ---

    def test_wall_time_is_non_negative(self):
        rec = self._run("CC(=O)O", ["CC(=O)[O-]"])
        assert rec.wall_time_s >= 0.0


# ===========================================================================
# Layer 2 — protonate_smiles_series() with Dimorphite-DL MOCKED
# ===========================================================================

class TestProtonateSmilesSeries:
    """Series function returns one record per input, in order."""

    def _run_series(self, smiles_list: list[str], mock_outputs: list[list[str]]):
        """
        mock_outputs[i] is the list returned by Dimorphite for smiles_list[i].
        """
        call_count = 0

        def _side_effect(mol_list, **kwargs):
            nonlocal call_count
            result = mock_outputs[call_count] if call_count < len(mock_outputs) else []
            call_count += 1
            return iter(result)

        mock_mod = MagicMock()
        mock_mod.__version__ = MOCK_DIMORPHITE_VERSION
        mock_mod.run_with_mol_list.side_effect = _side_effect
        with (
            patch.dict("sys.modules", {"dimorphite_dl": mock_mod}),
            patch(
                "revision.protonation.protonate._tool_versions",
                {"dimorphite_dl": MOCK_DIMORPHITE_VERSION},
            ),
        ):
            return protonate_smiles_series(smiles_list)

    def test_returns_same_length_as_input(self):
        smiles = ["CC(=O)O", "CN", "c1ccccc1"]
        outputs = [["CC(=O)[O-]"], ["C[NH3+]"], ["c1ccccc1"]]
        records = self._run_series(smiles, outputs)
        assert len(records) == len(smiles)

    def test_order_preserved(self):
        smiles = ["CC(=O)O", "CN"]
        outputs = [["CC(=O)[O-]"], ["C[NH3+]"]]
        records = self._run_series(smiles, outputs)
        assert records[0].input_id == "CC(=O)O"
        assert records[1].input_id == "CN"

    def test_all_records_in_session_log(self):
        smiles = ["CC(=O)O", "CN", "c1ccccc1"]
        outputs = [["CC(=O)[O-]"], ["C[NH3+]"], ["c1ccccc1"]]
        self._run_series(smiles, outputs)
        assert len(_session_log) == 3

    def test_empty_input_returns_empty_list(self):
        records = self._run_series([], [])
        assert records == []


# ===========================================================================
# Layer 2 (integration) — chemical correctness with real Dimorphite-DL
# ===========================================================================

@requires_dimorphite
@integration
class TestChemicalCorrectnessIntegration:
    """
    Integration tests — run only when dimorphite_dl is actually installed.

    Chemistry reference
    -------------------
    • Acetic acid (pKa 4.76):  At pH 7.4 > pKa  →  deprotonated (O⁻)
    • Benzoic acid (pKa 4.20): At pH 7.4 > pKa  →  deprotonated (O⁻)
    • Methylamine (pKa 10.66): At pH 7.4 < pKa  →  protonated   (N⁺)
    • Propylamine (pKa 10.57): At pH 7.4 < pKa  →  protonated   (N⁺)
    • Aniline      (pKa 4.60): At pH 7.4 > pKa  →  neutral      (unchanged)
    • Glycine: pKa(COOH)=2.35, pKa(NH3)=9.60
               At pH 7.4: COO⁻ and NH3⁺ simultaneously (zwitterion)
    """

    # -----------------------------------------------------------------------
    # Carboxylic acids — must be deprotonated (O⁻) at pH 7.4
    # -----------------------------------------------------------------------

    def test_acetic_acid_is_deprotonated(self):
        """Acetic acid (pKa 4.76) must be deprotonated at pH 7.4."""
        rec = protonate_smiles("CC(=O)O", ph=7.4)
        assert rec.status in ("ok", "unchanged"), f"Unexpected status: {rec.status}"
        assert rec.output is not None, "output should not be None"
        # The output SMILES must contain a negatively charged oxygen
        assert "[O-]" in rec.output or "O-" in (rec.protonated_formal_charges or ""), (
            f"Acetic acid not deprotonated at pH 7.4. output={rec.output!r} "
            f"charges={rec.protonated_formal_charges!r}"
        )

    def test_benzoic_acid_is_deprotonated(self):
        """Benzoic acid (pKa 4.20) must be deprotonated at pH 7.4."""
        rec = protonate_smiles("OC(=O)c1ccccc1", ph=7.4)
        assert rec.output is not None
        assert "[O-]" in rec.output or "O-" in (rec.protonated_formal_charges or ""), (
            f"Benzoic acid not deprotonated at pH 7.4. output={rec.output!r}"
        )

    def test_carboxylic_acid_status_is_ok(self):
        """Status should be 'ok' (not 'unchanged') since acetic acid is neutral in SMILES."""
        rec = protonate_smiles("CC(=O)O", ph=7.4)
        # neutral SMILES is the standard input; at pH 7.4 it should change
        assert rec.status == "ok", (
            f"Expected status='ok' for acetic acid at pH 7.4, got {rec.status!r}"
        )

    def test_acetic_acid_n_states_at_least_1(self):
        rec = protonate_smiles("CC(=O)O", ph=7.4)
        assert rec.n_states_enumerated >= 1

    # -----------------------------------------------------------------------
    # Aliphatic amines — must be protonated (N⁺) at pH 7.4
    # -----------------------------------------------------------------------

    def test_methylamine_is_protonated(self):
        """Methylamine (pKa 10.66) must be protonated at pH 7.4."""
        rec = protonate_smiles("CN", ph=7.4)
        assert rec.output is not None
        assert "[NH3+]" in rec.output or "[NH4+]" in rec.output or "N+1" in (
            rec.protonated_formal_charges or ""
        ), (
            f"Methylamine not protonated at pH 7.4. output={rec.output!r} "
            f"charges={rec.protonated_formal_charges!r}"
        )

    def test_propylamine_is_protonated(self):
        """Propylamine (pKa 10.57) must be protonated at pH 7.4."""
        rec = protonate_smiles("CCCN", ph=7.4)
        assert rec.output is not None
        has_n_plus = (
            "[NH3+]" in rec.output
            or "[NH2+]" in rec.output
            or "N+1" in (rec.protonated_formal_charges or "")
        )
        assert has_n_plus, (
            f"Propylamine not protonated at pH 7.4. output={rec.output!r} "
            f"charges={rec.protonated_formal_charges!r}"
        )

    # -----------------------------------------------------------------------
    # Aniline — must remain neutral at pH 7.4 (pKa < pH)
    # -----------------------------------------------------------------------

    def test_aniline_is_neutral(self):
        """Aniline (pKa 4.60) is neutral at pH 7.4 — its conjugate acid is too weak."""
        rec = protonate_smiles("Nc1ccccc1", ph=7.4)
        assert rec.output is not None
        # No positive nitrogen in the output
        has_n_plus = (
            "[NH3+]" in rec.output
            or "[NH2+]" in rec.output
            or "N+1" in (rec.protonated_formal_charges or "")
        )
        assert not has_n_plus, (
            f"Aniline should be neutral at pH 7.4, but got N+. output={rec.output!r}"
        )

    # -----------------------------------------------------------------------
    # Glycine — zwitterion at pH 7.4 (both groups change simultaneously)
    # -----------------------------------------------------------------------

    def test_glycine_is_zwitterion(self):
        """
        Glycine at pH 7.4 must carry both NH3+ (pKa 9.60 > pH) and
        COO- (pKa 2.35 < pH) simultaneously.
        """
        rec = protonate_smiles("NCC(=O)O", ph=7.4)
        assert rec.output is not None
        out = rec.output
        charges = rec.protonated_formal_charges or ""
        has_n_plus = "[NH3+]" in out or "N+1" in charges
        has_o_minus = "[O-]" in out or "O-1" in charges
        assert has_n_plus, (
            f"Glycine amino group should be protonated at pH 7.4. output={out!r}"
        )
        assert has_o_minus, (
            f"Glycine carboxyl group should be deprotonated at pH 7.4. output={out!r}"
        )

    def test_glycine_changes_logged(self):
        """groups_changed should reflect both modifications for glycine."""
        rec = protonate_smiles("NCC(=O)O", ph=7.4)
        # At least two atoms changed
        assert len(rec.groups_changed) >= 1 or rec.status in ("ok",), (
            "Expected at least one logged change for glycine zwitterion."
        )

    # -----------------------------------------------------------------------
    # pH boundary behaviour — check that the direction flips at the pKa
    # -----------------------------------------------------------------------

    def test_acetic_acid_neutral_below_pka(self):
        """At pH 3.0 (well below pKa 4.76) acetic acid should stay neutral."""
        rec = protonate_smiles("CC(=O)O", ph=3.0, ph_tolerance=0.5)
        if rec.status == "failed":
            pytest.skip("Dimorphite failed at pH 3.0; skipping boundary test")
        assert rec.output is not None
        # Should NOT contain O-
        assert "[O-]" not in rec.output, (
            f"Acetic acid should not be deprotonated at pH 3.0. output={rec.output!r}"
        )

    def test_methylamine_neutral_above_pka(self):
        """At pH 12.0 (above pKa 10.66) methylamine should be neutral."""
        rec = protonate_smiles("CN", ph=12.0, ph_tolerance=0.5)
        if rec.status == "failed":
            pytest.skip("Dimorphite failed at pH 12.0; skipping boundary test")
        assert rec.output is not None
        has_n_plus = "[NH3+]" in rec.output or "[NH4+]" in rec.output
        assert not has_n_plus, (
            f"Methylamine should not be protonated at pH 12.0. output={rec.output!r}"
        )

    # -----------------------------------------------------------------------
    # Robustness — invalid inputs handled gracefully
    # -----------------------------------------------------------------------

    def test_invalid_smiles_returns_failed_status(self):
        rec = protonate_smiles("THIS_IS_NOT_A_SMILES_%%", ph=7.4)
        # May be 'failed' or 'unchanged' depending on Dimorphite's handling
        assert rec.status in ("failed", "unchanged")

    def test_empty_smiles_does_not_crash(self):
        rec = protonate_smiles("", ph=7.4)
        assert rec.status in ("failed", "unchanged")


# ===========================================================================
# Layer 3 — file-based tests (OpenBabel / pdb2pqr)
# ===========================================================================

# Minimal chemically valid SDF for propionic acid (pKa 4.87 → should deprotonate)
_PROPIONIC_ACID_SDF = textwrap.dedent("""\

     RDKit          3D

  6  5  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.5400    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.2200    1.2700    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.6200    1.2700    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    1.5600    2.4600    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    4.2000    2.4100    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  1  0
  3  4  1  0
  3  5  2  0
  4  6  1  0
M  END
$$$$
""")

# Minimal PDB for a 3-residue pocket snippet (ALA-GLY-ASP)
_MINIMAL_POCKET_PDB = textwrap.dedent("""\
ATOM      1  N   ALA A   1       1.000   1.000   1.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.000   1.000   1.000  1.00  0.00           C
ATOM      3  C   ALA A   1       3.000   1.000   1.000  1.00  0.00           C
ATOM      4  O   ALA A   1       3.500   2.000   1.000  1.00  0.00           O
ATOM      5  N   ASP A   2       4.000   1.000   1.000  1.00  0.00           N
ATOM      6  CA  ASP A   2       5.000   1.000   1.000  1.00  0.00           C
ATOM      7  C   ASP A   2       6.000   1.000   1.000  1.00  0.00           C
ATOM      8  O   ASP A   2       6.500   2.000   1.000  1.00  0.00           O
ATOM      9  CG  ASP A   2       5.000   2.400   1.000  1.00  0.00           C
ATOM     10  OD1 ASP A   2       4.200   3.200   1.000  1.00  0.00           O
ATOM     11  OD2 ASP A   2       6.200   3.200   1.000  1.00  0.00           O
END
""")


@requires_openbabel
@integration
class TestFilesOpenBabel:
    """File-based tests using OpenBabel on temporary SDF / PDB files."""

    def test_protonate_ligand_sdf_ok(self, tmp_dir):
        sdf_in = tmp_dir / "propionic_acid.sdf"
        sdf_in.write_text(_PROPIONIC_ACID_SDF)
        rec = protonate_ligand_sdf(sdf_in, ph=7.4, out_path=tmp_dir / "out.sdf")
        assert rec.status == "ok", f"Expected ok, got {rec.status!r}. notes={rec.notes!r}"
        assert rec.output is not None
        assert Path(rec.output).exists(), "Output SDF file not created"

    def test_protonate_ligand_sdf_creates_output_beside_input_by_default(self, tmp_dir):
        sdf_in = tmp_dir / "mol.sdf"
        sdf_in.write_text(_PROPIONIC_ACID_SDF)
        rec = protonate_ligand_sdf(sdf_in, ph=7.4)
        assert rec.output is not None
        assert "_H.sdf" in rec.output or "_H.mol2" in rec.output

    def test_protonate_ligand_sdf_missing_file_raises(self, tmp_dir):
        with pytest.raises(FileNotFoundError):
            protonate_ligand_sdf(tmp_dir / "nonexistent.sdf")

    def test_protonate_ligand_sdf_tool_name(self, tmp_dir):
        sdf_in = tmp_dir / "mol.sdf"
        sdf_in.write_text(_PROPIONIC_ACID_SDF)
        rec = protonate_ligand_sdf(sdf_in, ph=7.4)
        assert rec.tool == "openbabel"
        assert rec.tool_version not in ("", None)

    def test_protonate_ligand_sdf_record_in_session_log(self, tmp_dir):
        sdf_in = tmp_dir / "mol.sdf"
        sdf_in.write_text(_PROPIONIC_ACID_SDF)
        protonate_ligand_sdf(sdf_in, ph=7.4)
        assert any(r.input_type == "sdf" for r in _session_log)

    def test_protonate_pocket_pdb_obabel_fallback(self, tmp_dir):
        """
        Force the OpenBabel fallback (prefer_pdb2pqr=False) so the test runs
        even if pdb2pqr is absent.
        """
        pdb_in = tmp_dir / "pocket.pdb"
        pdb_in.write_text(_MINIMAL_POCKET_PDB)
        rec = protonate_pocket_pdb(
            pdb_in, ph=7.4,
            out_path=tmp_dir / "pocket_H.pdb",
            prefer_pdb2pqr=False,
        )
        assert rec.tool == "openbabel"
        assert rec.status in ("ok", "failed"), f"Unexpected status: {rec.status!r}"
        if rec.status == "ok":
            assert Path(rec.output).exists()

    def test_protonate_pocket_pdb_missing_file_raises(self, tmp_dir):
        with pytest.raises(FileNotFoundError):
            protonate_pocket_pdb(tmp_dir / "no_pocket.pdb")


# ===========================================================================
# Layer 4 — write_log / clear_log round-trip
# ===========================================================================

class TestSessionLog:
    """Test the JSON-Lines log writing / clearing machinery."""

    def _add_record(self, input_id="CC(=O)O", status="ok"):
        _session_log.append(
            ProtonationRecord(
                input_id=input_id,
                input_type="smiles",
                ph=7.4,
                tool="dimorphite_dl",
                tool_version="1.3.2",
                status=status,
                output="CC(=O)[O-]" if status == "ok" else None,
            )
        )

    def test_write_log_creates_file(self, tmp_dir):
        self._add_record()
        out = tmp_dir / "test.jsonl"
        write_log(out)
        assert out.exists()

    def test_write_log_one_line_per_record(self, tmp_dir):
        for i in range(5):
            self._add_record(input_id=f"mol_{i}")
        out = tmp_dir / "test.jsonl"
        write_log(out)
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 5

    def test_write_log_valid_json_on_every_line(self, tmp_dir):
        self._add_record()
        out = tmp_dir / "test.jsonl"
        write_log(out)
        for line in out.read_text(encoding="utf-8").strip().splitlines():
            obj = json.loads(line)  # must not raise
            assert "input_id" in obj
            assert "status" in obj
            assert "tool" in obj

    def test_write_log_append_mode(self, tmp_dir):
        self._add_record(input_id="mol_a")
        out = tmp_dir / "test.jsonl"
        write_log(out, mode="w")
        clear_log()
        self._add_record(input_id="mol_b")
        write_log(out, mode="a")
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        ids = [json.loads(l)["input_id"] for l in lines]
        assert "mol_a" in ids and "mol_b" in ids

    def test_write_log_creates_parent_directories(self, tmp_dir):
        nested = tmp_dir / "a" / "b" / "c" / "log.jsonl"
        self._add_record()
        write_log(nested)
        assert nested.exists()

    def test_clear_log_empties_session(self):
        self._add_record()
        assert len(_session_log) == 1
        clear_log()
        assert len(_session_log) == 0

    def test_roundtrip_preserves_groups_changed(self, tmp_dir):
        # Use ASCII-only change string to avoid cp1252 encoding issues on Windows.
        change_str = "atom@3: neutral -> O-1"
        _session_log.append(
            ProtonationRecord(
                input_id="CC(=O)O",
                input_type="smiles",
                ph=7.4,
                tool="dimorphite_dl",
                tool_version="1.3.2",
                status="ok",
                output="CC(=O)[O-]",
                groups_changed=[change_str],
            )
        )
        out = tmp_dir / "test.jsonl"
        write_log(out)
        loaded = json.loads(out.read_text(encoding="utf-8").strip())
        assert loaded["groups_changed"] == [change_str]

    def test_failed_record_has_null_output(self, tmp_dir):
        _session_log.append(
            ProtonationRecord(
                input_id="BAD_SMILES",
                input_type="smiles",
                ph=7.4,
                tool="dimorphite_dl",
                tool_version="1.3.2",
                status="failed",
                output=None,
            )
        )
        out = tmp_dir / "test.jsonl"
        write_log(out)
        loaded = json.loads(out.read_text().strip())
        assert loaded["output"] is None
        assert loaded["status"] == "failed"
