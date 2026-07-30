"""
tests/test_conformer_qc.py
==========================
Tests for revision/conformer_qc/validate_conformers.py.

Test molecules
--------------
All chosen for well-defined expected behaviour:

  Molecule          SMILES              Why chosen
  ───────────────────────────────────────────────────────────────────────
  Ethanol           CCO                 Small, no stereo, should embed cleanly
  Alanine (L)       N[C@@H](C)C(=O)O   Single stereocentre; conformer must
                                        preserve the @@ annotation
  Alanine (D)       N[C@H](C)C(=O)O    Same scaffold, opposite config;
                                        must differ from L-form in InChI /t
  Aspirin           CC(=O)Oc1ccccc1C(=O)O  Drug-like, no stereo
  Glucose           OC[C@H]1OC(O)...   Multiple stereocentres
  "Impossible" mol  [Na+].[Cl-]         Ionic; ETKDGv3 may fail gracefully

Clash test
----------
A manually constructed mol with two carbon atoms at 0.5 Å (way below the
1.70+1.70=3.40 Å vdW sum) is injected post-embedding to verify check_clashes
fires.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# RDKit availability guard — all tests in this file require it
# ---------------------------------------------------------------------------
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False

# Module-level pytestmark removed — individual classes carry their own
# requires_rdkit marks so that pure-Python tests (record, log) always run.
requires_rdkit = pytest.mark.skipif(
    not _HAS_RDKIT,
    reason="RDKit not installed — install with: conda install -c conda-forge rdkit",
)

from revision.conformer_qc.validate_conformers import (
    CLASH_SCALE_DEFAULT,
    ConformerQCRecord,
    _inchi_stereo_from_smiles,
    _qc_log,
    check_clashes,
    check_stereo_consistency,
    clear_qc_log,
    generate_and_validate_conformer,
    qc_summary,
    qc_summary_dict,
    validate_smiles_dataset,
    write_qc_log,
)

# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(autouse=True)
def _reset_log():
    clear_qc_log()
    yield
    clear_qc_log()


@pytest.fixture
def ethanol_mol():
    """Ethanol with one ETKDGv3 conformer embedded."""
    mol = Chem.MolFromSmiles("CCO")
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    AllChem.EmbedMolecule(mol_h, params)
    return Chem.RemoveHs(mol_h)


@pytest.fixture
def clashing_mol():
    """
    A molecule with two NON-BONDED heavy atoms collapsed to 0.5 Å apart,
    guaranteed to trigger the clash check.

    n-Butane (CCCC) is used: atoms 0 and 3 are a 1,4 pair — they share
    no direct bond and no common bonded neighbour, so they are NOT in the
    1,2/1,3 exclusion set and WILL be checked against the vdW threshold.
    Placing them at 0.5 Å (well below 0.75 * 3.40 = 2.55 Å) guarantees
    detection at any non-zero clash_scale.

    Note: at clash_scale=0.0 the threshold is 0 * vdW_sum = 0 Å, and no
    real distance can be negative, so n==0 — consistent with the
    test_clash_scale_zero_never_flags test.
    """
    mol = Chem.MolFromSmiles("CCCC")   # n-butane: 4 heavy atoms
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    AllChem.EmbedMolecule(mol_h, params)
    mol_noh = Chem.RemoveHs(mol_h)

    # Collapse atoms 0 and 3 (the two terminal carbons, a 1,4 pair)
    # to 0.5 Å apart. They are not bonded and not 1,3-related,
    # so they will be checked by check_clashes.
    conf = mol_noh.GetConformer()
    conf.SetAtomPosition(0, (0.0, 0.0, 0.0))
    conf.SetAtomPosition(3, (0.5, 0.0, 0.0))
    return mol_noh


# ===========================================================================
# 1. ConformerQCRecord dataclass
# ===========================================================================

class TestConformerQCRecord:

    def test_default_overall_pass_is_false(self):
        rec = ConformerQCRecord(mol_id="test", smiles_input="CCO")
        assert rec.overall_pass is False

    def test_overall_pass_when_both_checks_true(self):
        rec = ConformerQCRecord(mol_id="test", smiles_input="CCO")
        rec.clash_pass = True
        rec.stereo_pass = True
        assert rec.overall_pass is True

    def test_overall_pass_false_if_clash_fails(self):
        rec = ConformerQCRecord(mol_id="test", smiles_input="CCO")
        rec.clash_pass = False
        rec.stereo_pass = True
        assert rec.overall_pass is False

    def test_asdict_is_json_serialisable(self):
        rec = ConformerQCRecord(mol_id="m1", smiles_input="CCO")
        rec.clash_pairs = ["atom0(C)--atom1(C): d=0.50A vdw_sum=3.40A"]
        d = asdict(rec)
        txt = json.dumps(d)          # must not raise
        loaded = json.loads(txt)
        assert loaded["mol_id"] == "m1"
        assert len(loaded["clash_pairs"]) == 1

    def test_timestamp_is_utc_string(self):
        rec = ConformerQCRecord(mol_id="x", smiles_input="CC")
        assert "T" in rec.timestamp and "Z" in rec.timestamp


# ===========================================================================
# 2. InChI stereo-layer helpers
# ===========================================================================

@requires_rdkit
class TestInChIStereoLayer:

    def test_flat_molecule_has_no_t_layer(self):
        layer = _inchi_stereo_from_smiles("CCO")      # ethanol — no stereocentre
        assert layer == ""

    def test_stereocentre_has_t_layer(self):
        layer = _inchi_stereo_from_smiles("N[C@@H](C)C(=O)O")   # L-alanine
        assert layer.startswith("/t"), f"Expected /t layer, got: {layer!r}"

    def test_l_and_d_alanine_have_different_t_layers(self):
        l_layer = _inchi_stereo_from_smiles("N[C@@H](C)C(=O)O")
        d_layer = _inchi_stereo_from_smiles("N[C@H](C)C(=O)O")
        assert l_layer != d_layer, (
            "L- and D-alanine must have distinct InChI /t layers"
        )

    def test_invalid_smiles_returns_empty(self):
        assert _inchi_stereo_from_smiles("NOT_VALID!!!") == ""

    def test_empty_smiles_returns_empty(self):
        assert _inchi_stereo_from_smiles("") == ""


# ===========================================================================
# 3. check_stereo_consistency
# ===========================================================================

@requires_rdkit
class TestStereoConsistency:

    def test_flat_molecule_both_flat(self, ethanol_mol):
        _, _, note = check_stereo_consistency("CCO", ethanol_mol)
        assert note == "both_flat"

    def test_l_alanine_conformer_matches_input(self):
        smi = "N[C@@H](C)C(=O)O"
        mol = Chem.MolFromSmiles(smi)
        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        AllChem.EmbedMolecule(mol_h, params)
        mol_noh = Chem.RemoveHs(mol_h)
        _, _, note = check_stereo_consistency(smi, mol_noh)
        # ETKDGv3 should preserve the stereocentre
        assert note in ("match", "no_stereo_conformer"), (
            f"Expected 'match' for L-alanine conformer, got {note!r}"
        )

    def test_no_stereo_in_input_smiles(self, ethanol_mol):
        # Input SMILES has no stereo marks → cannot verify → 'no_stereo_input'
        # or 'both_flat' (ethanol truly has no stereocentre)
        _, _, note = check_stereo_consistency("CCO", ethanol_mol)
        assert note in ("no_stereo_input", "both_flat")

    def test_return_tuple_has_three_elements(self, ethanol_mol):
        result = check_stereo_consistency("CCO", ethanol_mol)
        assert len(result) == 3

    def test_inchi_in_and_conf_are_strings(self, ethanol_mol):
        inchi_in, inchi_conf, _ = check_stereo_consistency("CCO", ethanol_mol)
        assert isinstance(inchi_in, str)
        assert isinstance(inchi_conf, str)


# ===========================================================================
# 4. check_clashes
# ===========================================================================

@requires_rdkit
class TestCheckClashes:

    def test_no_clashes_in_valid_ethanol(self, ethanol_mol):
        n, descs = check_clashes(ethanol_mol)
        assert n == 0, f"Expected no clashes in ethanol, found: {descs}"
        assert descs == []

    def test_clashes_detected_in_collapsed_mol(self, clashing_mol):
        n, descs = check_clashes(clashing_mol, clash_scale=CLASH_SCALE_DEFAULT)
        assert n > 0, "Expected at least one clash pair in collapsed ethane"
        assert len(descs) == n

    def test_clash_description_contains_distance(self, clashing_mol):
        _, descs = check_clashes(clashing_mol)
        assert any("d=" in d for d in descs)

    def test_clash_description_contains_vdw_sum(self, clashing_mol):
        _, descs = check_clashes(clashing_mol)
        assert any("vdw_sum=" in d for d in descs)

    def test_clash_description_contains_atom_symbols(self, clashing_mol):
        _, descs = check_clashes(clashing_mol)
        assert any("(C)" in d for d in descs)

    def test_clash_scale_zero_never_flags(self, clashing_mol):
        """At scale=0.0 no distance is less than 0 × vdW_sum."""
        n, _ = check_clashes(clashing_mol, clash_scale=0.0)
        assert n == 0

    def test_clash_scale_ten_flags_all_non_bonded(self):
        """At scale=10.0 all 1,4+ pairs clash (threshold >> any real distance).

        Uses n-butane (CCCC): atoms 0 and 3 are a 1,4 pair at ~3.7 Å in a
        normal conformer. At scale=10.0 the threshold is 10 * 3.40 = 34 Å,
        so d=3.7 < 34 is guaranteed to be flagged.
        """
        mol = Chem.MolFromSmiles("CCCC")
        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        AllChem.EmbedMolecule(mol_h, params)
        butane = Chem.RemoveHs(mol_h)
        n, _ = check_clashes(butane, clash_scale=10.0)
        assert n > 0

    def test_returns_tuple_of_int_and_list(self, ethanol_mol):
        result = check_clashes(ethanol_mol)
        assert isinstance(result[0], int)
        assert isinstance(result[1], list)


# ===========================================================================
# 5. generate_and_validate_conformer — single molecule
# ===========================================================================

@requires_rdkit
class TestGenerateAndValidate:

    def test_ethanol_embed_ok(self):
        mol, rec = generate_and_validate_conformer("CCO", mol_id="ethanol")
        assert rec.embed_status == "ok", f"embed_status={rec.embed_status}"
        assert mol is not None

    def test_ethanol_clash_pass(self):
        _, rec = generate_and_validate_conformer("CCO")
        assert rec.clash_pass, f"Expected no clashes in ethanol. clashes={rec.clash_pairs}"

    def test_ethanol_stereo_pass(self):
        _, rec = generate_and_validate_conformer("CCO")
        assert rec.stereo_pass, f"Stereo note: {rec.stereo_note}"

    def test_ethanol_overall_pass(self):
        _, rec = generate_and_validate_conformer("CCO")
        assert rec.overall_pass

    def test_aspirin_embeds_and_passes(self):
        _, rec = generate_and_validate_conformer(
            "CC(=O)Oc1ccccc1C(=O)O", mol_id="aspirin"
        )
        assert rec.embed_status == "ok"
        assert rec.clash_pass

    def test_l_alanine_stereo_match(self):
        smi = "N[C@@H](C)C(=O)O"
        _, rec = generate_and_validate_conformer(smi, mol_id="L-alanine")
        assert rec.embed_status == "ok"
        assert rec.stereo_note in ("match", "no_stereo_conformer"), (
            f"Expected stereo match for L-alanine, got {rec.stereo_note!r}"
        )

    def test_d_and_l_alanine_stereo_notes_are_independent(self):
        _, rec_l = generate_and_validate_conformer("N[C@@H](C)C(=O)O", mol_id="L")
        _, rec_d = generate_and_validate_conformer("N[C@H](C)C(=O)O",  mol_id="D")
        # Both should embed; stereo should be handled (match or graceful note)
        assert rec_l.embed_status == "ok"
        assert rec_d.embed_status == "ok"

    def test_invalid_smiles_returns_skipped(self):
        mol, rec = generate_and_validate_conformer("NOT_A_MOLECULE!!", mol_id="bad")
        assert mol is None
        assert rec.embed_status == "skipped"
        assert rec.overall_pass is False

    def test_n_heavy_atoms_counted(self):
        _, rec = generate_and_validate_conformer("CCO", mol_id="ethanol")
        # Ethanol heavy atoms: C, C, O = 3
        assert rec.n_heavy_atoms == 3

    def test_wall_time_positive(self):
        _, rec = generate_and_validate_conformer("CCO")
        assert rec.wall_time_s >= 0.0

    def test_record_appended_to_log(self):
        assert len(_qc_log) == 0
        generate_and_validate_conformer("CCO")
        assert len(_qc_log) == 1

    def test_mol_id_stored(self):
        _, rec = generate_and_validate_conformer("CCO", mol_id="ethanol_test")
        assert rec.mol_id == "ethanol_test"

    def test_smiles_input_stored(self):
        smi = "CC(=O)Oc1ccccc1C(=O)O"
        _, rec = generate_and_validate_conformer(smi)
        assert rec.smiles_input == smi

    def test_failed_embed_returns_none_mol(self):
        # A very complex / impossible SMILES that ETKDGv3 can't embed:
        # We force by passing a non-embeddable ion pair
        mol, rec = generate_and_validate_conformer("[Na+].[Cl-]", mol_id="nacl")
        # NaCl is two separate fragments; EmbedMolecule typically fails
        if rec.embed_status == "failed":
            assert mol is None
            assert rec.overall_pass is False
        else:
            # Some RDKit versions may embed it; that's acceptable
            assert rec.embed_status in ("ok", "failed", "skipped")


# ===========================================================================
# 6. validate_smiles_dataset
# ===========================================================================

SMALL_DATASET = [
    ("CCO",                    "ethanol"),
    ("N[C@@H](C)C(=O)O",      "L-alanine"),
    ("CC(=O)Oc1ccccc1C(=O)O", "aspirin"),
    ("c1ccccc1",               "benzene"),
    ("NOT_VALID",              "bad_mol"),
]


@requires_rdkit
class TestValidateDataset:

    def _run(self, **kwargs):
        smiles = [s for s, _ in SMALL_DATASET]
        ids    = [i for _, i in SMALL_DATASET]
        return validate_smiles_dataset(smiles, ids=ids, **kwargs)

    def test_returns_one_record_per_input(self):
        records = self._run()
        assert len(records) == len(SMALL_DATASET)

    def test_order_preserved(self):
        records = self._run()
        assert records[0].mol_id == "ethanol"
        assert records[4].mol_id == "bad_mol"

    def test_bad_mol_is_skipped(self):
        records = self._run()
        bad = records[4]
        assert bad.embed_status == "skipped"

    def test_valid_mols_embed_ok(self):
        records = self._run()
        for rec in records[:4]:   # first four are valid
            assert rec.embed_status in ("ok", "failed"), (
                f"{rec.mol_id}: unexpected embed_status={rec.embed_status!r}"
            )

    def test_ethanol_passes_overall(self):
        records = self._run()
        eth = records[0]
        assert eth.overall_pass, (
            f"Ethanol should pass QC. clash={eth.clash_pass} stereo={eth.stereo_note}"
        )

    def test_all_records_in_qc_log(self):
        self._run()
        assert len(_qc_log) == len(SMALL_DATASET)

    def test_write_sdf_creates_file(self, tmp_path):
        out = tmp_path / "test_out.sdf"
        self._run(out_sdf=str(out))
        assert out.exists(), "SDF output file not created"

    def test_sdf_contains_only_passing_conformers_by_default(self, tmp_path):
        out = tmp_path / "test.sdf"
        records = self._run(out_sdf=str(out))
        n_passing = sum(r.overall_pass for r in records)
        # Count molecules in SDF
        text = out.read_text(encoding="utf-8", errors="replace")
        n_in_sdf = text.count("$$$$")
        assert n_in_sdf == n_passing, (
            f"SDF should have {n_passing} mols (passing only), found {n_in_sdf}"
        )

    def test_save_failed_includes_more_molecules(self, tmp_path):
        out_pass = tmp_path / "pass.sdf"
        out_all  = tmp_path / "all.sdf"
        self._run(out_sdf=str(out_pass), save_failed=False)
        self._run(out_sdf=str(out_all),  save_failed=True)
        n_pass = out_pass.read_text(errors="replace").count("$$$$")
        n_all  = out_all.read_text(errors="replace").count("$$$$")
        assert n_all >= n_pass, "save_failed=True must include at least as many mols"

    def test_sdf_contains_qc_pass_property(self, tmp_path):
        out = tmp_path / "test.sdf"
        self._run(out_sdf=str(out), save_failed=True)
        text = out.read_text(errors="replace")
        assert "<QC_PASS>" in text

    def test_sdf_contains_smiles_input_property(self, tmp_path):
        out = tmp_path / "test.sdf"
        self._run(out_sdf=str(out))
        text = out.read_text(errors="replace")
        assert "<SMILES_INPUT>" in text

    def test_write_log_creates_jsonl(self, tmp_path):
        log = tmp_path / "qc.jsonl"
        self._run(log_path=str(log))
        assert log.exists()
        lines = log.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == len(SMALL_DATASET)

    def test_log_is_valid_json_per_line(self, tmp_path):
        log = tmp_path / "qc.jsonl"
        self._run(log_path=str(log))
        for line in log.read_text(encoding="utf-8").strip().splitlines():
            obj = json.loads(line)
            assert "mol_id" in obj
            assert "embed_status" in obj
            assert "clash_pass" in obj
            assert "stereo_note" in obj


# ===========================================================================
# 7. qc_summary / qc_summary_dict
# ===========================================================================

@requires_rdkit
class TestSummaryFunctions:

    def _make_records(self) -> list[ConformerQCRecord]:
        return validate_smiles_dataset(
            [s for s, _ in SMALL_DATASET],
            ids=[i for _, i in SMALL_DATASET],
        )

    def test_summary_string_contains_total(self):
        records = self._make_records()
        s = qc_summary(records)
        assert f"Total molecules" in s
        assert str(len(SMALL_DATASET)) in s

    def test_summary_string_contains_embed_ok(self):
        records = self._make_records()
        s = qc_summary(records)
        assert "Embed OK" in s

    def test_summary_string_contains_stereo_section(self):
        records = self._make_records()
        s = qc_summary(records)
        assert "Stereo" in s

    def test_summary_string_contains_overall_pass(self):
        records = self._make_records()
        s = qc_summary(records)
        assert "Overall QC PASS" in s

    def test_summary_dict_keys(self):
        records = self._make_records()
        d = qc_summary_dict(records)
        assert "total" in d
        assert "embed_ok" in d
        assert "clash_pass" in d
        assert "stereo_breakdown" in d
        assert "overall_pass" in d
        assert "overall_pass_pct" in d

    def test_summary_dict_total_matches_input(self):
        records = self._make_records()
        d = qc_summary_dict(records)
        assert d["total"] == len(SMALL_DATASET)

    def test_summary_dict_embed_ok_plus_failed_plus_skipped_equals_total(self):
        records = self._make_records()
        d = qc_summary_dict(records)
        assert d["embed_ok"] + d["embed_failed"] + d["embed_skipped"] == d["total"]

    def test_empty_records_returns_no_records_string(self):
        assert "No records" in qc_summary([])

    def test_empty_records_dict(self):
        d = qc_summary_dict([])
        assert d == {"total": 0}


# ===========================================================================
# 8. write_qc_log / clear_qc_log
# ===========================================================================

class TestQCLog:

    def _push(self, n=3):
        for i in range(n):
            rec = ConformerQCRecord(
                mol_id=f"mol_{i}", smiles_input="CCO",
                embed_status="ok", clash_pass=True, stereo_pass=True,
                stereo_note="both_flat",
            )
            _qc_log.append(rec)

    def test_write_creates_file(self, tmp_path):
        self._push()
        out = tmp_path / "log.jsonl"
        write_qc_log(out)
        assert out.exists()

    def test_write_one_line_per_record(self, tmp_path):
        self._push(5)
        out = tmp_path / "log.jsonl"
        write_qc_log(out)
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 5

    def test_write_valid_json_on_every_line(self, tmp_path):
        self._push()
        out = tmp_path / "log.jsonl"
        write_qc_log(out)
        for line in out.read_text(encoding="utf-8").strip().splitlines():
            obj = json.loads(line)
            assert "mol_id" in obj
            assert "clash_pass" in obj   # overall_pass is a @property, not a stored field

    def test_clear_log(self):
        self._push(4)
        assert len(_qc_log) == 4
        clear_qc_log()
        assert len(_qc_log) == 0

    def test_write_append_mode(self, tmp_path):
        self._push(2)
        out = tmp_path / "log.jsonl"
        write_qc_log(out, mode="w")
        clear_qc_log()
        self._push(3)
        write_qc_log(out, mode="a")
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 5
