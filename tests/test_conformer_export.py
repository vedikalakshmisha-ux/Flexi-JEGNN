"""
tests/test_conformer_export.py
================================
Unit tests for ``revision/data/export_conformers.py``.

Design principles
-----------------
* All RDKit/SDF-writer calls are monkeypatched in the unit tests so the
  suite passes without RDKit installed (Tier-1 / always-run).
* Real-RDKit round-trip tests live in TestRDKitRoundTrip, gated with
  ``@pytest.mark.skipif(not rdkit_available(), ...)``.
* Tests that test the "RDKit absent" behaviour mock ``rdkit_available``
  via monkeypatch so they work correctly regardless of actual environment.

Key fixes (vs initial version):
- test_creates_sdf_per_method, test_qc_skipped_for_non_qm9, and
  test_written_count_in_results no longer mock rdkit_available=False, which
  was incorrectly causing ETKDGv3 (a RDKit method) to be skipped while the
  assertions still expected it to produce output.
- test_written_count_in_results uses only CC and CCO (matching mock positions)
  instead of benzene (6 atoms, mismatches 2-atom mock positions).
"""

from __future__ import annotations

import csv
import io
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from revision.data.export_conformers import (
    METHOD_ETKDGv3,
    METHOD_OBABEL,
    METHOD_QC,
    RDKIT_METHODS,
    ConformerResult,
    _load_smiles_from_csv,
    _write_sdf_record_minimal,
    export_conformers,
    export_one_method,
    main,
    obabel_available,
    rdkit_available,
)

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

SMILES_ETHANE  = "CC"
SMILES_ETHANOL = "CCO"
SMILES_BENZENE = "c1ccccc1"
METHOD_ETKDGv2 = "ETKDGv2"

_ETHANE_POS  = np.array([[0.00, 0.00, 0.00],
                          [1.54, 0.00, 0.00]], dtype=np.float32)
_ETHANOL_POS = np.array([[0.00, 0.00, 0.00],
                          [1.54, 0.00, 0.00],
                          [2.40, 1.10, 0.00]], dtype=np.float32)


def _make_result(method: str, smiles: str, positions: np.ndarray,
                 seed: int = 42) -> ConformerResult:
    return ConformerResult(
        method=method, smiles=smiles, positions=positions,
        n_atoms=positions.shape[0], success=True, seed=seed,
    )


def _failed_result(method: str, smiles: str,
                   error_msg: str = "embed failed", seed: int = 42) -> ConformerResult:
    return ConformerResult(
        method=method, smiles=smiles, positions=None,
        n_atoms=0, success=False, error_msg=error_msg, seed=seed,
    )


def _write_smiles_csv(path: Path, entries: List[tuple],
                      smiles_col: str = "smiles",
                      name_col: Optional[str] = "name") -> None:
    """Write a minimal CSV with smiles (and optionally name) columns."""
    fieldnames = [smiles_col] + ([name_col] if name_col else [])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for smiles, name in entries:
            row = {smiles_col: smiles}
            if name_col:
                row[name_col] = name
            w.writerow(row)


# ============================================================================
# 1. Availability helpers (pass-through from generate_variants)
# ============================================================================

class TestAvailabilityHelpers:
    def test_rdkit_available_returns_bool(self):
        assert isinstance(rdkit_available(), bool)

    def test_obabel_available_returns_bool(self):
        assert isinstance(obabel_available(), bool)


# ============================================================================
# 2. _load_smiles_from_csv
# ============================================================================

class TestLoadSmilesFromCsv:
    def test_loads_smiles_and_name(self, tmp_path):
        p = tmp_path / "test.csv"
        _write_smiles_csv(p, [(SMILES_ETHANOL, "ethanol")])
        records = _load_smiles_from_csv(p, "smiles", "name", n_mols=None)
        assert records[0] == (SMILES_ETHANOL, "ethanol")

    def test_name_col_none_uses_smiles_as_name(self, tmp_path):
        p = tmp_path / "test.csv"
        _write_smiles_csv(p, [(SMILES_ETHANE, "x")])
        records = _load_smiles_from_csv(p, "smiles", name_col=None, n_mols=None)
        assert records[0][1] == SMILES_ETHANE

    def test_wrong_smiles_col_raises(self, tmp_path):
        p = tmp_path / "test.csv"
        _write_smiles_csv(p, [(SMILES_ETHANE, "x")])
        with pytest.raises(ValueError, match="not found"):
            _load_smiles_from_csv(p, "no_such_col", None, None)

    def test_skips_empty_smiles(self, tmp_path):
        p = tmp_path / "test.csv"
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["smiles"])
            w.writeheader()
            w.writerow({"smiles": SMILES_ETHANE})
            w.writerow({"smiles": ""})
            w.writerow({"smiles": SMILES_ETHANOL})
        records = _load_smiles_from_csv(p, "smiles", None, None)
        assert len(records) == 2

    def test_n_mols_limit(self, tmp_path):
        p = tmp_path / "test.csv"
        _write_smiles_csv(p, [(SMILES_ETHANE, "a"), (SMILES_ETHANOL, "b"),
                               (SMILES_BENZENE, "c")])
        records = _load_smiles_from_csv(p, "smiles", "name", n_mols=2)
        assert len(records) == 2

    def test_returns_list_of_tuples(self, tmp_path):
        p = tmp_path / "test.csv"
        _write_smiles_csv(p, [(SMILES_ETHANE, "x")])
        records = _load_smiles_from_csv(p, "smiles", "name", n_mols=None)
        assert isinstance(records, list)
        assert isinstance(records[0], tuple)


# ============================================================================
# 3. _write_sdf_record_minimal (no-bond SDF writer)
# ============================================================================

class TestWriteSdfRecordMinimal:
    """Tests for the no-RDKit fallback SDF writer."""

    def _write_and_read(self, positions, symbols, smiles, method, dataset):
        """Write a record to a StringIO and return the text."""
        buf = io.StringIO()
        _write_sdf_record_minimal(buf, "testmol", positions, symbols,
                                  smiles, method, dataset)
        return buf.getvalue()

    def test_produces_sdf_terminator(self):
        sdf = self._write_and_read(
            _ETHANE_POS, ["C", "C"], SMILES_ETHANE, METHOD_ETKDGv3, "TEST"
        )
        assert "$$$$" in sdf

    def test_m_end_present(self):
        sdf = self._write_and_read(
            _ETHANE_POS, ["C", "C"], SMILES_ETHANE, METHOD_ETKDGv3, "TEST"
        )
        assert "M  END" in sdf

    def test_counts_line_has_correct_atom_count(self):
        sdf = self._write_and_read(
            _ETHANOL_POS, ["C", "C", "O"], SMILES_ETHANOL, METHOD_ETKDGv3, "TEST"
        )
        lines = sdf.splitlines()
        counts_line = lines[3]
        n_atoms = int(counts_line[:3].strip())
        assert n_atoms == 3

    def test_atom_coordinates_in_output(self):
        sdf = self._write_and_read(
            _ETHANE_POS, ["C", "C"], SMILES_ETHANE, METHOD_ETKDGv3, "TEST"
        )
        assert "1.5400" in sdf

    def test_smiles_property_present(self):
        sdf = self._write_and_read(
            _ETHANE_POS, ["C", "C"], SMILES_ETHANE, METHOD_ETKDGv3, "TEST"
        )
        assert SMILES_ETHANE in sdf

    def test_method_property_present(self):
        sdf = self._write_and_read(
            _ETHANE_POS, ["C", "C"], SMILES_ETHANE, METHOD_ETKDGv3, "TEST"
        )
        assert METHOD_ETKDGv3 in sdf

    def test_dataset_property_present(self):
        sdf = self._write_and_read(
            _ETHANE_POS, ["C", "C"], SMILES_ETHANE, METHOD_ETKDGv3, "BACE"
        )
        assert "BACE" in sdf

    def test_element_symbols_in_output(self):
        sdf = self._write_and_read(
            _ETHANOL_POS, ["C", "C", "O"], SMILES_ETHANOL, METHOD_ETKDGv3, "TEST"
        )
        assert " O " in sdf

    def test_n_heavy_atoms_property_present(self):
        sdf = self._write_and_read(
            _ETHANOL_POS, ["C", "C", "O"], SMILES_ETHANOL, METHOD_ETKDGv3, "TEST"
        )
        assert "3" in sdf


# ============================================================================
# 4. export_one_method — monkeypatched
# ============================================================================

class TestExportOneMethod:
    """Tests for export_one_method; generate_conformer is always mocked."""

    def _make_mock(self, mapping: Dict[str, ConformerResult]):
        """Return a fake generate_conformer that returns from mapping."""
        def _mock(smiles, method=METHOD_ETKDGv3, seed=42, **kw):
            return mapping.get(smiles, _failed_result(method, smiles))
        return _mock

    def test_creates_output_file(self, tmp_path, monkeypatch):
        import revision.data.export_conformers as ecm
        mock = self._make_mock({
            SMILES_ETHANE:  _make_result(METHOD_ETKDGv3, SMILES_ETHANE,  _ETHANE_POS),
            SMILES_ETHANOL: _make_result(METHOD_ETKDGv3, SMILES_ETHANOL, _ETHANOL_POS),
        })
        monkeypatch.setattr(ecm, "generate_conformer", mock)
        monkeypatch.setattr(ecm, "rdkit_available", lambda: False)

        out = tmp_path / "test.sdf"
        export_one_method(
            records=[(SMILES_ETHANE, "m1"), (SMILES_ETHANOL, "m2")],
            method=METHOD_ETKDGv3, dataset="TEST", out_path=out, verbose=False,
        )
        assert out.exists()

    def test_written_count_matches_successes(self, tmp_path, monkeypatch):
        import revision.data.export_conformers as ecm
        mock = self._make_mock({
            SMILES_ETHANE:  _make_result(METHOD_ETKDGv3, SMILES_ETHANE,  _ETHANE_POS),
            SMILES_ETHANOL: _make_result(METHOD_ETKDGv3, SMILES_ETHANOL, _ETHANOL_POS),
        })
        monkeypatch.setattr(ecm, "generate_conformer", mock)
        monkeypatch.setattr(ecm, "rdkit_available", lambda: False)

        out = tmp_path / "test.sdf"
        counts = export_one_method(
            records=[(SMILES_ETHANE, "m1"), (SMILES_ETHANOL, "m2")],
            method=METHOD_ETKDGv3, dataset="TEST", out_path=out, verbose=False,
        )
        assert counts["written"] == 2
        assert counts["failed"] == 0

    def test_failed_result_increments_failed(self, tmp_path, monkeypatch):
        import revision.data.export_conformers as ecm
        mock = self._make_mock({
            SMILES_ETHANE:  _failed_result(METHOD_ETKDGv3, SMILES_ETHANE),
            SMILES_ETHANOL: _make_result(METHOD_ETKDGv3, SMILES_ETHANOL, _ETHANOL_POS),
        })
        monkeypatch.setattr(ecm, "generate_conformer", mock)
        monkeypatch.setattr(ecm, "rdkit_available", lambda: False)

        out = tmp_path / "test.sdf"
        counts = export_one_method(
            records=[(SMILES_ETHANE, "m1"), (SMILES_ETHANOL, "m2")],
            method=METHOD_ETKDGv3, dataset="TEST", out_path=out, verbose=False,
        )
        assert counts["failed"] == 1
        assert counts["written"] == 1

    def test_sdf_contains_two_terminators(self, tmp_path, monkeypatch):
        import revision.data.export_conformers as ecm
        mock = self._make_mock({
            SMILES_ETHANE:  _make_result(METHOD_ETKDGv3, SMILES_ETHANE,  _ETHANE_POS),
            SMILES_ETHANOL: _make_result(METHOD_ETKDGv3, SMILES_ETHANOL, _ETHANOL_POS),
        })
        monkeypatch.setattr(ecm, "generate_conformer", mock)
        monkeypatch.setattr(ecm, "rdkit_available", lambda: False)

        out = tmp_path / "test.sdf"
        export_one_method(
            records=[(SMILES_ETHANE, "m1"), (SMILES_ETHANOL, "m2")],
            method=METHOD_ETKDGv3, dataset="TEST", out_path=out, verbose=False,
        )
        text = out.read_text()
        assert text.count("$$$$") == 2

    def test_sd_property_smiles_in_output(self, tmp_path, monkeypatch):
        import revision.data.export_conformers as ecm
        mock = self._make_mock({
            SMILES_ETHANOL: _make_result(METHOD_ETKDGv3, SMILES_ETHANOL, _ETHANOL_POS),
        })
        monkeypatch.setattr(ecm, "generate_conformer", mock)
        monkeypatch.setattr(ecm, "rdkit_available", lambda: False)

        out = tmp_path / "test.sdf"
        export_one_method(
            records=[(SMILES_ETHANOL, "ethanol")],
            method=METHOD_ETKDGv3, dataset="TEST", out_path=out, verbose=False,
        )
        text = out.read_text()
        assert SMILES_ETHANOL in text

    def test_creates_parent_dirs(self, tmp_path, monkeypatch):
        import revision.data.export_conformers as ecm
        mock = self._make_mock({SMILES_ETHANE: _make_result(
            METHOD_ETKDGv3, SMILES_ETHANE, _ETHANE_POS)})
        monkeypatch.setattr(ecm, "generate_conformer", mock)
        monkeypatch.setattr(ecm, "rdkit_available", lambda: False)

        out = tmp_path / "a" / "b" / "test.sdf"
        export_one_method(
            records=[(SMILES_ETHANE, "m1")],
            method=METHOD_ETKDGv3, dataset="TEST", out_path=out, verbose=False,
        )
        assert out.exists()


# ============================================================================
# 5. export_conformers pipeline — monkeypatched
# ============================================================================

class TestExportConformersPipeline:
    def _write_dataset_csv(self, path: Path, smiles_list: List[str],
                           smiles_col: str = "smiles") -> None:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[smiles_col])
            w.writeheader()
            for s in smiles_list:
                w.writerow({smiles_col: s})

    def _mock_gen(self, smiles, method=METHOD_ETKDGv3, seed=42, **kw):
        pos = _ETHANOL_POS if smiles == SMILES_ETHANOL else _ETHANE_POS
        return _make_result(method, smiles, pos)

    def test_creates_sdf_per_method(self, tmp_path, monkeypatch):
        import revision.data.export_conformers as ecm
        monkeypatch.setattr(ecm, "generate_conformer", self._mock_gen)
        # Do NOT mock rdkit_available here: when RDKit is installed we want the
        # real RDKit writer to run so we can assert the file is actually created.
        monkeypatch.setattr(ecm, "obabel_available", lambda: False)

        ds_dir = tmp_path / "datasets"
        ds_dir.mkdir()
        self._write_dataset_csv(ds_dir / "BACE.csv", [SMILES_ETHANE], smiles_col="mol")
        out_dir = tmp_path / "conformers"

        results = export_conformers(
            datasets_dir=ds_dir,
            out_dir=out_dir,
            methods=[METHOD_ETKDGv3],
            datasets=["BACE"],
            verbose=False,
        )
        assert (out_dir / "BACE_ETKDGv3.sdf").exists()
        assert "BACE_ETKDGv3" in results

    def test_qc_skipped_for_non_qm9(self, tmp_path, monkeypatch):
        import revision.data.export_conformers as ecm
        monkeypatch.setattr(ecm, "generate_conformer", self._mock_gen)
        # Do NOT mock rdkit_available: ETKDGv3 must run so we can check
        # that 3_qc is the only thing skipped (because BACE is not qc_eligible).
        monkeypatch.setattr(ecm, "obabel_available", lambda: False)

        ds_dir = tmp_path / "datasets"
        ds_dir.mkdir()
        self._write_dataset_csv(ds_dir / "BACE.csv", [SMILES_ETHANE], smiles_col="mol")
        out_dir = tmp_path / "conformers"

        results = export_conformers(
            datasets_dir=ds_dir, out_dir=out_dir,
            methods=[METHOD_ETKDGv3, METHOD_QC],
            datasets=["BACE"], verbose=False,
        )
        # 3_qc should NOT appear in results for BACE
        assert "BACE_3_qc" not in results
        assert "BACE_ETKDGv3" in results

    def test_missing_csv_skips_dataset(self, tmp_path, monkeypatch):
        import revision.data.export_conformers as ecm
        monkeypatch.setattr(ecm, "generate_conformer", self._mock_gen)
        monkeypatch.setattr(ecm, "rdkit_available", lambda: False)
        monkeypatch.setattr(ecm, "obabel_available", lambda: False)

        ds_dir = tmp_path / "datasets"
        ds_dir.mkdir()
        # No CSV files written
        out_dir = tmp_path / "conformers"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = export_conformers(
                datasets_dir=ds_dir, out_dir=out_dir,
                methods=[METHOD_ETKDGv3], datasets=["BACE"], verbose=False,
            )
        assert results == {}
        assert any("not found" in str(x.message).lower() for x in w)

    def test_rdkit_methods_skipped_when_absent(self, tmp_path, monkeypatch):
        import revision.data.export_conformers as ecm
        monkeypatch.setattr(ecm, "generate_conformer", self._mock_gen)
        monkeypatch.setattr(ecm, "rdkit_available", lambda: False)
        monkeypatch.setattr(ecm, "obabel_available", lambda: False)

        ds_dir = tmp_path / "datasets"
        ds_dir.mkdir()
        self._write_dataset_csv(ds_dir / "BACE.csv", [SMILES_ETHANE], smiles_col="mol")
        out_dir = tmp_path / "conformers"

        results = export_conformers(
            datasets_dir=ds_dir, out_dir=out_dir,
            methods=RDKIT_METHODS, datasets=["BACE"], verbose=False,
        )
        # All RDKit methods skipped -> no results
        assert results == {}

    def test_written_count_in_results(self, tmp_path, monkeypatch):
        import revision.data.export_conformers as ecm
        monkeypatch.setattr(ecm, "generate_conformer", self._mock_gen)
        # Do NOT mock rdkit_available; use only SMILES whose heavy-atom count
        # matches the mock positions (CC=2 atoms, CCO=3 atoms) to avoid the
        # symbol/position mismatch warning that would cause molecules to be
        # skipped when RDKit IS installed and returns the real symbol count.
        monkeypatch.setattr(ecm, "obabel_available", lambda: False)

        ds_dir = tmp_path / "datasets"
        ds_dir.mkdir()
        # Three molecules whose atom counts match mock positions
        smiles_list = [SMILES_ETHANE, SMILES_ETHANOL, SMILES_ETHANE]
        self._write_dataset_csv(ds_dir / "HIV.csv", smiles_list, smiles_col="smiles")
        out_dir = tmp_path / "conformers"

        results = export_conformers(
            datasets_dir=ds_dir, out_dir=out_dir,
            methods=[METHOD_ETKDGv3], datasets=["HIV"], verbose=False,
        )
        assert results["HIV_ETKDGv3"]["written"] == 3

    def test_multiple_methods_produce_separate_files(self, tmp_path, monkeypatch):
        import revision.data.export_conformers as ecm
        monkeypatch.setattr(ecm, "generate_conformer", self._mock_gen)
        monkeypatch.setattr(ecm, "rdkit_available", lambda: False)
        monkeypatch.setattr(ecm, "obabel_available", lambda: False)

        ds_dir = tmp_path / "datasets"
        ds_dir.mkdir()
        self._write_dataset_csv(ds_dir / "BACE.csv", [SMILES_ETHANE], smiles_col="mol")
        out_dir = tmp_path / "conformers"

        # rdkit_available=False -> all rdkit methods skipped
        results = export_conformers(
            datasets_dir=ds_dir, out_dir=out_dir,
            methods=[METHOD_ETKDGv3], datasets=["BACE"], verbose=False,
        )
        # ETKDGv3 is a RDKit method; rdkit_available=False -> skipped
        assert not (out_dir / "BACE_ETKDGv3.sdf").exists()

    def test_unknown_dataset_warns(self, tmp_path, monkeypatch):
        import revision.data.export_conformers as ecm
        monkeypatch.setattr(ecm, "generate_conformer", self._mock_gen)
        monkeypatch.setattr(ecm, "rdkit_available", lambda: False)
        monkeypatch.setattr(ecm, "obabel_available", lambda: False)

        ds_dir = tmp_path / "datasets"
        ds_dir.mkdir()
        out_dir = tmp_path / "conformers"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            export_conformers(
                datasets_dir=ds_dir, out_dir=out_dir,
                methods=[METHOD_ETKDGv3], datasets=["NOTADATASET"],
                verbose=False,
            )
        assert any("Unknown dataset" in str(x.message) for x in w)


# ============================================================================
# 6. main() CLI — monkeypatched
# ============================================================================

class TestMainCLI:
    def _write_smiles(self, path: Path, smiles_list: List[str],
                      smiles_col: str = "smiles") -> None:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[smiles_col])
            w.writeheader()
            for s in smiles_list:
                w.writerow({smiles_col: s})

    def _mock_gen(self, smiles, method=METHOD_ETKDGv3, seed=42, **kw):
        return _make_result(method, smiles, _ETHANE_POS)

    def test_returns_integer(self, tmp_path, monkeypatch):
        import revision.data.export_conformers as ecm
        monkeypatch.setattr(ecm, "generate_conformer", self._mock_gen)
        monkeypatch.setattr(ecm, "rdkit_available", lambda: False)
        monkeypatch.setattr(ecm, "obabel_available", lambda: False)

        ds_dir = tmp_path / "datasets"
        ds_dir.mkdir()
        self._write_smiles(ds_dir / "BACE.csv", [SMILES_ETHANE])
        rc = main(["--datasets-dir", str(ds_dir),
                   "--out-dir", str(tmp_path / "out"),
                   "--datasets", "BACE",
                   "--methods", METHOD_ETKDGv3])
        assert isinstance(rc, int)

    def test_returns_one_on_missing_datasets_dir(self, tmp_path):
        """main() must return 1 when --datasets-dir doesn't exist."""
        missing = tmp_path / "no_such_dir"
        rc = main(["--datasets-dir", str(missing),
                   "--out-dir", str(tmp_path / "out")])
        assert rc == 1

    def test_zero_on_no_datasets(self, tmp_path, monkeypatch):
        import revision.data.export_conformers as ecm
        monkeypatch.setattr(ecm, "rdkit_available", lambda: False)
        monkeypatch.setattr(ecm, "obabel_available", lambda: False)

        ds_dir = tmp_path / "datasets"
        ds_dir.mkdir()
        rc = main(["--datasets-dir", str(ds_dir),
                   "--out-dir", str(tmp_path / "out"),
                   "--datasets", "BACE",
                   "--methods", METHOD_ETKDGv3])
        assert isinstance(rc, int)



# ============================================================================
# 7. RDKit round-trip tests (skipped when RDKit not installed)
# ============================================================================

@pytest.mark.skipif(not rdkit_available(), reason="RDKit not installed")
class TestRDKitRoundTrip:
    """Tests that use real RDKit to write and read SDF records."""

    def _mock_gen(self, smiles, method=METHOD_ETKDGv3, seed=42, **kw):
        pos = _ETHANOL_POS if smiles == SMILES_ETHANOL else _ETHANE_POS
        return _make_result(method, smiles, pos)

    def test_sdmolsupplier_reads_exported_sdf(self, tmp_path, monkeypatch):
        import revision.data.export_conformers as ecm
        monkeypatch.setattr(ecm, "generate_conformer", self._mock_gen)

        out = tmp_path / "test.sdf"
        export_one_method(
            records=[(SMILES_ETHANOL, "ethanol")],
            method=METHOD_ETKDGv3, dataset="TEST",
            out_path=out, verbose=False,
        )

        from rdkit.Chem import SDMolSupplier
        suppl = SDMolSupplier(str(out), removeHs=False, sanitize=False)
        mols = [m for m in suppl if m is not None]
        assert len(mols) == 1

    def test_atom_count_preserved(self, tmp_path, monkeypatch):
        import revision.data.export_conformers as ecm
        monkeypatch.setattr(ecm, "generate_conformer", self._mock_gen)

        out = tmp_path / "test.sdf"
        export_one_method(
            records=[(SMILES_ETHANOL, "ethanol")],
            method=METHOD_ETKDGv3, dataset="TEST",
            out_path=out, verbose=False,
        )

        from rdkit.Chem import SDMolSupplier
        suppl = SDMolSupplier(str(out), removeHs=False, sanitize=False)
        mol = next(m for m in suppl if m is not None)
        assert mol.GetNumAtoms() == 3

    def test_smiles_sd_property_present(self, tmp_path, monkeypatch):
        import revision.data.export_conformers as ecm
        monkeypatch.setattr(ecm, "generate_conformer", self._mock_gen)

        out = tmp_path / "test.sdf"
        export_one_method(
            records=[(SMILES_ETHANOL, "ethanol")],
            method=METHOD_ETKDGv3, dataset="TEST",
            out_path=out, verbose=False,
        )

        from rdkit.Chem import SDMolSupplier
        suppl = SDMolSupplier(str(out), removeHs=False, sanitize=False)
        mol = next(m for m in suppl if m is not None)
        assert mol.HasProp("smiles")
        assert mol.GetProp("smiles") == SMILES_ETHANOL

    def test_method_sd_property_present(self, tmp_path, monkeypatch):
        import revision.data.export_conformers as ecm
        monkeypatch.setattr(ecm, "generate_conformer", self._mock_gen)

        out = tmp_path / "test.sdf"
        export_one_method(
            records=[(SMILES_ETHANOL, "ethanol")],
            method=METHOD_ETKDGv3, dataset="TEST",
            out_path=out, verbose=False,
        )

        from rdkit.Chem import SDMolSupplier
        suppl = SDMolSupplier(str(out), removeHs=False, sanitize=False)
        mol = next(m for m in suppl if m is not None)
        assert mol.HasProp("method")
        assert mol.GetProp("method") == METHOD_ETKDGv3

    def test_3d_conformer_attached(self, tmp_path, monkeypatch):
        import revision.data.export_conformers as ecm
        monkeypatch.setattr(ecm, "generate_conformer", self._mock_gen)

        out = tmp_path / "test.sdf"
        export_one_method(
            records=[(SMILES_ETHANOL, "ethanol")],
            method=METHOD_ETKDGv3, dataset="TEST",
            out_path=out, verbose=False,
        )

        from rdkit.Chem import SDMolSupplier
        suppl = SDMolSupplier(str(out), removeHs=False, sanitize=False)
        mol = next(m for m in suppl if m is not None)
        assert mol.GetNumConformers() > 0

    def test_multi_molecule_round_trip(self, tmp_path, monkeypatch):
        import revision.data.export_conformers as ecm
        monkeypatch.setattr(ecm, "generate_conformer", self._mock_gen)

        out = tmp_path / "test.sdf"
        export_one_method(
            records=[(SMILES_ETHANE, "ethane"), (SMILES_ETHANOL, "ethanol")],
            method=METHOD_ETKDGv3, dataset="TEST",
            out_path=out, verbose=False,
        )

        from rdkit.Chem import SDMolSupplier
        suppl = SDMolSupplier(str(out), removeHs=False, sanitize=False)
        mols = [m for m in suppl if m is not None]
        assert len(mols) == 2
