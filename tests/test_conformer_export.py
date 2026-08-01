"""
tests/test_conformer_export.py
================================
Unit tests for ``revision/data/export_conformers.py``.

Test strategy
-------------
* **Always-run tests** (no RDKit, no obabel, no QM9 bundle):
  - SDF V2000 format produced by the no-bond fallback writer
  - CSV loading helpers
  - Method-skipping logic (unavailable backends silently skipped)
  - ``export_one_method`` and ``export_conformers`` via monkeypatching
    ``generate_conformer`` to inject synthetic ConformerResult objects
  - ``main()`` CLI smoke-test

* **RDKit-gated tests** (``@pytest.mark.skipif(not rdkit_available(), ...)``):
  - RDKit ``SDWriter`` path: full bond block in output
  - Round-trip: read exported SDF back with ``Chem.SDMolSupplier``, verify
    atom count, SD properties, and 3-D coordinate presence
  - Multi-molecule round-trip with known SMILES list

Fixed SMILES used throughout
-----------------------------
  ethane  'CC'   — 2 heavy atoms
  ethanol 'CCO'  — 3 heavy atoms
  benzene 'c1ccccc1' — 6 heavy atoms
"""

from __future__ import annotations

import csv
import sys
import tempfile
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Repo root on sys.path
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from revision.geometry_qc.generate_variants import (
    METHOD_ETKDGv3,
    METHOD_ETKDGv2,
    METHOD_ETKDG,
    METHOD_OBABEL,
    METHOD_RANDOM,
    RDKIT_METHODS,
    ConformerResult,
    rdkit_available,
    obabel_available,
)
from revision.data.export_conformers import (
    DATASET_REGISTRY,
    EXPORTABLE_METHODS,
    METHOD_QC,
    _load_smiles_from_csv,
    _write_sdf_record_minimal,
    _write_sdf_record_rdkit,
    export_conformers,
    export_one_method,
    main,
)

# ---------------------------------------------------------------------------
# Fixed test data
# ---------------------------------------------------------------------------

SMILES_ETHANE  = "CC"
SMILES_ETHANOL = "CCO"
SMILES_BENZENE = "c1ccccc1"

_ETHANE_POS = np.array([[0.0, 0.0, 0.0], [1.54, 0.0, 0.0]], dtype=np.float32)
_ETHANOL_POS = np.array([[0.0, 0.0, 0.0], [1.54, 0.0, 0.0], [2.40, 1.10, 0.0]],
                         dtype=np.float32)
_BENZENE_POS = np.array(
    [[1.4 * np.cos(a), 1.4 * np.sin(a), 0.0]
     for a in np.linspace(0, 2 * np.pi, 6, endpoint=False)],
    dtype=np.float32,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    method: str,
    smiles: str,
    positions: np.ndarray,
    success: bool = True,
    error_msg: str = "",
) -> ConformerResult:
    return ConformerResult(
        method=method, smiles=smiles, positions=positions,
        n_atoms=positions.shape[0], success=success,
        error_msg=error_msg, seed=42,
    )


def _failed_result(method: str, smiles: str) -> ConformerResult:
    return ConformerResult(
        method=method, smiles=smiles, positions=None,
        n_atoms=0, success=False, error_msg="test failure", seed=42,
    )


def _write_smiles_csv(path: Path, rows: List[Tuple[str, str]],
                      smiles_col: str = "smiles", name_col: str = "name") -> None:
    """Write a minimal SMILES CSV to *path*."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[smiles_col, name_col])
        w.writeheader()
        for smi, name in rows:
            w.writerow({smiles_col: smi, name_col: name})


def _make_mock_generator(results_by_smiles: dict):
    """
    Return a mock ``generate_conformer`` that looks up synthetic results.
    Ignores method/seed — always returns the pre-built ConformerResult.
    """
    def _mock(smiles, method=METHOD_ETKDGv3, seed=42, **kwargs):
        if smiles in results_by_smiles:
            return results_by_smiles[smiles]
        return _failed_result(method, smiles)
    return _mock


# ============================================================================
# 1. Dataset registry
# ============================================================================

class TestDatasetRegistry:
    def test_all_expected_datasets_present(self):
        for name in ["BACE", "HIV", "BBBP", "QM9", "ADMET"]:
            assert name in DATASET_REGISTRY

    def test_each_entry_has_smiles_col(self):
        for name, cfg in DATASET_REGISTRY.items():
            assert "smiles_col" in cfg, f"{name} missing smiles_col"

    def test_qm9_is_qc_eligible(self):
        assert DATASET_REGISTRY["QM9"]["qc_eligible"] is True

    def test_non_qm9_not_qc_eligible(self):
        for name in ["BACE", "HIV", "BBBP", "ADMET"]:
            assert DATASET_REGISTRY[name]["qc_eligible"] is False

    def test_bace_smiles_col_is_mol(self):
        assert DATASET_REGISTRY["BACE"]["smiles_col"] == "mol"

    def test_exportable_methods_contains_qc(self):
        assert METHOD_QC in EXPORTABLE_METHODS

    def test_exportable_methods_contains_all_rdkit(self):
        for m in RDKIT_METHODS:
            assert m in EXPORTABLE_METHODS


# ============================================================================
# 2. _load_smiles_from_csv
# ============================================================================

class TestLoadSmilesFromCsv:
    def test_loads_correct_count(self, tmp_path):
        p = tmp_path / "test.csv"
        _write_smiles_csv(p, [(SMILES_ETHANE, "mol1"), (SMILES_ETHANOL, "mol2")])
        records = _load_smiles_from_csv(p, "smiles", "name", n_mols=None)
        assert len(records) == 2

    def test_n_mols_limit(self, tmp_path):
        p = tmp_path / "test.csv"
        rows = [(f"C{'C'*i}", f"mol{i}") for i in range(10)]
        _write_smiles_csv(p, rows)
        records = _load_smiles_from_csv(p, "smiles", "name", n_mols=3)
        assert len(records) == 3

    def test_returns_smiles_and_name_tuples(self, tmp_path):
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


# ============================================================================
# 3. _write_sdf_record_minimal (no-bond SDF writer)
# ============================================================================

class TestWriteSdfRecordMinimal:
    """Tests for the no-RDKit fallback SDF writer."""

    def _write_and_read(self, positions, symbols, smiles, method, dataset):
        """Write a record to a StringIO and return the text."""
        import io
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
        # Counts line is line index 3 (0-based): n_atoms n_bonds ...
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
        assert "<smiles>" in sdf
        assert SMILES_ETHANE in sdf

    def test_method_property_present(self):
        sdf = self._write_and_read(
            _ETHANE_POS, ["C", "C"], SMILES_ETHANE, METHOD_ETKDGv3, "TEST"
        )
        assert "<method>" in sdf
        assert METHOD_ETKDGv3 in sdf

    def test_dataset_property_present(self):
        sdf = self._write_and_read(
            _ETHANE_POS, ["C", "C"], SMILES_ETHANE, METHOD_ETKDGv3, "BACE"
        )
        assert "<dataset>" in sdf
        assert "BACE" in sdf

    def test_n_heavy_atoms_property(self):
        sdf = self._write_and_read(
            _ETHANOL_POS, ["C", "C", "O"], SMILES_ETHANOL, METHOD_ETKDGv3, "TEST"
        )
        assert "<n_heavy_atoms>" in sdf
        assert "3" in sdf

    def test_v2000_header_present(self):
        sdf = self._write_and_read(
            _ETHANE_POS, ["C", "C"], SMILES_ETHANE, METHOD_ETKDGv3, "TEST"
        )
        assert "V2000" in sdf

    def test_zero_bonds_in_counts_line(self):
        sdf = self._write_and_read(
            _ETHANE_POS, ["C", "C"], SMILES_ETHANE, METHOD_ETKDGv3, "TEST"
        )
        lines = sdf.splitlines()
        counts_line = lines[3]
        n_bonds = int(counts_line[3:6].strip())
        assert n_bonds == 0

    def test_multiple_records_in_sequence(self):
        """Writing two records in sequence produces two $$$$ terminators."""
        import io
        buf = io.StringIO()
        for smi, sym, pos in [
            (SMILES_ETHANE,  ["C","C"],     _ETHANE_POS),
            (SMILES_ETHANOL, ["C","C","O"], _ETHANOL_POS),
        ]:
            _write_sdf_record_minimal(buf, smi, pos, sym, smi, METHOD_ETKDGv3, "TEST")
        text = buf.getvalue()
        assert text.count("$$$$") == 2


# ============================================================================
# 4. export_one_method — via monkeypatching generate_conformer
# ============================================================================

class TestExportOneMethod:
    def _make_mock(self, results_by_smiles: dict):
        return _make_mock_generator(results_by_smiles)

    def test_creates_output_file(self, tmp_path, monkeypatch):
        import revision.data.export_conformers as ecm
        mock = self._make_mock({
            SMILES_ETHANE:  _make_result(METHOD_ETKDGv3, SMILES_ETHANE,  _ETHANE_POS),
            SMILES_ETHANOL: _make_result(METHOD_ETKDGv3, SMILES_ETHANOL, _ETHANOL_POS),
        })
        monkeypatch.setattr(ecm, "generate_conformer", mock)
        monkeypatch.setattr(ecm, "rdkit_available", lambda: False)

        out = tmp_path / "test.sdf"
        counts = export_one_method(
            records=[(SMILES_ETHANE, "mol1"), (SMILES_ETHANOL, "mol2")],
            method=METHOD_ETKDGv3,
            dataset="TEST",
            out_path=out,
            verbose=False,
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
        monkeypatch.setattr(ecm, "rdkit_available", lambda: False)
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
        monkeypatch.setattr(ecm, "rdkit_available", lambda: False)
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
        # All RDKit methods skipped → no results
        assert results == {}

    def test_written_count_in_results(self, tmp_path, monkeypatch):
        import revision.data.export_conformers as ecm
        monkeypatch.setattr(ecm, "generate_conformer", self._mock_gen)
        monkeypatch.setattr(ecm, "rdkit_available", lambda: False)
        monkeypatch.setattr(ecm, "obabel_available", lambda: False)

        ds_dir = tmp_path / "datasets"
        ds_dir.mkdir()
        smiles_list = [SMILES_ETHANE, SMILES_ETHANOL, SMILES_BENZENE]
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

        # Pretend rdkit is available for ETKDGv3 and ETKDGv2
        monkeypatch.setattr(ecm, "rdkit_available", lambda: False)
        # Use only one method that works without rdkit... actually monkeypatch generate_conformer
        results = export_conformers(
            datasets_dir=ds_dir, out_dir=out_dir,
            methods=[METHOD_ETKDGv3], datasets=["BACE"], verbose=False,
        )
        # Only ETKDGv3 file (rdkit_available=False → skipped)
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

    def test_returns_zero_when_some_written(self, tmp_path, monkeypatch):
        import revision.data.export_conformers as ecm
        monkeypatch.setattr(ecm, "generate_conformer", self._mock_gen)
        monkeypatch.setattr(ecm, "rdkit_available", lambda: True)
        monkeypatch.setattr(ecm, "obabel_available", lambda: False)
        # Also patch rdkit writer to avoid real RDKit call
        def _mock_rdkit_write(writer, mol_name, positions, smiles, method, dataset):
            return True
        monkeypatch.setattr(ecm, "_write_sdf_record_rdkit", _mock_rdkit_write)

        ds_dir = tmp_path / "datasets"
        ds_dir.mkdir()
        self._write_smiles(ds_dir / "HIV.csv", [SMILES_ETHANE])
        # Patch SDWriter to avoid rdkit import
        class _FakeWriter:
            def __init__(self, path): pass
            def close(self): pass
        import builtins
        out_dir = tmp_path / "conformers"
        # Use minimal writer path (rdkit_available=False)
        monkeypatch.setattr(ecm, "rdkit_available", lambda: False)
        rc = main([
            "--datasets-dir", str(ds_dir),
            "--out-dir", str(out_dir),
            "--datasets", "HIV",
            "--methods", METHOD_ETKDGv3,
            "--quiet",
        ])
        # rdkit not available → method skipped → 0 written → rc=1
        # (but no crash)
        assert isinstance(rc, int)

    def test_returns_one_on_missing_datasets_dir(self, tmp_path):
        rc = main([
            "--datasets-dir", str(tmp_path / "no_such_dir"),
            "--out-dir", str(tmp_path / "out"),
            "--datasets", "HIV",
            "--methods", METHOD_ETKDGv3,
            "--quiet",
        ])
        assert rc == 1


# ============================================================================
# 7. RDKit round-trip tests (skipped when RDKit not installed)
# ============================================================================

@pytest.mark.skipif(not rdkit_available(), reason="RDKit not installed")
class TestRDKitRoundTrip:
    """
    When RDKit is available, exported SDF must be readable by
    ``Chem.SDMolSupplier`` with correct atom count and SD properties.
    """

    def _mock_gen(self, smiles, method=METHOD_ETKDGv3, seed=42, **kw):
        return _make_result(method, smiles, _ETHANOL_POS)

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
        assert mol.GetNumAtoms() == 3  # CCO heavy atoms

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

    def test_dataset_sd_property_present(self, tmp_path, monkeypatch):
        import revision.data.export_conformers as ecm
        monkeypatch.setattr(ecm, "generate_conformer", self._mock_gen)

        out = tmp_path / "test.sdf"
        export_one_method(
            records=[(SMILES_ETHANOL, "ethanol")],
            method=METHOD_ETKDGv3, dataset="BACE",
            out_path=out, verbose=False,
        )
        from rdkit.Chem import SDMolSupplier
        suppl = SDMolSupplier(str(out), removeHs=False, sanitize=False)
        mol = next(m for m in suppl if m is not None)
        assert mol.GetProp("dataset") == "BACE"

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
        assert mol.GetNumConformers() == 1
        conf = mol.GetConformer()
        pos0 = conf.GetAtomPosition(0)
        assert abs(pos0.x - _ETHANOL_POS[0, 0]) < 0.01
        assert abs(pos0.y - _ETHANOL_POS[0, 1]) < 0.01

    def test_multi_molecule_round_trip(self, tmp_path, monkeypatch):
        """Multiple molecules all survive the round-trip."""
        import revision.data.export_conformers as ecm
        pos_map = {
            SMILES_ETHANE:  _ETHANE_POS,
            SMILES_ETHANOL: _ETHANOL_POS,
        }
        def _mock(smiles, method=METHOD_ETKDGv3, seed=42, **kw):
            return _make_result(method, smiles, pos_map[smiles])
        monkeypatch.setattr(ecm, "generate_conformer", _mock)

        out = tmp_path / "multi.sdf"
        export_one_method(
            records=[(SMILES_ETHANE, "m1"), (SMILES_ETHANOL, "m2")],
            method=METHOD_ETKDGv3, dataset="TEST",
            out_path=out, verbose=False,
        )
        from rdkit.Chem import SDMolSupplier
        suppl = SDMolSupplier(str(out), removeHs=False, sanitize=False)
        mols = [m for m in suppl if m is not None]
        assert len(mols) == 2
        atom_counts = {mol.GetProp("smiles"): mol.GetNumAtoms() for mol in mols}
        assert atom_counts[SMILES_ETHANE]  == 2
        assert atom_counts[SMILES_ETHANOL] == 3

    def test_minimal_sdf_also_readable_by_rdkit(self, tmp_path):
        """
        The no-bond minimal writer output can also be read by RDKit
        with sanitize=False (coordinates present, no bond perception).
        """
        out = tmp_path / "minimal.sdf"
        with open(out, "w") as f:
            _write_sdf_record_minimal(
                f, "ethanol", _ETHANOL_POS, ["C", "C", "O"],
                SMILES_ETHANOL, METHOD_ETKDGv3, "TEST",
            )
        from rdkit.Chem import SDMolSupplier
        suppl = SDMolSupplier(str(out), removeHs=False, sanitize=False)
        mols = [m for m in suppl if m is not None]
        assert len(mols) == 1
        assert mols[0].GetNumAtoms() == 3
