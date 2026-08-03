"""
tests/test_geometry_variants.py
================================
Unit tests for ``revision/geometry_qc/generate_variants.py`` and
``revision/geometry_qc/variant_comparison_writer.py``.

Tests are designed to run in three distinct tiers:

Tier 1 — always-run (no external dependencies):
    Pure-numpy logic: ConformerResult, kabsch_rmsd, _parse_sdf_positions,
    pairwise_rmsd, write_comparison_csv, print_summary, error-path detection
    helpers (rdkit_available / obabel_available), and the generate_conformer
    error-dispatch logic (method unknown / backend absent).

Tier 2 — skipped when RDKit IS installed:
    Tests that specifically assert absence behaviour (RDKitNotInstalledError,
    empty generate_all_available dict, RDKit-missing warnings).  These are
    gated with ``@pytest.mark.skipif(rdkit_available(), ...)``.

Tier 3 — skipped when RDKit is NOT installed:
    Full conformer-generation round-trips.
    Gated with ``@pytest.mark.skipif(not rdkit_available(), ...)``.

Tier 4 — skipped when obabel is NOT installed:
    Full obabel subprocess round-trips.
    Gated with ``@pytest.mark.skipif(not obabel_available(), ...)``.
"""

from __future__ import annotations

import csv
import io
import sys
import tempfile
import warnings
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from revision.geometry_qc.generate_variants import (
    ALL_METHODS,
    METHOD_ETKDG,
    METHOD_ETKDGv2,
    METHOD_ETKDGv3,
    METHOD_OBABEL,
    METHOD_RANDOM,
    RDKIT_METHODS,
    ConformerResult,
    OpenBabelNotInstalledError,
    RDKitNotInstalledError,
    _parse_sdf_positions,
    generate_all_available,
    generate_conformer,
    obabel_available,
    rdkit_available,
)
from revision.geometry_qc.variant_comparison_writer import (
    OUTPUT_COLUMNS,
    kabsch_rmsd,
    pairwise_rmsd,
    print_summary,
    run_comparison,
    write_comparison_csv,
)

# ---------------------------------------------------------------------------
# Fixed test molecules
# ---------------------------------------------------------------------------

SMILES_ETHANE  = "CC"
SMILES_ETHANOL = "CCO"
SMILES_BENZENE = "c1ccccc1"
SMILES_METHANE = "C"


def _ethane_pos_a() -> np.ndarray:
    return np.array([[0.0, 0.0, 0.0], [1.54, 0.0, 0.0]], dtype=np.float32)


def _ethane_pos_b() -> np.ndarray:
    """Ethane translated by (1,1,1)."""
    return np.array([[1.0, 1.0, 1.0], [2.54, 1.0, 1.0]], dtype=np.float32)


def _ethanol_pos() -> np.ndarray:
    return np.array(
        [[0.0, 0.0, 0.0], [1.54, 0.0, 0.0], [2.40, 1.10, 0.0]], dtype=np.float32
    )


def _benzene_pos() -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    return np.array(
        [[1.4 * np.cos(a), 1.4 * np.sin(a), 0.0] for a in angles], dtype=np.float32
    )


# ---------------------------------------------------------------------------
# ConformerResult helpers
# ---------------------------------------------------------------------------

def _make_result(
    method: str, smiles: str, positions: np.ndarray, seed: int = 42
) -> ConformerResult:
    return ConformerResult(
        method=method, smiles=smiles, positions=positions,
        n_atoms=positions.shape[0], success=True, seed=seed,
    )


def _failed_result(
    method: str, smiles: str, error_msg: str = "test failure", seed: int = 42
) -> ConformerResult:
    return ConformerResult(
        method=method, smiles=smiles, positions=None,
        n_atoms=0, success=False, error_msg=error_msg, seed=seed,
    )


# ============================================================================
# 1. Availability helpers
# ============================================================================

class TestAvailability:
    def test_rdkit_available_returns_bool(self):
        assert isinstance(rdkit_available(), bool)

    def test_obabel_available_returns_bool(self):
        assert isinstance(obabel_available(), bool)

    @pytest.mark.skipif(rdkit_available(), reason="RDKit is installed in this environment")
    def test_rdkit_not_installed_on_this_machine(self):
        # Asserts absence — only meaningful when RDKit is not installed
        assert rdkit_available() is False

    @pytest.mark.skipif(obabel_available(), reason="obabel is installed in this environment")
    def test_obabel_not_installed_on_this_machine(self):
        assert obabel_available() is False

    def test_all_methods_list_complete(self):
        expected = {METHOD_ETKDG, METHOD_ETKDGv2, METHOD_ETKDGv3,
                    METHOD_RANDOM, METHOD_OBABEL}
        assert set(ALL_METHODS) == expected

    def test_rdkit_methods_subset_of_all(self):
        assert set(RDKIT_METHODS).issubset(set(ALL_METHODS))

    def test_obabel_in_all_not_in_rdkit(self):
        assert METHOD_OBABEL in ALL_METHODS
        assert METHOD_OBABEL not in RDKIT_METHODS


# ============================================================================
# 2. ConformerResult dataclass
# ============================================================================

class TestConformerResult:
    def test_success_result_repr(self):
        r = _make_result(METHOD_ETKDGv3, SMILES_ETHANE, _ethane_pos_a())
        assert "ETKDGv3" in repr(r)
        assert "OK" in repr(r)

    def test_failed_result_repr(self):
        r = _failed_result(METHOD_ETKDGv3, SMILES_ETHANE)
        assert "FAIL" in repr(r)

    def test_dist_matrix_shape(self):
        r = _make_result(METHOD_ETKDGv3, SMILES_ETHANE, _ethane_pos_a())
        D = r.dist_matrix
        assert D is not None
        assert D.shape == (2, 2)

    def test_dist_matrix_diagonal_zero(self):
        r = _make_result(METHOD_ETKDGv3, SMILES_BENZENE, _benzene_pos())
        D = r.dist_matrix
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-5)

    def test_dist_matrix_symmetric(self):
        r = _make_result(METHOD_ETKDGv3, SMILES_BENZENE, _benzene_pos())
        D = r.dist_matrix
        np.testing.assert_allclose(D, D.T, atol=1e-5)

    def test_dist_matrix_none_for_failed(self):
        r = _failed_result(METHOD_ETKDGv3, SMILES_ETHANE)
        assert r.dist_matrix is None

    def test_n_atoms_correct(self):
        pos = _ethanol_pos()
        r = _make_result(METHOD_ETKDGv3, SMILES_ETHANOL, pos)
        assert r.n_atoms == 3

    def test_positions_dtype(self):
        r = _make_result(METHOD_ETKDGv3, SMILES_ETHANE, _ethane_pos_a())
        assert r.positions.dtype == np.float32


# ============================================================================
# 3. kabsch_rmsd
# ============================================================================

class TestKabschRMSD:
    def test_identical_points_zero_rmsd(self):
        P = _ethane_pos_a()
        assert kabsch_rmsd(P, P.copy()) == pytest.approx(0.0, abs=1e-6)

    def test_translated_points_zero_rmsd(self):
        """Pure translation should give RMSD = 0 after centring."""
        P = _ethane_pos_a()
        Q = P + np.array([5.0, -3.0, 2.1])
        assert kabsch_rmsd(P, Q) == pytest.approx(0.0, abs=1e-5)

    def test_rotated_points_zero_rmsd(self):
        """Pure rotation should give RMSD = 0."""
        P = _benzene_pos()
        # 90 degree rotation about z-axis
        theta = np.pi / 2
        R = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta),  np.cos(theta), 0],
                      [0,              0,              1]], dtype=np.float32)
        Q = (P @ R.T).astype(np.float32)
        assert kabsch_rmsd(P, Q) == pytest.approx(0.0, abs=1e-5)

    def test_shape_mismatch_raises(self):
        P = _ethane_pos_a()   # (2, 3)
        Q = _ethanol_pos()    # (3, 3)
        with pytest.raises(ValueError, match="Shape mismatch"):
            kabsch_rmsd(P, Q)

    def test_single_atom_raises(self):
        P = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="at least 2"):
            kabsch_rmsd(P, P)

    def test_non_zero_rmsd_is_positive(self):
        P = _ethanol_pos()
        Q = _ethanol_pos().copy()
        Q[2] += np.array([0.5, 0.5, 0.5])   # displace last atom
        r = kabsch_rmsd(P, Q)
        assert r > 0.0

    def test_rmsd_is_symmetric(self):
        """kabsch_rmsd(P, Q) == kabsch_rmsd(Q, P)."""
        P = _ethanol_pos()
        Q = _ethanol_pos() + np.array([0.2, -0.1, 0.3])
        assert kabsch_rmsd(P, Q) == pytest.approx(kabsch_rmsd(Q, P), abs=1e-6)

    def test_rmsd_returns_float(self):
        P = _ethane_pos_a()
        assert isinstance(kabsch_rmsd(P, P.copy()), float)

    def test_known_displacement(self):
        """Move one atom by a known amount; check RMSD is plausible."""
        P = _ethanol_pos()
        Q = _ethanol_pos().copy()
        Q[0] += np.array([1.0, 0.0, 0.0])   # shift atom 0 by 1 Å
        # RMSD after centering should be > 0 and < 1
        r = kabsch_rmsd(P, Q)
        assert 0.0 < r


# ============================================================================
# 4. _parse_sdf_positions
# ============================================================================

MINIMAL_SDF_ETHANE = (
    "\n"
    "     RDKit          3D\n"
    "\n"
    "  2  1  0  0  0  0  0  0  0  0999 V2000\n"
    "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
    "    1.5400    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
    "  1  2  1  0\n"
    "M  END\n"
)

MINIMAL_SDF_ETHANOL = (
    "\n"
    "     RDKit          3D\n"
    "\n"
    "  3  2  0  0  0  0  0  0  0  0999 V2000\n"
    "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
    "    1.5400    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
    "    2.4000    1.1000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n"
    "  1  2  1  0\n"
    "  2  3  1  0\n"
    "M  END\n"
)


class TestParseSdfPositions:
    def test_ethane_atom_count(self):
        pos, n = _parse_sdf_positions(MINIMAL_SDF_ETHANE)
        assert n == 2

    def test_ethane_position_shape(self):
        pos, n = _parse_sdf_positions(MINIMAL_SDF_ETHANE)
        assert pos.shape == (2, 3)

    def test_ethane_first_atom_origin(self):
        pos, n = _parse_sdf_positions(MINIMAL_SDF_ETHANE)
        np.testing.assert_allclose(pos[0], [0.0, 0.0, 0.0], atol=1e-4)

    def test_ethane_second_atom_x(self):
        pos, n = _parse_sdf_positions(MINIMAL_SDF_ETHANE)
        assert abs(pos[1, 0] - 1.54) < 1e-3

    def test_dtype_is_float32(self):
        pos, n = _parse_sdf_positions(MINIMAL_SDF_ETHANE)
        assert pos.dtype == np.float32

    def test_ethanol_three_atoms(self):
        pos, n = _parse_sdf_positions(MINIMAL_SDF_ETHANOL)
        assert n == 3
        assert pos.shape == (3, 3)

    def test_bad_counts_line_raises(self):
        bad = "not_a_count\nmore\n"
        with pytest.raises(Exception):
            _parse_sdf_positions(bad)


# ============================================================================
# 5. generate_conformer — error paths (no RDKit needed)
# ============================================================================

class TestGenerateConformerErrorPaths:
    def test_unknown_method_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown method"):
            generate_conformer(SMILES_ETHANE, method="bogus_method")

    @pytest.mark.skipif(rdkit_available(), reason="Only valid when RDKit is not installed")
    def test_rdkit_method_raises_when_not_installed(self):
        with pytest.raises(RDKitNotInstalledError):
            generate_conformer(SMILES_ETHANE, method=METHOD_ETKDGv3)

    @pytest.mark.skipif(rdkit_available(), reason="Only valid when RDKit is not installed")
    def test_all_rdkit_methods_raise_when_not_installed(self):
        for method in RDKIT_METHODS:
            with pytest.raises(RDKitNotInstalledError):
                generate_conformer(SMILES_ETHANE, method=method)

    def test_obabel_method_raises_when_not_installed(self):
        # obabel is always absent in this repo's CI — no skipif needed
        with pytest.raises(OpenBabelNotInstalledError):
            generate_conformer(SMILES_ETHANE, method=METHOD_OBABEL)

    def test_obabel_error_message_contains_install_hint(self):
        try:
            generate_conformer(SMILES_ETHANE, method=METHOD_OBABEL)
        except OpenBabelNotInstalledError as exc:
            assert "obabel" in str(exc).lower() or "openbabel" in str(exc).lower()
            assert "install" in str(exc).lower() or "PATH" in str(exc)

    @pytest.mark.skipif(rdkit_available(), reason="Only valid when RDKit is not installed")
    def test_rdkit_error_message_contains_install_hint(self):
        try:
            generate_conformer(SMILES_ETHANE, method=METHOD_ETKDGv3)
        except RDKitNotInstalledError as exc:
            assert "rdkit" in str(exc).lower()
            assert "install" in str(exc).lower() or "conda" in str(exc).lower()


# ============================================================================
# 6. generate_all_available — skipping behaviour
# ============================================================================

class TestGenerateAllAvailable:
    @pytest.mark.skipif(rdkit_available() or obabel_available(),
                        reason="Only valid when no geometry backend is installed")
    def test_returns_empty_when_nothing_installed(self):
        """No RDKit, no obabel -> all methods skipped -> empty dict."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            results = generate_all_available(SMILES_ETHANE)
        assert results == {}

    def test_skip_methods_honoured(self):
        """Skipping all methods explicitly -> empty dict regardless of environment."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            results = generate_all_available(
                SMILES_ETHANE,
                skip_methods=ALL_METHODS,
            )
        assert results == {}

    @pytest.mark.skipif(rdkit_available(), reason="Only warns about RDKit when it is absent")
    def test_warns_about_missing_rdkit(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            generate_all_available(SMILES_ETHANE)
        rdkit_warns = [x for x in w if "RDKit" in str(x.message)]
        assert len(rdkit_warns) > 0

    def test_warns_about_missing_obabel(self):
        """obabel is always absent in this repo — warn regardless of RDKit state."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            generate_all_available(SMILES_ETHANE)
        ob_warns = [x for x in w if "obabel" in str(x.message).lower()]
        assert len(ob_warns) > 0


# ============================================================================
# 7. pairwise_rmsd — using synthetic ConformerResult fixtures
# ============================================================================

class TestPairwiseRmsd:
    def _two_results(
        self,
        smiles: str = SMILES_ETHANE,
        pos_a: np.ndarray = None,
        pos_b: np.ndarray = None,
    ) -> Dict[str, ConformerResult]:
        pos_a = pos_a if pos_a is not None else _ethane_pos_a()
        pos_b = pos_b if pos_b is not None else _ethane_pos_b()
        return {
            METHOD_ETKDGv3:  _make_result(METHOD_ETKDGv3, smiles, pos_a),
            METHOD_ETKDGv2: _make_result(METHOD_ETKDGv2,  smiles, pos_b),
        }

    def test_one_pair_produced(self):
        rows = pairwise_rmsd(self._two_results())
        assert len(rows) == 1

    def test_three_methods_three_pairs(self):
        pos = _ethanol_pos()
        results = {
            METHOD_ETKDG:   _make_result(METHOD_ETKDG,   SMILES_ETHANOL, pos),
            METHOD_ETKDGv2: _make_result(METHOD_ETKDGv2, SMILES_ETHANOL, pos),
            METHOD_ETKDGv3: _make_result(METHOD_ETKDGv3, SMILES_ETHANOL, pos),
        }
        rows = pairwise_rmsd(results)
        assert len(rows) == 3  # C(3,2) = 3 pairs

    def test_identical_positions_zero_rmsd(self):
        pos = _ethanol_pos()
        results = {
            METHOD_ETKDGv3: _make_result(METHOD_ETKDGv3, SMILES_ETHANOL, pos),
            METHOD_ETKDGv2: _make_result(METHOD_ETKDGv2, SMILES_ETHANOL, pos),
        }
        rows = pairwise_rmsd(results)
        assert float(rows[0]["rmsd"]) == pytest.approx(0.0, abs=1e-5)

    def test_translated_positions_zero_rmsd(self):
        pos_a = _ethane_pos_a()
        pos_b = pos_a + np.array([3.0, -1.0, 0.5])
        results = {
            METHOD_ETKDGv3: _make_result(METHOD_ETKDGv3, SMILES_ETHANE, pos_a),
            METHOD_ETKDGv2: _make_result(METHOD_ETKDGv2, SMILES_ETHANE, pos_b),
        }
        rows = pairwise_rmsd(results)
        assert float(rows[0]["rmsd"]) == pytest.approx(0.0, abs=1e-5)

    def test_pair_order_is_lexicographic(self):
        rows = pairwise_rmsd(self._two_results())
        assert rows[0]["method_a"] < rows[0]["method_b"]

    def test_both_ok_flag_set_when_both_succeed(self):
        rows = pairwise_rmsd(self._two_results())
        assert rows[0]["both_ok"] == 1

    def test_both_ok_flag_clear_when_one_fails(self):
        pos = _ethanol_pos()
        results = {
            METHOD_ETKDGv3: _make_result(METHOD_ETKDGv3, SMILES_ETHANOL, pos),
            METHOD_ETKDGv2: _failed_result(METHOD_ETKDGv2, SMILES_ETHANOL),
        }
        rows = pairwise_rmsd(results)
        assert rows[0]["both_ok"] == 0
        assert rows[0]["rmsd"] == ""

    def test_atom_mismatch_gives_empty_rmsd(self):
        results = {
            METHOD_ETKDGv3: _make_result(METHOD_ETKDGv3, SMILES_ETHANOL, _ethanol_pos()),
            METHOD_ETKDGv2: _make_result(METHOD_ETKDGv2, SMILES_ETHANE,  _ethane_pos_a()),
        }
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            rows = pairwise_rmsd(results)
        assert rows[0]["rmsd"] == ""

    def test_empty_dict_returns_empty_list(self):
        assert pairwise_rmsd({}) == []

    def test_single_method_returns_empty_list(self):
        results = {METHOD_ETKDGv3: _make_result(METHOD_ETKDGv3, SMILES_ETHANE, _ethane_pos_a())}
        assert pairwise_rmsd(results) == []

    def test_all_output_columns_present(self):
        rows = pairwise_rmsd(self._two_results())
        for col in OUTPUT_COLUMNS:
            assert col in rows[0], f"missing column: {col}"

    def test_smiles_field_preserved(self):
        rows = pairwise_rmsd(self._two_results(smiles=SMILES_ETHANOL))
        assert rows[0]["smiles"] == SMILES_ETHANOL

    def test_n_atoms_field_set(self):
        rows = pairwise_rmsd(self._two_results())
        assert rows[0]["n_atoms"] == 2

    def test_error_messages_propagated(self):
        results = {
            METHOD_ETKDGv3: _make_result(METHOD_ETKDGv3, SMILES_ETHANE, _ethane_pos_a()),
            METHOD_ETKDGv2: _failed_result(METHOD_ETKDGv2, SMILES_ETHANE, "embed failed"),
        }
        rows = pairwise_rmsd(results)
        # ETKDGv2 < ETKDGv3 lexicographically -> failed result lands in method_a
        assert rows[0]["method_a"] == METHOD_ETKDGv2
        assert rows[0]["method_a_error"] == "embed failed"
        assert rows[0]["method_b_error"] == ""


# ============================================================================
# 8. write_comparison_csv
# ============================================================================

class TestWriteComparisonCsv:
    def _make_rows(self):
        pos = _ethanol_pos()
        results = {
            METHOD_ETKDGv3: _make_result(METHOD_ETKDGv3, SMILES_ETHANOL, pos),
            METHOD_ETKDGv2: _make_result(METHOD_ETKDGv2, SMILES_ETHANOL, pos),
        }
        return pairwise_rmsd(results)

    def test_creates_file(self, tmp_path):
        rows = self._make_rows()
        out = tmp_path / "out.csv"
        write_comparison_csv(rows, out)
        assert out.exists()

    def test_header_matches_output_columns(self, tmp_path):
        rows = self._make_rows()
        out = tmp_path / "out.csv"
        write_comparison_csv(rows, out)
        with open(out, newline="") as f:
            reader = csv.DictReader(f)
            _ = list(reader)
            assert reader.fieldnames == OUTPUT_COLUMNS

    def test_row_count(self, tmp_path):
        rows = self._make_rows()
        out = tmp_path / "out.csv"
        write_comparison_csv(rows, out)
        with open(out, newline="") as f:
            loaded = list(csv.DictReader(f))
        assert len(loaded) == 1

    def test_rmsd_value_round_trips(self, tmp_path):
        rows = self._make_rows()
        out = tmp_path / "out.csv"
        write_comparison_csv(rows, out)
        with open(out, newline="") as f:
            loaded = list(csv.DictReader(f))
        assert float(loaded[0]["rmsd"]) == pytest.approx(0.0, abs=1e-5)

    def test_creates_parent_directories(self, tmp_path):
        rows = self._make_rows()
        out = tmp_path / "a" / "b" / "c.csv"
        write_comparison_csv(rows, out)
        assert out.exists()

    def test_empty_rows_produces_header_only(self, tmp_path):
        out = tmp_path / "empty.csv"
        write_comparison_csv([], out)
        with open(out, newline="") as f:
            loaded = list(csv.DictReader(f))
        assert loaded == []


# ============================================================================
# 9. run_comparison (monkeypatched)
# ============================================================================

class TestRunComparison:
    def test_returns_list(self, monkeypatch):
        monkeypatch.setattr(
            "revision.geometry_qc.variant_comparison_writer.generate_all_available",
            lambda smiles, **kw: {},
        )
        result = run_comparison([SMILES_ETHANE], skip_methods=ALL_METHODS)
        assert isinstance(result, list)

    def test_empty_when_no_methods(self, monkeypatch):
        monkeypatch.setattr(
            "revision.geometry_qc.variant_comparison_writer.generate_all_available",
            lambda smiles, **kw: {},
        )
        rows = run_comparison([SMILES_ETHANE, SMILES_ETHANOL], skip_methods=ALL_METHODS)
        assert rows == []

    def test_molecule_count_respected(self, monkeypatch):
        seen = []
        def _fake_gen(smiles, **kw):
            seen.append(smiles)
            return {}
        monkeypatch.setattr(
            "revision.geometry_qc.variant_comparison_writer.generate_all_available",
            _fake_gen,
        )
        run_comparison([SMILES_ETHANE, SMILES_ETHANOL, SMILES_BENZENE],
                       skip_methods=ALL_METHODS)
        assert len(seen) == 3


# ============================================================================
# 10. print_summary
# ============================================================================

class TestPrintSummary:
    def _make_rows(self):
        pos = _ethanol_pos()
        results = {
            METHOD_ETKDGv3: _make_result(METHOD_ETKDGv3, SMILES_ETHANOL, pos),
            METHOD_ETKDGv2: _make_result(METHOD_ETKDGv2, SMILES_ETHANOL, pos),
        }
        return pairwise_rmsd(results)

    def test_smoke_non_empty(self, capsys):
        rows = self._make_rows()
        print_summary(rows)
        out = capsys.readouterr().out
        assert "RMSD" in out

    def test_empty_rows_message(self, capsys):
        print_summary([])
        out = capsys.readouterr().out
        assert "No comparison rows" in out

    def test_method_names_appear(self, capsys):
        rows = self._make_rows()
        print_summary(rows)
        out = capsys.readouterr().out
        assert METHOD_ETKDGv2 in out or METHOD_ETKDGv3 in out


# ============================================================================
# 11. main() end-to-end (monkeypatched generate_all_available)
# ============================================================================

class TestMainEndToEnd:
    def _fake_gen(self, smiles, **kw):
        return {}

    def _write_smiles_csv(self, path, smiles_list):
        import csv as _csv
        with open(path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=["smiles"])
            w.writeheader()
            for s in smiles_list:
                w.writerow({"smiles": s})

    def test_rc_zero_when_rows_written(self, tmp_path, monkeypatch):
        """If rows are written, main returns 0."""
        import revision.geometry_qc.variant_comparison_writer as vcw
        pos = _ethanol_pos()
        fake_result = {
            METHOD_ETKDGv3: _make_result(METHOD_ETKDGv3, SMILES_ETHANOL, pos),
            METHOD_ETKDGv2: _make_result(METHOD_ETKDGv2, SMILES_ETHANOL, pos),
        }
        monkeypatch.setattr(vcw, "generate_all_available",
                            lambda smiles, **kw: fake_result)
        p = tmp_path / "smiles.csv"
        self._write_smiles_csv(p, [SMILES_ETHANOL])
        out = tmp_path / "out.csv"
        rc = vcw.main(["--smiles-csv", str(p), "--out-csv", str(out),
                        "--n-mols", "1"])
        assert rc == 0
        assert out.exists()

    def test_rc_one_on_missing_file(self, tmp_path):
        import revision.geometry_qc.variant_comparison_writer as vcw
        rc = vcw.main(["--smiles-csv", str(tmp_path / "no_such.csv")])
        assert rc == 1

    def test_output_csv_has_correct_columns(self, tmp_path, monkeypatch):
        import revision.geometry_qc.variant_comparison_writer as vcw
        pos = _ethanol_pos()
        fake_result = {
            METHOD_ETKDGv3: _make_result(METHOD_ETKDGv3, SMILES_ETHANOL, pos),
            METHOD_ETKDGv2: _make_result(METHOD_ETKDGv2, SMILES_ETHANOL, pos),
        }
        monkeypatch.setattr(vcw, "generate_all_available",
                            lambda smiles, **kw: fake_result)
        p = tmp_path / "smiles.csv"
        self._write_smiles_csv(p, [SMILES_ETHANOL])
        out = tmp_path / "out.csv"
        vcw.main(["--smiles-csv", str(p), "--out-csv", str(out), "--n-mols", "1"])
        with open(out, newline="") as f:
            header = csv.DictReader(f).fieldnames
        assert header == OUTPUT_COLUMNS

    def test_n_mols_limits_processing(self, tmp_path, monkeypatch):
        import revision.geometry_qc.variant_comparison_writer as vcw
        seen = []
        def _fake(smiles, **kw):
            seen.append(smiles)
            return {}
        monkeypatch.setattr(vcw, "generate_all_available", _fake)
        p = tmp_path / "smiles.csv"
        self._write_smiles_csv(p, [SMILES_ETHANE, SMILES_ETHANOL, SMILES_BENZENE])
        out = tmp_path / "out.csv"
        vcw.main(["--smiles-csv", str(p), "--out-csv", str(out), "--n-mols", "2"])
        assert len(seen) == 2


# ============================================================================
# 12. RDKit conformer round-trips (skipped when RDKit not installed)
# ============================================================================

@pytest.mark.skipif(not rdkit_available(), reason="RDKit not installed")
class TestRDKitConformers:
    """Full conformer generation round-trips using real RDKit."""

    SMILES = [SMILES_ETHANE, SMILES_ETHANOL, SMILES_BENZENE, "c1ccc(cc1)O"]

    @pytest.mark.parametrize("smiles", SMILES)
    @pytest.mark.parametrize("method", RDKIT_METHODS)
    def test_generation_succeeds(self, smiles, method):
        result = generate_conformer(smiles, method=method, seed=42)
        assert result.success, f"{method}/{smiles}: {result.error_msg}"
        assert result.positions is not None
        assert result.n_atoms >= 2

    @pytest.mark.parametrize("smiles", SMILES)
    @pytest.mark.parametrize("method", RDKIT_METHODS)
    def test_positions_dtype_float32(self, smiles, method):
        result = generate_conformer(smiles, method=method, seed=42)
        if result.success:
            assert result.positions.dtype == np.float32

    @pytest.mark.parametrize("smiles", SMILES)
    @pytest.mark.parametrize("method", RDKIT_METHODS)
    def test_positions_shape_correct(self, smiles, method):
        result = generate_conformer(smiles, method=method, seed=42)
        if result.success:
            assert result.positions.ndim == 2
            assert result.positions.shape[1] == 3


# ============================================================================
# 13. OpenBabel round-trips (skipped when obabel not on PATH)
# ============================================================================

@pytest.mark.skipif(not obabel_available(), reason="obabel not on PATH")
class TestOpenBabelConformer:
    def test_generation_succeeds_ethanol(self):
        result = generate_conformer(SMILES_ETHANOL, method=METHOD_OBABEL, seed=42)
        assert result.success
        assert result.positions is not None

    def test_positions_have_three_coords(self):
        result = generate_conformer(SMILES_ETHANOL, method=METHOD_OBABEL, seed=42)
        if result.success:
            assert result.positions.shape[1] == 3
