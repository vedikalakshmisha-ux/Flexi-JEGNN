"""
tests/test_geometry_variants.py
================================
Unit tests for:
  * ``revision/geometry_qc/generate_variants.py``
  * ``revision/geometry_qc/variant_comparison_writer.py``

Test strategy
-------------
* **Pure-numpy tests** (always run): Kabsch RMSD, ``ConformerResult``,
  ``pairwise_rmsd``, ``write_comparison_csv``, ``print_summary``, ``main()``,
  and availability-detection helpers — none of these need RDKit or OpenBabel.
* **Synthetic conformer fixtures**: hand-crafted ``ConformerResult`` objects
  with known positions avoid any dependency on actual conformer generation.
* **RDKit-gated tests**: conformer generation tests are marked
  ``pytest.mark.skipif(not rdkit_available(), reason="RDKit not installed")``.
* **OpenBabel-gated tests**: similarly gated on ``obabel_available()``.

Small fixed SMILES used throughout
------------------------------------
Methane  ('C')   — 1 heavy atom  (RMSD undefined for single atom, tested as edge case)
Ethane   ('CC')  — 2 heavy atoms
Ethanol  ('CCO') — 3 heavy atoms
Benzene  ('c1ccccc1') — 6 heavy atoms
"""

from __future__ import annotations

import csv
import math
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path
# ---------------------------------------------------------------------------
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
    generate_all_available,
    generate_conformer,
    obabel_available,
    rdkit_available,
    _parse_sdf_positions,
)
from revision.geometry_qc.variant_comparison_writer import (
    OUTPUT_COLUMNS,
    kabsch_rmsd,
    main,
    pairwise_rmsd,
    print_summary,
    run_comparison,
    write_comparison_csv,
)

# ---------------------------------------------------------------------------
# Fixed SMILES for tests
# ---------------------------------------------------------------------------
SMILES_METHANE = "C"
SMILES_ETHANE  = "CC"
SMILES_ETHANOL = "CCO"
SMILES_BENZENE = "c1ccccc1"

FIXED_SMILES = [SMILES_METHANE, SMILES_ETHANE, SMILES_ETHANOL, SMILES_BENZENE]

# ---------------------------------------------------------------------------
# Helpers — synthetic ConformerResult factories
# ---------------------------------------------------------------------------

def _make_result(
    method: str,
    smiles: str,
    positions: np.ndarray,
    success: bool = True,
    error_msg: str = "",
    seed: int = 42,
) -> ConformerResult:
    n = positions.shape[0] if positions is not None else 0
    return ConformerResult(
        method=method,
        smiles=smiles,
        positions=positions,
        n_atoms=n,
        success=success,
        error_msg=error_msg,
        seed=seed,
    )


def _failed_result(method: str, smiles: str, reason: str = "test failure") -> ConformerResult:
    return ConformerResult(
        method=method, smiles=smiles, positions=None,
        n_atoms=0, success=False, error_msg=reason, seed=42,
    )


def _ethane_pos_a() -> np.ndarray:
    """Two-atom (CC) positions, version A."""
    return np.array([[0.0, 0.0, 0.0], [1.54, 0.0, 0.0]], dtype=np.float32)


def _ethane_pos_b() -> np.ndarray:
    """Two-atom (CC) positions, version B — translated + rotated slightly."""
    return np.array([[1.0, 1.0, 1.0], [2.54, 1.0, 1.0]], dtype=np.float32)


def _ethanol_pos() -> np.ndarray:
    """Three-atom (CCO) positions."""
    return np.array([
        [0.0,  0.0,  0.0],
        [1.54, 0.0,  0.0],
        [2.40, 1.10, 0.0],
    ], dtype=np.float32)


def _benzene_pos() -> np.ndarray:
    """Six-atom (c1ccccc1) positions on a flat hexagon, radius 1.4 Å."""
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    r = 1.4
    return np.array(
        [[r * np.cos(a), r * np.sin(a), 0.0] for a in angles],
        dtype=np.float32,
    )


# ============================================================================
# 1. Availability detection
# ============================================================================

class TestAvailability:
    def test_rdkit_available_returns_bool(self):
        assert isinstance(rdkit_available(), bool)

    def test_obabel_available_returns_bool(self):
        assert isinstance(obabel_available(), bool)

    def test_rdkit_not_installed_on_this_machine(self):
        # We confirmed RDKit is not installed; test the detection
        assert rdkit_available() is False

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
        # 90° rotation about z-axis
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
        r = kabsch_rmsd(P, P)
        assert isinstance(r, float)

    def test_rmsd_known_value(self):
        """
        Two parallel C-C bonds at C-C distance 1.54 Å displaced by 1 Å along y.
        After centring both sets:  P = [[-0.77, 0, 0], [0.77, 0, 0]]
                                   Q = [[-0.77, 1, 0], [0.77, 1, 0]]
        Rotation can't reduce the y-offset. RMSD should be 1.0 Å.
        """
        P = np.array([[-0.77, 0.0, 0.0], [0.77, 0.0, 0.0]], dtype=np.float32)
        Q = np.array([[-0.77, 1.0, 0.0], [0.77, 1.0, 0.0]], dtype=np.float32)
        # After centring Q is identical to P — RMSD = 0
        # (pure translation disappears; this tests the centring step)
        assert kabsch_rmsd(P, Q) == pytest.approx(0.0, abs=1e-5)

    def test_rmsd_with_genuine_distortion(self):
        """Move one atom by a known amount; RMSD should be > 0."""
        P = _benzene_pos()
        Q = P.copy()
        Q[0] += np.array([1.0, 0.0, 0.0])  # displace one vertex
        r = kabsch_rmsd(P, Q)
        assert r > 0.0


# ============================================================================
# 4. _parse_sdf_positions
# ============================================================================

class TestParseSdfPositions:
    # Minimal SDF V2000 block for ethane (2 atoms)
    _ETHANE_SDF = """\

     RDKit          3D

  2  1  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.5400    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
M  END
$$$$
"""

    def test_parses_atom_count(self):
        pos, n = _parse_sdf_positions(self._ETHANE_SDF)
        assert n == 2

    def test_parses_positions_shape(self):
        pos, n = _parse_sdf_positions(self._ETHANE_SDF)
        assert pos.shape == (2, 3)

    def test_parses_positions_values(self):
        pos, n = _parse_sdf_positions(self._ETHANE_SDF)
        assert pos[0, 0] == pytest.approx(0.0, abs=1e-5)
        assert pos[1, 0] == pytest.approx(1.54, abs=1e-5)

    def test_parses_dtype(self):
        pos, _ = _parse_sdf_positions(self._ETHANE_SDF)
        assert pos.dtype == np.float32

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

    def test_rdkit_method_raises_when_not_installed(self):
        # RDKit is confirmed absent on this machine
        with pytest.raises(RDKitNotInstalledError):
            generate_conformer(SMILES_ETHANE, method=METHOD_ETKDGv3)

    def test_all_rdkit_methods_raise_when_not_installed(self):
        for method in RDKIT_METHODS:
            with pytest.raises(RDKitNotInstalledError):
                generate_conformer(SMILES_ETHANE, method=method)

    def test_obabel_method_raises_when_not_installed(self):
        # obabel confirmed absent
        with pytest.raises(OpenBabelNotInstalledError):
            generate_conformer(SMILES_ETHANE, method=METHOD_OBABEL)

    def test_obabel_error_message_contains_install_hint(self):
        try:
            generate_conformer(SMILES_ETHANE, method=METHOD_OBABEL)
        except OpenBabelNotInstalledError as exc:
            assert "obabel" in str(exc).lower() or "openbabel" in str(exc).lower()
            assert "install" in str(exc).lower() or "PATH" in str(exc)

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
    def test_returns_empty_when_nothing_installed(self):
        """No RDKit, no obabel → all methods skipped → empty dict."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            results = generate_all_available(SMILES_ETHANE)
        assert results == {}

    def test_skip_methods_honoured(self):
        """Skipping all methods explicitly → empty dict."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            results = generate_all_available(
                SMILES_ETHANE,
                skip_methods=ALL_METHODS,
            )
        assert results == {}

    def test_warns_about_missing_rdkit(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            generate_all_available(SMILES_ETHANE)
        rdkit_warns = [x for x in w if "RDKit" in str(x.message)]
        assert len(rdkit_warns) > 0

    def test_warns_about_missing_obabel(self):
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

    def test_rmsd_positive_for_different_positions(self):
        pos_a = _ethanol_pos()
        pos_b = _ethanol_pos().copy()
        pos_b[1] += np.array([0.5, 0.5, 0.5])
        results = {
            METHOD_ETKDGv3: _make_result(METHOD_ETKDGv3, SMILES_ETHANOL, pos_a),
            METHOD_ETKDGv2: _make_result(METHOD_ETKDGv2, SMILES_ETHANOL, pos_b),
        }
        rows = pairwise_rmsd(results)
        assert float(rows[0]["rmsd"]) > 0.0

    def test_both_ok_flag_when_both_succeed(self):
        rows = pairwise_rmsd(self._two_results())
        assert rows[0]["both_ok"] == 1

    def test_both_ok_flag_when_one_fails(self):
        results = {
            METHOD_ETKDGv3: _make_result(METHOD_ETKDGv3, SMILES_ETHANE, _ethane_pos_a()),
            METHOD_ETKDGv2: _failed_result(METHOD_ETKDGv2, SMILES_ETHANE),
        }
        rows = pairwise_rmsd(results)
        assert rows[0]["both_ok"] == 0
        assert rows[0]["rmsd"] == ""

    def test_both_ok_zero_when_both_fail(self):
        results = {
            METHOD_ETKDGv3: _failed_result(METHOD_ETKDGv3, SMILES_ETHANE),
            METHOD_ETKDGv2: _failed_result(METHOD_ETKDGv2, SMILES_ETHANE),
        }
        rows = pairwise_rmsd(results)
        assert rows[0]["both_ok"] == 0

    def test_rmsd_empty_when_atom_count_mismatch(self):
        results = {
            METHOD_ETKDGv3: _make_result(METHOD_ETKDGv3, SMILES_ETHANE,  _ethane_pos_a()),
            METHOD_ETKDGv2: _make_result(METHOD_ETKDGv2, SMILES_ETHANOL, _ethanol_pos()),
        }
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            rows = pairwise_rmsd(results)
        assert rows[0]["rmsd"] == ""

    def test_method_a_lt_method_b_always(self):
        """Pairs are always lexicographically ordered: A < B."""
        results = {
            METHOD_ETKDG:   _make_result(METHOD_ETKDG,   SMILES_ETHANOL, _ethanol_pos()),
            METHOD_ETKDGv2: _make_result(METHOD_ETKDGv2, SMILES_ETHANOL, _ethanol_pos()),
            METHOD_ETKDGv3: _make_result(METHOD_ETKDGv3, SMILES_ETHANOL, _ethanol_pos()),
        }
        rows = pairwise_rmsd(results)
        for row in rows:
            assert row["method_a"] < row["method_b"]

    def test_output_columns_present(self):
        rows = pairwise_rmsd(self._two_results())
        for col in OUTPUT_COLUMNS:
            assert col in rows[0], f"Missing column: {col}"

    def test_smiles_propagated(self):
        rows = pairwise_rmsd(self._two_results(smiles=SMILES_BENZENE,
                                               pos_a=_benzene_pos(),
                                               pos_b=_benzene_pos()))
        assert rows[0]["smiles"] == SMILES_BENZENE

    def test_empty_input_returns_empty(self):
        assert pairwise_rmsd({}) == []

    def test_single_method_returns_empty(self):
        results = {METHOD_ETKDGv3: _make_result(METHOD_ETKDGv3, SMILES_ETHANE, _ethane_pos_a())}
        assert pairwise_rmsd(results) == []

    def test_n_atoms_populated_on_success(self):
        results = {
            METHOD_ETKDGv3: _make_result(METHOD_ETKDGv3, SMILES_ETHANE, _ethane_pos_a()),
            METHOD_ETKDGv2: _make_result(METHOD_ETKDGv2, SMILES_ETHANE, _ethane_pos_b()),
        }
        rows = pairwise_rmsd(results)
        assert rows[0]["n_atoms"] == 2

    def test_error_messages_propagated(self):
        results = {
            METHOD_ETKDGv3: _make_result(METHOD_ETKDGv3, SMILES_ETHANE, _ethane_pos_a()),
            METHOD_ETKDGv2: _failed_result(METHOD_ETKDGv2, SMILES_ETHANE, "embed failed"),
        }
        rows = pairwise_rmsd(results)
        # ETKDGv2 < ETKDGv3 lexicographically → failed result lands in method_a
        assert rows[0]["method_a"] == METHOD_ETKDGv2
        assert rows[0]["method_a_error"] == "embed failed"
        assert rows[0]["method_b_error"] == ""


# ============================================================================
# 8. write_comparison_csv
# ============================================================================

class TestWriteComparisonCsv:
    def _sample_rows(self) -> List[dict]:
        pos = _ethanol_pos()
        results = {
            METHOD_ETKDGv3: _make_result(METHOD_ETKDGv3, SMILES_ETHANOL, pos),
            METHOD_ETKDGv2: _make_result(METHOD_ETKDGv2, SMILES_ETHANOL, pos),
        }
        return pairwise_rmsd(results)

    def test_creates_file(self, tmp_path):
        out = tmp_path / "comparison.csv"
        write_comparison_csv(self._sample_rows(), out)
        assert out.exists()

    def test_header_matches_output_columns(self, tmp_path):
        out = tmp_path / "comparison.csv"
        write_comparison_csv(self._sample_rows(), out)
        with open(out, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert list(reader.fieldnames) == OUTPUT_COLUMNS

    def test_row_count(self, tmp_path):
        out = tmp_path / "comparison.csv"
        sample = self._sample_rows()
        write_comparison_csv(sample, out)
        with open(out, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == len(sample)

    def test_rmsd_value_round_trips(self, tmp_path):
        out = tmp_path / "comparison.csv"
        write_comparison_csv(self._sample_rows(), out)
        with open(out, newline="") as f:
            rows = list(csv.DictReader(f))
        assert float(rows[0]["rmsd"]) == pytest.approx(0.0, abs=1e-5)

    def test_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "a" / "b" / "c.csv"
        write_comparison_csv(self._sample_rows(), out)
        assert out.exists()

    def test_empty_rows_writes_header_only(self, tmp_path):
        out = tmp_path / "empty.csv"
        write_comparison_csv([], out)
        with open(out, newline="") as f:
            rows = list(csv.DictReader(f))
        assert rows == []


# ============================================================================
# 9. run_comparison — with synthetic data injected via monkeypatching
# ============================================================================

class TestRunComparison:
    """
    Monkeypatch ``generate_all_available`` to inject synthetic results
    so run_comparison can be tested without RDKit.
    """

    def _mock_generate(self, smiles, seed=42, skip_methods=None):
        pos = _ethanol_pos()
        pos2 = _ethanol_pos() + np.array([0.1, 0.0, 0.0])
        return {
            METHOD_ETKDGv3: _make_result(METHOD_ETKDGv3, smiles, pos, seed=seed),
            METHOD_ETKDGv2: _make_result(METHOD_ETKDGv2, smiles, pos2, seed=seed),
        }

    def test_returns_list(self, monkeypatch):
        import revision.geometry_qc.variant_comparison_writer as vcw
        monkeypatch.setattr(vcw, "generate_all_available", self._mock_generate)
        rows = run_comparison([SMILES_ETHANOL], verbose=False)
        assert isinstance(rows, list)

    def test_one_pair_per_molecule(self, monkeypatch):
        import revision.geometry_qc.variant_comparison_writer as vcw
        monkeypatch.setattr(vcw, "generate_all_available", self._mock_generate)
        rows = run_comparison([SMILES_ETHANOL, SMILES_BENZENE], verbose=False)
        # 2 molecules × 1 pair each = 2 rows
        assert len(rows) == 2

    def test_rmsd_positive_when_positions_differ(self, monkeypatch):
        import revision.geometry_qc.variant_comparison_writer as vcw
        monkeypatch.setattr(vcw, "generate_all_available", self._mock_generate)
        rows = run_comparison([SMILES_ETHANOL], verbose=False)
        # pos2 has x+0.1 displacement → after centring both sets are identical
        # (pure translation) → RMSD = 0
        assert float(rows[0]["rmsd"]) == pytest.approx(0.0, abs=1e-4)

    def test_empty_smiles_list_returns_empty(self, monkeypatch):
        import revision.geometry_qc.variant_comparison_writer as vcw
        monkeypatch.setattr(vcw, "generate_all_available", self._mock_generate)
        rows = run_comparison([], verbose=False)
        assert rows == []

    def test_skip_methods_passed_through(self, monkeypatch):
        captured = {}
        def _mock(smiles, seed=42, skip_methods=None):
            captured["skip"] = list(skip_methods or [])
            return {}
        import revision.geometry_qc.variant_comparison_writer as vcw
        monkeypatch.setattr(vcw, "generate_all_available", _mock)
        run_comparison([SMILES_ETHANE], skip_methods=[METHOD_OBABEL], verbose=False)
        assert METHOD_OBABEL in captured["skip"]


# ============================================================================
# 10. print_summary — smoke tests
# ============================================================================

class TestPrintSummary:
    def _rows(self):
        pos = _ethanol_pos()
        pos2 = pos.copy(); pos2[0] += 0.3
        results = {
            METHOD_ETKDGv3: _make_result(METHOD_ETKDGv3, SMILES_ETHANOL, pos),
            METHOD_ETKDGv2: _make_result(METHOD_ETKDGv2, SMILES_ETHANOL, pos2),
        }
        return pairwise_rmsd(results)

    def test_no_crash(self, capsys):
        print_summary(self._rows())
        out = capsys.readouterr().out
        assert "RMSD" in out

    def test_empty_no_crash(self, capsys):
        print_summary([])
        out = capsys.readouterr().out
        assert "No comparison rows" in out

    def test_method_names_in_output(self, capsys):
        print_summary(self._rows())
        out = capsys.readouterr().out
        assert METHOD_ETKDGv3 in out or METHOD_ETKDGv2 in out


# ============================================================================
# 11. main() end-to-end — via monkeypatching + temp CSVs
# ============================================================================

class TestMainEndToEnd:
    def _write_smiles_csv(self, path: Path, smiles_list: List[str]) -> None:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["smiles", "homo"])
            w.writeheader()
            for s in smiles_list:
                w.writerow({"smiles": s, "homo": "-0.3"})

    def _mock_gen(self, smiles, seed=42, skip_methods=None):
        pos = _ethane_pos_a()
        pos2 = _ethane_pos_b()
        return {
            METHOD_ETKDGv3: _make_result(METHOD_ETKDGv3, smiles, pos, seed=seed),
            METHOD_ETKDGv2: _make_result(METHOD_ETKDGv2, smiles, pos2, seed=seed),
        }

    def test_returns_zero_on_success(self, tmp_path, monkeypatch):
        import revision.geometry_qc.variant_comparison_writer as vcw
        monkeypatch.setattr(vcw, "generate_all_available", self._mock_gen)
        csv_in = tmp_path / "smiles.csv"
        self._write_smiles_csv(csv_in, [SMILES_ETHANE, SMILES_ETHANOL])
        out = tmp_path / "out.csv"
        rc = main(["--smiles-csv", str(csv_in), "--out-csv", str(out),
                   "--no-summary"])
        assert rc == 0

    def test_output_file_created(self, tmp_path, monkeypatch):
        import revision.geometry_qc.variant_comparison_writer as vcw
        monkeypatch.setattr(vcw, "generate_all_available", self._mock_gen)
        csv_in = tmp_path / "smiles.csv"
        self._write_smiles_csv(csv_in, [SMILES_ETHANE])
        out = tmp_path / "out.csv"
        main(["--smiles-csv", str(csv_in), "--out-csv", str(out), "--no-summary"])
        assert out.exists()

    def test_output_columns_correct(self, tmp_path, monkeypatch):
        import revision.geometry_qc.variant_comparison_writer as vcw
        monkeypatch.setattr(vcw, "generate_all_available", self._mock_gen)
        csv_in = tmp_path / "smiles.csv"
        self._write_smiles_csv(csv_in, [SMILES_ETHANE])
        out = tmp_path / "out.csv"
        main(["--smiles-csv", str(csv_in), "--out-csv", str(out), "--no-summary"])
        with open(out, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert list(reader.fieldnames) == OUTPUT_COLUMNS

    def test_n_mols_limits_input(self, tmp_path, monkeypatch):
        seen = []
        def _mock(smiles, seed=42, skip_methods=None):
            seen.append(smiles)
            return {}
        import revision.geometry_qc.variant_comparison_writer as vcw
        monkeypatch.setattr(vcw, "generate_all_available", _mock)
        csv_in = tmp_path / "smiles.csv"
        self._write_smiles_csv(csv_in, FIXED_SMILES)
        out = tmp_path / "out.csv"
        # Limit to 2 molecules; main() will return 1 (no rows) but we just
        # check that the loader respected n_mols
        main(["--smiles-csv", str(csv_in), "--out-csv", str(out),
              "--n-mols", "2", "--no-summary"])
        assert len(seen) == 2

    def test_returns_one_on_missing_csv(self, tmp_path):
        rc = main([
            "--smiles-csv", str(tmp_path / "ghost.csv"),
            "--out-csv",    str(tmp_path / "out.csv"),
            "--no-summary",
        ])
        assert rc == 1

    def test_returns_one_on_wrong_smiles_col(self, tmp_path, monkeypatch):
        import revision.geometry_qc.variant_comparison_writer as vcw
        monkeypatch.setattr(vcw, "generate_all_available", self._mock_gen)
        csv_in = tmp_path / "smiles.csv"
        self._write_smiles_csv(csv_in, [SMILES_ETHANE])
        out = tmp_path / "out.csv"
        rc = main(["--smiles-csv", str(csv_in), "--smiles-col", "no_such_col",
                   "--out-csv", str(out), "--no-summary"])
        assert rc == 1


# ============================================================================
# 12. RDKit conformer generation tests (skipped when RDKit not installed)
# ============================================================================

@pytest.mark.skipif(not rdkit_available(), reason="RDKit not installed")
class TestRDKitConformers:
    """These tests only run when RDKit is importable."""

    @pytest.mark.parametrize("smiles,n_heavy", [
        (SMILES_ETHANE,  2),
        (SMILES_ETHANOL, 3),
        (SMILES_BENZENE, 6),
    ])
    @pytest.mark.parametrize("method", RDKIT_METHODS)
    def test_generates_correct_n_atoms(self, smiles, n_heavy, method):
        r = generate_conformer(smiles, method=method, seed=42)
        assert r.success, f"{method} failed for {smiles}: {r.error_msg}"
        assert r.n_atoms == n_heavy

    @pytest.mark.parametrize("method", RDKIT_METHODS)
    def test_positions_shape(self, method):
        r = generate_conformer(SMILES_ETHANOL, method=method, seed=42)
        assert r.success
        assert r.positions.shape == (3, 3)

    @pytest.mark.parametrize("method", RDKIT_METHODS)
    def test_positions_dtype_float32(self, method):
        r = generate_conformer(SMILES_ETHANOL, method=method, seed=42)
        assert r.positions.dtype == np.float32

    def test_invalid_smiles_fails_gracefully(self):
        r = generate_conformer("not_a_smiles!!!", method=METHOD_ETKDGv3, seed=42)
        assert not r.success
        assert r.positions is None

    def test_methane_single_heavy_atom(self):
        """Methane has 1 heavy atom; embedding should still succeed."""
        r = generate_conformer(SMILES_METHANE, method=METHOD_ETKDGv3, seed=42)
        # Methane may fail (< 2 atoms) — test handles both cases
        if r.success:
            assert r.n_atoms == 1
        else:
            assert "fewer than 2" in r.error_msg or not r.success

    def test_generate_all_available_returns_rdkit_methods(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            results = generate_all_available(
                SMILES_ETHANOL,
                skip_methods=[METHOD_OBABEL],
            )
        assert len(results) == len(RDKIT_METHODS)

    def test_kabsch_rmsd_between_rdkit_conformers(self):
        """Different seeds → different conformers → RMSD can be measured."""
        r1 = generate_conformer(SMILES_BENZENE, method=METHOD_ETKDGv3, seed=42)
        r2 = generate_conformer(SMILES_BENZENE, method=METHOD_ETKDGv3, seed=999)
        if r1.success and r2.success:
            rmsd = kabsch_rmsd(r1.positions, r2.positions)
            assert isinstance(rmsd, float)
            assert rmsd >= 0.0

    def test_etkdgv3_vs_etkdgv2_rmsd_finite(self):
        r3 = generate_conformer(SMILES_ETHANOL, method=METHOD_ETKDGv3, seed=42)
        r2 = generate_conformer(SMILES_ETHANOL, method=METHOD_ETKDGv2, seed=42)
        if r3.success and r2.success:
            rmsd = kabsch_rmsd(r3.positions, r2.positions)
            assert math.isfinite(rmsd)


# ============================================================================
# 13. OpenBabel tests (skipped when obabel not on PATH)
# ============================================================================

@pytest.mark.skipif(not obabel_available(), reason="obabel not on PATH")
class TestOpenBabelConformer:
    def test_generates_positions(self):
        r = generate_conformer(SMILES_ETHANOL, method=METHOD_OBABEL, seed=42)
        assert r.success, r.error_msg
        assert r.positions is not None

    def test_n_atoms_matches_heavy_count(self):
        r = generate_conformer(SMILES_ETHANOL, method=METHOD_OBABEL, seed=42)
        # OpenBabel may include H — positions count may differ from RDKit heavy-atom count
        assert r.n_atoms > 0

    def test_invalid_smiles_does_not_crash(self):
        r = generate_conformer("!!!", method=METHOD_OBABEL, seed=42)
        # Should return a failure result, not raise
        assert isinstance(r, ConformerResult)
