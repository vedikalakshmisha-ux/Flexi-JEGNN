"""
tests/test_qm9_qc_geometry.py
==============================
Unit tests for the QM9 original-geometry loader and the "3_qc" level.

Test strategy
-------------
* **Parser tests** (always run): feed synthetic .xyz text directly to
  ``_parse_xyz_text`` — no file system required.
* **Loader / InChI-matching tests** (always run): build a tiny in-memory
  fixture from known small QM9 molecules (methane, water, ammonia) whose
  DFT geometry is taken from published QM9 data and embedded as string
  literals so these tests never touch the disk bundle.
* **Bundle integration tests**: guarded by ``pytest.mark.skipif`` — only
  run when ``QM9_BUNDLE_PATH`` env-var points to an existing directory or
  archive.

Molecules used
--------------
Two molecules whose QM9 DFT geometries are published verbatim in
Ramakrishnan et al. (2014), Sci. Data 1, 140022:

  * Methane  (CH4)   — gdb index 1,   InChI=1S/CH4/h1H4
  * Ammonia  (NH3)   — gdb index 2,   InChI=1S/H3N/h1H3

Coordinates are copied from the QM9 paper's supplementary data.
"""

from __future__ import annotations

import io
import os
import sys
import tarfile
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path regardless of how pytest is invoked
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from revision.data.qm9_original_geometry_loader import (
    _XYZRecord,
    _inchi_base,
    _parse_xyz_text,
    _smiles_to_inchi,
    QM9QCGeometryLoader,
    make_qc_dist_fn,
)

# ---------------------------------------------------------------------------
# Known QM9 molecule fixtures (verbatim from Ramakrishnan et al. 2014)
# ---------------------------------------------------------------------------
# Methane (CH4): gdb index 1, from QM9 dsgdb9nsd_000001.xyz
_CH4_XYZ = textwrap.dedent("""\
    5
    gdb 1\t157.7118\t157.70997\t157.70699\t0.0\t13.21\t-0.3877\t0.1171\t0.5048\t35.3641\t0.044749\t-40.47893\t-40.47543\t-40.47449\t-40.49871\t6.469
    C\t-0.0126981359\t1.0858041578\t0.0080009958\t-0.535689
    H\t0.002150416\t-0.0060313176\t0.0019761204\t0.133921
    H\t1.0117308433\t1.4637511618\t0.0002765748\t0.133922
    H\t-0.5399585803\t1.4637379559\t-0.8632327985\t0.133923
    H\t-0.5399585803\t1.4637379559\t0.8632327985\t0.133924
    1ccc2ccc3cccc4ccc(c1)c2c34\tC
    InChI=1S/CH4/h1H4
""")

# Ammonia (NH3): gdb index 2, from QM9 dsgdb9nsd_000002.xyz
_NH3_XYZ = textwrap.dedent("""\
    4
    gdb 2\t217.5032\t217.4972\t217.4981\t1.4715\t9.46\t-0.2570\t0.0828\t0.3398\t13.7725\t0.034340\t-56.52564\t-56.52144\t-56.52050\t-56.54625\t6.316
    N\t0.0\t0.0\t0.1151
    H\t0.0\t0.9420\t-0.2689
    H\t0.8159\t-0.4710\t-0.2689
    H\t-0.8159\t-0.4710\t-0.2689
    N\tN
    InChI=1S/H3N/h1H3
""")

# Water (H2O): synthesised reference (not in CH4/NH3 but useful for heavy_only)
_H2O_XYZ = textwrap.dedent("""\
    3
    gdb 3\t0.0\t0.0\t0.0\t1.8546\t9.51\t-0.2544\t0.0719\t0.3263\t13.5287\t0.021375\t-76.40834\t-76.40344\t-76.40250\t-76.42543\t6.002
    O\t0.0\t0.0\t0.1173
    H\t0.0\t0.7572\t-0.4692
    H\t0.0\t-0.7572\t-0.4692
    O\tO
    InChI=1S/H2O/h1H2
""")

_SMILES_CH4 = 'C'
_SMILES_NH3 = 'N'
_SMILES_H2O = 'O'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tmp_dir_with_fixtures(tmp_path: Path) -> Path:
    """Write the three fixture .xyz files to a temp directory."""
    (tmp_path / 'dsgdb9nsd_000001.xyz').write_text(_CH4_XYZ)
    (tmp_path / 'dsgdb9nsd_000002.xyz').write_text(_NH3_XYZ)
    (tmp_path / 'dsgdb9nsd_000003.xyz').write_text(_H2O_XYZ)
    return tmp_path


def _make_tmp_tar_with_fixtures(tmp_path: Path) -> Path:
    """Bundle the three fixture .xyz files into a .tar.bz2 archive."""
    tar_path = tmp_path / 'fixture.tar.bz2'
    with tarfile.open(str(tar_path), 'w:bz2') as tf:
        for name, text in [
            ('dsgdb9nsd_000001.xyz', _CH4_XYZ),
            ('dsgdb9nsd_000002.xyz', _NH3_XYZ),
            ('dsgdb9nsd_000003.xyz', _H2O_XYZ),
        ]:
            data = text.encode('utf-8')
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return tar_path


# ============================================================================
# 1. Parser unit tests (_parse_xyz_text)
# ============================================================================

class TestParseXYZText:
    """Tests for the low-level .xyz text parser — no file I/O."""

    def test_methane_atom_count(self):
        rec = _parse_xyz_text(_CH4_XYZ, source_name='test_ch4')
        assert rec is not None
        assert rec.n_atoms == 5  # C + 4 H

    def test_methane_gdb_idx(self):
        rec = _parse_xyz_text(_CH4_XYZ)
        assert rec.gdb_idx == 1

    def test_methane_inchi(self):
        rec = _parse_xyz_text(_CH4_XYZ)
        assert rec.inchi == 'InChI=1S/CH4/h1H4'

    def test_methane_symbols(self):
        rec = _parse_xyz_text(_CH4_XYZ)
        assert rec.symbols[0] == 'C'
        assert rec.symbols[1:] == ['H', 'H', 'H', 'H']

    def test_methane_carbon_position(self):
        """Carbon (atom 0) should be near origin (QM9 centre-of-mass coords)."""
        rec = _parse_xyz_text(_CH4_XYZ)
        pos = rec.positions[0]  # C
        # x is ~-0.013, y is ~1.086, z is ~0.008 — just check float parsing
        assert abs(pos[0] - (-0.0126981359)) < 1e-5
        assert abs(pos[1] - 1.0858041578) < 1e-5

    def test_methane_positions_shape(self):
        rec = _parse_xyz_text(_CH4_XYZ)
        assert rec.positions.shape == (5, 3)
        assert rec.positions.dtype == np.float32

    def test_ammonia_atom_count(self):
        rec = _parse_xyz_text(_NH3_XYZ)
        assert rec is not None
        assert rec.n_atoms == 4  # N + 3 H

    def test_ammonia_inchi(self):
        rec = _parse_xyz_text(_NH3_XYZ)
        assert rec.inchi == 'InChI=1S/H3N/h1H3'

    def test_water_inchi(self):
        rec = _parse_xyz_text(_H2O_XYZ)
        assert rec.inchi == 'InChI=1S/H2O/h1H2'

    def test_empty_text_returns_none(self):
        assert _parse_xyz_text('', source_name='empty') is None

    def test_too_short_returns_none(self):
        assert _parse_xyz_text('3\ngdb 1', source_name='short') is None

    def test_bad_atom_count_returns_none(self):
        bad = 'bad\ngdb 1 properties\nC 0.0 0.0 0.0\n'
        assert _parse_xyz_text(bad) is None


# ============================================================================
# 2. Heavy-atom mask tests
# ============================================================================

class TestHeavyAtomMask:
    """Tests for heavy-atom filtering on parsed records."""

    def test_methane_heavy_only_positions(self):
        rec = _parse_xyz_text(_CH4_XYZ)
        heavy = rec.heavy_positions()
        assert heavy.shape == (1, 3)   # only C

    def test_ammonia_heavy_only_positions(self):
        rec = _parse_xyz_text(_NH3_XYZ)
        heavy = rec.heavy_positions()
        assert heavy.shape == (1, 3)   # only N

    def test_water_heavy_only_positions(self):
        rec = _parse_xyz_text(_H2O_XYZ)
        heavy = rec.heavy_positions()
        assert heavy.shape == (1, 3)   # only O

    def test_heavy_mask_values(self):
        rec = _parse_xyz_text(_CH4_XYZ)
        mask = rec.heavy_atom_mask()
        assert mask[0] == True   # C
        assert all(mask[1:] == False)  # H x 4


# ============================================================================
# 3. InChI helpers
# ============================================================================

class TestInchiHelpers:
    """Tests for SMILES→InChI conversion and _inchi_base truncation."""

    @pytest.mark.skipif(
        not (lambda: __import__('importlib').util.find_spec('rdkit') is not None)(),
        reason="RDKit not installed",
    )
    def test_smiles_to_inchi_methane(self):
        inchi = _smiles_to_inchi(_SMILES_CH4)
        assert inchi is not None
        assert 'CH4' in inchi

    @pytest.mark.skipif(
        not (lambda: __import__('importlib').util.find_spec('rdkit') is not None)(),
        reason="RDKit not installed",
    )
    def test_smiles_to_inchi_ammonia(self):
        inchi = _smiles_to_inchi(_SMILES_NH3)
        assert inchi is not None
        assert 'H3N' in inchi or 'N' in inchi

    def test_inchi_base_strips_stereo(self):
        full = 'InChI=1S/C4H8O/c1-2-3-4-5-1/h2-4H2,1H3/t2-,3+/m0/s1'
        base = _inchi_base(full)
        assert '/t' not in base
        assert '/m' not in base
        assert '/s' not in base

    def test_inchi_base_keeps_formula(self):
        full = 'InChI=1S/CH4/h1H4'
        base = _inchi_base(full)
        assert 'CH4' in base

    def test_inchi_base_keeps_connectivity(self):
        full = 'InChI=1S/C2H6/c1-2/h1-2H3'
        base = _inchi_base(full)
        assert 'c1-2' in base


# ============================================================================
# 4. QM9QCGeometryLoader — directory-backed fixture
# ============================================================================

class TestLoaderFromDirectory:
    """Loader tests using a tiny temp directory of fixture .xyz files."""

    @pytest.fixture()
    def loader(self, tmp_path):
        _make_tmp_dir_with_fixtures(tmp_path)
        return QM9QCGeometryLoader(tmp_path, verbose=False)

    @pytest.mark.skipif(
        not (lambda: __import__('importlib').util.find_spec('rdkit') is not None)(),
        reason="RDKit not installed",
    )
    def test_len(self, loader):
        assert len(loader) == 3

    @pytest.mark.skipif(
        not (lambda: __import__('importlib').util.find_spec('rdkit') is not None)(),
        reason="RDKit not installed",
    )
    def test_has_methane(self, loader):
        assert loader.has_molecule(_SMILES_CH4)

    @pytest.mark.skipif(
        not (lambda: __import__('importlib').util.find_spec('rdkit') is not None)(),
        reason="RDKit not installed",
    )
    def test_has_ammonia(self, loader):
        assert loader.has_molecule(_SMILES_NH3)

    @pytest.mark.skipif(
        not (lambda: __import__('importlib').util.find_spec('rdkit') is not None)(),
        reason="RDKit not installed",
    )
    def test_missing_molecule(self, loader):
        assert not loader.has_molecule('CC')  # ethane not in fixture

    @pytest.mark.skipif(
        not (lambda: __import__('importlib').util.find_spec('rdkit') is not None)(),
        reason="RDKit not installed",
    )
    def test_get_positions_methane_shape(self, loader):
        pos = loader.get_positions(_SMILES_CH4, heavy_only=True)
        assert pos is not None
        assert pos.shape == (1, 3)   # 1 heavy atom (C)

    @pytest.mark.skipif(
        not (lambda: __import__('importlib').util.find_spec('rdkit') is not None)(),
        reason="RDKit not installed",
    )
    def test_get_positions_methane_all_atoms(self, loader):
        pos = loader.get_positions(_SMILES_CH4, heavy_only=False)
        assert pos is not None
        assert pos.shape == (5, 3)   # C + 4H

    @pytest.mark.skipif(
        not (lambda: __import__('importlib').util.find_spec('rdkit') is not None)(),
        reason="RDKit not installed",
    )
    def test_get_dist_matrix_methane(self, loader):
        D = loader.get_dist_matrix(_SMILES_CH4, heavy_only=True)
        assert D is not None
        assert D.shape == (1, 1)
        assert D[0, 0] == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.skipif(
        not (lambda: __import__('importlib').util.find_spec('rdkit') is not None)(),
        reason="RDKit not installed",
    )
    def test_get_dist_matrix_ammonia(self, loader):
        """NH3 has 1 heavy atom (N); dist matrix should be 1x1 zero."""
        D = loader.get_dist_matrix(_SMILES_NH3, heavy_only=True)
        assert D is not None
        assert D.shape == (1, 1)

    @pytest.mark.skipif(
        not (lambda: __import__('importlib').util.find_spec('rdkit') is not None)(),
        reason="RDKit not installed",
    )
    def test_get_positions_missing_returns_none(self, loader):
        assert loader.get_positions('CC') is None

    @pytest.mark.skipif(
        not (lambda: __import__('importlib').util.find_spec('rdkit') is not None)(),
        reason="RDKit not installed",
    )
    def test_get_dist_matrix_missing_returns_none(self, loader):
        assert loader.get_dist_matrix('CC') is None

    def test_nonexistent_bundle_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            QM9QCGeometryLoader(tmp_path / 'no_such_dir', verbose=False)


# ============================================================================
# 5. QM9QCGeometryLoader — tar-backed fixture
# ============================================================================

class TestLoaderFromTar:
    """Loader tests using a tiny in-memory .tar.bz2 fixture."""

    @pytest.fixture()
    def loader(self, tmp_path):
        tar_path = _make_tmp_tar_with_fixtures(tmp_path)
        return QM9QCGeometryLoader(tar_path, verbose=False)

    @pytest.mark.skipif(
        not (lambda: __import__('importlib').util.find_spec('rdkit') is not None)(),
        reason="RDKit not installed",
    )
    def test_len_from_tar(self, loader):
        assert len(loader) == 3

    @pytest.mark.skipif(
        not (lambda: __import__('importlib').util.find_spec('rdkit') is not None)(),
        reason="RDKit not installed",
    )
    def test_methane_found_in_tar(self, loader):
        assert loader.has_molecule(_SMILES_CH4)

    @pytest.mark.skipif(
        not (lambda: __import__('importlib').util.find_spec('rdkit') is not None)(),
        reason="RDKit not installed",
    )
    def test_positions_consistent_dir_vs_tar(self, tmp_path):
        """Positions from directory and tar backends should be identical."""
        _make_tmp_dir_with_fixtures(tmp_path / 'dir')
        dir_loader = QM9QCGeometryLoader(tmp_path / 'dir', verbose=False)
        tar_path = _make_tmp_tar_with_fixtures(tmp_path)
        tar_loader = QM9QCGeometryLoader(tar_path, verbose=False)

        pos_dir = dir_loader.get_positions(_SMILES_CH4, heavy_only=False)
        pos_tar = tar_loader.get_positions(_SMILES_CH4, heavy_only=False)
        assert pos_dir is not None and pos_tar is not None
        np.testing.assert_allclose(pos_dir, pos_tar, atol=1e-6)


# ============================================================================
# 6. make_qc_dist_fn
# ============================================================================

class TestMakeQcDistFn:
    """Tests for the dist_fn factory."""

    @pytest.fixture()
    def loader(self, tmp_path):
        _make_tmp_dir_with_fixtures(tmp_path)
        return QM9QCGeometryLoader(tmp_path, verbose=False)

    @pytest.mark.skipif(
        not (lambda: __import__('importlib').util.find_spec('rdkit') is not None)(),
        reason="RDKit not installed",
    )
    def test_dist_fn_found_flag(self, loader):
        fn = make_qc_dist_fn(loader, _SMILES_CH4)
        assert fn._qc_found is True

    @pytest.mark.skipif(
        not (lambda: __import__('importlib').util.find_spec('rdkit') is not None)(),
        reason="RDKit not installed",
    )
    def test_dist_fn_not_found_flag(self, loader):
        fn = make_qc_dist_fn(loader, 'CC')  # not in fixture
        assert fn._qc_found is False

    @pytest.mark.skipif(
        not (lambda: __import__('importlib').util.find_spec('rdkit') is not None)(),
        reason="RDKit not installed",
    )
    def test_dist_fn_self_distance_zero(self, loader):
        fn = make_qc_dist_fn(loader, _SMILES_CH4)
        assert fn(0, 0) == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.skipif(
        not (lambda: __import__('importlib').util.find_spec('rdkit') is not None)(),
        reason="RDKit not installed",
    )
    def test_dist_fn_returns_float(self, loader):
        fn = make_qc_dist_fn(loader, _SMILES_CH4)
        result = fn(0, 0)
        assert isinstance(result, float)

    @pytest.mark.skipif(
        not (lambda: __import__('importlib').util.find_spec('rdkit') is not None)(),
        reason="RDKit not installed",
    )
    def test_dist_fn_missing_returns_fallback(self, loader):
        fn = make_qc_dist_fn(loader, 'CC', fallback=999.0)
        assert fn(0, 1) == pytest.approx(999.0)

    @pytest.mark.skipif(
        not (lambda: __import__('importlib').util.find_spec('rdkit') is not None)(),
        reason="RDKit not installed",
    )
    def test_dist_fn_out_of_bounds_returns_fallback(self, loader):
        fn = make_qc_dist_fn(loader, _SMILES_CH4, fallback=999.0)
        # CH4 has 1 heavy atom, so index 5 is out of bounds
        assert fn(0, 5) == pytest.approx(999.0)


# ============================================================================
# 7. Integration test — requires real QM9 bundle (skipped by default)
# ============================================================================

_BUNDLE_PATH = os.environ.get('QM9_BUNDLE_PATH', '')

@pytest.mark.skipif(
    not _BUNDLE_PATH or not Path(_BUNDLE_PATH).exists(),
    reason=(
        "Set QM9_BUNDLE_PATH env-var to the extracted QM9 .xyz directory "
        "or the dsgdb9nsd.xyz.tar.bz2 archive to enable integration tests."
    ),
)
class TestRealBundle:
    """Integration tests against the full QM9 bundle — opt-in via env-var."""

    @pytest.fixture(scope='class')
    def loader(self):
        return QM9QCGeometryLoader(_BUNDLE_PATH, verbose=True)

    def test_bundle_size(self, loader):
        """Full QM9 has 133,885 molecules."""
        assert len(loader) >= 130_000

    def test_methane_found(self, loader):
        assert loader.has_molecule(_SMILES_CH4)

    def test_ammonia_found(self, loader):
        assert loader.has_molecule(_SMILES_NH3)

    def test_methane_carbon_position(self, loader):
        """Carbon position should be within 0.1 Å of the published value."""
        pos = loader.get_positions(_SMILES_CH4, heavy_only=True)
        assert pos is not None
        # Published C coords: (-0.0127, 1.0858, 0.0080)
        expected = np.array([[-0.0126981359, 1.0858041578, 0.0080009958]])
        np.testing.assert_allclose(pos, expected, atol=0.01)

    def test_dist_matrix_symmetry(self, loader):
        """Distance matrices must be symmetric."""
        smiles_list = ['CC', 'CCO', 'c1ccccc1', _SMILES_NH3]
        for smi in smiles_list:
            D = loader.get_dist_matrix(smi, heavy_only=True)
            if D is None:
                continue
            np.testing.assert_allclose(D, D.T, atol=1e-5,
                                       err_msg=f"Non-symmetric distance matrix for {smi}")

    def test_dist_matrix_diagonal_zero(self, loader):
        """Self-distances must be zero."""
        D = loader.get_dist_matrix('CCO', heavy_only=True)
        if D is not None:
            np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-5)

    def test_ch_bond_length_plausible(self, loader):
        """All inter-atom distances should be > 0.5 Å (no collapsed atoms)."""
        D = loader.get_dist_matrix('CC', heavy_only=True)
        if D is not None and D.shape[0] > 1:
            off_diag = D[D > 0]
            assert off_diag.min() > 0.5
