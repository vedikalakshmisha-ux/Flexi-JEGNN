"""
tests/test_qm9_geometry_comparison.py
======================================
Unit tests for ``revision/benchmarks/qm9_geometry_comparison.py``.

Test strategy
-------------
All tests are self-contained: they build synthetic CSV content in memory
(or in ``tmp_path``) and never read from ``results/qm9_raw_seeds.csv``.
No RDKit, PyTorch, or QM9 bundle is required.

Coverage
--------
* ``load_rows_for_level``  — happy path, level filtering, missing file
* ``pair_rows``            — full match, partial match, no match, duplicates
* ``write_comparison_csv`` — column completeness, value round-trips
* ``main()``              — end-to-end via synthetic CSVs on disk
* Delta arithmetic         — signs, NaN propagation
* ``print_summary``        — smoke test (no crash, captures stdout)
"""

from __future__ import annotations

import csv
import io
import math
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path regardless of how pytest is invoked
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from revision.benchmarks.qm9_geometry_comparison import (
    OUTPUT_COLUMNS,
    LEVEL_QC,
    LEVEL_RDKIT,
    _float_or_nan,
    load_rows_for_level,
    main,
    pair_rows,
    print_summary,
    write_comparison_csv,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

# Columns present in qm9_raw_seeds.csv
_RAW_COLS = [
    "key", "pearson_r", "mae", "rmse", "train_time", "ms_per_mol",
    "n_params", "epochs_run", "stopped_early",
    "dataset", "model", "level_id", "seed",
]


def _make_row(
    model: str,
    seed: str,
    level_id: str,
    pearson_r: float = 0.9,
    mae: float = 0.005,
    rmse: float = 0.008,
    ms_per_mol: float = 2.5,
    epochs_run: int = 80,
    stopped_early: int = 0,
) -> dict:
    """Build a synthetic row dict matching qm9_raw_seeds.csv structure."""
    key = f"QM9_{model}_{level_id}_{seed}"
    return {
        "key": key,
        "pearson_r": str(pearson_r),
        "mae": str(mae),
        "rmse": str(rmse),
        "train_time": "60.0",
        "ms_per_mol": str(ms_per_mol),
        "n_params": "471041",
        "epochs_run": str(epochs_run),
        "stopped_early": str(stopped_early),
        "dataset": "QM9",
        "model": model,
        "level_id": level_id,
        "seed": seed,
    }


def _write_csv(path: Path, rows: List[dict]) -> None:
    """Write rows to *path* using _RAW_COLS as fieldnames."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_RAW_COLS)
        writer.writeheader()
        writer.writerows(rows)


# Pre-built synthetic data used by multiple tests
_MODELS = ["PharmaJEGNN", "SchNet"]
_SEEDS = ["42", "123"]

_L3_ROWS = [
    _make_row(m, s, "3", pearson_r=0.85, mae=0.010, rmse=0.015)
    for m in _MODELS for s in _SEEDS
]

_L3QC_ROWS = [
    _make_row(m, s, "3_qc", pearson_r=0.90, mae=0.008, rmse=0.012)
    for m in _MODELS for s in _SEEDS
]


# ---------------------------------------------------------------------------
# 1. _float_or_nan
# ---------------------------------------------------------------------------

class TestFloatOrNan:
    def test_valid_float(self):
        assert _float_or_nan("0.123") == pytest.approx(0.123)

    def test_valid_negative(self):
        assert _float_or_nan("-0.05") == pytest.approx(-0.05)

    def test_empty_string_is_nan(self):
        assert math.isnan(_float_or_nan(""))

    def test_none_is_nan(self):
        assert math.isnan(_float_or_nan(None))

    def test_non_numeric_is_nan(self):
        assert math.isnan(_float_or_nan("N/A"))

    def test_scientific_notation(self):
        assert _float_or_nan("1.5e-3") == pytest.approx(0.0015)


# ---------------------------------------------------------------------------
# 2. load_rows_for_level
# ---------------------------------------------------------------------------

class TestLoadRowsForLevel:
    def test_loads_correct_level(self, tmp_path):
        csv_path = tmp_path / "raw.csv"
        all_rows = _L3_ROWS + _L3QC_ROWS
        _write_csv(csv_path, all_rows)

        rows = load_rows_for_level(csv_path, "3")
        assert len(rows) == len(_L3_ROWS)

    def test_loads_qc_level(self, tmp_path):
        csv_path = tmp_path / "raw.csv"
        _write_csv(csv_path, _L3_ROWS + _L3QC_ROWS)

        rows = load_rows_for_level(csv_path, "3_qc")
        assert len(rows) == len(_L3QC_ROWS)

    def test_key_is_model_seed_tuple(self, tmp_path):
        csv_path = tmp_path / "raw.csv"
        _write_csv(csv_path, _L3_ROWS)

        rows = load_rows_for_level(csv_path, "3")
        assert ("PharmaJEGNN", "42") in rows

    def test_excludes_other_levels(self, tmp_path):
        csv_path = tmp_path / "raw.csv"
        other = [_make_row("PharmaJEGNN", "42", "2")]
        _write_csv(csv_path, _L3_ROWS + other)

        rows = load_rows_for_level(csv_path, "3")
        # level-2 row must not appear
        assert all(r["level_id"] == "3" for r in rows.values())

    def test_empty_level_returns_empty_dict(self, tmp_path):
        csv_path = tmp_path / "raw.csv"
        _write_csv(csv_path, _L3_ROWS)  # no 3_qc rows

        rows = load_rows_for_level(csv_path, "3_qc")
        assert rows == {}

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_rows_for_level(tmp_path / "nonexistent.csv", "3")

    def test_row_values_preserved(self, tmp_path):
        csv_path = tmp_path / "raw.csv"
        _write_csv(csv_path, _L3_ROWS)

        rows = load_rows_for_level(csv_path, "3")
        row = rows[("PharmaJEGNN", "42")]
        assert row["model"] == "PharmaJEGNN"
        assert row["seed"] == "42"
        assert float(row["pearson_r"]) == pytest.approx(0.85)

    def test_duplicate_key_keeps_first(self, tmp_path):
        csv_path = tmp_path / "raw.csv"
        dup = _make_row("PharmaJEGNN", "42", "3", pearson_r=0.99)
        _write_csv(csv_path, _L3_ROWS + [dup])

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rows = load_rows_for_level(csv_path, "3")
        assert len(rows) == len(_L3_ROWS)
        # The first occurrence (pearson_r=0.85) is kept
        assert float(rows[("PharmaJEGNN", "42")]["pearson_r"]) == pytest.approx(0.85)
        assert any("Duplicate" in str(warning.message) for warning in w)


# ---------------------------------------------------------------------------
# 3. pair_rows
# ---------------------------------------------------------------------------

class TestPairRows:
    def _make_index(self, rows: List[dict]) -> Dict[Tuple[str, str], dict]:
        return {(r["model"], r["seed"]): r for r in rows}

    def test_full_match_count(self):
        rdkit = self._make_index(_L3_ROWS)
        qc = self._make_index(_L3QC_ROWS)
        paired = pair_rows(rdkit, qc)
        assert len(paired) == len(_L3_ROWS)

    def test_paired_model_seed_preserved(self):
        rdkit = self._make_index(_L3_ROWS)
        qc = self._make_index(_L3QC_ROWS)
        paired = pair_rows(rdkit, qc)
        keys = {(r["model"], r["seed"]) for r in paired}
        expected = {(m, s) for m in _MODELS for s in _SEEDS}
        assert keys == expected

    def test_partial_match_skips_unmatched(self, capsys):
        rdkit = self._make_index(_L3_ROWS)
        # QC has only one model
        qc_partial = self._make_index([
            _make_row("PharmaJEGNN", s, "3_qc") for s in _SEEDS
        ])
        paired = pair_rows(rdkit, qc_partial)
        assert len(paired) == len(_SEEDS)  # only PharmaJEGNN matched
        captured = capsys.readouterr()
        assert "WARN" in captured.err  # should warn about unmatched

    def test_no_match_returns_empty(self, capsys):
        rdkit = self._make_index([_make_row("PharmaJEGNN", "42", "3")])
        qc = self._make_index([_make_row("SchNet", "999", "3_qc")])
        paired = pair_rows(rdkit, qc)
        assert paired == []

    def test_delta_pearson_r_sign(self):
        """delta_pearson_r = qc − rdkit; QC is better here so delta > 0."""
        rdkit = self._make_index([_make_row("PharmaJEGNN", "42", "3", pearson_r=0.80)])
        qc = self._make_index([_make_row("PharmaJEGNN", "42", "3_qc", pearson_r=0.90)])
        paired = pair_rows(rdkit, qc)
        assert len(paired) == 1
        assert paired[0]["delta_pearson_r"] == pytest.approx(0.10, abs=1e-8)

    def test_delta_mae_sign(self):
        """delta_mae = qc − rdkit; lower is better, so negative = QC better."""
        rdkit = self._make_index([_make_row("PharmaJEGNN", "42", "3", mae=0.010)])
        qc = self._make_index([_make_row("PharmaJEGNN", "42", "3_qc", mae=0.008)])
        paired = pair_rows(rdkit, qc)
        assert paired[0]["delta_mae"] == pytest.approx(-0.002, abs=1e-8)

    def test_delta_rmse_sign(self):
        rdkit = self._make_index([_make_row("SchNet", "42", "3", rmse=0.015)])
        qc = self._make_index([_make_row("SchNet", "42", "3_qc", rmse=0.012)])
        paired = pair_rows(rdkit, qc)
        assert paired[0]["delta_rmse"] == pytest.approx(-0.003, abs=1e-8)

    def test_nan_propagates_in_delta(self):
        rdkit = self._make_index([_make_row("PharmaJEGNN", "42", "3", pearson_r=0.85)])
        # Manually corrupt the qc row
        qc_rows = self._make_index([_make_row("PharmaJEGNN", "42", "3_qc")])
        qc_rows[("PharmaJEGNN", "42")]["pearson_r"] = "N/A"
        paired = pair_rows(rdkit, qc_rows)
        assert math.isnan(paired[0]["delta_pearson_r"])

    def test_rdkit_key_correct(self):
        rdkit = self._make_index(_L3_ROWS)
        qc = self._make_index(_L3QC_ROWS)
        paired = pair_rows(rdkit, qc)
        row = next(r for r in paired if r["model"] == "PharmaJEGNN" and r["seed"] == "42")
        assert row["rdkit_key"] == "QM9_PharmaJEGNN_3_42"

    def test_qc_key_correct(self):
        rdkit = self._make_index(_L3_ROWS)
        qc = self._make_index(_L3QC_ROWS)
        paired = pair_rows(rdkit, qc)
        row = next(r for r in paired if r["model"] == "PharmaJEGNN" and r["seed"] == "42")
        assert row["qc_key"] == "QM9_PharmaJEGNN_3_qc_42"

    def test_pass_through_fields_present(self):
        rdkit = self._make_index(_L3_ROWS)
        qc = self._make_index(_L3QC_ROWS)
        paired = pair_rows(rdkit, qc)
        row = paired[0]
        for field in ["rdkit_ms_per_mol", "qc_ms_per_mol",
                      "rdkit_epochs_run", "qc_epochs_run",
                      "rdkit_stopped_early", "qc_stopped_early"]:
            assert field in row, f"Missing field: {field}"

    def test_sorted_by_model_then_seed(self):
        """pair_rows should return rows sorted by (model, seed)."""
        rdkit = self._make_index(_L3_ROWS)
        qc = self._make_index(_L3QC_ROWS)
        paired = pair_rows(rdkit, qc)
        keys = [(r["model"], r["seed"]) for r in paired]
        # All seeds are numeric so sorted numerically
        assert keys == sorted(keys, key=lambda k: (k[0], int(k[1])))


# ---------------------------------------------------------------------------
# 4. write_comparison_csv
# ---------------------------------------------------------------------------

class TestWriteComparisonCsv:
    def _make_paired_rows(self) -> List[dict]:
        rdkit = {(r["model"], r["seed"]): r for r in _L3_ROWS}
        qc = {(r["model"], r["seed"]): r for r in _L3QC_ROWS}
        return pair_rows(rdkit, qc)

    def test_creates_file(self, tmp_path):
        out = tmp_path / "comparison.csv"
        write_comparison_csv(self._make_paired_rows(), out)
        assert out.exists()

    def test_column_headers_match_output_columns(self, tmp_path):
        out = tmp_path / "comparison.csv"
        write_comparison_csv(self._make_paired_rows(), out)
        with open(out, newline="") as fh:
            reader = csv.DictReader(fh)
            assert list(reader.fieldnames) == OUTPUT_COLUMNS

    def test_row_count(self, tmp_path):
        out = tmp_path / "comparison.csv"
        paired = self._make_paired_rows()
        write_comparison_csv(paired, out)
        with open(out, newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == len(paired)

    def test_delta_values_round_trip(self, tmp_path):
        """Values written and read back should be numerically consistent."""
        out = tmp_path / "comparison.csv"
        write_comparison_csv(self._make_paired_rows(), out)
        with open(out, newline="") as fh:
            rows = list(csv.DictReader(fh))
        # delta_pearson_r should be ~+0.05 for all rows (0.90 - 0.85)
        for row in rows:
            assert float(row["delta_pearson_r"]) == pytest.approx(0.05, abs=1e-8)
            assert float(row["delta_mae"]) == pytest.approx(-0.002, abs=1e-8)
            assert float(row["delta_rmse"]) == pytest.approx(-0.003, abs=1e-8)

    def test_creates_parent_directory(self, tmp_path):
        out = tmp_path / "subdir" / "nested" / "comparison.csv"
        write_comparison_csv(self._make_paired_rows(), out)
        assert out.exists()

    def test_empty_paired_writes_header_only(self, tmp_path):
        out = tmp_path / "empty.csv"
        write_comparison_csv([], out)
        with open(out, newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert rows == []


# ---------------------------------------------------------------------------
# 5. print_summary — smoke test
# ---------------------------------------------------------------------------

class TestPrintSummary:
    def test_no_crash_with_valid_data(self, capsys):
        rdkit = {(r["model"], r["seed"]): r for r in _L3_ROWS}
        qc = {(r["model"], r["seed"]): r for r in _L3QC_ROWS}
        paired = pair_rows(rdkit, qc)
        print_summary(paired)   # should not raise
        captured = capsys.readouterr()
        assert "geometry comparison" in captured.out.lower()
        assert "OVERALL" in captured.out

    def test_no_crash_with_empty_input(self, capsys):
        print_summary([])
        captured = capsys.readouterr()
        assert "No paired rows" in captured.out

    def test_summary_contains_all_models(self, capsys):
        rdkit = {(r["model"], r["seed"]): r for r in _L3_ROWS}
        qc = {(r["model"], r["seed"]): r for r in _L3QC_ROWS}
        paired = pair_rows(rdkit, qc)
        print_summary(paired)
        captured = capsys.readouterr()
        for model in _MODELS:
            assert model in captured.out


# ---------------------------------------------------------------------------
# 6. main() end-to-end via CLI args
# ---------------------------------------------------------------------------

class TestMainEndToEnd:
    """Write synthetic CSVs to tmp_path and call main() directly."""

    def _setup(self, tmp_path: Path):
        """Write level-3 and level-3_qc rows to the same CSV."""
        csv_path = tmp_path / "qm9_raw_seeds.csv"
        _write_csv(csv_path, _L3_ROWS + _L3QC_ROWS)
        out_path = tmp_path / "comparison.csv"
        return csv_path, out_path

    def test_returns_zero_on_success(self, tmp_path):
        csv_path, out_path = self._setup(tmp_path)
        rc = main([
            "--results-csv", str(csv_path),
            "--qc-csv", str(csv_path),
            "--out-csv", str(out_path),
            "--no-summary",
        ])
        assert rc == 0

    def test_output_file_created(self, tmp_path):
        csv_path, out_path = self._setup(tmp_path)
        main([
            "--results-csv", str(csv_path),
            "--qc-csv", str(csv_path),
            "--out-csv", str(out_path),
            "--no-summary",
        ])
        assert out_path.exists()

    def test_output_row_count(self, tmp_path):
        csv_path, out_path = self._setup(tmp_path)
        main([
            "--results-csv", str(csv_path),
            "--qc-csv", str(csv_path),
            "--out-csv", str(out_path),
            "--no-summary",
        ])
        with open(out_path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == len(_MODELS) * len(_SEEDS)

    def test_returns_one_on_missing_level3(self, tmp_path):
        """If the results CSV has no level-3 rows, main() returns 1."""
        csv_path = tmp_path / "only_qc.csv"
        _write_csv(csv_path, _L3QC_ROWS)  # no level-3 rows
        out_path = tmp_path / "out.csv"
        rc = main([
            "--results-csv", str(csv_path),
            "--qc-csv", str(csv_path),
            "--out-csv", str(out_path),
            "--no-summary",
        ])
        assert rc == 1

    def test_returns_one_on_missing_qc(self, tmp_path):
        """If the QC CSV has no level-3_qc rows, main() returns 1."""
        csv_path = tmp_path / "only_rdkit.csv"
        _write_csv(csv_path, _L3_ROWS)  # no 3_qc rows
        out_path = tmp_path / "out.csv"
        rc = main([
            "--results-csv", str(csv_path),
            "--qc-csv", str(csv_path),
            "--out-csv", str(out_path),
            "--no-summary",
        ])
        assert rc == 1

    def test_returns_one_on_nonexistent_file(self, tmp_path):
        rc = main([
            "--results-csv", str(tmp_path / "ghost.csv"),
            "--out-csv", str(tmp_path / "out.csv"),
            "--no-summary",
        ])
        assert rc == 1

    def test_separate_qc_csv(self, tmp_path):
        """Level-3 and level-3_qc rows in different files."""
        rdkit_csv = tmp_path / "rdkit.csv"
        qc_csv = tmp_path / "qc.csv"
        out_path = tmp_path / "comparison.csv"
        _write_csv(rdkit_csv, _L3_ROWS)
        _write_csv(qc_csv, _L3QC_ROWS)
        rc = main([
            "--results-csv", str(rdkit_csv),
            "--qc-csv", str(qc_csv),
            "--out-csv", str(out_path),
            "--no-summary",
        ])
        assert rc == 0
        with open(out_path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == len(_MODELS) * len(_SEEDS)

    def test_all_output_columns_present(self, tmp_path):
        csv_path, out_path = self._setup(tmp_path)
        main([
            "--results-csv", str(csv_path),
            "--qc-csv", str(csv_path),
            "--out-csv", str(out_path),
            "--no-summary",
        ])
        with open(out_path, newline="") as fh:
            reader = csv.DictReader(fh)
            assert list(reader.fieldnames) == OUTPUT_COLUMNS
