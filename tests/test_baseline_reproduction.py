"""
tests/test_baseline_reproduction.py
====================================
Smoke tests for revision/benchmarks/reproduce_baselines.py.

Structure
---------
TestBenchmarkDataclasses   - BenchmarkResult / SeedResult: always run, no torch needed.
TestLiteratureStructure    - LITERATURE dict shape / citation strings: always run.
TestModuleFlags            - _WORKING_MODELS, _TODO_MODELS, _HAS_*: always run.
TestFunctionSignatures     - no **kwargs, correct params: always run.
TestLevelWarnings          - UserWarning fires before RuntimeError: always run.
TestSmokeRunDMPNN          - 1 seed x 2 epochs, tiny CSV: skipped without torch.
TestSmokeRunGIN            - same.
TestSmokeRunSchNet         - same, level=3.
TestSmokeRunDimeNet        - same, level=3.
TestSmokeRunUniMol         - same, level=3.
TestSmokeRunAttentiveFP    - same, level=0: also skipped without PyG >= 2.0.

Tiny-CSV format
---------------
load_dataset() for BACE expects columns 'mol' + 'Class', but _resolve_columns()
accepts the aliases 'smiles' and 'label' (case-insensitive) for any dataset.
The synthetic CSV uses 'smiles' + 'label' so it works regardless of dataset name.
load_dataset() hard-rejects fewer than 50 rows, so the fixture produces 60 rows
(30 active, 30 inactive) with drug-like SMILES to hit that minimum.
"""

from __future__ import annotations

import inspect
import json
import math
import warnings
from dataclasses import asdict

import pytest

# ---------------------------------------------------------------------------
# Module import -- always succeeds (torch not required at import time)
# ---------------------------------------------------------------------------
from revision.benchmarks.reproduce_baselines import (
    BenchmarkResult,
    SeedResult,
    LITERATURE,
    _WORKING_MODELS,
    _TODO_MODELS,
    _HAS_EXPERIMENT_MODULE,
    _HAS_ATTENTIVEFP,
    _EPOCHS_DEFAULT,
    run_dmpnn,
    run_gin,
    run_schnet,
    run_dimenet,
    run_unimol,
    run_attentivefp,
)

# ---------------------------------------------------------------------------
# Shared skip markers
# ---------------------------------------------------------------------------
needs_experiment = pytest.mark.skipif(
    not _HAS_EXPERIMENT_MODULE,
    reason=(
        "experiments/classification.py not importable "
        "(torch / PyG not installed in this environment)"
    ),
)

needs_attentivefp = pytest.mark.skipif(
    not _HAS_ATTENTIVEFP,
    reason="torch_geometric.nn.AttentiveFP not available (PyG < 2.0)",
)


# ---------------------------------------------------------------------------
# Tiny synthetic dataset fixture
# ---------------------------------------------------------------------------
# 30 drug-like actives + 30 simple inactives = 60 rows.
# Column names 'smiles' and 'label' resolve via _resolve_columns aliases.
_ACTIVE_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",          # aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",   # caffeine
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",     # ibuprofen
    "CC(=O)Nc1ccc(O)cc1",             # paracetamol
    "c1ccc(cc1)C(=O)O",               # benzoic acid
    "OC(=O)c1ccccc1O",                # salicylic acid
    "c1ccc2ccccc2c1",                  # naphthalene
    "C1=CC=C(C=C1)N",                 # aniline
    "c1ccncc1",                        # pyridine
    "C1CCNCC1",                        # piperidine
    "CC(O)c1ccccc1",                   # 1-phenylethan-1-ol
    "c1ccc(cc1)O",                     # phenol
    "CC1=CC=CC=C1",                    # toluene
    "COc1ccccc1",                      # anisole
    "Cc1cccc(C)c1",                    # m-xylene
    "CC(=O)c1ccccc1",                  # acetophenone
    "c1ccc(cc1)CC(=O)O",               # phenylacetic acid
    "OCC1OC(O)C(O)C(O)C1O",           # glucose
    "NC(=O)c1ccccc1",                 # benzamide
    "Clc1ccccc1",                      # chlorobenzene
    "Brc1ccccc1",                      # bromobenzene
    "c1ccc(F)cc1",                     # fluorobenzene
    "CC(C)(C)c1ccccc1",                # tert-butylbenzene
    "c1ccc(-c2ccccc2)cc1",             # biphenyl
    "c1ccc(cc1)S(=O)(=O)O",           # benzenesulphonic acid
    "CC1=CC=C(C=C1)O",                 # p-cresol
    "NC(Cc1ccccc1)C(=O)O",            # phenylalanine
    "c1ccc2[nH]cccc2c1",              # indole
    "O=C1NC(=O)c2ccccc21",            # isatoic anhydride (approx)
    "c1cnc2ccccc2n1",                  # quinoxaline
]

_INACTIVE_SMILES = [
    "C",
    "CC",
    "CCC",
    "CCCC",
    "CCCCC",
    "CCCCCC",
    "CCO",
    "CCCO",
    "CCCCO",
    "CCCCCO",
    "CN",
    "CCN",
    "CCCN",
    "CCCCN",
    "CO",
    "CCS",
    "CCCS",
    "CC(C)O",
    "CC(C)N",
    "CC(C)C",
    "CCOC",
    "CC(=O)C",
    "CCC(=O)C",
    "CCCC=O",
    "CCC=O",
    "CC=O",
    "C=O",
    "C#N",
    "CC#N",
    "CCOCC",
]

assert len(_ACTIVE_SMILES) == 30, f"expected 30, got {len(_ACTIVE_SMILES)}"
assert len(_INACTIVE_SMILES) == 30, f"expected 30, got {len(_INACTIVE_SMILES)}"


@pytest.fixture(scope="session")
def tiny_csv_dir(tmp_path_factory):
    """
    Write a 60-row synthetic CSV (smiles, label) to a temp directory.
    Returns the directory Path.  The same file is written as BACE.csv,
    HIV.csv, and BBBP.csv so every runner can use it.
    Column names 'smiles'/'label' resolve via _resolve_columns aliases.
    """
    d = tmp_path_factory.mktemp("tiny_datasets")
    rows = (
        [f"{s},1" for s in _ACTIVE_SMILES] +
        [f"{s},0" for s in _INACTIVE_SMILES]
    )
    body = "smiles,label\n" + "\n".join(rows) + "\n"
    for name in ("BACE", "HIV", "BBBP"):
        (d / f"{name}.csv").write_text(body, encoding="utf-8")
    return d


# ===========================================================================
# 1. BenchmarkResult / SeedResult dataclasses
# ===========================================================================
class TestBenchmarkDataclasses:
    """Always-run: no torch needed."""

    def _make_sr(self, seed: int = 42, auc: float = 0.82) -> SeedResult:
        return SeedResult(
            seed=seed,
            train_auc=0.90,
            val_auc=0.85,
            test_auc=auc,
            n_params=50_000,
            epochs_run=60,
            stopped_early=False,
            wall_time_s=12.3,
        )

    def test_seed_result_fields(self):
        sr = self._make_sr()
        assert sr.seed == 42
        assert sr.test_auc == 0.82

    def test_compute_aggregate_single_seed(self):
        br = BenchmarkResult(dataset="BACE", model_key="dmpnn", level=0)
        br.seed_results.append(self._make_sr(auc=0.82))
        br.compute_aggregate()
        assert math.isclose(br.replicated_roc_auc_mean, 0.82)
        assert br.replicated_roc_auc_std == 0.0
        assert br.delta is None   # original_roc_auc is None

    def test_compute_aggregate_multi_seed(self):
        br = BenchmarkResult(dataset="HIV", model_key="gin", level=0)
        for auc in (0.80, 0.84, 0.82):
            br.seed_results.append(self._make_sr(auc=auc))
        br.compute_aggregate()
        assert math.isclose(br.replicated_roc_auc_mean, 0.82, abs_tol=1e-9)
        assert br.replicated_roc_auc_std > 0.0

    def test_delta_computed_when_original_known(self):
        br = BenchmarkResult(
            dataset="BACE", model_key="dmpnn", level=0,
            original_roc_auc=0.80,
        )
        br.seed_results.append(self._make_sr(auc=0.82))
        br.compute_aggregate()
        assert br.delta is not None
        assert math.isclose(br.delta, 0.82 - 0.80, abs_tol=1e-4)

    def test_json_round_trip(self):
        br = BenchmarkResult(dataset="BBBP", model_key="schnet", level=3)
        br.seed_results.append(self._make_sr(auc=0.75))
        br.compute_aggregate()
        recovered = json.loads(json.dumps(asdict(br)))
        assert recovered["dataset"] == "BBBP"
        assert recovered["model_key"] == "schnet"
        assert recovered["replicated_roc_auc_mean"] == pytest.approx(0.75)

    def test_summary_line_is_string(self):
        br = BenchmarkResult(dataset="BACE", model_key="gin", level=0)
        br.seed_results.append(self._make_sr(auc=0.83))
        br.compute_aggregate()
        line = br.summary_line()
        assert isinstance(line, str)
        assert "GIN" in line
        assert "BACE" in line


# ===========================================================================
# 2. LITERATURE structure
# ===========================================================================
class TestLiteratureStructure:
    """Always-run: validates every entry in the LITERATURE dict."""

    MODELS = ("dmpnn", "gin", "schnet", "dimenet", "unimol", "attentivefp")
    DATASETS = ("BACE", "HIV", "BBBP")

    def test_all_six_models_present(self):
        for m in self.MODELS:
            assert m in LITERATURE, f"LITERATURE missing model '{m}'"

    def test_all_three_datasets_per_model(self):
        for m in self.MODELS:
            for ds in self.DATASETS:
                assert ds in LITERATURE[m], (
                    f"LITERATURE['{m}'] missing dataset '{ds}'"
                )

    def test_original_roc_auc_all_none(self):
        """No number has been guessed; all originals must be None."""
        for m in self.MODELS:
            for ds in self.DATASETS:
                val = LITERATURE[m][ds]["original_roc_auc"]
                assert val is None, (
                    f"LITERATURE['{m}']['{ds}']['original_roc_auc'] = {val!r}; "
                    "must be None until exact paper table/split is confirmed."
                )

    def test_citations_non_empty(self):
        for m in self.MODELS:
            for ds in self.DATASETS:
                cit = LITERATURE[m][ds]["citation"]
                assert isinstance(cit, str) and len(cit) > 10, (
                    f"LITERATURE['{m}']['{ds}']['citation'] blank or too short"
                )

    def test_gin_cites_xu_not_hu(self):
        """GIN cites Xu et al. 2019 (architecture paper), not Hu et al. 2020."""
        cit = LITERATURE["gin"]["BACE"]["citation"]
        assert "Xu" in cit, f"GIN citation should be Xu et al. 2019, got: {cit!r}"
        assert "Hu" not in cit, (
            f"GIN citation should NOT reference Hu et al., got: {cit!r}"
        )

    def test_schnet_cites_2018_paper(self):
        """SchNet cites schutt2018schnet (J. Chem. Phys.), doi confirmed."""
        cit = LITERATURE["schnet"]["BACE"]["citation"]
        assert "2018" in cit, f"SchNet citation should be 2018, got: {cit!r}"
        assert "10.1063/1.5019779" in cit, (
            f"SchNet citation missing correct DOI, got: {cit!r}"
        )

    def test_unimol_cites_chemrxiv_v4(self):
        """Uni-Mol cites ChemRxiv preprint v4 (what sn-bibliography.bib cites)."""
        cit = LITERATURE["unimol"]["BACE"]["citation"]
        assert "chemrxiv" in cit.lower(), (
            f"Uni-Mol citation should be ChemRxiv, got: {cit!r}"
        )
        assert "jjm0j-v4" in cit, (
            f"Uni-Mol citation missing v4 suffix, got: {cit!r}"
        )

    def test_attentivefp_split_mismatch_flagged(self):
        """AttentiveFP BACE/BBBP split field must mention random splits."""
        for ds in ("BACE", "BBBP"):
            split = LITERATURE["attentivefp"][ds].get("split", "")
            assert "random" in split.lower(), (
                f"attentivefp/{ds} split should note random-split mismatch, "
                f"got: {split!r}"
            )


# ===========================================================================
# 3. Module flags and sets
# ===========================================================================
class TestModuleFlags:
    """Always-run."""

    def test_working_models_contains_all_six(self):
        expected = {"dmpnn", "gin", "schnet", "dimenet", "unimol", "attentivefp"}
        assert _WORKING_MODELS == expected

    def test_todo_models_empty(self):
        assert len(_TODO_MODELS) == 0, (
            f"Expected no remaining stubs, got: {_TODO_MODELS}"
        )

    def test_epochs_default_positive(self):
        assert isinstance(_EPOCHS_DEFAULT, int) and _EPOCHS_DEFAULT > 0

    def test_has_experiment_module_is_bool(self):
        assert isinstance(_HAS_EXPERIMENT_MODULE, bool)

    def test_has_attentivefp_is_bool(self):
        assert isinstance(_HAS_ATTENTIVEFP, bool)


# ===========================================================================
# 4. Function signatures
# ===========================================================================
class TestFunctionSignatures:
    """Always-run."""

    _RUNNERS = {
        "dmpnn":       run_dmpnn,
        "gin":         run_gin,
        "schnet":      run_schnet,
        "dimenet":     run_dimenet,
        "unimol":      run_unimol,
        "attentivefp": run_attentivefp,
    }
    _REQUIRED = (
        "dataset_name", "datasets_dir", "level", "seeds",
        "epochs", "batch_size", "lr",
    )

    def test_no_kwargs_catchall(self):
        for name, fn in self._RUNNERS.items():
            params = list(inspect.signature(fn).parameters.values())
            var_kw = [p for p in params if p.kind == p.VAR_KEYWORD]
            assert not var_kw, (
                f"run_{name} still has **kwargs; replace with explicit parameters."
            )

    def test_all_required_params_present(self):
        for name, fn in self._RUNNERS.items():
            sig = inspect.signature(fn)
            for p in self._REQUIRED:
                assert p in sig.parameters, (
                    f"run_{name} is missing required parameter '{p}'"
                )

    def test_level_defaults(self):
        """2-D models default level=0; 3-D models default level=3."""
        assert inspect.signature(run_dmpnn).parameters["level"].default == 0
        assert inspect.signature(run_gin).parameters["level"].default == 0
        assert inspect.signature(run_attentivefp).parameters["level"].default == 0
        assert inspect.signature(run_schnet).parameters["level"].default == 3
        assert inspect.signature(run_dimenet).parameters["level"].default == 3
        assert inspect.signature(run_unimol).parameters["level"].default == 3


# ===========================================================================
# 5. Level warnings fire before RuntimeError / ImportError
# ===========================================================================
class TestLevelWarnings:
    """Always-run: warning must fire even when deps are missing."""

    def _captured_call(self, fn, **kwargs):
        """Return (warnings_list, exception_or_None)."""
        caught = []
        exc = None
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                fn(**kwargs)
            except Exception as e:
                exc = e
        return list(w), exc

    def test_schnet_warns_at_level_0(self):
        w, _ = self._captured_call(
            run_schnet, dataset_name="BACE", datasets_dir=".", level=0, seeds=[0]
        )
        texts = [str(x.message) for x in w]
        assert any("3-D" in t or "level=3" in t for t in texts), (
            f"Expected level-3 UserWarning from run_schnet(level=0), got: {texts}"
        )

    def test_dimenet_warns_at_level_1(self):
        w, _ = self._captured_call(
            run_dimenet, dataset_name="BACE", datasets_dir=".", level=1, seeds=[0]
        )
        texts = [str(x.message) for x in w]
        assert any("3-D" in t or "level=3" in t for t in texts), (
            f"Expected level-3 UserWarning from run_dimenet(level=1), got: {texts}"
        )

    def test_unimol_warns_at_level_2(self):
        w, _ = self._captured_call(
            run_unimol, dataset_name="BACE", datasets_dir=".", level=2, seeds=[0]
        )
        texts = [str(x.message) for x in w]
        assert any("3-D" in t or "level=3" in t for t in texts), (
            f"Expected level-3 UserWarning from run_unimol(level=2), got: {texts}"
        )

    def test_dmpnn_no_level_warning_at_level_0(self):
        w, _ = self._captured_call(
            run_dmpnn, dataset_name="BACE", datasets_dir=".", level=0, seeds=[0]
        )
        level_warns = [
            x for x in w
            if issubclass(x.category, UserWarning)
            and ("level" in str(x.message).lower() or "3-D" in str(x.message))
        ]
        assert not level_warns, (
            f"run_dmpnn emitted unexpected level warning at level=0: {level_warns}"
        )

    def test_gin_no_level_warning_at_level_0(self):
        w, _ = self._captured_call(
            run_gin, dataset_name="BACE", datasets_dir=".", level=0, seeds=[0]
        )
        level_warns = [
            x for x in w
            if issubclass(x.category, UserWarning)
            and ("level" in str(x.message).lower() or "3-D" in str(x.message))
        ]
        assert not level_warns, (
            f"run_gin emitted unexpected level warning at level=0: {level_warns}"
        )

    def test_attentivefp_raises_importerror_when_pyg_missing(self):
        if _HAS_ATTENTIVEFP:
            pytest.skip("PyG AttentiveFP is available; ImportError path not triggered")
        with pytest.raises(ImportError, match="torch_geometric"):
            run_attentivefp(dataset_name="BACE", datasets_dir=".", seeds=[0])


# ===========================================================================
# Shared result validator used by all smoke tests
# ===========================================================================
def _assert_result_valid(
    result: BenchmarkResult,
    model_key: str,
    dataset: str,
    n_seeds: int = 1,
) -> None:
    """Shared post-conditions for every smoke-test BenchmarkResult."""
    assert isinstance(result, BenchmarkResult)
    assert result.model_key == model_key
    assert result.dataset == dataset
    assert len(result.seed_results) == n_seeds

    sr = result.seed_results[0]
    assert isinstance(sr, SeedResult)
    assert 0.0 <= sr.test_auc <= 1.0, f"test_auc={sr.test_auc} not in [0,1]"
    assert 0.0 <= sr.val_auc  <= 1.0, f"val_auc={sr.val_auc} not in [0,1]"
    assert sr.epochs_run >= 1,       "epochs_run must be >= 1"
    assert sr.n_params > 0,          "n_params must be positive"
    assert sr.wall_time_s >= 0.0

    # Must round-trip to JSON without errors
    json.dumps(asdict(result))

    # Timestamp must be set after a real run
    assert result.timestamp != "", "timestamp not set after run"


# ===========================================================================
# 6-11. Functional smoke tests (auto-skipped without torch / experiment)
# ===========================================================================

@needs_experiment
class TestSmokeRunDMPNN:
    def test_bace_one_seed(self, tiny_csv_dir):
        result = run_dmpnn(
            dataset_name="BACE", datasets_dir=tiny_csv_dir,
            level=0, seeds=[0], epochs=2, batch_size=32,
        )
        _assert_result_valid(result, "dmpnn", "BACE", n_seeds=1)

    def test_original_value_remains_none(self, tiny_csv_dir):
        result = run_dmpnn("BACE", tiny_csv_dir, level=0, seeds=[0], epochs=2)
        assert result.original_roc_auc is None
        assert result.delta is None


@needs_experiment
class TestSmokeRunGIN:
    def test_bace_one_seed(self, tiny_csv_dir):
        result = run_gin(
            dataset_name="BACE", datasets_dir=tiny_csv_dir,
            level=0, seeds=[0], epochs=2, batch_size=32,
        )
        _assert_result_valid(result, "gin", "BACE", n_seeds=1)

    def test_multi_seed_aggregate(self, tiny_csv_dir):
        result = run_gin(
            "BACE", tiny_csv_dir, level=0, seeds=[0, 1], epochs=2,
        )
        assert len(result.seed_results) == 2
        assert not math.isnan(result.replicated_roc_auc_mean)


@needs_experiment
class TestSmokeRunSchNet:
    def test_bace_level3_one_seed(self, tiny_csv_dir):
        result = run_schnet(
            dataset_name="BACE", datasets_dir=tiny_csv_dir,
            level=3, seeds=[0], epochs=2, batch_size=32,
        )
        _assert_result_valid(result, "schnet", "BACE", n_seeds=1)
        assert result.level == 3

    def test_wrong_level_emits_warning(self, tiny_csv_dir):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                run_schnet("BACE", tiny_csv_dir, level=0, seeds=[0], epochs=2)
            except Exception:
                pass
        texts = [str(x.message) for x in w]
        assert any("3-D" in t or "level=3" in t for t in texts)


@needs_experiment
class TestSmokeRunDimeNet:
    def test_bace_level3_one_seed(self, tiny_csv_dir):
        result = run_dimenet(
            dataset_name="BACE", datasets_dir=tiny_csv_dir,
            level=3, seeds=[0], epochs=2, batch_size=32,
        )
        _assert_result_valid(result, "dimenet", "BACE", n_seeds=1)
        assert result.level == 3


@needs_experiment
class TestSmokeRunUniMol:
    def test_bace_level3_one_seed(self, tiny_csv_dir):
        result = run_unimol(
            dataset_name="BACE", datasets_dir=tiny_csv_dir,
            level=3, seeds=[0], epochs=2, batch_size=32,
        )
        _assert_result_valid(result, "unimol", "BACE", n_seeds=1)
        assert result.level == 3

    def test_pretraining_gap_noted_in_literature(self):
        """architecture_note must mention the pre-training gap."""
        note = LITERATURE["unimol"]["BACE"].get("architecture_note", "")
        assert "pre-train" in note.lower() or "pretrain" in note.lower(), (
            f"unimol BACE architecture_note should mention pre-training gap, "
            f"got: {note!r}"
        )


@needs_experiment
@needs_attentivefp
class TestSmokeRunAttentiveFP:
    def test_bace_level0_one_seed(self, tiny_csv_dir):
        result = run_attentivefp(
            dataset_name="BACE", datasets_dir=tiny_csv_dir,
            level=0, seeds=[0], epochs=2, batch_size=32,
        )
        _assert_result_valid(result, "attentivefp", "BACE", n_seeds=1)
        assert result.level == 0

    def test_split_mismatch_noted_in_literature(self):
        """BACE and BBBP entries must flag the random-split mismatch."""
        for ds in ("BACE", "BBBP"):
            split = LITERATURE["attentivefp"][ds].get("split", "")
            assert "random" in split.lower(), (
                f"attentivefp/{ds} split should note random-split mismatch, "
                f"got: {split!r}"
            )
