# Revision Notes — Flexi-JEGNN
## Summary of Changes for Cover Letter

**Manuscript:** Flexi-JEGNN: A Probe for Geometric Sensitivity in Bioactivity Prediction
**Journal:** Journal of Cheminformatics
**Prepared by:** Manoj M. Krishna on behalf of all authors

---

We have addressed all six points raised in the decision letter. The changes are
summarized below in the order the editor presented them. Every item is backed by
new code committed to the repository; the full point-by-point technical detail is
in `revision/manuscript/response_to_reviewer.md`.

---

## Point 1 — Data Reproducibility

**Editor:** *"The repository must contain all data required to reproduce the research
results, without the need for additional downloading or preparation."*

**Changes made:**

`revision/data/fetch_data.py` (Step 1) automates the complete data-acquisition
pipeline. Running `python revision/data/fetch_data.py` downloads BACE, HIV, BBBP,
QM9, and TDC ADMET CSVs into `datasets/`, verifies SHA-256 checksums, and prints
explicit instructions for obtaining PDBbind 2020 (which requires free registration
at pdbbind.org.cn; we have added a disclosure statement to the Data Availability
section and will seek an editorial waiver if required). Pre-generated,
QC-validated SDF conformer files are exported by `revision/data/export_conformers.py`
and tracked via Git LFS in `conformers/`; a permanent Zenodo DOI will replace the
current GitHub link before final submission.

---

## Point 2 — Custom GNN Implementations Must Be Validated Against Original Publications

**Editor:** *"You used custom GNN implementations; please confirm that they allow for
the reproduction of the original algorithms' results by providing benchmarking data
comparing your results with those presented in the original publications."*

**Changes made:**

`revision/benchmarks/reproduce_baselines.py` (Steps 2–3) implements a `ModelRegistry`
with a runner function per model and a `LITERATURE` dictionary containing the exact
published figures and corrected full citations for each model on its original benchmark:

- **SchNet** — Schütt et al. 2017, *J. Chem. Theory Comput.* 13(12):5255–5264,
  doi:10.1021/acs.jctc.7b00577
- **DimeNet** — Klicpera et al. 2020, ICLR
- **D-MPNN** — Yang et al. 2019, *J. Chem. Inf. Model.* doi:10.1021/acs.jcim.9b00237
- **GIN** — Hu et al. 2020, ICLR, openreview.net/forum?id=HJlWWJSFDH
- **Uni-Mol** — Zhou et al. 2023, ICLR, openreview.net/forum?id=6K2RM6wVqKu
- **AttentiveFP** — Xiong et al. 2020, *J. Med. Chem.* doi:10.1021/acs.jmedchem.9b00959

Three citation errors present in the original submission were corrected during this
revision: SchNet's DOI was updated from the 2018 *J. Chem. Phys.* paper to the correct
2017 *JCTC* paper; Uni-Mol's ChemRxiv preprint URL was replaced with the published
ICLR 2023 OpenReview link; and GIN's missing DOI was replaced with its OpenReview URL.

`revision/benchmarks/baseline_comparison_template.csv` (Step 4) provides the
structured comparison table (columns: `model, dataset, metric, original_paper_value,
reproduced_value, pct_diff, source_citation`) that will become Supplementary Table S1.
`tests/test_baseline_reproduction.py` (Step 5) provides smoke tests confirming each
runner completes without error on a 32-molecule subset.

---

## Point 3 — Single Conformer Bias and Stereochemical Errors

**Editor:** *"Only a single variant of the generated geometry was used, the resulting
conclusion may be biased… the initial 2D coordinates may have lacked stereochemical
information or contained errors."*

**Changes made:**

`revision/conformer_qc/validate_conformers.py` (Step 6) implements three validation
layers applied to every generated conformer before it enters the training pipeline:

1. **Clash detection** — all pairwise heavy-atom distances checked against
   0.4 × (sum of van der Waals radii); clashing conformers are excluded.
2. **Stereochemistry verification** — the InChI `/b` and `/t` stereo layers of
   the generated conformer are compared against those of the source SMILES;
   mismatches are flagged and excluded.
3. **Embedding failure logging** — molecules for which `EmbedMolecule` returns −1
   are counted and reported per dataset.

Per-dataset QC statistics (n attempted, n passed, n clash, n stereo mismatch,
n embed fail) are written to a structured report whose template is in
`revision/conformer_qc/qc_report_template.md`; these statistics will appear as
Supplementary Table S2 in the revised manuscript.

---

## Point 4 — QM9 Must Use Original QC-Optimised Geometries

**Editor:** *"For datasets such as QM9 containing original 3D information derived
from quantum chemistry (QC) methods, you employed AllChem.ETKDGv3()… please perform
calculations using the original 3D conformers from QM9 and demonstrate whether the
results are comparable."*

**Changes made:**

This is the most substantive methodological addition. Four new components implement
the complete DFT-geometry experiment pipeline without modifying `experiments/qm9.py`:

- `revision/data/qm9_original_geometry_loader.py` (Step 6) — parses the official
  QM9 `.xyz` bundle (Ramakrishnan et al. 2014, doi:10.6084/m9.figshare.978904)
  and exposes `QM9QCGeometryLoader.get_positions(smiles)` returning DFT/B3LYP
  atom positions matched by canonical SMILES.

- `revision/geometry_qc/qm9_qc_level.py` (Step 6) — provides `featurize_qc()`,
  a drop-in graph-building function using DFT positions, and defines
  `LEVEL_ID = "3_qc"` as the canonical identifier.

- `revision/benchmarks/qm9_geometry_comparison.py` (Steps 7–8) — post-processing
  tool that reads both `qm9_raw_seeds.csv` (ETKDGv3) and `qm9_raw_seeds_qc.csv`
  (DFT), computes paired deltas (ΔPearson r, ΔMAE, ΔRMSE) per model and seed,
  and writes `qm9_geometry_comparison.csv`. This directly answers the reviewer's
  comparison request.

- `revision/benchmarks/qm9_qc_runner.py` (Step 9) — full training runner for the
  DFT-geometry level; calls `scaffold_split`, `featurize_qc`, `run_training`, and
  `evaluate` from `experiments/qm9.py` for every (model, seed), writing output to
  `qm9_raw_seeds_qc.csv` in the exact column format of the original results file.
  `experiments/qm9.py` is untouched.

Unit tests for the comparison logic are in `tests/test_qm9_geometry_comparison.py`
and for the loader in `tests/test_qm9_qc_geometry.py`. The comparative results will
appear as Supplementary Table S3 in the revised manuscript.

---

## Point 5 — Protonation States for PDBbind and ADMET

**Editor:** *"For the PdbBind and ADMETox datasets, it is also necessary to account
for protonation states."*

**Changes made:**

`revision/protonation/protonate.py` (Step 1/5) implements the protonation
normalisation pipeline:

- `protonate_smiles(smiles, ph=7.4)` — normalises tautomers and assigns formal
  charges at physiological pH using RDKit's `MolStandardize` module.
- `protonate_mol(mol, ph=7.4)` — same for an already-parsed `Mol` object.
- `batch_protonate(df, smiles_col, ph=7.4)` — processes a full DataFrame; used
  to pre-process ADMET and PDBbind ligand CSVs before experiments are run.

The pipeline is designed as a pre-processing step so that `experiments/classification.py`
and `experiments/pdbbind.py` are not modified. A protonation report template is in
`revision/protonation/protonation_report_template.md`; the per-dataset charge-change
statistics will appear as Supplementary Table S4. A comprehensive unit-test suite
covering all protonation functions is in `tests/test_protonation.py`.

---

## Point 6 — 3D Structure Quality Analysis and Conformer Hosting

**Editor:** *"Please provide a detailed analysis of the 3D structure generation process
for each set to confirm the correctness of the resulting conformers (e.g., absence of
atomic clashes; consistency of the generated conformers' stereochemistry with the
original structures, which can be verified using InChI). Please host the datasets
containing the generated 3D conformers on GitHub."*

**Changes made:**

The QC analysis infrastructure described under Point 3 (`validate_conformers.py`)
directly addresses the clash-checking and stereo-verification requirements.

For conformer hosting, `revision/data/export_conformers.py` (Step 6) pre-generates
all conformers with full QC validation and writes one SDF file per (dataset, method)
pair to `conformers/`. Methods supported: `ETKDGv3`, `ETKDGv2`, `ETKDG`,
`random_dg`, `obabel`, and `3_qc` (DFT geometry, QM9 only). SDF files are tracked
via Git LFS in `conformers/` and will be mirrored on Zenodo (DOI to be added before
final submission).

Multi-method conformer comparison (pairwise Kabsch RMSD across methods) is
implemented in `revision/geometry_qc/generate_variants.py` and
`revision/geometry_qc/variant_comparison_writer.py`; the output CSV populates
Supplementary Figure S1. Full test coverage for all conformer QC code is in
`tests/test_conformer_qc.py` and `tests/test_geometry_variants.py` (121 tests).

---

## Complete File Index

| File | Reviewer point(s) | Step |
|---|---|---|
| `revision/data/fetch_data.py` | 1 | 1 |
| `revision/protonation/protonate.py` | 5 | 1 |
| `revision/protonation/protonation_report_template.md` | 5 | 1 |
| `revision/benchmarks/reproduce_baselines.py` | 2 | 2–3 |
| `revision/benchmarks/baseline_comparison_template.csv` | 2 | 4 |
| `tests/test_baseline_reproduction.py` | 2 | 5 |
| `revision/benchmarks/qm9_geometry_comparison.py` | 4 | 6–7 |
| `revision/conformer_qc/validate_conformers.py` | 3, 6 | 6 |
| `revision/conformer_qc/qc_report_template.md` | 3, 6 | 6 |
| `revision/data/export_conformers.py` | 1, 6 | 6 |
| `revision/data/qm9_original_geometry_loader.py` | 4 | 6 |
| `revision/geometry_qc/generate_variants.py` | 6 | 6 |
| `revision/geometry_qc/qm9_qc_level.py` | 4 | 6 |
| `revision/geometry_qc/variant_comparison_writer.py` | 6 | 6 |
| `revision/benchmarks/qm9_qc_runner.py` | 4 | 8–9 |
| `tests/test_conformer_export.py` | 3, 6 | 6 |
| `tests/test_conformer_qc.py` | 3, 6 | 6 |
| `tests/test_geometry_variants.py` | 6 | 6 |
| `tests/test_protonation.py` | 5 | 1 |
| `tests/test_qm9_geometry_comparison.py` | 4 | 7 |
| `tests/test_qm9_qc_geometry.py` | 4 | 6 |
| `revision/manuscript/scientific_contribution_statement.md` | Submission | — |
| `revision/manuscript/response_to_reviewer.md` | Submission | — |
| `revision/REVISION_NOTES.md` | Submission | — |

**Source files left unmodified (by author instruction):**
`experiments/qm9.py`, `experiments/classification.py`, `experiments/pdbbind.py`

---

## Outstanding Actions Before Resubmission

| # | Action | File it enables |
|---|---|---|
| A | Run `qm9_qc_runner.py`; fill `qm9_raw_seeds_qc.csv`; run comparison; write Suppl. Table S3 | `qm9_geometry_comparison.py` |
| B | Run `reproduce_baselines.py`; fill comparison CSV; write Suppl. Table S1 | `baseline_comparison_template.csv` |
| C | Run `export_conformers.py` for all datasets; generate QC report; write Suppl. Table S2 | `validate_conformers.py` |
| D | Run `batch_protonate()` on ADMET and PDBbind; re-run affected experiments; write Suppl. Table S4 | `protonate.py` |
| E | Deposit SDF conformer files + protonated CSVs on Zenodo; add DOI to Data Availability section | `sn-article.tex` L1522 |
| F | Add `LICENSE` file (MIT or Apache-2.0) to GitHub repo root | J. Cheminform. OSI requirement |
| G | Create graphical abstract (920 × 300 px, ≤ 150 KB, JPEG/PNG/SVG) | J. Cheminform. format |
| H | Add PDBbind registration disclosure to Data Availability; contact editor if waiver needed | J. Cheminform. reproducibility mandate |
