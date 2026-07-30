# Rejection Letter Analysis: Flexi-JEGNN

## Comprehensive mapping of each reviewer point to the code and paper

---

## Overview of Current Project Structure

| File | Description |
|------|-------------|
| [sn-article.tex](file:///c:/Users/Amulya/Desktop/Flexi-JEGNN/sn-article.tex) | Full LaTeX paper (1,628 lines) |
| [experiments/classification.py](file:///c:/Users/Amulya/Desktop/Flexi-JEGNN/experiments/classification.py) | BACE, HIV, BBBP, ADMET experiment (704 lines) |
| [experiments/qm9.py](file:///c:/Users/Amulya/Desktop/Flexi-JEGNN/experiments/qm9.py) | QM9 regression experiment (667 lines) |
| [experiments/pdbbind.py](file:///c:/Users/Amulya/Desktop/Flexi-JEGNN/experiments/pdbbind.py) | PDBbind regression experiment (797 lines) |
| [unified_experiment.py](file:///c:/Users/Amulya/Desktop/Flexi-JEGNN/unified_experiment.py) | CLI entry point (74 lines) |
| [README.md](file:///c:/Users/Amulya/Desktop/Flexi-JEGNN/README.md) | Repository documentation |
| `results/classification_results.csv` | Existing classification results |
| `results/PDBbind_results.csv` | Existing PDBbind results |
| `results/qm9_raw_seeds.csv` | Existing QM9 results |
| Root-level CSVs: ADMET, BACE, BBBP, HIV, QM9 | Raw dataset files (at root, NOT in a `datasets/` subfolder) |

> [!CAUTION]
> The `datasets/` folder referenced in README does NOT exist — raw CSVs are at the project root. The `refined-set/` folder for PDBbind is also absent. These are the most immediate reproducibility failures.

---

## Rejection Point 1: Repository Must Contain All Data Required to Reproduce Results

**Reviewer says:** _"the repository must contain all data required to reproduce the research results, without the need for additional downloading or preparation"_

### Current State
- **5 CSV files are at the project root** (not in a `datasets/` subfolder): `ADMET.csv`, `BACE.csv`, `BBBP.csv`, `HIV.csv`, `QM9.csv`
- **`datasets/` directory does NOT exist** — but README and code expect it: [`classification.py` line 593](file:///c:/Users/Amulya/Desktop/Flexi-JEGNN/experiments/classification.py#L593), [`qm9.py` line 561](file:///c:/Users/Amulya/Desktop/Flexi-JEGNN/experiments/qm9.py#L561)
- **PDBbind `refined-set/` directory does NOT exist** in the repo at all (it's a ~800MB dataset)
- The paper's Data Availability Statement ([tex line 1522–1523](file:///c:/Users/Amulya/Desktop/Flexi-JEGNN/sn-article.tex#L1522)) says results are deposited, but the `datasets/` subfolder and PDBbind data are missing
- `results/` folder does contain the three pre-computed CSV files

### What Needs to Change
- Create the `datasets/` subdirectory and move/place the 5 CSV files inside it **OR** update the code and README to look in the root
- For PDBbind: since the refined-set is ~800 MB, it should be hosted on Zenodo with a download script
- The generated 3D conformer datasets (see Point 5 below) should be hosted on GitHub or Zenodo

---

## Rejection Point 2: Custom GNN Implementations — Benchmarking Against Original Publications

**Reviewer says:** _"You used custom GNN implementations; please confirm that they allow for the reproduction of the original algorithms' results by providing benchmarking data comparing your results with those presented in the original publications."_

### Current State
All six model implementations are **custom re-implementations from scratch** — not using official library versions. Specifically:

| Model | Paper Location | Code Location | Nature of Custom Impl |
|-------|---------------|---------------|----------------------|
| SchNet | [tex L158, L688](file:///c:/Users/Amulya/Desktop/Flexi-JEGNN/sn-article.tex#L158) | [classification.py L335–365](file:///c:/Users/Amulya/Desktop/Flexi-JEGNN/experiments/classification.py#L335) | `SchNetLayer` MessagePassing — RBF filter replicated manually |
| DimeNet | [tex L160, L689](file:///c:/Users/Amulya/Desktop/Flexi-JEGNN/sn-article.tex#L160) | [classification.py L368–398](file:///c:/Users/Amulya/Desktop/Flexi-JEGNN/experiments/classification.py#L368) | `DimeNetBlock` — simplified, missing full angular interactions |
| D-MPNN | [tex L154, L686](file:///c:/Users/Amulya/Desktop/Flexi-JEGNN/sn-article.tex#L154) | [classification.py L279–309](file:///c:/Users/Amulya/Desktop/Flexi-JEGNN/experiments/classification.py#L279) | `DMPNNConv` — bond-message passing simplified |
| GIN | [tex L690](file:///c:/Users/Amulya/Desktop/Flexi-JEGNN/sn-article.tex#L690) | [classification.py L312–327](file:///c:/Users/Amulya/Desktop/Flexi-JEGNN/experiments/classification.py#L312) | Uses PyG's `GINConv` but wraps differently |
| Uni-Mol | [tex L161, L690](file:///c:/Users/Amulya/Desktop/Flexi-JEGNN/sn-article.tex#L161) | [classification.py L401–432](file:///c:/Users/Amulya/Desktop/Flexi-JEGNN/experiments/classification.py#L401) | `UniMolLite` — transformer with pair-bias; heavily simplified |
| AttentiveFP | (PDBbind only) | [pdbbind.py L602–640](file:///c:/Users/Amulya/Desktop/Flexi-JEGNN/experiments/pdbbind.py#L602) | Custom attention-based GNN |

The paper acknowledges this (Table 1 in the tex, and text like "re-implemented under the identical scaffold-split protocol"), but **no benchmarking table comparing custom implementations to original published numbers exists**.

> [!IMPORTANT]
> The paper claims these are re-implemented for fair comparison, but reviewers need proof that the custom implementations are functionally correct — i.e., produce results consistent with the originals on their own benchmarks.

### What Needs to Change
- Add a benchmarking table (or supplementary section) comparing each custom model on its **original publication's benchmark** (e.g., SchNet on QM9 MAE, DimeNet on QM9 MAE, D-MPNN on MoleculeNet AUC, GIN on TUD graph classification)
- Note: The paper already shows cross-model results at Table `tab:qm9_cross_model` ([tex L1239](file:///c:/Users/Amulya/Desktop/Flexi-JEGNN/sn-article.tex#L1239)), but these are under the constrained scaffold-split protocol, not the original papers' settings

---

## Rejection Point 3: Single Conformer Geometry — Potential Bias from Conformer Quality

**Reviewer says:** _"only a single variant of the generated geometry was used, the resulting conclusion may be biased: the geometry generation tool itself might be of insufficient quality. Furthermore, the initial 2D coordinates may have lacked stereochemical information or contained errors, potentially leading to the formation of incorrect 3D conformers."_

### Current State
**Level 3 (L3) code — in all three experiment files:**

In [classification.py L157–174](file:///c:/Users/Amulya/Desktop/Flexi-JEGNN/experiments/classification.py#L157):
```python
elif level == 3:
    mol_h = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3()) == -1:
        return None          # silently drops failed conformers
    try:
        AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
    except Exception:
        pass
    conf = mol_h.GetConformer()
    pos = np.array([[conf.GetAtomPosition(i).x, ...] for i in range(n)], ...)
```

The identical code appears in [qm9.py L151–168](file:///c:/Users/Amulya/Desktop/Flexi-JEGNN/experiments/qm9.py#L151) and [pdbbind.py L186–188](file:///c:/Users/Amulya/Desktop/Flexi-JEGNN/experiments/pdbbind.py#L186).

**Problems identified:**
1. **Single conformer only** — `EmbedMolecule` generates one conformer; no ensemble is used
2. **No stereochemical validation** — SMILES input parsed with `Chem.MolFromSmiles()` at [classification.py L118](file:///c:/Users/Amulya/Desktop/Flexi-JEGNN/experiments/classification.py#L118), which may strip or not resolve stereocenters from flat SMILES
3. **No clash checking** — failed embedding returns `None` (line 160), but partial/bad conformers that succeed silently are not checked for atomic clashes
4. **No InChI comparison** — no verification that generated conformer stereochemistry matches the original SMILES
5. **Paper mentions this in limitations** ([tex L1447](file:///c:/Users/Amulya/Desktop/Flexi-JEGNN/sn-article.tex#L1447)): _"L3 encodes a single minimum-energy conformer, which fails to capture dynamic conformational flexibility"_ — but does not address whether that single conformer is even geometrically correct

### What Needs to Change
- Add conformer quality validation: check for atomic clashes (minimum interatomic distance > van der Waals radii sum), verify stereochemistry matches original via InChI comparison
- Optionally: generate multiple conformers (e.g., `EmbedMultipleConfs`) and use the lowest-energy one
- Report the failure/skip rate at L3 for each dataset
- Host the generated 3D conformer datasets on GitHub (as requested by reviewer in Point 5)

---

## Rejection Point 4: QM9 — Must Use Original QC-Optimized Geometries

**Reviewer says:** _"for datasets (such as QM9) containing original 3D information derived from quantum chemistry (QC) methods, you employed the AllChem.ETKDGv3() method...please perform calculations using the original 3D conformers from QM9 and demonstrate whether the results are comparable to those obtained using AllChem.ETKDGv3()"_

### Current State
The QM9 experiment reads from `QM9.csv` which contains only **SMILES strings and HOMO values**:

In [qm9.py L53–56](file:///c:/Users/Amulya/Desktop/Flexi-JEGNN/experiments/qm9.py#L53):
```python
DATASET_NAME = 'QM9'
SMILES_COL = 'smiles'
LABEL_COL = 'homo'
```

And [qm9.py L111–168](file:///c:/Users/Amulya/Desktop/Flexi-JEGNN/experiments/qm9.py#L111) — the `_build_graph_tensors()` function generates conformers from scratch using ETKDGv3 for Level 3. **The original DFT-optimized QM9 3D coordinates are completely ignored/unused.**

**Paper claims (tex line 628):** _"QM9: quantum chemical property prediction. The HOMO–LUMO gap (Δε) is the regression target"_ — but never states that QC geometries are used; in fact, the code clearly overwrites them.

This is a **fundamental methodological flaw** for a paper studying geometric sensitivity: QM9's native 3D is DFT-quality, yet the paper uses ETKDGv3 (force-field level) as "ground truth" (L3). The reviewer correctly points out that if ETKDGv3 geometries are worse than DFT, the conclusions about "no benefit from 3D" may simply reflect introducing noise.

### What Needs to Change
- Add a **Level 3b** (or redefine L3 for QM9): load the original DFT/B3LYP 3D coordinates from the QM9 dataset's original `.xyz` files
- The QM9 dataset comes from Ramakrishnan et al. 2014 and is available with original 3D coords from [figshare](https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904)
- The `QM9.csv` currently in the repo needs to be augmented with 3D coordinate columns (x, y, z per atom), OR a separate pipeline must parse the original QM9 xyz files
- Report results with both ETKDGv3 and original DFT conformers — the comparison determines whether the "no geometric benefit" conclusion holds or was an artifact of noisy conformers

---

## Rejection Point 5: PDBbind and ADMETox — Protonation States

**Reviewer says:** _"For the PdbBind and ADMETox datasets, it is also necessary to account for protonation states."_

### Current State

**PDBbind** ([pdbbind.py L117–146](file:///c:/Users/Amulya/Desktop/Flexi-JEGNN/experiments/pdbbind.py#L117)):
```python
def load_ligand_mol(pdbid, data_root):
    ...
    suppl = Chem.SDMolSupplier(str(fpath), removeHs=True, sanitize=False)
    for mol in suppl:
        ...
        Chem.SanitizeMol(mol, sanitize_no_valence)
```
- `removeHs=True` discards all hydrogens including ionizable ones
- No `protonate_mol()` or pH-adjustment step
- `Chem.SanitizeMol` with `sanitize_no_valence` skips valence checks that would normally catch protonation errors
- Pocket parsing ([pdbbind.py L93–114](file:///c:/Users/Amulya/Desktop/Flexi-JEGNN/experiments/pdbbind.py#L93)) reads raw PDB ATOM/HETATM records with no protonation state correction

**ADMET** ([classification.py L591–606](file:///c:/Users/Amulya/Desktop/Flexi-JEGNN/experiments/classification.py#L591)):
```python
def load_dataset(name, datasets_dir):
    ...
    df = pd.read_csv(fpath, usecols=[sc, lc])
    df = df.dropna(subset=[sc]).copy()
```
- Reads raw SMILES from CSV; no protonation state normalization
- `Chem.MolFromSmiles()` at [classification.py L118](file:///c:/Users/Amulya/Desktop/Flexi-JEGNN/experiments/classification.py#L118) uses RDKit's default protonation (neutral form, no pH adjustment)
- For ADMET endpoints like hERG (cardiotoxicity), the ionization state at physiological pH directly determines binding; ignoring this is a real scientific concern

**Paper mentions** in [tex L656](file:///c:/Users/Amulya/Desktop/Flexi-JEGNN/sn-article.tex#L656) (PDBbind): _"protein atoms receive their element one-hot...with the six RDKit chemical descriptors...set to zero"_ — acknowledges the crude protein encoding but says nothing about protonation.

### What Needs to Change
- **PDBbind ligands**: Apply protonation normalization at physiological pH (7.4) using RDKit's `MolStandardize` or an external tool (e.g., Dimorphite-DL, OpenBabel with `--ph 7.4`)
- **PDBbind pocket**: Parse protonation from the co-crystal PDB; or use a protonation predictor
- **ADMET SMILES**: Normalize protonation before featurization (e.g., using `rdMolStandardize.Cleanup()` + enumerate tautomers + assign ionization state)
- Document the protonation handling protocol in the paper

---

## Rejection Point 6: 3D Structure Quality Analysis — Clash Checking and Stereochemistry Verification

**Reviewer says:** _"Please provide a detailed analysis of the 3D structure generation process for each set to confirm the correctness of the resulting conformers (e.g., absence of atomic clashes; consistency of the generated conformers' stereochemistry with the original structures, which can be verified using InChI). Please host the datasets containing the generated 3D conformers on GitHub."_

### Current State

The code generates conformers silently (see [classification.py L157–174](file:///c:/Users/Amulya/Desktop/Flexi-JEGNN/experiments/classification.py#L157)) with **zero validation**:
- `EmbedMolecule(...) == -1` → return None (skipped molecule)
- No minimum distance check → atomic clashes not detected
- No InChI comparison → stereo inversion not caught
- No conformer saved to disk → generated conformers are ephemeral (computed on-the-fly each run)

The paper nowhere reports:
- How many molecules failed embedding at L3 per dataset
- Whether any generated conformers had atomic clashes
- Whether stereochemistry was preserved

**Crucially, since conformers are generated on-the-fly during training (not pre-generated and cached), they cannot be "hosted on GitHub" in the current design.** This must change architecturally.

### What Needs to Change
- **Pre-generate** all L3 conformers as SDF files before training, with full validation:
  - Clash check: all pairwise distances > 0.4 × (sum of van der Waals radii) 
  - Stereo check: compare SMILES InChI stereo layer with conformer InChI stereo layer
  - Log: number of failures, skips, clashes, stereo mismatches per dataset
- Save validated conformer SDF files to a `conformers/` directory
- **Host these SDF files on GitHub** (or Zenodo if too large)
- Add a `conformer_analysis.py` script that generates this QC report
- Report summary statistics in a new paper section (e.g., Table: "Conformer generation quality statistics per dataset")

---

## Summary Table: Reviewer Points → Code/Paper Locations

| # | Reviewer Point | Primary Code Location | Primary Paper Section |
|---|---------------|----------------------|----------------------|
| 1 | Data reproducibility — missing `datasets/` dir, missing PDBbind | README, all `load_dataset()` calls | Data Availability Statement (tex L1522) |
| 2 | Custom GNN implementations — no benchmarking vs. originals | All model classes in all 3 experiment files | Results §4.4 Table cross_model, §4.5 SOTA Context |
| 3 | Single conformer bias, stereo errors in ETKDGv3 | `_build_graph_tensors()` L3 branch (all 3 files) | Limitations §6 "Single conformer" |
| 4 | QM9: should use original DFT 3D, not ETKDGv3 | `qm9.py` entire L3 branch (lines 151–168) | Methods §3.5 (QM9 dataset desc.), Limitations §6 |
| 5 | Protonation states for PDBbind and ADMET | `pdbbind.py` L117–146 (ligand loading), `classification.py` L118 (SMILES parse) | Methods §3.6 (PDBbind featurization tex L649–681) |
| 6 | 3D structure quality: clash checking, stereo via InChI, host conformers | `_build_graph_tensors()` L3 branch — no validation code exists | Missing entirely from paper |

---

## Additional Issues Observed During Exploration

### Dataset location mismatch
The README says datasets go in `datasets/` subdirectory, but the actual CSV files (`BACE.csv`, `HIV.csv`, etc.) are at the **project root**. Running `python unified_experiment.py` with defaults would fail immediately with `FileNotFoundError`.

### The `bst` directory
There is an unexplored `bst/` directory in the root. This likely contains BibTeX style files for the journal template, not research code.

### Results CSVs are pre-computed
The `results/` directory already contains all three results CSVs, meaning the experiments have been run. But without the `datasets/` directory properly set up, nobody else can reproduce them.

### Paper claims 3,250 runs completed; code supports 3,600 candidates
`6 datasets × 6 models × 5 levels × 20 seeds = 3,600` — paper says 3,250 completed. This means ~350 runs diverged or returned None. The current code silently skips failed runs (`if m is None: continue` at [classification.py L697](file:///c:/Users/Amulya/Desktop/Flexi-JEGNN/experiments/classification.py#L697)) without logging which ones failed or why.

### QM9.csv column name
Code expects `homo` as the label column ([qm9.py L55](file:///c:/Users/Amulya/Desktop/Flexi-JEGNN/experiments/qm9.py#L55)), but the paper calls the target "HOMO–LUMO gap (Δε)" — need to verify the actual CSV column matches, since the real QM9 dataset calls this column `gap` in many versions.
