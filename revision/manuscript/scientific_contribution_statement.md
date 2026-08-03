# Scientific Contribution Statement
## *Journal of Cheminformatics* — Research Article

**Manuscript title:** Flexi-JEGNN: A Probe for Geometric Sensitivity in Bioactivity Prediction

**Corresponding author:** Pooja Gupta (pooja.gupta@jaipur.manipal.edu)

---

> **Guideline source:** Journal of Cheminformatics Research article requirements
> ([link.springer.com/journal/13321/submission-guidelines/research](https://link.springer.com/journal/13321/submission-guidelines/research))
>
> *"The abstract should contain a section titled 'Scientific Contribution'. In this section,
> please use a maximum of 3 sentences to specifically highlight the scientific contributions
> that advance the field and what differentiates your contribution from prior work on this topic."*

---

## Scientific Contribution

*(≤ 3 sentences — insert verbatim into the abstract immediately below the main abstract body)*

Flexi-JEGNN is the first controlled probe framework designed to isolate the predictive
contribution of geometric precision **within** the scalar-distance message-passing paradigm
by accepting five distance-representation levels — from a constant scalar to a genuine
ETKDGv3 3D conformer — without any architectural modification, enabling a clean
within-architecture causal estimate that eliminates the confound between geometric fidelity
and model capacity present in all prior comparative studies.
Across 3,250 training runs on six benchmarks (BACE, BBBP, HIV, QM9, PDBbind, TDC ADMET)
with 20 random seeds and six model families, a one-way restricted ANOVA over physically
meaningful levels (L0–L3) is non-significant on every benchmark (*p* = 0.070–0.549;
Bayes Factors BF01 = 84.6–631.1 on five of six tasks), and the same insensitivity is
reproduced by SchNet, DimeNet, and Uni-Mol at the same distance levels.
These results establish pure 2D topological encodings (zero spatial preprocessing cost) as
the Pareto-efficient choice for virtual screening on standard MoleculeNet/TDC bioactivity
benchmarks within the scalar-distance paradigm, and provide quantitative evidence that
current bioactivity benchmarks lack the geometric sensitivity needed to discriminate between
3D architectures — motivating the community to develop geometry-sensitive evaluation sets
before attributing empirical gains to 3D representations.

---

## Full Declarations Block

*(Paste as-is into the Declarations section of the manuscript — already present in
sn-article.tex but reproduced here for the submission system's structured fields.)*

### Availability of Data and Materials

The datasets analysed in this study (BACE, BBBP, HIV, PDBbind 2020 Refined Set, QM9,
TDC ADMET) are publicly available through MoleculeNet (http://moleculenet.org) and the
Therapeutic Data Commons (https://tdcommons.ai).
The Flexi-JEGNN source code, trained model checkpoints, and full per-seed metric arrays
for all 3,250 experimental runs (ROC-AUC, Pearson r, RMSE, MAE per seed x level x
dataset x model) are openly available at:

  GitHub repository: https://github.com/ManojK2K80/Flexi-JEGNN

Raw results are deposited in the repository, enabling independent replication of all
ANOVA, Bayes Factor, and Cohen's f calculations reported in this paper.

> [!IMPORTANT]
> **Action required before submission:** Deposit raw conformer SDF files and any
> PDBbind-derived data on Zenodo or Figshare (see rejection_analysis.md section 3.1).
> Replace the GitHub-only data link above with the permanent DOI once obtained.

---

### Competing Interests

The authors declare that they have no competing interests.

---

### Funding

This research received no specific grant from any funding agency in the public, commercial,
or not-for-profit sectors. Computational resources were provided by JSS Academy of
Technical Education (Bengaluru) and Manipal University Jaipur.

---

### Authors' Contributions

Using CRediT (Contributor Roles Taxonomy — https://credit.niso.org/):

| Author | Role(s) |
|---|---|
| Nethravathi B (N.B.) | Conceptualization; Methodology; Supervision |
| Pooja Gupta (P.G.) | Conceptualization; Methodology; Supervision; Project administration |
| Manoj M. Krishna (M.M.K.) | Software; Formal analysis; Investigation; Writing – Original Draft |
| Parnika Madhukumar (P.M.) | Data curation; Validation |
| Vedika Lakshmisha (V.L.) | Data curation; Validation |
| Abhinav Trivedi (A.T.) | Writing – Review & Editing |

All authors read and approved the final manuscript.

---

### Acknowledgements

The authors thank JSS Academy of Technical Education (Bengaluru) and Manipal University
Jaipur for access to computational infrastructure. The QM9 dataset was originally
compiled and released by Ramakrishnan et al. (2014); PDBbind 2020 is maintained by the
Wang group at Shanghai Institute of Organic Chemistry.

---

## Checklist Against J. Cheminform. Research Article Requirements

| Requirement | Status | Notes |
|---|---|---|
| Abstract <= 350 words | Check before submission | Count before submission |
| "Scientific Contribution" subsection in abstract | Drafted above | Insert after main abstract body |
| Graphical abstract (920 x 300 px, <= 150 KB, JPEG/PNG/SVG) | TODO | Generate from FigFindings.drawio.pdf |
| All datasets/code accessible without login or non-CC license | Partial | PDBbind requires registration — add statement |
| Availability of Data and Materials section | Present in .tex | Add Zenodo DOI when available |
| Competing interests | Done | |
| Funding | Done | |
| Authors' contributions (CRediT) | Done | |
| Acknowledgements | Done | |
| Keywords (3–10) | Done | 5 keywords in .tex |
| Reference format (Vancouver numbered) | Done | sn-vancouver-num class used |
| Source code provided (OSI-approved licence) | TODO | Add LICENSE file to GitHub repo |

---

## Notes on Reproducibility Requirement

J. Cheminform. has an unusually strict reproducibility policy:

  "...any datasets, software and algorithms that are required to reach the conclusions
  stated in the paper must be provided as supplemental materials, or be otherwise
  accessible without the need for registration, login or agreement with license terms
  other than Creative Commons licenses for data and OSI-approved Open Source Licenses
  for software."

PDBbind 2020 requires free registration at http://pdbbind.org.cn, which technically
violates this clause. The manuscript should either:

  1. Use only the publicly mirrored subset available without login (e.g., via Zenodo), OR
  2. Add an explicit statement in the Data Availability section noting the registration
     requirement and citing the PDBbind registration page, and consult the editor before
     submission to obtain a waiver.

Conformer SDF files should be deposited on Zenodo/Figshare (no login required) and
cited with a permanent DOI; the GitHub repository link alone does not satisfy the
"accessible without license agreement" clause for large binary assets tracked via Git LFS.
