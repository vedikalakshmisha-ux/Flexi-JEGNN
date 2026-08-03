"""
revision/data/fetch_data.py
===========================
Data reproducibility helper for the Flexi-JEGNN revision.

Scope (this file)
-----------------
* Verifies SHA-256 checksums of the four primary classification/regression
  datasets (BACE, HIV, BBBP, QM9) and the in-house ADMET table.
* Provides Zenodo / canonical download stubs for each dataset so that any
  reviewer or collaborator can reproduce the exact files used.
* PDBbind refined-set is NOT yet sourced locally — see TODO below.

Out of scope (teammate handles separately)
------------------------------------------
* Protonation pipeline          -> revision/protonation/
* Conformer QC / validation     -> revision/conformer_qc/
* Baseline benchmark scripts    -> revision/benchmarks/reproduce_baselines.py

Usage
-----
    python revision/data/fetch_data.py --verify-only
    python revision/data/fetch_data.py --download-missing
    python revision/data/fetch_data.py --help
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
# Each entry:
#   key          : short dataset name used throughout the codebase
#   filename     : expected filename on disk
#   sha256       : real SHA-256 of the file in datasets/ (computed 2026-08-01)
#   size_bytes   : nominal file size (informational; not enforced)
#   zenodo_url   : canonical download URL (None = TODO / not yet public)
#   notes        : free-text provenance note
# ---------------------------------------------------------------------------

DATASETS: dict[str, dict] = {
    "BACE": {
        "filename": "BACE.csv",
        "sha256": "503e806bce4181d15622c0090fcb778125e7c0d8b3dbe3364801a9d5c502482d",
        "size_bytes": 104742,
        "zenodo_url": "https://deepchemio.s3-us-west-1.amazonaws.com/datasets/bace.csv",
        "notes": (
            "BACE-1 inhibitor dataset (MoleculeNet / DeepChem). "
            "Binary classification: inhibitor vs non-inhibitor."
        ),
    },
    "HIV": {
        "filename": "HIV.csv",
        "sha256": "9ffa7fe57dc86c342627ee1d5255e937e2ab812393c73c4d16c697022f6e1d22",
        "size_bytes": 2193844,
        "zenodo_url": "https://deepchemio.s3-us-west-1.amazonaws.com/datasets/HIV.csv",
        "notes": (
            "HIV replication inhibition dataset (DUD-E / MoleculeNet). "
            "Binary classification."
        ),
    },
    "BBBP": {
        "filename": "BBBP.csv",
        "sha256": "d07a38487aeac5cee5508413e468043ef3097451d2a112701c2d60be9ec6b662",
        "size_bytes": 148743,
        "zenodo_url": "https://deepchemio.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
        "notes": (
            "Blood-brain barrier penetration dataset (MoleculeNet). "
            "Binary classification."
        ),
    },
    "QM9": {
        "filename": "QM9.csv",
        "sha256": "2bc6156fe9050ce92101843b4fdc38edbd3126d9ebf101883f70b47c98ddf46a",
        "size_bytes": 3361581,
        "zenodo_url": "https://deepchemio.s3-us-west-1.amazonaws.com/datasets/qm9.csv",
        "notes": (
            "QM9 quantum-chemistry dataset (Ramakrishnan et al. 2014). "
            "134k small organic molecules; 12 regression targets. "
            "NOTE: geometry column in this CSV is RDKit-generated, NOT the "
            "original DFT-optimised B3LYP/6-31G(2df,p) geometry from the "
            "QM9 SDF archive. See revision/geometry/ for the full geometry "
            "comparison and the original XYZ/SDF download."
        ),
    },
    "ADMET": {
        "filename": "ADMET.csv",
        "sha256": "7d7e7facd853a63e79ddce4e9c3fcb7a0d83a1c300b603031c0f2c64fbe77761",
        "size_bytes": 525119,
        "zenodo_url": None,  # TODO: confirm public Zenodo DOI with teammate
        "notes": (
            "In-house ADMET endpoint table. Zenodo DOI pending — teammate is "
            "finalising the upload. Set zenodo_url above once the DOI is known."
        ),
    },
    # -----------------------------------------------------------------------
    # TODO: PDBbind refined set
    # -----------------------------------------------------------------------
    # "PDBbind": {
    #     "filename": "PDBbind_refined.tar.gz",
    #     "sha256": None,          # not yet sourced locally
    #     "size_bytes": None,
    #     "zenodo_url": "http://www.pdbbind.org.cn/download/PDBbind_v2020_refined.tar.gz",
    #     "notes": (
    #         "PDBbind v2020 refined set. Requires free registration at "
    #         "http://www.pdbbind.org.cn/. Teammate is sourcing this "
    #         "independently. Uncomment and fill sha256 once available."
    #     ),
    # },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_of_file(path: Path, chunk: int = 1 << 20) -> str:
    """Return lowercase hex SHA-256 of *path*."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def verify_dataset(
    name: str,
    meta: dict,
    data_dir: Path,
) -> bool:
    """
    Check that *data_dir / meta['filename']* exists and matches the recorded
    SHA-256.  Returns True on success, False on any failure.
    """
    path = data_dir / meta["filename"]

    if not path.exists():
        print(f"  [MISSING]  {name}: {path}")
        return False

    actual = sha256_of_file(path)
    expected = meta["sha256"]

    if expected is None:
        print(f"  [SKIP]     {name}: no reference checksum recorded (TODO)")
        return True  # treat as soft pass

    if actual.lower() == expected.lower():
        print(f"  [OK]       {name}: SHA-256 verified ({actual[:16]}...)")
        return True

    print(
        f"  [MISMATCH] {name}: expected {expected[:16]}... "
        f"got {actual[:16]}..."
    )
    return False


def download_dataset(name: str, meta: dict, data_dir: Path) -> None:
    """
    Attempt to download *meta['filename']* from *meta['zenodo_url']* into
    *data_dir*.  Skips if the file already exists and its checksum matches.
    """
    path = data_dir / meta["filename"]

    # Already present and verified?
    if path.exists() and meta["sha256"] is not None:
        if sha256_of_file(path).lower() == meta["sha256"].lower():
            print(f"  [SKIP]     {name}: already present and verified.")
            return

    url: Optional[str] = meta.get("zenodo_url")
    if url is None:
        print(
            f"  [TODO]     {name}: no download URL recorded. "
            "Please add it to DATASETS in this file once the source is known."
        )
        return

    print(f"  [DOWNLOAD] {name}: {url}")
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, path)
        print(f"             Saved to {path}")
        # Re-verify after download
        if meta["sha256"] is not None:
            actual = sha256_of_file(path)
            if actual.lower() != meta["sha256"].lower():
                print(
                    f"  [WARNING]  {name}: checksum mismatch after download! "
                    f"Expected {meta['sha256'][:16]}... got {actual[:16]}..."
                )
            else:
                print(f"             Checksum OK.")
    except Exception as exc:
        print(f"  [ERROR]    {name}: download failed -- {exc}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Flexi-JEGNN data reproducibility helper.\n\n"
            "Verifies SHA-256 checksums for all registered datasets and "
            "optionally downloads any that are missing."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--data-dir",
        default=None,
        help=(
            "Directory containing the CSV/archive files. "
            "Defaults to <repo_root>/datasets/."
        ),
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--verify-only",
        action="store_true",
        default=True,
        help="Only verify checksums; do not download anything (default).",
    )
    mode.add_argument(
        "--download-missing",
        action="store_true",
        help="Download any dataset whose file is missing or whose URL is known.",
    )
    p.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        default=None,
        help="Operate on a single dataset instead of all.",
    )
    return p


def resolve_data_dir(cli_arg: Optional[str]) -> Path:
    if cli_arg:
        return Path(cli_arg).expanduser().resolve()
    # Default: <repo_root>/datasets/  (this file lives at repo/revision/data/)
    here = Path(__file__).resolve().parent          # revision/data/
    return here.parent.parent / "datasets"          # repo/datasets/


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    data_dir = resolve_data_dir(args.data_dir)
    print(f"\nData directory : {data_dir}")
    print(f"Mode           : {'download-missing' if args.download_missing else 'verify-only'}")
    print("-" * 60)

    targets: dict[str, dict] = (
        {args.dataset: DATASETS[args.dataset]} if args.dataset else DATASETS
    )

    all_ok = True
    for name, meta in targets.items():
        if args.download_missing:
            download_dataset(name, meta, data_dir)
        ok = verify_dataset(name, meta, data_dir)
        all_ok = all_ok and ok

    print("-" * 60)
    if all_ok:
        print("All datasets verified successfully.\n")
        return 0
    else:
        print(
            "One or more datasets failed verification. "
            "Run with --download-missing to attempt automatic retrieval, "
            "or manually place the files in the data directory.\n"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
