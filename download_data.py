"""Fetcher for Codabench data from Hugging Face.

Downloads the BACE dataset and saves it in the expected bundle structure:
    data/
    ├── dev_phase/
        ├── input_data/
        │   ├── train.csv          # SMILES + Label  (training set)
        │   └── test.csv           # SMILES only     (test features, no labels)
        └── reference_data/
            └── test_labels.csv    # Label only      (ground truth for scoring)
"""

import argparse
from pathlib import Path
from zlib import adler32
import pandas as pd
from rdkit import Chem  # type: ignore
from rdkit.Chem import Descriptors  # type: ignore

# ──────────────────────────────────────────────
# Paths  — mirror the codabench bundle structure
# ──────────────────────────────────────────────

ROOT               = Path(__file__).parent
INPUT_DATA_DIR     = ROOT /"deta" / "dev_phase" / "input_data"
REFERENCE_DATA_DIR = ROOT / "deta" / "dev_phase" / "reference_data"

# ──────────────────────────────────────────────
# Challenge config
# ──────────────────────────────────────────────

CHALLENGE_NAME = "bace_classifier"

DATASET_CONFIG = {
    "public": dict(
        hf_url="hf://datasets/molvision/BACE-V-SMILES-0/data/train-00000-of-00001.parquet",
        data_checksum=None,  # fill with hash_folder(INPUT_DATA_DIR) after first run
    ),
}


# ──────────────────────────────────────────────
# Data processing helpers
# ──────────────────────────────────────────────

def property_split(df, smiles_col="SMILES", label_col="Answer",
                   property_type="MW", train_frac=0.8):
    """Split into train/test based on a sorted molecular property."""
    df_split = df.copy()

    def calculate_property(smiles):
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        return (Descriptors.MolWt(mol) if property_type == "MW"
                else Descriptors.MolLogP(mol))

    df_split[property_type] = df_split[smiles_col].apply(calculate_property)
    df_split = df_split.dropna(subset=[property_type])
    df_split = (df_split
                .sort_values(by=property_type, ascending=True)
                .reset_index(drop=True))

    split_index = int(len(df_split) * train_frac)
    return df_split.iloc[:split_index].copy(), df_split.iloc[split_index:].copy()


def clean_and_format_dataset(df):
    """Rename columns and encode labels."""
    df_clean = df[["TargetMolecule", "Answer"]].copy()
    df_clean = df_clean.rename(columns={"TargetMolecule": "SMILES",
                                        "Answer": "Label"})
    df_clean["Label"] = df_clean["Label"].replace({
        "<boolean>No</boolean>": 0,
        "<boolean>Yes</boolean>": 1,
    })
    df_clean = df_clean.dropna(subset=["SMILES", "Label"])
    return df_clean


# ──────────────────────────────────────────────
# Checksum helpers
# ──────────────────────────────────────────────

def hash_folder(folder_path):
    """Return the Adler32 hash of an entire directory."""
    folder = Path(folder_path)
    checksum = 1
    for f in sorted(folder.rglob("*")):
        if f.is_file():
            checksum = adler32(f.read_bytes(), checksum)
        else:
            checksum = adler32(f.name.encode(), checksum)
    return checksum


def checksum_data(raise_error=False):
    data_checksum = DATASET_CONFIG["public"]["data_checksum"]
    if data_checksum is None:
        return True  # not configured yet, skip
    local_checksum = hash_folder(INPUT_DATA_DIR)
    if raise_error and data_checksum != local_checksum:
        raise ValueError(
            f"Checksum mismatch. Expected {data_checksum}, got {local_checksum}. "
            f"Try removing dev_phase/ and re-running."
        )
    return data_checksum == local_checksum


# ──────────────────────────────────────────────
# Main download routine
# ──────────────────────────────────────────────

def download_data():
    """Download and save data in the codabench bundle structure."""

    # Guard: do not overwrite existing data
    if INPUT_DATA_DIR.exists() and any(INPUT_DATA_DIR.iterdir()):
        print(
            f"{INPUT_DATA_DIR} is not empty. Please empty dev_phase/ "
            "if you wish to re-download."
        )
        return

    # Create directories
    INPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download from Hugging Face
    hf_url = DATASET_CONFIG["public"]["hf_url"]
    print("Downloading data from Hugging Face...", end="", flush=True)
    df = pd.read_parquet(hf_url)
    print("Ok.")

    # 2. Process
    print("Processing data...", end="", flush=True)
    df = df.drop_duplicates(subset=["TargetMolecule"]).reset_index(drop=True)
    train_df, test_df = property_split(
        df, smiles_col="TargetMolecule", label_col="Answer", property_type="MW"
    )
    train_clean = clean_and_format_dataset(train_df)
    test_clean  = clean_and_format_dataset(test_df)
    print("Ok.")

    # 3. Save input_data
    #    - train.csv : SMILES + Label  (the model will fit on this)
    #    - train_labels.csv : ground truth labels consumed by scoring.py
    #    - test.csv  : SMILES only     (the model predicts on this)
    print("Saving input_data...", end="", flush=True)
    train_clean[["SMILES"]].to_csv(INPUT_DATA_DIR / "train.csv", index=False)
    train_clean[["Label"]].to_csv(INPUT_DATA_DIR / "train_labels.csv", index=False)
    test_clean[["SMILES"]].to_csv(INPUT_DATA_DIR / "test.csv", index=False)
    print("Ok.")

    # 4. Save reference_data
    #    - test_labels.csv : ground truth labels consumed by scoring.py
    print("Saving reference_data...", end="", flush=True)
    test_clean[["Label"]].to_csv(
        REFERENCE_DATA_DIR / "test_labels.csv", index=False
    )
    print("Ok.")

    # 5. Integrity check
    print("Checking the data...", end="", flush=True)
    checksum_data(raise_error=True)
    print("Ok.")

    print("\nDone! Data saved in data/dev_phase/")
    print(f"  input_data/train.csv           : {len(train_clean)} rows  (SMILES + Label)")
    print(f"  input_data/train_labels.csv    : {len(train_clean)} rows  (Label only)")
    print(f"  input_data/test.csv            : {len(test_clean)} rows  (SMILES only)")
    print(f"  reference_data/test_labels.csv : {len(test_clean)} rows  (Label only)")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Data loader for the {CHALLENGE_NAME} challenge."
    )
    # Kept for API compatibility with the original OSF-based template
    parser.add_argument("--private", action="store_true",
                        help="(unused) kept for template compatibility.")
    parser.add_argument("--username", type=str, default=None)
    parser.add_argument("--password", type=str, default=None)
    args = parser.parse_args()

    if args.private:
        print("Warning: --private is not supported for this challenge. "
              "Downloading public data.")

    download_data()