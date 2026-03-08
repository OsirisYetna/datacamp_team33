# Script to download the BACE dataset from Hugging Face and create splits
# formatted for the Codabench phase architecture (train and test only).
import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem  # type: ignore
from rdkit.Chem import Descriptors  # type: ignore

PHASE = 'dev_phase'

DATA_DIR = Path(PHASE) / 'input_data'
REF_DIR = Path(PHASE) / 'reference_data'

HF_URL = "hf://datasets/molvision/BACE-V-SMILES-0/data/train-00000-of-00001.parquet"


def make_csv(data, filepath):
    """Helper to ensure directories exist and save a DataFrame to CSV."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(filepath, index=False)


def property_split(df, property_type="MW", train_frac=0.8):
    """Split into train/test based on a sorted molecular property."""
    df_split = df.copy()

    def calculate_property(smiles):
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        return (Descriptors.MolWt(mol) if property_type == "MW"
                else Descriptors.MolLogP(mol))

    print(f"Calculating {property_type} for splitting...")
    df_split[property_type] = df_split["TargetMolecule"].apply(calculate_property)
    df_split = df_split.dropna(subset=[property_type])
    df_split = (df_split
                .sort_values(by=property_type, ascending=True)
                .reset_index(drop=True))

    split_index = int(len(df_split) * train_frac)
    
    train_df = df_split.iloc[:split_index].copy()
    test_df = df_split.iloc[split_index:].copy()
    
    return train_df, test_df


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Load BACE data from HF and generate train/test splits for the benchmark'
    )
    # Kept for template compatibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (unused in MW property split but kept for API compatibility)')
    args = parser.parse_args()

    # 1. Download from Hugging Face
    print("Downloading data from Hugging Face...")
    df = pd.read_parquet(HF_URL)
    df = df.drop_duplicates(subset=["TargetMolecule"]).reset_index(drop=True)

    # 2. Generate and split the data based on Molecular Weight (Train/Test only)
    train_df, test_df = property_split(
        df, property_type="MW", train_frac=0.8
    )
    
    # 3. Clean and format
    print("Formatting datasets...")
    splits_data = {
        'train': clean_and_format_dataset(train_df),
        'test': clean_and_format_dataset(test_df)
    }

    # 4. Store the data in the correct folders
    # - input_data contains train data (both features and labels) and only
    #   test features so the test labels are kept secret
    # - reference_data contains the test labels for scoring
    print("Saving data into Codabench folder structure...")
    for split, df_split in splits_data.items():
        split_dir = DATA_DIR / split
        
        # Save Features (SMILES)
        make_csv(df_split[["SMILES"]], split_dir / f'{split}_features.csv')
        
        # Save Labels
        label_dir = split_dir if split == "train" else REF_DIR
        make_csv(df_split[["Label"]], label_dir / f'{split}_labels.csv')
        
        print(f"  Saved {split}: {len(df_split)} rows")

    print("\nDone! Data successfully setup in dev_phase/")