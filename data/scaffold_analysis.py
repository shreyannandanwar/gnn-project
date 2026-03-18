"""
data/scaffold_analysis.py
Scaffold split quality verification.
"""
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from typing import Dict


def get_scaffold(smiles: str) -> str:
    """Extract Bemis-Murcko scaffold from SMILES."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=False
        )
        return scaffold
    except:
        return None


def analyze_scaffold_split(df: pd.DataFrame) -> Dict:
    """
    Verify scaffold split quality.
    
    Parameters
    ----------
    df : DataFrame with columns ['smiles', 'split']
    
    Returns
    -------
    dict with scaffold statistics and overlap metrics
    """
    df = df.copy()
    df["scaffold"] = df["smiles"].apply(get_scaffold)
    df = df[df["scaffold"].notna()]  # Drop failed scaffolds
    
    stats = {}
    
    # Per-split scaffold counts
    for split in ["train", "valid", "test"]:
        split_df = df[df["split"] == split]
        scaffolds = set(split_df["scaffold"].unique())
        
        stats[split] = {
            "num_samples": len(split_df),
            "num_scaffolds": len(scaffolds),
            "scaffolds": scaffolds,
        }
    
    # Scaffold overlap between splits
    train_sc = stats["train"]["scaffolds"]
    valid_sc = stats["valid"]["scaffolds"]
    test_sc = stats["test"]["scaffolds"]
    
    overlap_train_valid = len(train_sc & valid_sc)
    overlap_train_test = len(train_sc & test_sc)
    overlap_valid_test = len(valid_sc & test_sc)
    
    total_scaffolds = len(train_sc | valid_sc | test_sc)
    
    stats["overlap"] = {
        "train_valid": overlap_train_valid,
        "train_test": overlap_train_test,
        "valid_test": overlap_valid_test,
        "total_scaffolds": total_scaffolds,
    }
    
    # Leak percentages
    stats["leak_pct"] = {
        "train_valid": 100 * overlap_train_valid / total_scaffolds,
        "train_test": 100 * overlap_train_test / total_scaffolds,
        "valid_test": 100 * overlap_valid_test / total_scaffolds,
    }
    
    return stats


def print_scaffold_report(dataset_name: str, stats: Dict):
    """Pretty-print scaffold analysis."""
    print(f"\n{'='*60}")
    print(f"Scaffold Split Analysis: {dataset_name.upper()}")
    print(f"{'='*60}")
    
    for split in ["train", "valid", "test"]:
        s = stats[split]
        print(f"\n{split.upper()}:")
        print(f"  Samples:   {s['num_samples']:5d}")
        print(f"  Scaffolds: {s['num_scaffolds']:5d}")
    
    print(f"\nOVERLAP:")
    o = stats["overlap"]
    l = stats["leak_pct"]
    print(f"  Train-Valid: {o['train_valid']:4d} ({l['train_valid']:5.2f}%)")
    print(f"  Train-Test:  {o['train_test']:4d} ({l['train_test']:5.2f}%)")
    print(f"  Valid-Test:  {o['valid_test']:4d} ({l['valid_test']:5.2f}%)")
    print(f"  Total Scaffolds: {o['total_scaffolds']}")
    
    # Quality check
    if l["train_test"] > 5.0:
        print(f"\n⚠️  WARNING: High train-test scaffold leak (>{5}%)")
    else:
        print(f"\n✅ Split quality: GOOD")