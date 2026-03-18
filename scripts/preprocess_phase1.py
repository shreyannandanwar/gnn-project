"""
scripts/preprocess_phase1.py
Phase 1 preprocessing pipeline:
  1. Load all MoleculeNet datasets
  2. Verify scaffold splits
  3. Generate 3D molecular graphs
  4. Save processed PyG datasets
"""
import os
import pickle
import sys
import pandas as pd  # ← ADD THIS LINE
import numpy as np 
from tqdm import tqdm

sys.path.insert(0, ".")

from data.moleculenet import load_all_datasets
from data.scaffold_analysis import analyze_scaffold_split, print_scaffold_report
from data.featurizer import Molecule3DFeaturizer


def main():
    print("\n" + "="*80)
    print("PHASE 1: DATASET PREPROCESSING")
    print("="*80)

    # ──────────────────────────────────────────────────────────────────────
    # Step 1: Load all datasets
    # ──────────────────────────────────────────────────────────────────────
    print("\n[1/3] Loading MoleculeNet datasets...")
    registry = load_all_datasets(data_dir="data")

    # ──────────────────────────────────────────────────────────────────────
    # Step 2: Verify scaffold splits
    # ──────────────────────────────────────────────────────────────────────
    print("\n[2/3] Verifying scaffold splits...")
    
    for dataset_name, info in registry.items():
        df = info["df"]
        stats = analyze_scaffold_split(df)
        print_scaffold_report(dataset_name, stats)
        
        # Save stats
        stats_path = f"data/processed/{dataset_name}_scaffold_stats.pkl"
        with open(stats_path, "wb") as f:
            pickle.dump(stats, f)

    # ──────────────────────────────────────────────────────────────────────
    # Step 3: Generate 3D molecular graphs
    # ──────────────────────────────────────────────────────────────────────
    print("\n[3/3] Generating 3D molecular graphs...")
    
    featurizer = Molecule3DFeaturizer(use_3d=True)
    node_dim, edge_dim = featurizer.get_feature_dims()
    
    print(f"\nFeature dimensions:")
    print(f"  Node features: {node_dim}")
    print(f"  Edge features: {edge_dim}")
    print(f"  Positions: 3 (x, y, z)\n")

    for dataset_name, info in registry.items():
        print(f"\n{'─'*60}")
        print(f"Processing {dataset_name.upper()}")
        print(f"{'─'*60}")
        
        df = info["df"]
        tasks = info["tasks"]
        num_tasks = info["num_tasks"]
        
        graphs_by_split = {"train": [], "valid": [], "test": []}
        failed = 0
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{dataset_name}"):
            smiles = row["smiles"]
            split = row["split"]
            
            mol = featurizer.smiles_to_mol(smiles)
            if mol is None:
                failed += 1
                continue
            
            # For single-task datasets
            if num_tasks == 1:
                task = tasks[0]
                label = row[task]
                
                # Skip missing labels
                if label == -1 or pd.isna(label):
                    continue
                
                graph = featurizer.mol_to_graph(mol, label=label, task_name=task)
                if graph is not None:
                    graph.smiles = smiles
                    graphs_by_split[split].append(graph)
            
            # For multi-task datasets (e.g., Tox21)
            else:
                for task in tasks:
                    label = row[task]
                    
                    # Skip missing labels
                    if label == -1 or pd.isna(label):
                        continue
                    
                    graph = featurizer.mol_to_graph(mol, label=label, task_name=task)
                    if graph is not None:
                        graph.smiles = smiles
                        graphs_by_split[split].append(graph)
        
        print(f"  Failed: {failed}/{len(df)}")
        print(f"  Train: {len(graphs_by_split['train'])}")
        print(f"  Valid: {len(graphs_by_split['valid'])}")
        print(f"  Test:  {len(graphs_by_split['test'])}")
        
        # Save
        save_path = f"data/processed/{dataset_name}_graphs_3d.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(graphs_by_split, f)
        print(f"  ✓ Saved → {save_path}")

    print("\n" + "="*80)
    print("✅ PHASE 1 COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()