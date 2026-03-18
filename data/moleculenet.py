"""
data/moleculenet.py
MoleculeNet dataset loaders with scaffold splitting verification.
"""
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from deepchem.molnet import (
    load_bace_classification,
    load_bbbp,
    load_clintox,
    load_hiv,
    load_tox21,
)


DATASET_LOADERS = {
    "tox21": load_tox21,
    "clintox": load_clintox,
    "bbbp": load_bbbp,
    "bace": load_bace_classification,
    "hiv": load_hiv,
}


class MoleculeNetLoader:
    """
    Unified interface for loading MoleculeNet classification datasets.
    
    All datasets use scaffold splitting by default.
    """

    def __init__(self, dataset_name: str, data_dir: str = "data/raw"):
        self.dataset_name = dataset_name.lower()
        if self.dataset_name not in DATASET_LOADERS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def load(self) -> Tuple[List[str], Tuple, object]:
        """
        Load dataset using DeepChem.
        
        Returns
        -------
        tasks       : list of str – task names
        datasets    : tuple (train, valid, test) – DeepChem Dataset objects
        transformers : DeepChem transformer object
        """
        loader_fn = DATASET_LOADERS[self.dataset_name]
        
        tasks, datasets, transformers = loader_fn(
            featurizer="Raw",  # We'll generate our own features
            splitter="scaffold",
            data_dir=self.data_dir,
        )
        
        return tasks, datasets, transformers

    def to_dataframe(self, tasks, datasets) -> pd.DataFrame:
        """
        Convert DeepChem datasets to unified DataFrame.
        
        Columns: smiles, split, task1, task2, ...
        """
        train_ds, valid_ds, test_ds = datasets

        def _convert(dc_dataset, split_name):
            smiles = dc_dataset.ids
            labels = dc_dataset.y
            
            # Handle both single-task and multi-task
            if labels.ndim == 1:
                labels = labels[:, None]  # [N] -> [N, 1]
            
            data = {"smiles": smiles, "split": split_name}
            for i, task in enumerate(tasks):
                data[task] = labels[:, i]
            
            return pd.DataFrame(data)

        dfs = [
            _convert(train_ds, "train"),
            _convert(valid_ds, "valid"),
            _convert(test_ds, "test"),
        ]
        
        return pd.concat(dfs, ignore_index=True)


def load_all_datasets(data_dir: str = "data") -> Dict[str, Dict]:
    """
    Load all 5 MoleculeNet classification datasets.
    
    Returns
    -------
    dict mapping dataset_name -> {
        'tasks': list of task names,
        'df': pd.DataFrame with columns [smiles, split, task1, ...],
        'num_tasks': int,
    }
    """
    registry = {}
    
    for name in DATASET_LOADERS.keys():
        print(f"\n{'='*60}")
        print(f"Loading {name.upper()}")
        print(f"{'='*60}")
        
        loader = MoleculeNetLoader(name, data_dir=f"{data_dir}/raw")
        tasks, datasets, _ = loader.load()
        df = loader.to_dataframe(tasks, datasets)
        
        # Basic statistics
        print(f"Tasks: {tasks}")
        print(f"Total samples: {len(df)}")
        print(f"  Train: {len(df[df.split == 'train'])}")
        print(f"  Valid: {len(df[df.split == 'valid'])}")
        print(f"  Test:  {len(df[df.split == 'test'])}")
        
        # Count missing labels
        for task in tasks:
            missing = df[task].isna().sum()
            if missing > 0:
                print(f"  Missing labels in {task}: {missing}")
        
        registry[name] = {
            "tasks": tasks,
            "df": df,
            "num_tasks": len(tasks),
        }
        
        # Save to disk
        save_path = f"{data_dir}/processed/{name}_raw.pkl"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(registry[name], f)
        
        print(f"✓ Saved to {save_path}")
    
    return registry