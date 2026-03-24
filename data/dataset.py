"""
data/dataset.py
PyTorch Geometric dataset wrapper for single-task molecular property prediction.
"""
import os
import pickle
from typing import List, Optional

import torch
from torch_geometric.data import Dataset, Data


class MoleculeDataset(Dataset):
    """
    Single-task molecular property prediction dataset.
    
    For multi-task datasets (e.g., Tox21), specify which task to use.
    
    Parameters
    ----------
    dataset_name : str
        One of: 'tox21', 'clintox', 'bbbp', 'bace', 'hiv'
    split : str
        One of: 'train', 'valid', 'test'
    task_name : str, optional
        For multi-task datasets, which task to use.
        For single-task datasets, can be None (auto-detected).
    root : str
        Path to data directory (default: 'data')
    """
    
    def __init__(
        self,
        dataset_name: str,
        split: str,
        task_name: Optional[str] = None,
        root: str = "data",
    ):
        self.dataset_name = dataset_name.lower()
        self.split = split.lower()
        self.task_name = task_name
        self.root = root
        
        # Load processed graphs
        graphs_path = os.path.join(root, "processed", f"{dataset_name}_graphs_3d.pkl")
        if not os.path.exists(graphs_path):
            raise FileNotFoundError(
                f"Processed graphs not found: {graphs_path}\n"
                "Please run: python scripts/preprocess_phase1.py"
            )
        
        with open(graphs_path, "rb") as f:
            graphs_by_split = pickle.load(f)
        
        self.graphs = graphs_by_split[split]
        
        # Load dataset info to get task names
        info_path = os.path.join(root, "processed", f"{dataset_name}_raw.pkl")
        with open(info_path, "rb") as f:
            info = pickle.load(f)
        
        self.all_tasks = info["tasks"]
        self.num_tasks = info["num_tasks"]
        
        # Determine task for single-task datasets
        if self.num_tasks == 1 and task_name is None:
            self.task_name = self.all_tasks[0]
        
        # Filter graphs for specified task (multi-task datasets)
        if self.num_tasks > 1 and task_name is not None:
            self._filter_by_task(task_name)
        
        super().__init__(root)
    
    def _filter_by_task(self, task_name: str):
        """
        For multi-task datasets, filter to only include samples with valid labels for the specified task.
        """
        if task_name not in self.all_tasks:
            raise ValueError(f"Task {task_name} not in {self.all_tasks}")
        
        task_idx = self.all_tasks.index(task_name)
        filtered = []
        
        for graph in self.graphs:
            # Multi-task labels are stored as vectors
            if hasattr(graph, "y") and graph.y.dim() == 1:
                label = graph.y[task_idx].item()
                # Skip missing labels (-1 or NaN)
                if label >= 0:
                    # Create new graph with single-task label
                    g = graph.clone()
                    g.y = torch.tensor([label], dtype=torch.float)
                    g.task_name = task_name
                    filtered.append(g)
        
        self.graphs = filtered
    
    def len(self) -> int:
        return len(self.graphs)
    
    def get(self, idx: int) -> Data:
        return self.graphs[idx]
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"dataset={self.dataset_name}, "
            f"split={self.split}, "
            f"task={self.task_name}, "
            f"size={len(self)})"
        )


def get_task_names(dataset_name: str, root: str = "data") -> List[str]:
    """
    Get list of tasks for a dataset.
    """
    info_path = os.path.join(root, "processed", f"{dataset_name}_raw.pkl")
    with open(info_path, "rb") as f:
        info = pickle.load(f)
    return info["tasks"]