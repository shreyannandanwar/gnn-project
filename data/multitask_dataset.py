"""
data/multitask_dataset.py
Multi-task dataset combining all MoleculeNet classification datasets.
"""
import os
import pickle
import random
from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader


# Task registry with metadata
TASK_REGISTRY = {
    # Dataset: (task_names, is_multitask)
    "bbbp": (["p_np"], False),
    "bace": (["Class"], False),
    "hiv": (["HIV_active"], False),
    "clintox": (["FDA_APPROVED", "CT_TOX"], True),
    "tox21": ([
        "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER",
        "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5",
        "SR-HSE", "SR-MMP", "SR-p53"
    ], True),
}


def get_all_task_names() -> List[str]:
    """Get flat list of all task names across datasets."""
    all_tasks = []
    for dataset, (tasks, _) in TASK_REGISTRY.items():
        for task in tasks:
            all_tasks.append(f"{dataset}_{task}")
    return all_tasks


def get_task_id(dataset: str, task: str) -> int:
    """Get global task ID for a dataset-task pair."""
    all_tasks = get_all_task_names()
    task_name = f"{dataset}_{task}"
    return all_tasks.index(task_name)


def get_num_tasks() -> int:
    """Get total number of tasks."""
    return len(get_all_task_names())


class MultiTaskDataset(Dataset):
    """
    Combined multi-task dataset for joint training.
    
    Each sample includes:
    - Molecular graph (x, edge_index, edge_attr, pos)
    - Task ID (integer)
    - Binary label
    
    Parameters
    ----------
    datasets : list of str
        Which datasets to include (e.g., ['bbbp', 'bace', 'tox21'])
    split : str
        'train', 'valid', or 'test'
    root : str
        Data directory
    balance_tasks : bool
        If True, balance samples across tasks during iteration
    """
    
    def __init__(
        self,
        datasets: List[str],
        split: str,
        root: str = "data",
        balance_tasks: bool = False,
    ):
        self.datasets = [d.lower() for d in datasets]
        self.split = split.lower()
        self.root = root
        self.balance_tasks = balance_tasks
        
        self.samples = []  # List of (graph, task_id, dataset_name, task_name)
        self.task_to_samples = {}  # task_id -> list of sample indices
        
        self._load_all_datasets()
        
        super().__init__(root)
    
    def _load_all_datasets(self):
        """Load and combine all specified datasets."""
        sample_idx = 0
        
        for dataset_name in self.datasets:
            if dataset_name not in TASK_REGISTRY:
                raise ValueError(f"Unknown dataset: {dataset_name}")
            
            task_names, is_multitask = TASK_REGISTRY[dataset_name]
            
            # Load processed graphs
            graphs_path = os.path.join(
                self.root, "processed", f"{dataset_name}_graphs_3d.pkl"
            )
            
            if not os.path.exists(graphs_path):
                raise FileNotFoundError(f"Missing: {graphs_path}")
            
            with open(graphs_path, "rb") as f:
                graphs_by_split = pickle.load(f)
            
            graphs = graphs_by_split[self.split]
            
            if is_multitask:
                # Multi-task dataset (e.g., Tox21)
                # Each graph has vector labels
                for graph in graphs:
                    if not hasattr(graph, "y") or graph.y is None:
                        continue
                    
                    labels = graph.y
                    if labels.dim() == 0:
                        labels = labels.unsqueeze(0)
                    
                    for task_idx, task_name in enumerate(task_names):
                        if task_idx >= len(labels):
                            continue
                        
                        label = labels[task_idx].item()
                        
                        # Skip missing labels (-1)
                        if label < 0:
                            continue
                        
                        # Create sample
                        task_id = get_task_id(dataset_name, task_name)
                        
                        # Clone graph with single label
                        g = Data(
                            x=graph.x,
                            edge_index=graph.edge_index,
                            edge_attr=graph.edge_attr,
                            pos=graph.pos,
                            y=torch.tensor([label], dtype=torch.float),
                            task_id=torch.tensor([task_id], dtype=torch.long),
                        )
                        
                        if hasattr(graph, "smiles"):
                            g.smiles = graph.smiles
                        
                        self.samples.append((g, task_id, dataset_name, task_name))
                        
                        if task_id not in self.task_to_samples:
                            self.task_to_samples[task_id] = []
                        self.task_to_samples[task_id].append(sample_idx)
                        
                        sample_idx += 1
            
            else:
                # Single-task dataset
                task_name = task_names[0]
                task_id = get_task_id(dataset_name, task_name)
                
                for graph in graphs:
                    if not hasattr(graph, "y") or graph.y is None:
                        continue
                    
                    label = graph.y
                    if label.dim() == 0:
                        label = label.unsqueeze(0)
                    
                    label_val = label[0].item()
                    
                    # Skip invalid labels
                    if label_val < 0:
                        continue
                    
                    g = Data(
                        x=graph.x,
                        edge_index=graph.edge_index,
                        edge_attr=graph.edge_attr,
                        pos=graph.pos,
                        y=torch.tensor([label_val], dtype=torch.float),
                        task_id=torch.tensor([task_id], dtype=torch.long),
                    )
                    
                    if hasattr(graph, "smiles"):
                        g.smiles = graph.smiles
                    
                    self.samples.append((g, task_id, dataset_name, task_name))
                    
                    if task_id not in self.task_to_samples:
                        self.task_to_samples[task_id] = []
                    self.task_to_samples[task_id].append(sample_idx)
                    
                    sample_idx += 1
    
    def len(self) -> int:
        return len(self.samples)
    
    def get(self, idx: int) -> Data:
        graph, task_id, dataset_name, task_name = self.samples[idx]
        return graph
    
    def get_task_counts(self) -> Dict[int, int]:
        """Get number of samples per task."""
        return {tid: len(indices) for tid, indices in self.task_to_samples.items()}
    
    def get_task_name(self, task_id: int) -> str:
        """Get human-readable task name from ID."""
        all_tasks = get_all_task_names()
        return all_tasks[task_id]
    
    def __repr__(self) -> str:
        task_counts = self.get_task_counts()
        return (
            f"{self.__class__.__name__}(\n"
            f"  datasets={self.datasets},\n"
            f"  split={self.split},\n"
            f"  total_samples={len(self)},\n"
            f"  num_tasks={len(task_counts)},\n"
            f")"
        )


class TaskBalancedSampler(torch.utils.data.Sampler):
    """
    Sampler that balances samples across tasks.
    
    Each batch contains equal numbers of samples from each task.
    """
    
    def __init__(self, dataset: MultiTaskDataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.task_to_samples = dataset.task_to_samples
        self.num_tasks = len(self.task_to_samples)
        
        # Samples per task per batch
        self.samples_per_task = max(1, batch_size // self.num_tasks)
        
        # Calculate epoch length
        min_task_size = min(len(s) for s in self.task_to_samples.values())
        self.num_batches = min_task_size // self.samples_per_task
    
    def __iter__(self):
        # Shuffle samples within each task
        task_iterators = {
            tid: iter(random.sample(indices, len(indices)))
            for tid, indices in self.task_to_samples.items()
        }
        
        for _ in range(self.num_batches):
            batch_indices = []
            for tid in sorted(self.task_to_samples.keys()):
                for _ in range(self.samples_per_task):
                    try:
                        idx = next(task_iterators[tid])
                        batch_indices.append(idx)
                    except StopIteration:
                        break
            
            random.shuffle(batch_indices)
            yield from batch_indices
    
    def __len__(self):
        return self.num_batches * self.samples_per_task * self.num_tasks