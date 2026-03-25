"""
evaluation/gradient_analysis.py
Utilities for analyzing gradient conflicts between tasks.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.loader import DataLoader


def compute_task_gradients(
    model: nn.Module,
    data_loaders: Dict[int, DataLoader],
    device: str = "cuda",
    num_batches: int = 10,
) -> Dict[int, torch.Tensor]:
    """
    Compute average gradients for each task.
    
    Parameters
    ----------
    model : nn.Module
        Model with compute_loss method
    data_loaders : dict
        Mapping from task_id to DataLoader
    device : str
        Device to use
    num_batches : int
        Number of batches to average over
    
    Returns
    -------
    dict mapping task_id -> flattened gradient vector
    """
    model.to(device)
    model.train()
    
    task_gradients = {}
    
    for task_id, loader in data_loaders.items():
        accumulated_grads = None
        batch_count = 0
        
        for batch in loader:
            if batch_count >= num_batches:
                break
            
            batch = batch.to(device)
            model.zero_grad()
            
            loss = model.compute_loss(batch)
            loss.backward()
            
            # Collect gradients
            grads = []
            for param in model.parameters():
                if param.grad is not None:
                    grads.append(param.grad.view(-1).clone())
            
            flat_grad = torch.cat(grads)
            
            if accumulated_grads is None:
                accumulated_grads = flat_grad
            else:
                accumulated_grads += flat_grad
            
            batch_count += 1
        
        if accumulated_grads is not None and batch_count > 0:
            task_gradients[task_id] = accumulated_grads / batch_count
    
    return task_gradients


def compute_gradient_cosine_similarity(
    grad1: torch.Tensor,
    grad2: torch.Tensor,
) -> float:
    """
    Compute cosine similarity between two gradient vectors.
    
    Returns value in [-1, 1]:
    - 1: gradients aligned (cooperative)
    - 0: gradients orthogonal (neutral)
    - -1: gradients opposing (conflicting)
    """
    cos_sim = torch.nn.functional.cosine_similarity(
        grad1.unsqueeze(0),
        grad2.unsqueeze(0),
    )
    return cos_sim.item()


def compute_gradient_similarity_matrix(
    task_gradients: Dict[int, torch.Tensor],
    task_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute pairwise gradient cosine similarity matrix.
    
    Parameters
    ----------
    task_gradients : dict
        Mapping from task_id to gradient vector
    task_names : list, optional
        Human-readable task names
    
    Returns
    -------
    similarity_matrix : np.ndarray [num_tasks, num_tasks]
    task_labels : list of str
    """
    task_ids = sorted(task_gradients.keys())
    n_tasks = len(task_ids)
    
    similarity_matrix = np.zeros((n_tasks, n_tasks))
    
    for i, tid1 in enumerate(task_ids):
        for j, tid2 in enumerate(task_ids):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                sim = compute_gradient_cosine_similarity(
                    task_gradients[tid1],
                    task_gradients[tid2],
                )
                similarity_matrix[i, j] = sim
    
    if task_names is None:
        task_labels = [f"Task {tid}" for tid in task_ids]
    else:
        task_labels = [task_names[tid] for tid in task_ids]
    
    return similarity_matrix, task_labels


def compute_gradient_magnitude(grad: torch.Tensor) -> float:
    """Compute L2 norm of gradient vector."""
    return torch.norm(grad, p=2).item()


def compute_gradient_statistics(
    task_gradients: Dict[int, torch.Tensor],
) -> Dict[str, float]:
    """
    Compute summary statistics of gradient relationships.
    
    Returns
    -------
    dict with keys:
        - mean_similarity: average pairwise cosine similarity
        - min_similarity: minimum (most conflicting)
        - max_similarity: maximum (most cooperative)
        - conflict_ratio: fraction of negative similarities
        - mean_magnitude: average gradient norm
    """
    task_ids = sorted(task_gradients.keys())
    n_tasks = len(task_ids)
    
    similarities = []
    magnitudes = []
    
    for i, tid1 in enumerate(task_ids):
        magnitudes.append(compute_gradient_magnitude(task_gradients[tid1]))
        
        for j, tid2 in enumerate(task_ids):
            if i < j:  # Upper triangle only
                sim = compute_gradient_cosine_similarity(
                    task_gradients[tid1],
                    task_gradients[tid2],
                )
                similarities.append(sim)
    
    similarities = np.array(similarities)
    
    return {
        "mean_similarity": float(np.mean(similarities)),
        "min_similarity": float(np.min(similarities)),
        "max_similarity": float(np.max(similarities)),
        "std_similarity": float(np.std(similarities)),
        "conflict_ratio": float(np.mean(similarities < 0)),
        "mean_magnitude": float(np.mean(magnitudes)),
    }


def plot_gradient_similarity_matrix(
    similarity_matrix: np.ndarray,
    task_labels: List[str],
    save_path: Optional[str] = None,
    title: str = "Task Gradient Cosine Similarity",
) -> plt.Figure:
    """
    Plot heatmap of gradient similarities.
    
    Parameters
    ----------
    similarity_matrix : np.ndarray
        [num_tasks, num_tasks] similarity matrix
    task_labels : list of str
        Task names for axes
    save_path : str, optional
        Path to save figure
    title : str
        Plot title
    
    Returns
    -------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(
        similarity_matrix,
        xticklabels=task_labels,
        yticklabels=task_labels,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",  # Red (negative) to Green (positive)
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
    )
    
    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    return fig


def identify_conflicting_pairs(
    similarity_matrix: np.ndarray,
    task_labels: List[str],
    threshold: float = 0.0,
) -> List[Tuple[str, str, float]]:
    """
    Identify task pairs with gradient conflict.
    
    Parameters
    ----------
    similarity_matrix : np.ndarray
    task_labels : list of str
    threshold : float
        Pairs with similarity below this are considered conflicting
    
    Returns
    -------
    list of (task1, task2, similarity) tuples, sorted by similarity
    """
    n_tasks = len(task_labels)
    conflicts = []
    
    for i in range(n_tasks):
        for j in range(i + 1, n_tasks):
            sim = similarity_matrix[i, j]
            if sim < threshold:
                conflicts.append((task_labels[i], task_labels[j], sim))
    
    # Sort by similarity (most negative first)
    conflicts.sort(key=lambda x: x[2])
    
    return conflicts