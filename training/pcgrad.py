"""
training/pcgrad.py
Projected Conflict Gradient (PCGrad) optimizer.

Reference:
    Yu et al. "Gradient Surgery for Multi-Task Learning" (NeurIPS 2020)
"""
import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import List, Dict, Optional
import random
import copy


def project_conflicting_gradients(
    grads: List[torch.Tensor],
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Apply PCGrad projection to remove conflicting gradient components.
    
    For each gradient g_i, project onto the normal plane of any gradient g_j
    that conflicts with it (negative cosine similarity).
    
    Parameters
    ----------
    grads : list of Tensor
        List of gradient vectors for each task
    reduction : str
        How to combine projected gradients: 'mean' or 'sum'
    
    Returns
    -------
    Tensor : combined gradient after conflict removal
    """
    num_tasks = len(grads)
    
    if num_tasks == 0:
        raise ValueError("No gradients provided")
    
    if num_tasks == 1:
        return grads[0]
    
    # Clone gradients to avoid modifying originals
    projected_grads = [g.clone() for g in grads]
    
    # Random order for projection (as in original paper)
    task_order = list(range(num_tasks))
    random.shuffle(task_order)
    
    for i in task_order:
        for j in task_order:
            if i == j:
                continue
            
            g_i = projected_grads[i]
            g_j = grads[j]  # Use original gradient for projection
            
            # Compute cosine similarity
            dot_product = torch.dot(g_i, g_j)
            
            # If conflicting (negative dot product), project
            if dot_product < 0:
                # Project g_i onto normal plane of g_j
                # g_i' = g_i - (g_i · g_j / ||g_j||^2) * g_j
                g_j_norm_sq = torch.dot(g_j, g_j)
                if g_j_norm_sq > 1e-8:
                    projection = (dot_product / g_j_norm_sq) * g_j
                    projected_grads[i] = g_i - projection
    
    # Combine projected gradients
    stacked = torch.stack(projected_grads)
    
    if reduction == "mean":
        return stacked.mean(dim=0)
    elif reduction == "sum":
        return stacked.sum(dim=0)
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


class PCGradOptimizer:
    """
    Wrapper that applies PCGrad to any base optimizer.
    
    Usage:
        optimizer = PCGradOptimizer(
            Adam(model.parameters(), lr=0.001),
            num_tasks=5,
        )
        
        for batch in loader:
            optimizer.zero_grad()
            
            # Compute per-task losses
            losses = [compute_loss(batch, task_id=i) for i in range(5)]
            
            # Backward with PCGrad
            optimizer.backward(losses)
            
            # Step
            optimizer.step()
    
    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Base optimizer (e.g., Adam, SGD)
    reduction : str
        How to combine gradients: 'mean' or 'sum'
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        reduction: str = "mean",
    ):
        self.optimizer = optimizer
        self.reduction = reduction
        self._param_groups = optimizer.param_groups
    
    @property
    def param_groups(self):
        return self.optimizer.param_groups
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def step(self):
        self.optimizer.step()
    
    def state_dict(self):
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
    
    def backward(self, losses: List[torch.Tensor]):
        """
        Compute PCGrad-adjusted gradients and apply to parameters.
        
        Parameters
        ----------
        losses : list of Tensor
            Per-task losses (scalar tensors)
        """
        if len(losses) == 0:
            return
        
        # Collect parameters
        params = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    params.append(p)
        
        # Compute per-task gradients
        task_grads = []
        
        for loss in losses:
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            
            # Flatten gradients
            grads = []
            for p in params:
                if p.grad is not None:
                    grads.append(p.grad.view(-1).clone())
                else:
                    grads.append(torch.zeros(p.numel(), device=p.device))
            
            task_grads.append(torch.cat(grads))
        
        # Apply PCGrad
        combined_grad = project_conflicting_gradients(
            task_grads,
            reduction=self.reduction,
        )
        
        # Unflatten and assign to parameters
        self.optimizer.zero_grad()
        
        offset = 0
        for p in params:
            numel = p.numel()
            p.grad = combined_grad[offset : offset + numel].view(p.shape).clone()
            offset += numel
    
    def backward_and_step(self, losses: List[torch.Tensor]):
        """Convenience method: backward + step."""
        self.backward(losses)
        self.step()


def compute_pcgrad_statistics(
    grads: List[torch.Tensor],
) -> Dict[str, float]:
    """
    Compute statistics about gradient conflicts before/after PCGrad.
    
    Returns
    -------
    dict with:
        - num_conflicts: number of conflicting pairs
        - conflict_ratio: fraction of pairs that conflict
        - avg_similarity_before: mean cosine similarity before PCGrad
        - avg_similarity_after: mean cosine similarity after PCGrad
    """
    num_tasks = len(grads)
    if num_tasks < 2:
        return {}
    
    # Before PCGrad
    similarities_before = []
    num_conflicts = 0
    
    for i in range(num_tasks):
        for j in range(i + 1, num_tasks):
            cos_sim = torch.nn.functional.cosine_similarity(
                grads[i].unsqueeze(0),
                grads[j].unsqueeze(0),
            ).item()
            similarities_before.append(cos_sim)
            if cos_sim < 0:
                num_conflicts += 1
    
    # After PCGrad
    projected = [g.clone() for g in grads]
    
    for i in range(num_tasks):
        for j in range(num_tasks):
            if i == j:
                continue
            dot = torch.dot(projected[i], grads[j])
            if dot < 0:
                norm_sq = torch.dot(grads[j], grads[j])
                if norm_sq > 1e-8:
                    projected[i] = projected[i] - (dot / norm_sq) * grads[j]
    
    similarities_after = []
    for i in range(num_tasks):
        for j in range(i + 1, num_tasks):
            cos_sim = torch.nn.functional.cosine_similarity(
                projected[i].unsqueeze(0),
                projected[j].unsqueeze(0),
            ).item()
            similarities_after.append(cos_sim)
    
    num_pairs = num_tasks * (num_tasks - 1) // 2
    
    return {
        "num_conflicts": num_conflicts,
        "conflict_ratio": num_conflicts / num_pairs,
        "avg_similarity_before": sum(similarities_before) / len(similarities_before),
        "avg_similarity_after": sum(similarities_after) / len(similarities_after),
    }