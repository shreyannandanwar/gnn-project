"""
GNNExplainer
============
Learns soft masks over molecular edges and node features to identify
the most important substructure for a given prediction.

Reference: Ying et al., NeurIPS 2019
"GNNExplainer: Generating Explanations for Graph Neural Networks"

This implementation:
- Works with frozen F1/F2 models via MaskableModelWrapper
- Learns edge_mask [E]: per-bond importance
- Learns node_feat_mask [F]: per-feature importance
- Produces node_importance [N]: aggregated atom importance
- Computes loss = prediction_fidelity + edge_sparsity + entropy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import Dict, List, Optional, Tuple
import numpy as np


class GNNExplainer:
    """
    Generates explanations for molecular GNN predictions.

    Args:
        model: MaskableModelWrapper (wraps F1/F2 model)
        num_hops: Number of GNN layers (determines receptive field)
        epochs: Mask optimization steps per molecule
        lr: Learning rate for mask optimization
        edge_size: L1 penalty on edge mask (encourages sparsity)
        node_feat_size: L1 penalty on feature mask
        edge_entropy: Entropy penalty (encourages discrete 0/1 masks)
        node_feat_entropy: Feature mask entropy penalty
        temperature: Sigmoid temperature for mask sharpness

    Example:
        model = MaskableModelWrapper(base_model)
        explainer = GNNExplainer(model, epochs=100)
        result = explainer.explain(mol_data, task_idx=0)
        print(result['node_importance'])  # [N] per-atom importance
    """

    def __init__(
        self,
        model: nn.Module,
        num_hops: int = 4,
        epochs: int = 100,
        lr: float = 0.01,
        edge_size: float = 0.005,
        node_feat_size: float = 1.0,
        edge_entropy: float = 1.0,
        node_feat_entropy: float = 0.1,
        temperature: float = 1.0,
    ):
        self.model = model
        self.num_hops = num_hops
        self.epochs = epochs
        self.lr = lr
        self.edge_size = edge_size
        self.node_feat_size = node_feat_size
        self.edge_entropy = edge_entropy
        self.node_feat_entropy = node_feat_entropy
        self.temperature = temperature

        # Keep model frozen and in eval mode
        self.model.eval()

    def explain(
        self,
        data: Data,
        task_idx: int,
        target: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> Dict:
        """
        Generate explanation for a single molecule.

        Args:
            data: PyG Data (single molecule)
            task_idx: Which prediction to explain (0-16)
            target: Ground truth label. If None, uses model prediction.
            device: Computation device

        Returns:
            Dictionary containing:
                edge_mask      [E]  per-bond importance in [0,1]
                node_feat_mask [F]  per-feature importance in [0,1]
                node_importance[N]  per-atom importance (aggregated)
                prediction     float  original prediction
                target         float  target label
                loss_curve     list   loss per epoch
                converged      bool   whether loss converged
        """
        # Device setup
        if device is None:
            try:
                device = next(self.model.base_model.parameters()).device
            except StopIteration:
                device = torch.device('cpu')

        data = data.to(device)

        # Ensure single molecule (add batch if missing)
        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = torch.zeros(
                data.x.size(0), dtype=torch.long, device=device
            )

        num_edges = data.edge_index.shape[1]
        num_node_features = data.x.shape[1]

        # Initialize learnable masks
        # sigmoid(0) = 0.5, so we start at half importance
        edge_mask = nn.Parameter(
            torch.zeros(num_edges, device=device)
        )
        node_feat_mask = nn.Parameter(
            torch.zeros(num_node_features, device=device)
        )

        optimizer = torch.optim.Adam(
            [edge_mask, node_feat_mask], lr=self.lr
        )

        # Scheduler to reduce LR if loss plateaus
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=15, factor=0.5, min_lr=1e-4
        )

        # Get target if not provided
        if target is None:
            with torch.no_grad():
                self.model.eval()
                raw = self.model(data, task_idx=task_idx)
                pred_prob = torch.sigmoid(raw)
                target = (pred_prob > 0.5).float()
        target = target.view(1).to(device)

        # Optimization loop
        loss_curve = []
        best_loss = float('inf')
        patience_counter = 0
        EARLY_STOP_PATIENCE = 30

        masked_data = Data(
            x=data.x.clone(),
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            pos=getattr(data, 'pos', None),
            batch=data.batch,
        )

        for epoch in range(self.epochs):
            optimizer.zero_grad()

            # Compute soft masks
            edge_weight = torch.sigmoid(edge_mask / self.temperature)
            feat_weight = torch.sigmoid(node_feat_mask / self.temperature)

            # Apply node feature mask to input features
            masked_x = data.x * feat_weight.unsqueeze(0)
            masked_data.x = masked_x

            # Forward with edge weight mask
            logits = self.model(
                masked_data,
                task_idx=task_idx,
                edge_weight=edge_weight,
            )

            # Prediction fidelity loss
            pred_loss = F.binary_cross_entropy_with_logits(
                logits.view(1), target
            )

            # Edge mask losses
            # L1: encourage sparsity (few important edges)
            edge_l1 = self.edge_size * edge_weight.mean()

            # Entropy: encourage discrete masks (not 0.5)
            edge_ent = self.edge_entropy * self._binary_entropy(edge_weight)

            # Node feature mask losses
            feat_l1 = self.node_feat_size * feat_weight.mean()
            feat_ent = self.node_feat_entropy * self._binary_entropy(feat_weight)

            # Total loss
            total_loss = pred_loss + edge_l1 + edge_ent + feat_l1 + feat_ent
            total_loss.backward()
            optimizer.step()

            loss_val = total_loss.item()
            loss_curve.append(loss_val)
            scheduler.step(loss_val)

            # Early stopping
            if loss_val < best_loss - 1e-4:
                best_loss = loss_val
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= EARLY_STOP_PATIENCE:
                break

        # Extract final masks
        final_edge_mask = torch.sigmoid(edge_mask).detach().cpu()
        final_feat_mask = torch.sigmoid(node_feat_mask).detach().cpu()

        # Aggregate edge importance to node importance
        node_importance = self._aggregate_to_nodes(
            data.edge_index.cpu(),
            final_edge_mask,
            data.num_nodes,
        )

        # Get clean prediction (no masking)
        with torch.no_grad():
            self.model.eval()
            clean_logit = self.model(data, task_idx=task_idx)
            prediction = torch.sigmoid(clean_logit).item()

        # Check convergence
        if len(loss_curve) > 10:
            recent = loss_curve[-10:]
            converged = (max(recent) - min(recent)) < 0.01
        else:
            converged = False

        return {
            'edge_mask': final_edge_mask,
            'node_feat_mask': final_feat_mask,
            'node_importance': node_importance,
            'prediction': prediction,
            'target': target.cpu().item(),
            'loss_curve': loss_curve,
            'converged': converged,
            'epochs_run': len(loss_curve),
            'num_edges': num_edges,
            'num_nodes': data.num_nodes,
        }

    def explain_batch(
        self,
        data_list: List[Data],
        task_idx: int,
        device: Optional[torch.device] = None,
        verbose: bool = True,
    ) -> List[Dict]:
        """
        Explain a list of molecules.

        Args:
            data_list: List of PyG Data objects
            task_idx: Task to explain
            device: Computation device
            verbose: Print progress

        Returns:
            List of explanation dicts
        """
        results = []
        n = len(data_list)

        for i, data in enumerate(data_list):
            if verbose and (i % 5 == 0 or i == n - 1):
                print(f"  Explaining molecule {i+1}/{n}...", end='\r')

            try:
                exp = self.explain(data, task_idx, device=device)
                exp['mol_idx'] = i
                results.append(exp)
            except Exception as e:
                # Graceful failure
                if verbose:
                    print(f"\n  Warning: Failed on molecule {i}: {e}")
                results.append({
                    'edge_mask': torch.ones(data.edge_index.shape[1]) * 0.5,
                    'node_feat_mask': torch.ones(data.x.shape[1]) * 0.5,
                    'node_importance': torch.ones(data.num_nodes) * 0.5,
                    'prediction': 0.5,
                    'target': 0.0,
                    'loss_curve': [],
                    'converged': False,
                    'epochs_run': 0,
                    'num_edges': data.edge_index.shape[1],
                    'num_nodes': data.num_nodes,
                    'mol_idx': i,
                    'failed': True,
                })

        if verbose:
            print()  # newline after progress
        return results

    @staticmethod
    def _binary_entropy(
        mask: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Binary entropy of mask values.
        H(p) = -p*log(p) - (1-p)*log(1-p)
        Encourages masks toward 0 or 1 (discrete).
        """
        return -(
            mask * torch.log(mask + eps)
            + (1 - mask) * torch.log(1 - mask + eps)
        ).mean()

    @staticmethod
    def _aggregate_to_nodes(
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Convert edge importance to node importance.
        Node score = mean importance of all incident edges.

        Args:
            edge_index: [2, E]
            edge_mask: [E] importance per edge
            num_nodes: N

        Returns:
            [N] node importance scores
        """
        src, dst = edge_index[0], edge_index[1]
        node_scores = torch.zeros(num_nodes, device=edge_mask.device, dtype=edge_mask.dtype)
        node_counts = torch.zeros(num_nodes, device=edge_mask.device, dtype=edge_mask.dtype)

        node_scores.scatter_add_(0, src, edge_mask)
        node_scores.scatter_add_(0, dst, edge_mask)
        one_mask = torch.ones_like(edge_mask)
        node_counts.scatter_add_(0, src, one_mask)
        node_counts.scatter_add_(0, dst, one_mask)

        return node_scores / node_counts.clamp(min=1.0)