"""
models/task_conditioned_egnn.py
Task-conditioned E(n)-Equivariant GNN for multi-task learning.

Key innovation: Task embeddings modulate message passing, allowing
shared structural learning with task-specific adaptation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import add_self_loops
from typing import Optional


class TaskConditionedEGNNLayer(nn.Module):
    """
    E(n)-equivariant layer with task conditioning.
    
    Node updates are modulated by a task embedding vector:
        h' = f(h, neighbors, geometry, task_emb)
    
    This allows the layer to learn shared geometric patterns while
    adapting computation to specific tasks.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        task_dim: int,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.task_dim = task_dim
        
        # Edge message network (with task conditioning)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim + 1 + task_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        
        # Node update network (with task conditioning)
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + task_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Coordinate update
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        
        # Task modulation (FiLM-style)
        self.task_scale = nn.Linear(task_dim, hidden_dim)
        self.task_shift = nn.Linear(task_dim, hidden_dim)
    
    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        task_emb: torch.Tensor,
    ):
        """
        Parameters
        ----------
        h : [N, hidden_dim] - node features
        pos : [N, 3] - 3D coordinates
        edge_index : [2, E] - edge indices
        edge_attr : [E, edge_dim] - edge features
        task_emb : [N, task_dim] - task embedding (broadcasted to all nodes)
        
        Returns
        -------
        h_new : [N, hidden_dim]
        pos_new : [N, 3]
        """
        row, col = edge_index
        num_nodes = h.size(0)
        num_edges = edge_index.size(1)
        
        # Expand task embedding to edges
        task_emb_edge = task_emb[row]  # [E, task_dim]
        
        # Compute pairwise distances
        rel_pos = pos[row] - pos[col]  # [E, 3]
        dist = torch.norm(rel_pos, dim=1, keepdim=True)  # [E, 1]
        
        # Edge messages (task-conditioned)
        edge_input = torch.cat([
            h[row], h[col], edge_attr, dist, task_emb_edge
        ], dim=1)
        m_ij = self.edge_mlp(edge_input)  # [E, hidden_dim]
        
        # Aggregate messages
        m_i = torch.zeros(num_nodes, self.hidden_dim, device=h.device)
        m_i.index_add_(0, col, m_ij)
        
        # Node update (task-conditioned)
        node_input = torch.cat([h, m_i, task_emb], dim=1)
        h_update = self.node_mlp(node_input)
        
        # FiLM modulation
        scale = self.task_scale(task_emb)  # [N, hidden_dim]
        shift = self.task_shift(task_emb)  # [N, hidden_dim]
        h_update = scale * h_update + shift
        
        h_new = h + h_update
        
        # Coordinate update
        coord_weights = self.coord_mlp(m_ij)  # [E, 1]
        coord_diff = coord_weights * rel_pos / (dist + 1e-8)
        
        pos_update = torch.zeros_like(pos)
        pos_update.index_add_(0, col, coord_diff)
        pos_new = pos + pos_update
        
        return h_new, pos_new


class TaskConditionedEGNN(nn.Module):
    """
    Task-conditioned E(n)-equivariant encoder for multi-task learning.
    
    Parameters
    ----------
    node_dim : int
        Input node feature dimension
    edge_dim : int
        Input edge feature dimension
    hidden_dim : int
        Hidden dimension
    num_layers : int
        Number of message passing layers
    num_tasks : int
        Number of tasks (for task embedding table)
    task_dim : int
        Task embedding dimension
    dropout : float
        Dropout probability
    """
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_tasks: int = 17,
        task_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_tasks = num_tasks
        self.task_dim = task_dim
        
        # Task embeddings
        self.task_embeddings = nn.Embedding(num_tasks, task_dim)
        
        # Input projection
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        
        # Task-conditioned EGNN layers
        self.layers = nn.ModuleList([
            TaskConditionedEGNNLayer(hidden_dim, edge_dim, task_dim)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize task embeddings
        nn.init.xavier_uniform_(self.task_embeddings.weight)
    
    def forward(
        self,
        batch,
        task_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        batch : PyG Batch
            Batched molecular graphs with task_id attribute
        task_id : Tensor, optional
            Override task IDs (for inference on specific task)
        
        Returns
        -------
        graph_emb : [batch_size, hidden_dim]
        """
        h = self.node_embedding(batch.x)
        pos = batch.pos
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        
        # Get task IDs
        if task_id is not None:
            task_ids = task_id
        elif hasattr(batch, "task_id"):
            task_ids = batch.task_id.squeeze()
        else:
            raise ValueError("No task_id provided")
        
        # Get task embeddings for each node
        # task_ids is [batch_size], need to expand to [num_nodes]
        if task_ids.dim() == 0:
            task_ids = task_ids.unsqueeze(0)
        
        # Map graph-level task IDs to node-level
        node_task_ids = task_ids[batch.batch]  # [num_nodes]
        task_emb = self.task_embeddings(node_task_ids)  # [num_nodes, task_dim]
        
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=h.size(0))
        
        # Pad edge_attr for self-loops
        num_self_loops = h.size(0)
        self_loop_attr = torch.zeros(
            num_self_loops, edge_attr.size(1),
            device=edge_attr.device, dtype=edge_attr.dtype
        )
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)
        
        # Message passing
        for layer in self.layers:
            h_new, pos_new = layer(h, pos, edge_index, edge_attr, task_emb)
            h = self.dropout(h_new)
            pos = pos_new
        
        # Graph-level pooling
        graph_emb = global_mean_pool(h, batch.batch)
        
        return graph_emb


class MultiTaskClassifier(nn.Module):
    """
    Multi-task classifier with task-conditioned encoder and per-task heads.
    
    Architecture:
        Molecular Graph + Task ID
               ↓
        Task-Conditioned EGNN Encoder
               ↓
        Graph Embedding
               ↓
        Task-Specific Head (selected by task_id)
               ↓
        Binary Prediction
    """
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_tasks: int = 17,
        task_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_tasks = num_tasks
        
        # Shared encoder
        self.encoder = TaskConditionedEGNN(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_tasks=num_tasks,
            task_dim=task_dim,
            dropout=dropout,
        )
        
        # Per-task classification heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
            for _ in range(num_tasks)
        ])
    
    def forward(self, batch, task_id: Optional[torch.Tensor] = None):
        """
        Forward pass returning logits.
        
        Parameters
        ----------
        batch : PyG Batch
        task_id : Tensor, optional
            Shape [batch_size] or scalar
        
        Returns
        -------
        logits : [batch_size, 1]
        """
        # Get task IDs
        if task_id is not None:
            task_ids = task_id
        elif hasattr(batch, "task_id"):
            task_ids = batch.task_id.squeeze()
        else:
            raise ValueError("No task_id provided")
        
        if task_ids.dim() == 0:
            task_ids = task_ids.unsqueeze(0)
        
        # Encode
        graph_emb = self.encoder(batch, task_ids)  # [batch_size, hidden_dim]
        
        # Apply task-specific heads
        # Each sample may have a different task, so we process per-task
        batch_size = graph_emb.size(0)
        logits = torch.zeros(batch_size, 1, device=graph_emb.device)
        
        unique_tasks = task_ids.unique()
        
        for tid in unique_tasks:
            mask = task_ids == tid
            if mask.sum() > 0:
                head = self.heads[tid.item()]
                logits[mask] = head(graph_emb[mask])
        
        return logits
    
    def predict(self, batch, task_id: Optional[torch.Tensor] = None):
        """Predict probabilities."""
        logits = self.forward(batch, task_id)
        return torch.sigmoid(logits)
    
    def compute_loss(self, batch, task_id: Optional[torch.Tensor] = None):
        """Compute BCE loss."""
        logits = self.forward(batch, task_id)
        return F.binary_cross_entropy_with_logits(logits, batch.y)
    
    def compute_per_task_losses(self, batch) -> Dict[int, torch.Tensor]:
        """
        Compute separate loss for each task in batch.
        Used for PCGrad training.
        
        Returns
        -------
        dict mapping task_id -> loss tensor
        """
        task_ids = batch.task_id.squeeze()
        if task_ids.dim() == 0:
            task_ids = task_ids.unsqueeze(0)
        
        unique_tasks = task_ids.unique()
        losses = {}
        
        for tid in unique_tasks:
            mask = task_ids == tid
            if mask.sum() > 0:
                # Get subset of batch for this task
                task_logits = self.forward(batch, task_ids)[mask]
                task_labels = batch.y[mask]
                
                losses[tid.item()] = F.binary_cross_entropy_with_logits(
                    task_logits, task_labels
                )
        
        return losses


class HardSharingClassifier(nn.Module):
    """
    Hard parameter sharing baseline (no task conditioning).
    
    Uses standard EGNN encoder shared across all tasks,
    with per-task classification heads.
    """
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_tasks: int = 17,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_tasks = num_tasks
        
        # Import standard EGNN
        from models.egnn import EGNN
        
        self.encoder = EGNN(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        
        # Per-task heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
            for _ in range(num_tasks)
        ])
    
    def forward(self, batch, task_id: Optional[torch.Tensor] = None):
        """Forward pass."""
        if task_id is not None:
            task_ids = task_id
        elif hasattr(batch, "task_id"):
            task_ids = batch.task_id.squeeze()
        else:
            raise ValueError("No task_id provided")
        
        if task_ids.dim() == 0:
            task_ids = task_ids.unsqueeze(0)
        
        graph_emb = self.encoder(batch)
        
        batch_size = graph_emb.size(0)
        logits = torch.zeros(batch_size, 1, device=graph_emb.device)
        
        unique_tasks = task_ids.unique()
        
        for tid in unique_tasks:
            mask = task_ids == tid
            if mask.sum() > 0:
                head = self.heads[tid.item()]
                logits[mask] = head(graph_emb[mask])
        
        return logits
    
    def predict(self, batch, task_id: Optional[torch.Tensor] = None):
        logits = self.forward(batch, task_id)
        return torch.sigmoid(logits)
    
    def compute_loss(self, batch, task_id: Optional[torch.Tensor] = None):
        logits = self.forward(batch, task_id)
        return F.binary_cross_entropy_with_logits(logits, batch.y)
    
    def compute_per_task_losses(self, batch):
        task_ids = batch.task_id.squeeze()
        if task_ids.dim() == 0:
            task_ids = task_ids.unsqueeze(0)
        
        unique_tasks = task_ids.unique()
        losses = {}
        
        for tid in unique_tasks:
            mask = task_ids == tid
            if mask.sum() > 0:
                task_logits = self.forward(batch, task_ids)[mask]
                task_labels = batch.y[mask]
                losses[tid.item()] = F.binary_cross_entropy_with_logits(
                    task_logits, task_labels
                )
        
        return losses