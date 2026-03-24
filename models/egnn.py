"""
models/egnn.py
E(n)-Equivariant Graph Neural Network for 3D molecular graphs.

Reference:
    Satorras et al. "E(n) Equivariant Graph Neural Networks" (ICML 2021)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import add_self_loops


class EGNNLayer(nn.Module):
    """
    Single E(n)-equivariant message passing layer.
    
    Updates both node features (h) and positions (x) while maintaining
    E(n) equivariance (translation and rotation invariance).
    """
    
    def __init__(self, hidden_dim: int, edge_dim: int = 6):
        super().__init__()
        
        # Edge message network
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim + 1, hidden_dim),  # +1 for distance
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        
        # Node update network
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Position update (coordinate net)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
    
    def forward(self, h, pos, edge_index, edge_attr):
        """
        Parameters
        ----------
        h : [N, hidden_dim] - node features
        pos : [N, 3] - 3D coordinates
        edge_index : [2, E] - edge indices
        edge_attr : [E, edge_dim] - edge features
        
        Returns
        -------
        h_new : [N, hidden_dim]
        pos_new : [N, 3]
        """
        row, col = edge_index
        
        # Compute pairwise distances
        rel_pos = pos[row] - pos[col]  # [E, 3]
        dist = torch.norm(rel_pos, dim=1, keepdim=True)  # [E, 1]
        
        # Edge messages
        edge_input = torch.cat([h[row], h[col], edge_attr, dist], dim=1)  # [E, 2*h + e + 1]
        m_ij = self.edge_mlp(edge_input)  # [E, hidden_dim]
        
        # Aggregate messages
        m_i = torch.zeros(h.size(0), m_ij.size(1), device=h.device)  # [N, hidden_dim]
        m_i.index_add_(0, col, m_ij)
        
        # Update node features
        h_new = h + self.node_mlp(torch.cat([h, m_i], dim=1))
        
        # Update positions (coordinate update)
        coord_weights = self.coord_mlp(m_ij)  # [E, 1]
        coord_diff = coord_weights * rel_pos / (dist + 1e-8)  # Normalized direction
        
        pos_update = torch.zeros_like(pos)  # [N, 3]
        pos_update.index_add_(0, col, coord_diff)
        pos_new = pos + pos_update
        
        return h_new, pos_new


class EGNN(nn.Module):
    """
    E(n)-Equivariant Graph Neural Network encoder.
    
    Parameters
    ----------
    node_dim : int
        Input node feature dimension
    edge_dim : int
        Input edge feature dimension
    hidden_dim : int
        Hidden dimension for all layers
    num_layers : int
        Number of message passing layers
    dropout : float
        Dropout probability
    """
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input embedding
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        
        # EGNN layers
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim, edge_dim)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, batch):
        """
        Parameters
        ----------
        batch : torch_geometric.data.Batch
            Batch of molecular graphs with attributes:
            - x: [N, node_dim] node features
            - edge_index: [2, E] edge indices
            - edge_attr: [E, edge_dim] edge features
            - pos: [N, 3] 3D coordinates
            - batch: [N] batch assignment
        
        Returns
        -------
        graph_embedding : [batch_size, hidden_dim]
        """
        h = self.node_embedding(batch.x)
        pos = batch.pos
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        
        # Add self-loops for better propagation
        edge_index, _ = add_self_loops(edge_index, num_nodes=h.size(0))
        
        # Pad edge_attr for self-loops (zeros)
        num_self_loops = h.size(0)
        self_loop_attr = torch.zeros(
            num_self_loops, edge_attr.size(1),
            device=edge_attr.device, dtype=edge_attr.dtype
        )
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)
        
        # Message passing
        for layer in self.layers:
            h_new, pos_new = layer(h, pos, edge_index, edge_attr)
            h = self.dropout(h_new)
            pos = pos_new
        
        # Graph-level pooling
        graph_emb = global_mean_pool(h, batch.batch)
        
        return graph_emb


class EGNNClassifier(nn.Module):
    """
    Full EGNN model for binary classification.
    
    Wraps EGNN encoder + MLP head.
    """
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.encoder = EGNN(node_dim, edge_dim, hidden_dim, num_layers, dropout)
        
        # Classification head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, batch):
        """Forward pass returning logits."""
        graph_emb = self.encoder(batch)
        logits = self.head(graph_emb)
        return logits
    
    def predict(self, batch):
        """Predict probabilities (for evaluation)."""
        logits = self.forward(batch)
        return torch.sigmoid(logits)
    
    def compute_loss(self, batch):
        """Compute BCE loss."""
        logits = self.forward(batch)
        loss = F.binary_cross_entropy_with_logits(logits, batch.y)
        return loss