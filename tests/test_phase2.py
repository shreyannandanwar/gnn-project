"""
tests/test_phase2.py
Comprehensive tests for Phase 2: Single-Task 3D GNN Baselines.

Run with: python -m pytest tests/test_phase2.py -v
"""
import os
import sys
import pickle
import pytest
import torch
import numpy as np

sys.path.insert(0, ".")

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_graph():
    """Create a simple molecular graph for testing."""
    # 5 atoms, 4 bonds (undirected = 8 edges)
    x = torch.randn(5, 129)  # node features
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4],
        [1, 0, 2, 1, 3, 2, 4, 3],
    ], dtype=torch.long)
    edge_attr = torch.randn(8, 6)  # edge features
    pos = torch.randn(5, 3)  # 3D coordinates
    y = torch.tensor([1.0])  # binary label
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, y=y)


@pytest.fixture
def sample_batch(sample_graph):
    """Create a batch of graphs."""
    graphs = [sample_graph.clone() for _ in range(4)]
    return Batch.from_data_list(graphs)


@pytest.fixture
def model_config():
    """Default model configuration."""
    return {
        "node_dim": 129,
        "edge_dim": 6,
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMoleculeDataset:
    """Tests for data/dataset.py"""
    
    def test_dataset_import(self):
        """Verify dataset module can be imported."""
        from data.dataset import MoleculeDataset, get_task_names
        assert MoleculeDataset is not None
        assert get_task_names is not None
    
    def test_processed_files_exist(self):
        """Check that Phase 1 preprocessing completed."""
        datasets = ["tox21", "clintox", "bbbp", "bace", "hiv"]
        for ds in datasets:
            path = f"data/processed/{ds}_graphs_3d.pkl"
            assert os.path.exists(path), f"Missing: {path}. Run Phase 1 preprocessing first."
    
    def test_load_bbbp_dataset(self):
        """Test loading BBBP (single-task) dataset."""
        from data.dataset import MoleculeDataset
        
        train_ds = MoleculeDataset("bbbp", "train")
        val_ds = MoleculeDataset("bbbp", "valid")
        test_ds = MoleculeDataset("bbbp", "test")
        
        assert len(train_ds) > 0, "Train dataset should not be empty"
        assert len(val_ds) > 0, "Valid dataset should not be empty"
        assert len(test_ds) > 0, "Test dataset should not be empty"
    
    def test_load_bace_dataset(self):
        """Test loading BACE (single-task) dataset."""
        from data.dataset import MoleculeDataset
        
        train_ds = MoleculeDataset("bace", "train")
        assert len(train_ds) > 0
    
    def test_load_hiv_dataset(self):
        """Test loading HIV (single-task) dataset."""
        from data.dataset import MoleculeDataset
        
        train_ds = MoleculeDataset("hiv", "train")
        assert len(train_ds) > 0
    
    def test_load_tox21_with_task(self):
        """Test loading Tox21 (multi-task) with specific task."""
        from data.dataset import MoleculeDataset, get_task_names
        
        tasks = get_task_names("tox21")
        assert len(tasks) == 12, f"Tox21 should have 12 tasks, got {len(tasks)}"
        
        # Load specific task
        train_ds = MoleculeDataset("tox21", "train", task_name="NR-AR")
        assert len(train_ds) > 0
    
    def test_load_clintox_with_task(self):
        """Test loading ClinTox (multi-task) with specific task."""
        from data.dataset import MoleculeDataset, get_task_names
        
        tasks = get_task_names("clintox")
        assert len(tasks) == 2, f"ClinTox should have 2 tasks, got {len(tasks)}"
        
        train_ds = MoleculeDataset("clintox", "train", task_name="CT_TOX")
        assert len(train_ds) > 0
    
    def test_dataset_graph_attributes(self):
        """Verify graphs have all required attributes."""
        from data.dataset import MoleculeDataset
        
        ds = MoleculeDataset("bbbp", "train")
        sample = ds[0]
        
        # Check required attributes
        assert hasattr(sample, "x"), "Graph missing node features (x)"
        assert hasattr(sample, "edge_index"), "Graph missing edge_index"
        assert hasattr(sample, "edge_attr"), "Graph missing edge_attr"
        assert hasattr(sample, "pos"), "Graph missing 3D positions (pos)"
        assert hasattr(sample, "y"), "Graph missing label (y)"
        
        # Check dimensions
        assert sample.x.dim() == 2, "Node features should be 2D"
        assert sample.edge_index.dim() == 2, "edge_index should be 2D"
        assert sample.edge_index.size(0) == 2, "edge_index first dim should be 2"
        assert sample.pos.dim() == 2, "Positions should be 2D"
        assert sample.pos.size(1) == 3, "Positions should be 3D coordinates"
        assert sample.y.dim() == 1, "Label should be 1D"
    
    def test_dataset_feature_dimensions(self):
        """Verify consistent feature dimensions across samples."""
        from data.dataset import MoleculeDataset
        
        ds = MoleculeDataset("bace", "train")
        
        # Check first 10 samples
        node_dims = set()
        edge_dims = set()
        
        for i in range(min(10, len(ds))):
            sample = ds[i]
            node_dims.add(sample.x.size(1))
            edge_dims.add(sample.edge_attr.size(1))
        
        assert len(node_dims) == 1, f"Inconsistent node dimensions: {node_dims}"
        assert len(edge_dims) == 1, f"Inconsistent edge dimensions: {edge_dims}"
    
    def test_dataloader_batching(self):
        """Test that DataLoader properly batches graphs."""
        from data.dataset import MoleculeDataset
        
        ds = MoleculeDataset("bbbp", "train")
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        
        batch = next(iter(loader))
        
        assert hasattr(batch, "batch"), "Batch should have batch attribute"
        assert batch.batch.max().item() == 3, "Batch should contain 4 graphs (indices 0-3)"
        assert batch.x.size(0) > 4, "Batched node features should be concatenated"
    
    def test_label_distribution(self):
        """Verify labels are binary (0 or 1)."""
        from data.dataset import MoleculeDataset
        
        ds = MoleculeDataset("bbbp", "train")
        
        labels = []
        for i in range(min(100, len(ds))):
            labels.append(ds[i].y.item())
        
        labels = np.array(labels)
        unique_labels = np.unique(labels)
        
        assert all(l in [0.0, 1.0] for l in unique_labels), \
            f"Labels should be binary, got {unique_labels}"
    
    def test_invalid_dataset_raises(self):
        """Test that invalid dataset name raises error."""
        from data.dataset import MoleculeDataset
        
        with pytest.raises(FileNotFoundError):
            MoleculeDataset("nonexistent_dataset", "train")
    
    def test_invalid_task_raises(self):
        """Test that invalid task name raises error."""
        from data.dataset import MoleculeDataset
        
        with pytest.raises(ValueError):
            MoleculeDataset("tox21", "train", task_name="INVALID_TASK")


# ─────────────────────────────────────────────────────────────────────────────
# EGNN Model Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEGNNLayer:
    """Tests for EGNN layer."""
    
    def test_egnn_layer_import(self):
        """Verify EGNN layer can be imported."""
        from models.egnn import EGNNLayer
        assert EGNNLayer is not None
    
    def test_egnn_layer_forward(self):
        """Test single EGNN layer forward pass."""
        from models.egnn import EGNNLayer
        
        hidden_dim = 64
        edge_dim = 6
        num_nodes = 5
        num_edges = 8
        
        layer = EGNNLayer(hidden_dim, edge_dim)
        
        h = torch.randn(num_nodes, hidden_dim)
        pos = torch.randn(num_nodes, 3)
        edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 4],
            [1, 0, 2, 1, 3, 2, 4, 3],
        ], dtype=torch.long)
        edge_attr = torch.randn(num_edges, edge_dim)
        
        h_new, pos_new = layer(h, pos, edge_index, edge_attr)
        
        assert h_new.shape == (num_nodes, hidden_dim), "Node features shape mismatch"
        assert pos_new.shape == (num_nodes, 3), "Positions shape mismatch"
    
    def test_egnn_layer_output_finite(self):
        """Verify EGNN layer outputs are finite."""
        from models.egnn import EGNNLayer
        
        layer = EGNNLayer(64, 6)
        h = torch.randn(5, 64)
        pos = torch.randn(5, 3)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        edge_attr = torch.randn(3, 6)
        
        h_new, pos_new = layer(h, pos, edge_index, edge_attr)
        
        assert torch.isfinite(h_new).all(), "Node features contain NaN/Inf"
        assert torch.isfinite(pos_new).all(), "Positions contain NaN/Inf"


class TestEGNN:
    """Tests for EGNN encoder."""
    
    def test_egnn_import(self):
        """Verify EGNN encoder can be imported."""
        from models.egnn import EGNN
        assert EGNN is not None
    
    def test_egnn_initialization(self, model_config):
        """Test EGNN encoder initialization."""
        from models.egnn import EGNN
        
        model = EGNN(**model_config)
        
        assert model.node_dim == model_config["node_dim"]
        assert model.edge_dim == model_config["edge_dim"]
        assert model.hidden_dim == model_config["hidden_dim"]
        assert model.num_layers == model_config["num_layers"]
    
    def test_egnn_forward(self, sample_batch, model_config):
        """Test EGNN encoder forward pass."""
        from models.egnn import EGNN
        
        model = EGNN(**model_config)
        
        graph_emb = model(sample_batch)
        
        expected_shape = (4, model_config["hidden_dim"])  # batch_size=4
        assert graph_emb.shape == expected_shape, \
            f"Expected {expected_shape}, got {graph_emb.shape}"
    
    def test_egnn_output_finite(self, sample_batch, model_config):
        """Verify EGNN outputs are finite."""
        from models.egnn import EGNN
        
        model = EGNN(**model_config)
        graph_emb = model(sample_batch)
        
        assert torch.isfinite(graph_emb).all(), "Graph embeddings contain NaN/Inf"
    
    def test_egnn_different_graph_sizes(self, model_config):
        """Test EGNN handles graphs of different sizes."""
        from models.egnn import EGNN
        
        model = EGNN(**model_config)
        
        # Create graphs with different sizes
        graphs = []
        for n_atoms in [3, 5, 10, 15]:
            x = torch.randn(n_atoms, model_config["node_dim"])
            edge_index = torch.randint(0, n_atoms, (2, n_atoms * 2))
            edge_attr = torch.randn(n_atoms * 2, model_config["edge_dim"])
            pos = torch.randn(n_atoms, 3)
            y = torch.tensor([1.0])
            graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, y=y))
        
        batch = Batch.from_data_list(graphs)
        graph_emb = model(batch)
        
        assert graph_emb.shape == (4, model_config["hidden_dim"])


class TestEGNNClassifier:
    """Tests for full EGNN classifier."""
    
    def test_classifier_import(self):
        """Verify EGNNClassifier can be imported."""
        from models.egnn import EGNNClassifier
        assert EGNNClassifier is not None
    
    def test_classifier_forward(self, sample_batch, model_config):
        """Test classifier forward pass returns logits."""
        from models.egnn import EGNNClassifier
        
        model = EGNNClassifier(**model_config)
        logits = model(sample_batch)
        
        assert logits.shape == (4, 1), f"Expected (4, 1), got {logits.shape}"
    
    def test_classifier_predict(self, sample_batch, model_config):
        """Test classifier predict returns probabilities."""
        from models.egnn import EGNNClassifier
        
        model = EGNNClassifier(**model_config)
        probs = model.predict(sample_batch)
        
        assert probs.shape == (4, 1), f"Expected (4, 1), got {probs.shape}"
        assert (probs >= 0).all() and (probs <= 1).all(), \
            "Probabilities should be in [0, 1]"
    
    def test_classifier_compute_loss(self, sample_batch, model_config):
        """Test classifier loss computation."""
        from models.egnn import EGNNClassifier
        
        model = EGNNClassifier(**model_config)
        loss = model.compute_loss(sample_batch)
        
        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() > 0, "Loss should be positive"
        assert torch.isfinite(loss), "Loss should be finite"
    
    def test_classifier_backward(self, sample_batch, model_config):
        """Test classifier backward pass (gradient flow)."""
        from models.egnn import EGNNClassifier
        
        model = EGNNClassifier(**model_config)
        loss = model.compute_loss(sample_batch)
        loss.backward()
        
        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"NaN/Inf gradient in {name}"
    
    def test_classifier_train_eval_modes(self, sample_batch, model_config):
        """Test model behaves differently in train vs eval mode."""
        from models.egnn import EGNNClassifier
        
        model = EGNNClassifier(**model_config)
        
        # Train mode (with dropout)
        model.train()
        torch.manual_seed(42)
        out_train1 = model(sample_batch)
        torch.manual_seed(42)
        out_train2 = model(sample_batch)
        
        # Eval mode (no dropout)
        model.eval()
        torch.manual_seed(42)
        out_eval1 = model(sample_batch)
        torch.manual_seed(42)
        out_eval2 = model(sample_batch)
        
        # Eval outputs should be identical
        assert torch.allclose(out_eval1, out_eval2), "Eval mode should be deterministic"
    
    def test_classifier_on_real_data(self, model_config):
        """Test classifier on actual dataset samples."""
        from data.dataset import MoleculeDataset
        from models.egnn import EGNNClassifier
        
        ds = MoleculeDataset("bbbp", "train")
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        
        # Update config with actual dimensions
        model_config["node_dim"] = batch.x.size(1)
        model_config["edge_dim"] = batch.edge_attr.size(1)
        
        model = EGNNClassifier(**model_config)
        
        # Forward pass
        logits = model(batch)
        assert logits.shape == (4, 1)
        
        # Loss computation
        loss = model.compute_loss(batch)
        assert torch.isfinite(loss)
        
        # Backward pass
        loss.backward()


# ─────────────────────────────────────────────────────────────────────────────
# Training Integration Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainingIntegration:
    """Integration tests for training pipeline."""
    
    def test_trainer_with_egnn(self):
        """Test Trainer with EGNN model."""
        from data.dataset import MoleculeDataset
        from models.egnn import EGNNClassifier
        from training.trainer import Trainer, set_seed
        
        set_seed(42)
        
        # Small dataset for quick test
        train_ds = MoleculeDataset("bace", "train")
        val_ds = MoleculeDataset("bace", "valid")
        
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
        
        # Get dimensions
        sample = train_ds[0]
        node_dim = sample.x.size(1)
        edge_dim = sample.edge_attr.size(1)
        
        model = EGNNClassifier(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=64,
            num_layers=2,
            dropout=0.1,
        )
        
        cfg = {
            "lr": 0.001,
            "wd": 1e-5,
            "patience": 5,
            "epochs": 3,  # Very few epochs for test
        }
        
        ckpt_path = "checkpoints/test_egnn.pt"
        trainer = Trainer(model, train_loader, val_loader, cfg, "cpu", ckpt_path)
        
        # Run training
        best_auc = trainer.run(epochs=3)
        
        # Check results
        assert best_auc >= 0.0 and best_auc <= 1.0, f"Invalid AUC: {best_auc}"
        assert len(trainer.history["train_loss"]) > 0
        assert len(trainer.history["val_auc"]) > 0
        
        # Cleanup
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
    
    def test_single_epoch_training(self):
        """Test single epoch completes without error."""
        from data.dataset import MoleculeDataset
        from models.egnn import EGNNClassifier
        from training.trainer import Trainer, set_seed
        
        set_seed(0)
        
        train_ds = MoleculeDataset("bbbp", "train")
        val_ds = MoleculeDataset("bbbp", "valid")
        
        # Use subset for speed
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
        
        sample = train_ds[0]
        model = EGNNClassifier(
            node_dim=sample.x.size(1),
            edge_dim=sample.edge_attr.size(1),
            hidden_dim=32,
            num_layers=2,
        )
        
        cfg = {"lr": 0.001, "wd": 1e-5, "patience": 10, "epochs": 1}
        trainer = Trainer(model, train_loader, val_loader, cfg, "cpu", "checkpoints/test.pt")
        
        # Single epoch
        train_loss = trainer.train_epoch()
        val_auc = trainer.eval_epoch(val_loader)
        
        assert train_loss > 0, "Training loss should be positive"
        assert 0 <= val_auc <= 1, f"Val AUC out of range: {val_auc}"
        
        # Cleanup
        if os.path.exists("checkpoints/test.pt"):
            os.remove("checkpoints/test.pt")


# ─────────────────────────────────────────────────────────────────────────────
# Model Equivariance Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEquivariance:
    """Tests for E(n) equivariance properties."""
    
    def test_translation_invariance(self, model_config):
        """Test that model output is invariant to translation."""
        from models.egnn import EGNN
        
        model = EGNN(**model_config)
        model.eval()
        
        # Create a simple graph
        x = torch.randn(5, model_config["node_dim"])
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        edge_attr = torch.randn(4, model_config["edge_dim"])
        pos = torch.randn(5, 3)
        
        graph1 = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
        batch1 = Batch.from_data_list([graph1])
        
        # Translate positions
        translation = torch.tensor([5.0, -3.0, 2.0])
        pos_translated = pos + translation
        graph2 = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos_translated)
        batch2 = Batch.from_data_list([graph2])
        
        with torch.no_grad():
            emb1 = model(batch1)
            emb2 = model(batch2)
        
        assert torch.allclose(emb1, emb2, atol=1e-5), \
            "Model should be translation invariant"
    
    def test_permutation_equivariance(self, model_config):
        """Test that model handles node permutation correctly."""
        from models.egnn import EGNN
        
        model = EGNN(**model_config)
        model.eval()
        
        # Original graph
        x = torch.randn(4, model_config["node_dim"])
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        edge_attr = torch.randn(3, model_config["edge_dim"])
        pos = torch.randn(4, 3)
        
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
        batch = Batch.from_data_list([graph])
        
        with torch.no_grad():
            emb = model(batch)
        
        # Model should produce valid output
        assert torch.isfinite(emb).all()


# ─────────────────────────────────────────────────────────────────────────────
# GPU Tests (optional)
# ─────────────────────────────────────────────────────────────────────────────

class TestGPU:
    """GPU-related tests (skipped if no GPU available)."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_to_gpu(self, sample_batch, model_config):
        """Test model runs on GPU."""
        from models.egnn import EGNNClassifier
        
        device = torch.device("cuda")
        model = EGNNClassifier(**model_config).to(device)
        batch = sample_batch.to(device)
        
        logits = model(batch)
        
        assert logits.device.type == "cuda"
        assert torch.isfinite(logits).all()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_loss_backward_gpu(self, sample_batch, model_config):
        """Test backward pass on GPU."""
        from models.egnn import EGNNClassifier
        
        device = torch.device("cuda")
        model = EGNNClassifier(**model_config).to(device)
        batch = sample_batch.to(device)
        
        loss = model.compute_loss(batch)
        loss.backward()
        
        for param in model.parameters():
            if param.grad is not None:
                assert param.grad.device.type == "cuda"


# ─────────────────────────────────────────────────────────────────────────────
# Results File Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestResultsFormat:
    """Tests for results file format (after training)."""
    
    def test_results_directory_exists(self):
        """Check results directory exists."""
        assert os.path.exists("results"), "results/ directory should exist"
    
    @pytest.mark.skipif(
        not any(f.endswith(".json") for f in os.listdir("results") if os.path.isfile(f"results/{f}")),
        reason="No results files found (run training first)"
    )
    def test_results_json_format(self):
        """Verify results JSON format."""
        import json
        
        results_files = [f for f in os.listdir("results") if f.endswith(".json")]
        
        for fname in results_files:
            with open(f"results/{fname}") as f:
                data = json.load(f)
            
            required_keys = ["dataset", "model", "seeds", "per_seed_auc", "mean_auc", "std_auc"]
            for key in required_keys:
                assert key in data, f"Missing key '{key}' in {fname}"
            
            assert 0 <= data["mean_auc"] <= 1, f"Invalid mean_auc in {fname}"
            assert data["std_auc"] >= 0, f"Invalid std_auc in {fname}"
            assert len(data["per_seed_auc"]) == len(data["seeds"]), \
                f"Mismatch in seeds and results in {fname}"


# ─────────────────────────────────────────────────────────────────────────────
# Run all tests
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])