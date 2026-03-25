"""
tests/test_phase3.py
Tests for Phase 3: Multi-task architecture and gradient analysis.

Run with: python -m pytest tests/test_phase3.py -v
"""
import os
import sys
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
def sample_multitask_batch():
    """Create batch with task IDs."""
    graphs = []
    for i in range(8):
        x = torch.randn(5, 129)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        edge_attr = torch.randn(4, 6)
        pos = torch.randn(5, 3)
        y = torch.tensor([float(i % 2)])
        task_id = torch.tensor([i % 4])  # 4 different tasks
        
        graphs.append(Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr,
            pos=pos, y=y, task_id=task_id
        ))
    
    return Batch.from_data_list(graphs)


@pytest.fixture
def model_config():
    return {
        "node_dim": 129,
        "edge_dim": 6,
        "hidden_dim": 64,
        "num_layers": 2,
        "num_tasks": 4,
        "task_dim": 32,
        "dropout": 0.1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Task Dataset Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiTaskDataset:
    """Tests for multi-task dataset."""
    
    def test_import(self):
        from data.multitask_dataset import (
            MultiTaskDataset,
            get_all_task_names,
            get_num_tasks,
        )
        assert MultiTaskDataset is not None
        assert get_all_task_names is not None
        assert get_num_tasks is not None
    
    def test_get_all_task_names(self):
        from data.multitask_dataset import get_all_task_names
        
        names = get_all_task_names()
        assert len(names) > 0
        assert all(isinstance(n, str) for n in names)
        # Should have format "dataset_task"
        assert all("_" in n for n in names)
    
    def test_get_num_tasks(self):
        from data.multitask_dataset import get_num_tasks
        
        n = get_num_tasks()
        assert n > 0
        assert isinstance(n, int)
        # Should be 17 tasks total (1+1+1+2+12)
        assert n == 17
    
    def test_load_single_dataset(self):
        from data.multitask_dataset import MultiTaskDataset
        
        ds = MultiTaskDataset(["bbbp"], "train")
        assert len(ds) > 0
        print(f"  BBBP train samples: {len(ds)}")
    
    def test_load_multiple_datasets(self):
        from data.multitask_dataset import MultiTaskDataset
        
        ds = MultiTaskDataset(["bbbp", "bace"], "train")
        assert len(ds) > 0
        assert len(ds.task_to_samples) >= 2
        print(f"  Multi-dataset samples: {len(ds)}, tasks: {len(ds.task_to_samples)}")
    
    def test_sample_has_task_id(self):
        from data.multitask_dataset import MultiTaskDataset
        
        ds = MultiTaskDataset(["bbbp"], "train")
        sample = ds[0]
        
        assert hasattr(sample, "task_id")
        assert sample.task_id.dim() == 1
        assert sample.task_id.size(0) == 1
    
    def test_task_counts(self):
        from data.multitask_dataset import MultiTaskDataset
        
        ds = MultiTaskDataset(["bbbp", "bace"], "train")
        counts = ds.get_task_counts()
        
        assert isinstance(counts, dict)
        assert all(c > 0 for c in counts.values())
        print(f"  Task counts: {counts}")
    
    def test_get_task_name(self):
        from data.multitask_dataset import MultiTaskDataset
        
        ds = MultiTaskDataset(["bbbp"], "train")
        task_id = list(ds.task_to_samples.keys())[0]
        task_name = ds.get_task_name(task_id)
        
        assert isinstance(task_name, str)
        assert "bbbp" in task_name.lower()
    
    def test_all_samples_have_labels(self):
        from data.multitask_dataset import MultiTaskDataset
        
        ds = MultiTaskDataset(["bace"], "train")
        
        for i in range(min(10, len(ds))):
            sample = ds[i]
            assert hasattr(sample, "y")
            assert sample.y[0] in [0.0, 1.0]


# ─────────────────────────────────────────────────────────────────────────────
# Task-Conditioned EGNN Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTaskConditionedEGNN:
    """Tests for task-conditioned model."""
    
    def test_import(self):
        from models.task_conditioned_egnn import (
            TaskConditionedEGNNLayer,
            TaskConditionedEGNN,
            MultiTaskClassifier,
            HardSharingClassifier,
        )
        assert TaskConditionedEGNNLayer is not None
        assert TaskConditionedEGNN is not None
        assert MultiTaskClassifier is not None
        assert HardSharingClassifier is not None
    
    def test_layer_forward(self):
        from models.task_conditioned_egnn import TaskConditionedEGNNLayer
        
        layer = TaskConditionedEGNNLayer(
            hidden_dim=64,
            edge_dim=6,
            task_dim=32,
        )
        
        h = torch.randn(5, 64)
        pos = torch.randn(5, 3)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        edge_attr = torch.randn(3, 6)
        task_emb = torch.randn(5, 32)
        
        h_new, pos_new = layer(h, pos, edge_index, edge_attr, task_emb)
        
        assert h_new.shape == (5, 64)
        assert pos_new.shape == (5, 3)
        assert torch.isfinite(h_new).all()
        assert torch.isfinite(pos_new).all()
    
    def test_encoder_forward(self, sample_multitask_batch, model_config):
        from models.task_conditioned_egnn import TaskConditionedEGNN
        
        encoder = TaskConditionedEGNN(**model_config)
        graph_emb = encoder(sample_multitask_batch)
        
        assert graph_emb.shape == (8, model_config["hidden_dim"])
        assert torch.isfinite(graph_emb).all()
    
    def test_classifier_forward(self, sample_multitask_batch, model_config):
        from models.task_conditioned_egnn import MultiTaskClassifier
        
        model = MultiTaskClassifier(**model_config)
        logits = model(sample_multitask_batch)
        
        assert logits.shape == (8, 1)
        assert torch.isfinite(logits).all()
    
    def test_classifier_predict(self, sample_multitask_batch, model_config):
        from models.task_conditioned_egnn import MultiTaskClassifier
        
        model = MultiTaskClassifier(**model_config)
        probs = model.predict(sample_multitask_batch)
        
        assert probs.shape == (8, 1)
        assert (probs >= 0).all() and (probs <= 1).all()
    
    def test_classifier_compute_loss(self, sample_multitask_batch, model_config):
        from models.task_conditioned_egnn import MultiTaskClassifier
        
        model = MultiTaskClassifier(**model_config)
        loss = model.compute_loss(sample_multitask_batch)
        
        assert loss.dim() == 0
        assert loss.item() > 0
        assert torch.isfinite(loss)
    
    def test_classifier_per_task_losses(self, sample_multitask_batch, model_config):
        from models.task_conditioned_egnn import MultiTaskClassifier
        
        model = MultiTaskClassifier(**model_config)
        losses = model.compute_per_task_losses(sample_multitask_batch)
        
        assert isinstance(losses, dict)
        assert len(losses) > 0
        assert all(torch.isfinite(loss) for loss in losses.values())
    
    def test_classifier_backward(self, sample_multitask_batch, model_config):
        from models.task_conditioned_egnn import MultiTaskClassifier
        
        model = MultiTaskClassifier(**model_config)
        loss = model.compute_loss(sample_multitask_batch)
        loss.backward()
        
        # Check gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"NaN/Inf in {name}"
    
    def test_task_embeddings_learned(self, model_config):
        from models.task_conditioned_egnn import TaskConditionedEGNN
        
        encoder = TaskConditionedEGNN(**model_config)
        
        # Check task embeddings exist
        assert hasattr(encoder, "task_embeddings")
        assert encoder.task_embeddings.num_embeddings == model_config["num_tasks"]
        assert encoder.task_embeddings.embedding_dim == model_config["task_dim"]
        
        # Check they have gradients
        emb = encoder.task_embeddings(torch.tensor([0]))
        assert emb.requires_grad


class TestHardSharingClassifier:
    """Tests for hard sharing baseline."""
    
    def test_import(self):
        from models.task_conditioned_egnn import HardSharingClassifier
        assert HardSharingClassifier is not None
    
    def test_forward(self, sample_multitask_batch, model_config):
        from models.task_conditioned_egnn import HardSharingClassifier
        
        model = HardSharingClassifier(
            node_dim=model_config["node_dim"],
            edge_dim=model_config["edge_dim"],
            hidden_dim=model_config["hidden_dim"],
            num_layers=model_config["num_layers"],
            num_tasks=model_config["num_tasks"],
            dropout=model_config["dropout"],
        )
        
        logits = model(sample_multitask_batch)
        assert logits.shape == (8, 1)
    
    def test_compute_loss(self, sample_multitask_batch, model_config):
        from models.task_conditioned_egnn import HardSharingClassifier
        
        model = HardSharingClassifier(
            node_dim=model_config["node_dim"],
            edge_dim=model_config["edge_dim"],
            hidden_dim=model_config["hidden_dim"],
            num_layers=model_config["num_layers"],
            num_tasks=model_config["num_tasks"],
        )
        
        loss = model.compute_loss(sample_multitask_batch)
        assert torch.isfinite(loss)
    
    def test_per_task_losses(self, sample_multitask_batch, model_config):
        from models.task_conditioned_egnn import HardSharingClassifier
        
        model = HardSharingClassifier(
            node_dim=model_config["node_dim"],
            edge_dim=model_config["edge_dim"],
            hidden_dim=model_config["hidden_dim"],
            num_layers=model_config["num_layers"],
            num_tasks=model_config["num_tasks"],
        )
        
        losses = model.compute_per_task_losses(sample_multitask_batch)
        assert isinstance(losses, dict)
        assert len(losses) > 0


# ─────────────────────────────────────────────────────────────────────────────
# PCGrad Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPCGrad:
    """Tests for PCGrad optimizer."""
    
    def test_import(self):
        from training.pcgrad import (
            project_conflicting_gradients,
            PCGradOptimizer,
        )
        assert project_conflicting_gradients is not None
        assert PCGradOptimizer is not None
    
    def test_project_single_gradient(self):
        from training.pcgrad import project_conflicting_gradients
        
        grads = [torch.randn(100)]
        result = project_conflicting_gradients(grads)
        
        assert result.shape == (100,)
        assert torch.allclose(result, grads[0])
    
    def test_project_aligned_gradients(self):
        from training.pcgrad import project_conflicting_gradients
        
        # Aligned gradients (same direction)
        g1 = torch.randn(100)
        g2 = g1 * 0.8
        
        grads = [g1, g2]
        result = project_conflicting_gradients(grads, reduction="mean")
        
        assert result.shape == (100,)
        assert torch.isfinite(result).all()
    
    def test_project_conflicting_gradients(self):
        from training.pcgrad import project_conflicting_gradients
        
        # Conflicting gradients (opposite directions)
        g1 = torch.randn(100)
        g2 = -g1
        
        grads = [g1, g2]
        result = project_conflicting_gradients(grads, reduction="mean")
        
        assert result.shape == (100,)
        assert torch.isfinite(result).all()
        
        # Result should be different from mean
        naive_mean = (g1 + g2) / 2
        assert not torch.allclose(result, naive_mean)
    
    def test_pcgrad_optimizer_initialization(self):
        from training.pcgrad import PCGradOptimizer
        from torch.optim import Adam
        
        model = torch.nn.Linear(10, 1)
        base_opt = Adam(model.parameters())
        
        pcgrad_opt = PCGradOptimizer(base_opt)
        
        assert pcgrad_opt.optimizer is base_opt
        assert hasattr(pcgrad_opt, "zero_grad")
        assert hasattr(pcgrad_opt, "step")
        assert hasattr(pcgrad_opt, "backward")
    
    def test_pcgrad_backward(self):
        from training.pcgrad import PCGradOptimizer
        from torch.optim import Adam
        
        model = torch.nn.Linear(10, 1)
        optimizer = PCGradOptimizer(Adam(model.parameters()))
        
        # Create dummy losses
        x = torch.randn(5, 10)
        
        loss1 = model(x).mean()
        loss2 = -model(x).mean()  # Conflicting
        
        optimizer.zero_grad()
        optimizer.backward([loss1, loss2])
        
        # Check gradients were set
        for param in model.parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()
    
    def test_pcgrad_step(self):
        from training.pcgrad import PCGradOptimizer
        from torch.optim import Adam
        
        model = torch.nn.Linear(10, 1)
        optimizer = PCGradOptimizer(Adam(model.parameters()))
        
        # Get initial parameters
        initial_params = [p.clone() for p in model.parameters()]
        
        # Backward and step
        x = torch.randn(5, 10)
        losses = [model(x).mean(), -model(x).mean()]
        
        optimizer.zero_grad()
        optimizer.backward(losses)
        optimizer.step()
        
        # Check parameters changed
        for p_init, p_new in zip(initial_params, model.parameters()):
            assert not torch.allclose(p_init, p_new)


# ─────────────────────────────────────────────────────────────────────────────
# Gradient Analysis Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGradientAnalysis:
    """Tests for gradient analysis utilities."""
    
    def test_import(self):
        from evaluation.gradient_analysis import (
            compute_gradient_cosine_similarity,
            compute_gradient_similarity_matrix,
            compute_gradient_statistics,
        )
        assert compute_gradient_cosine_similarity is not None
        assert compute_gradient_similarity_matrix is not None
        assert compute_gradient_statistics is not None
    
    def test_cosine_similarity(self):
        from evaluation.gradient_analysis import compute_gradient_cosine_similarity
        
        # Same direction
        g1 = torch.randn(100)
        g2 = g1 * 2
        sim = compute_gradient_cosine_similarity(g1, g2)
        assert abs(sim - 1.0) < 1e-5
        
        # Opposite direction
        g3 = -g1
        sim = compute_gradient_cosine_similarity(g1, g3)
        assert abs(sim - (-1.0)) < 1e-5
        
        # Orthogonal (approximately)
        g4 = torch.randn(100)
        sim = compute_gradient_cosine_similarity(g1, g4)
        assert -1 <= sim <= 1
    
    def test_similarity_matrix(self):
        from evaluation.gradient_analysis import compute_gradient_similarity_matrix
        
        grads = {
            0: torch.randn(100),
            1: torch.randn(100),
            2: torch.randn(100),
        }
        
        matrix, labels = compute_gradient_similarity_matrix(grads)
        
        assert matrix.shape == (3, 3)
        assert len(labels) == 3
        
        # Diagonal should be 1
        assert all(abs(matrix[i, i] - 1.0) < 1e-5 for i in range(3))
        
        # Symmetric
        for i in range(3):
            for j in range(3):
                assert abs(matrix[i, j] - matrix[j, i]) < 1e-5
    
    def test_gradient_statistics(self):
        from evaluation.gradient_analysis import compute_gradient_statistics
        
        grads = {
            0: torch.randn(100),
            1: torch.randn(100),
            2: -torch.randn(100),  # Include some conflicts
        }
        
        stats = compute_gradient_statistics(grads)
        
        assert "mean_similarity" in stats
        assert "min_similarity" in stats
        assert "max_similarity" in stats
        assert "conflict_ratio" in stats
        assert "mean_magnitude" in stats
        
        assert -1 <= stats["mean_similarity"] <= 1
        assert 0 <= stats["conflict_ratio"] <= 1
        assert stats["mean_magnitude"] > 0
    
    def test_identify_conflicts(self):
        from evaluation.gradient_analysis import (
            compute_gradient_similarity_matrix,
            identify_conflicting_pairs,
        )
        
        # Create deliberately conflicting gradients
        g1 = torch.ones(100)
        g2 = -torch.ones(100)
        g3 = torch.ones(100) * 0.5
        
        grads = {0: g1, 1: g2, 2: g3}
        matrix, labels = compute_gradient_similarity_matrix(
            grads, task_names=["Task0", "Task1", "Task2"]
        )
        
        conflicts = identify_conflicting_pairs(matrix, labels, threshold=0.0)
        
        # Should find conflict between Task0 and Task1
        assert len(conflicts) > 0
        assert any("Task0" in c[0] or "Task0" in c[1] for c in conflicts)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Task Trainer Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiTaskTrainer:
    """Tests for multi-task trainer."""
    
    def test_import(self):
        from training.multitask_trainer import MultiTaskTrainer
        assert MultiTaskTrainer is not None
    
    def test_trainer_initialization(self):
        from training.multitask_trainer import MultiTaskTrainer
        from models.task_conditioned_egnn import MultiTaskClassifier
        from data.multitask_dataset import MultiTaskDataset
        
        # Create minimal dataset
        train_ds = MultiTaskDataset(["bbbp"], "train")
        val_ds = MultiTaskDataset(["bbbp"], "valid")
        
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        
        # Create per-task val loaders
        val_loaders = {}
        for task_id in val_ds.task_to_samples.keys():
            indices = val_ds.task_to_samples[task_id]
            graphs = [val_ds.samples[i][0] for i in indices[:50]]
            val_loaders[task_id] = DataLoader(graphs, batch_size=32)
        
        # Create model
        sample = train_ds[0]
        model = MultiTaskClassifier(
            node_dim=sample.x.size(1),
            edge_dim=sample.edge_attr.size(1),
            hidden_dim=32,
            num_layers=2,
            num_tasks=4,
            task_dim=16,
        )
        
        cfg = {"lr": 0.001, "wd": 1e-5, "patience": 5, "epochs": 2}
        
        trainer = MultiTaskTrainer(
            model=model,
            train_loader=train_loader,
            val_loaders=val_loaders,
            cfg=cfg,
            device="cpu",
            ckpt_path="checkpoints/test_multitask.pt",
            use_pcgrad=False,
        )
        
        assert trainer.model is not None
        assert trainer.optimizer is not None
    
    def test_trainer_single_epoch(self):
        from training.multitask_trainer import MultiTaskTrainer
        from models.task_conditioned_egnn import HardSharingClassifier
        from data.multitask_dataset import MultiTaskDataset
        
        train_ds = MultiTaskDataset(["bace"], "train")
        val_ds = MultiTaskDataset(["bace"], "valid")
        
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        
        val_loaders = {}
        for task_id in val_ds.task_to_samples.keys():
            indices = val_ds.task_to_samples[task_id][:50]
            graphs = [val_ds.samples[i][0] for i in indices]
            val_loaders[task_id] = DataLoader(graphs, batch_size=32)
        
        sample = train_ds[0]
        model = HardSharingClassifier(
            node_dim=sample.x.size(1),
            edge_dim=sample.edge_attr.size(1),
            hidden_dim=32,
            num_layers=2,
            num_tasks=4,
        )
        
        cfg = {"lr": 0.001, "wd": 1e-5, "patience": 5}
        
        trainer = MultiTaskTrainer(
            model=model,
            train_loader=train_loader,
            val_loaders=val_loaders,
            cfg=cfg,
            device="cpu",
            ckpt_path="checkpoints/test_multitask_epoch.pt",
        )
        
        # Single epoch
        loss = trainer.train_epoch()
        assert loss > 0
        
        # Eval
        task_aucs = trainer.eval_all_tasks()
        assert len(task_aucs) > 0
        assert all(0 <= auc <= 1 for auc in task_aucs.values() if not np.isnan(auc))
        
        # Cleanup
        if os.path.exists("checkpoints/test_multitask_epoch.pt"):
            os.remove("checkpoints/test_multitask_epoch.pt")


# ─────────────────────────────────────────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:
    """End-to-end integration tests."""
    
    def test_full_pipeline_task_conditioned(self):
        """Test complete training pipeline."""
        from data.multitask_dataset import MultiTaskDataset
        from models.task_conditioned_egnn import MultiTaskClassifier
        from training.multitask_trainer import MultiTaskTrainer
        from training.trainer import set_seed
        
        set_seed(42)
        
        # Small subset for speed
        train_ds = MultiTaskDataset(["bbbp"], "train")
        val_ds = MultiTaskDataset(["bbbp"], "valid")
        
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        
        val_loaders = {}
        for task_id in val_ds.task_to_samples.keys():
            indices = val_ds.task_to_samples[task_id][:50]
            graphs = [val_ds.samples[i][0] for i in indices]
            val_loaders[task_id] = DataLoader(graphs, batch_size=32)
        
        sample = train_ds[0]
        model = MultiTaskClassifier(
            node_dim=sample.x.size(1),
            edge_dim=sample.edge_attr.size(1),
            hidden_dim=32,
            num_layers=2,
            num_tasks=4,
            task_dim=16,
        )
        
        cfg = {"lr": 0.001, "wd": 1e-5, "patience": 3, "epochs": 3}
        
        trainer = MultiTaskTrainer(
            model=model,
            train_loader=train_loader,
            val_loaders=val_loaders,
            cfg=cfg,
            device="cpu",
            ckpt_path="checkpoints/test_integration.pt",
        )
        
        results = trainer.run(epochs=3)
        
        assert "best_avg_auc" in results
        assert 0 <= results["best_avg_auc"] <= 1
        
        # Cleanup
        if os.path.exists("checkpoints/test_integration.pt"):
            os.remove("checkpoints/test_integration.pt")
    
    def test_full_pipeline_with_pcgrad(self):
        """Test pipeline with PCGrad."""
        from data.multitask_dataset import MultiTaskDataset
        from models.task_conditioned_egnn import MultiTaskClassifier
        from training.multitask_trainer import MultiTaskTrainer
        from training.trainer import set_seed
        
        set_seed(0)
        
        train_ds = MultiTaskDataset(["bace"], "train")
        val_ds = MultiTaskDataset(["bace"], "valid")
        
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        
        val_loaders = {}
        for task_id in val_ds.task_to_samples.keys():
            indices = val_ds.task_to_samples[task_id][:50]
            graphs = [val_ds.samples[i][0] for i in indices]
            val_loaders[task_id] = DataLoader(graphs, batch_size=32)
        
        sample = train_ds[0]
        model = MultiTaskClassifier(
            node_dim=sample.x.size(1),
            edge_dim=sample.edge_attr.size(1),
            hidden_dim=32,
            num_layers=2,
            num_tasks=4,
            task_dim=16,
        )
        
        cfg = {"lr": 0.001, "wd": 1e-5, "patience": 3, "epochs": 3}
        
        trainer = MultiTaskTrainer(
            model=model,
            train_loader=train_loader,
            val_loaders=val_loaders,
            cfg=cfg,
            device="cpu",
            ckpt_path="checkpoints/test_pcgrad.pt",
            use_pcgrad=True,
        )
        
        results = trainer.run(epochs=3)
        
        assert "best_avg_auc" in results
        
        # Cleanup
        if os.path.exists("checkpoints/test_pcgrad.pt"):
            os.remove("checkpoints/test_pcgrad.pt")


# ─────────────────────────────────────────────────────────────────────────────
# Run all tests
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])