"""
tests/test_phase1.py
Verify Phase 1 outputs.
"""
import pickle
import os
import pytest


class TestPhase1:
    def test_all_datasets_processed(self):
        """Check that all 5 datasets have processed graph files."""
        datasets = ["tox21", "clintox", "bbbp", "bace", "hiv"]
        for ds in datasets:
            path = f"data/processed/{ds}_graphs_3d.pkl"
            assert os.path.exists(path), f"Missing: {path}"

    def test_scaffold_stats_saved(self):
        """Check scaffold statistics files exist."""
        datasets = ["tox21", "clintox", "bbbp", "bace", "hiv"]
        for ds in datasets:
            path = f"data/processed/{ds}_scaffold_stats.pkl"
            assert os.path.exists(path), f"Missing: {path}"

    def test_graph_structure(self):
        """Verify graph objects have required attributes."""
        with open("data/processed/bbbp_graphs_3d.pkl", "rb") as f:
            graphs = pickle.load(f)
        
        sample = graphs["train"][0]
        assert hasattr(sample, "x"), "Missing node features"
        assert hasattr(sample, "edge_index"), "Missing edge indices"
        assert hasattr(sample, "edge_attr"), "Missing edge features"
        assert hasattr(sample, "pos"), "Missing 3D positions"
        assert hasattr(sample, "y"), "Missing label"
        assert sample.pos.shape[1] == 3, "Positions must be 3D"

    def test_scaffold_leak_acceptable(self):
        """Verify train-test scaffold overlap < 5%."""
        datasets = ["tox21", "clintox", "bbbp", "bace", "hiv"]
        for ds in datasets:
            with open(f"data/processed/{ds}_scaffold_stats.pkl", "rb") as f:
                stats = pickle.load(f)
            
            leak = stats["leak_pct"]["train_test"]
            assert leak < 5.0, f"{ds}: train-test leak {leak:.2f}% > 5%"

    def test_feature_dimensions(self):
        """Check feature dimensionality consistency."""
        from data.featurizer import Molecule3DFeaturizer
        
        feat = Molecule3DFeaturizer()
        node_dim, edge_dim = feat.get_feature_dims()
        
        with open("data/processed/bace_graphs_3d.pkl", "rb") as f:
            graphs = pickle.load(f)
        
        sample = graphs["train"][0]
        assert sample.x.shape[1] == node_dim
        assert sample.edge_attr.shape[1] == edge_dim