"""
tests/test_phase0.py
Sanity-checks for the Phase 0 infrastructure.
Run with: python -m pytest tests/test_phase0.py -v
"""
import numpy as np
import torch
import pytest
import sys, os
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.metrics import compute_roc_auc
from training.trainer import set_seed, EarlyStopping


# ─────────────────────────────────────────────────────────────────────────────
# compute_roc_auc tests
# ─────────────────────────────────────────────────────────────────────────────

class TestROCAUC:
    def test_perfect_predictions_returns_1(self):
        y_true = torch.tensor([0., 0., 1., 1.])
        y_pred = torch.tensor([0.1, 0.2, 0.8, 0.9])
        assert compute_roc_auc(y_true, y_pred) == pytest.approx(1.0)

    def test_random_predictions_near_05(self):
        rng = np.random.default_rng(42)
        y_true = torch.from_numpy(rng.integers(0, 2, 1000).astype(np.float32))
        y_pred = torch.from_numpy(rng.uniform(0, 1, 1000).astype(np.float32))
        auc = compute_roc_auc(y_true, y_pred)
        assert 0.4 < auc < 0.6, f"Random AUC should be ~0.5, got {auc:.4f}"

    def test_worst_predictions_returns_0(self):
        # Inverted perfect prediction → AUC = 0
        y_true = torch.tensor([0., 0., 1., 1.])
        y_pred = torch.tensor([0.9, 0.8, 0.1, 0.2])
        assert compute_roc_auc(y_true, y_pred) == pytest.approx(0.0)

    def test_missing_labels_masked(self):
        # Label -1 should be ignored
        y_true = torch.tensor([-1., 0., 1., 1.])
        y_pred = torch.tensor([0.99, 0.1, 0.8, 0.9])
        auc = compute_roc_auc(y_true, y_pred)
        assert auc == pytest.approx(1.0), "Missing label (-1) should be masked out"

    def test_all_same_label_returns_nan(self):
        y_true = torch.tensor([1., 1., 1., 1.])
        y_pred = torch.tensor([0.5, 0.6, 0.7, 0.8])
        assert np.isnan(compute_roc_auc(y_true, y_pred))

    def test_multitask_averages_tasks(self):
        # Task 0: perfect, Task 1: perfect → avg = 1.0
        y_true = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        y_pred = torch.tensor([[0.1, 0.1], [0.2, 0.9], [0.8, 0.2], [0.9, 0.8]])
        auc = compute_roc_auc(y_true, y_pred)
        assert auc == pytest.approx(1.0)

    def test_multitask_skips_invalid_task(self):
        # Task 1 all-same label → skip; Task 0 perfect → auc = 1.0
        y_true = torch.tensor([[0., 1.], [1., 1.], [1., 1.]])
        y_pred = torch.tensor([[0.1, 0.5], [0.9, 0.5], [0.9, 0.5]])
        auc = compute_roc_auc(y_true, y_pred)
        assert auc == pytest.approx(1.0)

    def test_accepts_numpy_arrays(self):
        y_true = np.array([0, 0, 1, 1], dtype=np.float32)
        y_pred = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
        assert compute_roc_auc(y_true, y_pred) == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# set_seed tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSetSeed:
    def test_same_seed_same_tensor(self):
        set_seed(42)
        a = torch.randn(10)
        set_seed(42)
        b = torch.randn(10)
        assert torch.allclose(a, b), "Same seed should produce identical tensors"

    def test_different_seed_different_tensor(self):
        set_seed(0)
        a = torch.randn(10)
        set_seed(1)
        b = torch.randn(10)
        assert not torch.allclose(a, b), "Different seeds should produce different tensors"


# ─────────────────────────────────────────────────────────────────────────────
# EarlyStopping tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEarlyStopping:
    def _dummy_model(self):
        """Minimal model with state_dict for checkpoint testing."""
        import torch.nn as nn
        return nn.Linear(2, 1)

    def test_stops_after_patience(self, tmp_path):
        es = EarlyStopping(patience=3, min_delta=0.0)
        model = self._dummy_model()
        ckpt = str(tmp_path / "ckpt.pt")

        es.step(0.80, model, ckpt)  # new best
        es.step(0.79, model, ckpt)  # no improvement — counter=1
        es.step(0.79, model, ckpt)  # counter=2
        es.step(0.79, model, ckpt)  # counter=3 → stop
        assert es.stop

    def test_does_not_stop_on_improvement(self, tmp_path):
        es = EarlyStopping(patience=2, min_delta=0.0)
        model = self._dummy_model()
        ckpt = str(tmp_path / "ckpt.pt")

        for score in [0.80, 0.81, 0.82, 0.83]:
            es.step(score, model, ckpt)
        assert not es.stop

    def test_best_score_tracked(self, tmp_path):
        es = EarlyStopping(patience=5)
        model = self._dummy_model()
        ckpt = str(tmp_path / "ckpt.pt")

        for score in [0.75, 0.82, 0.79, 0.85, 0.83]:
            es.step(score, model, ckpt)
        assert es.best_score == pytest.approx(0.85)

    def test_checkpoint_saved(self, tmp_path):
        es = EarlyStopping(patience=5)
        model = self._dummy_model()
        ckpt = str(tmp_path / "ckpt.pt")

        es.step(0.80, model, ckpt)
        assert os.path.exists(ckpt), "Checkpoint file should be saved on improvement"