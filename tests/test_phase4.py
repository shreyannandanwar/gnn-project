"""
tests/test_phase4.py
Tests for Phase 4: Ablations, visualizations, and paper-ready analysis.

Run with: python -m pytest tests/test_phase4.py -v
"""
import os
import sys
import pytest
import torch
import numpy as np
import json

sys.path.insert(0, ".")


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_ablation_results():
    """Mock ablation results."""
    return {
        "16": {"value": 16, "mean_auc": 0.72, "std_auc": 0.01, "num_params": 50000, "avg_training_time": 100},
        "32": {"value": 32, "mean_auc": 0.74, "std_auc": 0.02, "num_params": 60000, "avg_training_time": 120},
        "64": {"value": 64, "mean_auc": 0.76, "std_auc": 0.01, "num_params": 80000, "avg_training_time": 150},
        "128": {"value": 128, "mean_auc": 0.75, "std_auc": 0.02, "num_params": 120000, "avg_training_time": 200},
    }


@pytest.fixture
def sample_model_results():
    """Mock per-task model results."""
    return {
        "hard_sharing": {"task_a": 0.72, "task_b": 0.75, "task_c": 0.70},
        "task_conditioned": {"task_a": 0.74, "task_b": 0.76, "task_c": 0.73},
        "task_cond_pcgrad": {"task_a": 0.75, "task_b": 0.77, "task_c": 0.74},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualization Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestVisualizations:
    """Tests for visualization functions."""

    def test_import(self):
        from evaluation.visualizations import (
            plot_ablation_curve,
            plot_improvement_delta,
            plot_per_task_comparison,
            plot_strategy_comparison,
            plot_task_embedding_heatmap,
            plot_task_embedding_tsne,
            plot_training_curves,
        )

        assert plot_ablation_curve is not None
        assert plot_strategy_comparison is not None
        assert plot_per_task_comparison is not None
        assert plot_training_curves is not None
        assert plot_improvement_delta is not None
        assert plot_task_embedding_tsne is not None
        assert plot_task_embedding_heatmap is not None

    def test_ablation_curve(self, tmp_path):
        from evaluation.visualizations import plot_ablation_curve

        values = [16, 32, 64, 128]
        means = [0.72, 0.74, 0.76, 0.75]
        stds = [0.01, 0.02, 0.01, 0.02]

        save_path = str(tmp_path / "ablation.png")
        fig = plot_ablation_curve(
            values, means, stds,
            xlabel="Task Dim",
            title="Test",
            save_path=save_path,
        )

        assert fig is not None
        assert os.path.exists(save_path)
        plt_close(fig)

    def test_strategy_comparison(self, tmp_path):
        from evaluation.visualizations import plot_strategy_comparison

        save_path = str(tmp_path / "strategy.png")
        fig = plot_strategy_comparison(
            ["Hard Sharing", "Task-Cond", "Task-Cond+PCGrad"],
            [0.74, 0.76, 0.77],
            [0.01, 0.01, 0.01],
            save_path=save_path,
        )

        assert fig is not None
        assert os.path.exists(save_path)
        plt_close(fig)

    def test_per_task_comparison(self, tmp_path, sample_model_results):
        from evaluation.visualizations import plot_per_task_comparison

        save_path = str(tmp_path / "per_task.png")
        fig = plot_per_task_comparison(
            ["task_a", "task_b", "task_c"],
            sample_model_results,
            save_path=save_path,
        )

        assert fig is not None
        assert os.path.exists(save_path)
        plt_close(fig)

    def test_improvement_delta(self, tmp_path):
        from evaluation.visualizations import plot_improvement_delta

        save_path = str(tmp_path / "delta.png")
        fig = plot_improvement_delta(
            ["task_a", "task_b", "task_c"],
            {"task_a": 0.72, "task_b": 0.75, "task_c": 0.70},
            {"task_a": 0.75, "task_b": 0.74, "task_c": 0.74},
            save_path=save_path,
        )

        assert fig is not None
        assert os.path.exists(save_path)
        plt_close(fig)

    def test_training_curves(self, tmp_path):
        from evaluation.visualizations import plot_training_curves

        histories = {
            "model_a": {
                "train_loss": [0.7, 0.6, 0.5, 0.4],
                "val_auc_avg": [0.6, 0.65, 0.7, 0.72],
            },
            "model_b": {
                "train_loss": [0.65, 0.55, 0.45, 0.35],
                "val_auc_avg": [0.62, 0.68, 0.73, 0.75],
            },
        }

        save_path = str(tmp_path / "curves.png")
        fig = plot_training_curves(histories, save_path=save_path)

        assert fig is not None
        assert os.path.exists(save_path)
        plt_close(fig)


class TestTaskEmbeddingVisualization:
    """Tests for task embedding visualizations."""

    def test_tsne_plot(self, tmp_path):
        from evaluation.visualizations import plot_task_embedding_tsne

        # Create fake embeddings
        embeddings = np.random.randn(10, 64)
        task_names = [
            "bbbp_p_np", "bace_Class", "hiv_HIV_active",
            "clintox_FDA", "clintox_TOX",
            "tox21_NR-AR", "tox21_NR-ER", "tox21_NR-AhR",
            "tox21_SR-ARE", "tox21_SR-MMP",
        ]

        save_path = str(tmp_path / "tsne.png")
        fig = plot_task_embedding_tsne(
            embeddings, task_names,
            save_path=save_path,
            perplexity=3.0,
        )

        assert fig is not None
        assert os.path.exists(save_path)
        plt_close(fig)

    def test_heatmap(self, tmp_path):
        from evaluation.visualizations import plot_task_embedding_heatmap

        embeddings = np.random.randn(5, 32)
        task_names = ["bbbp_p_np", "bace_Class", "hiv_active", "ct_FDA", "ct_TOX"]

        save_path = str(tmp_path / "heatmap.png")
        fig = plot_task_embedding_heatmap(
            embeddings, task_names,
            save_path=save_path,
        )

        assert fig is not None
        assert os.path.exists(save_path)
        plt_close(fig)

    def test_extract_task_embeddings(self):
        from evaluation.visualizations import extract_task_embeddings
        from models.task_conditioned_egnn import MultiTaskClassifier
        from data.multitask_dataset import get_num_tasks

        num_tasks = get_num_tasks()
        model = MultiTaskClassifier(
            node_dim=129, edge_dim=6,
            hidden_dim=64, num_layers=2,
            num_tasks=num_tasks, task_dim=32,
        )

        embeddings, task_names = extract_task_embeddings(model)

        assert embeddings.shape == (num_tasks, 32)
        assert len(task_names) == num_tasks


# ─────────────────────────────────────────────────────────────────────────────
# Statistical Significance Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSignificance:
    """Tests for significance testing utilities."""

    def test_import(self):
        from evaluation.significance import paired_t_test
        assert paired_t_test is not None

    def test_paired_t_test_significant(self):
        from evaluation.significance import paired_t_test

        scores_a = [0.72, 0.73, 0.71, 0.74, 0.72]
        scores_b = [0.82, 0.83, 0.81, 0.84, 0.82]

        result = paired_t_test(scores_a, scores_b)

        assert result["significant"] is True
        assert result["mean_diff"] > 0
        assert result["p_value"] < 0.05

    def test_paired_t_test_not_significant(self):
        from evaluation.significance import paired_t_test

        scores_a = [0.72, 0.73, 0.74, 0.71, 0.75]
        scores_b = [0.73, 0.72, 0.75, 0.72, 0.74]

        result = paired_t_test(scores_a, scores_b)

        assert "t_stat" in result
        assert "p_value" in result
        assert "significant" in result
        assert "ci_lower" in result
        assert "ci_upper" in result

    def test_paired_t_test_identical(self):
        from evaluation.significance import paired_t_test

        scores = [0.75, 0.75, 0.75, 0.75, 0.75]
        result = paired_t_test(scores, scores)

        assert result["mean_diff"] == 0.0

    def test_confidence_interval_contains_mean(self):
        from evaluation.significance import paired_t_test

        scores_a = [0.70, 0.72, 0.71, 0.73, 0.69]
        scores_b = [0.75, 0.77, 0.76, 0.78, 0.74]

        result = paired_t_test(scores_a, scores_b)

        assert result["ci_lower"] <= result["mean_diff"] <= result["ci_upper"]


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX Table Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestLatexTables:
    """Tests for LaTeX table generation."""

    def test_import(self):
        from evaluation.latex_tables import (
            generate_ablation_table,
            generate_main_results_table,
        )

        assert generate_main_results_table is not None
        assert generate_ablation_table is not None

    def test_main_table_generation(self, tmp_path):
        from evaluation.latex_tables import generate_main_results_table

        output_path = str(tmp_path / "main.tex")

        try:
            latex = generate_main_results_table(
                results_dir="results",
                output_path=output_path,
            )

            if os.path.exists(output_path):
                with open(output_path) as f:
                    content = f.read()
                assert "\\begin{table}" in content
                assert "\\end{table}" in content
        except (FileNotFoundError, json.JSONDecodeError):
            pytest.skip("No results files found")

    def test_ablation_table_generation(self, tmp_path):
        """Test ablation table with mock data."""
        import json

        # Create mock ablation result
        mock_data = {
            "ablation": "task_dim",
            "description": "Test",
            "variable": "task_dim",
            "fixed": {},
            "results": {
                "32": {"mean_auc": 0.74, "std_auc": 0.01, "num_params": 50000, "avg_training_time": 100},
                "64": {"mean_auc": 0.76, "std_auc": 0.01, "num_params": 60000, "avg_training_time": 120},
            },
        }

        input_dir = str(tmp_path / "ablations")
        os.makedirs(input_dir)

        with open(os.path.join(input_dir, "ablation_task_dim.json"), "w") as f:
            json.dump(mock_data, f)

        from evaluation.latex_tables import generate_ablation_table

        output_path = str(tmp_path / "ablation.tex")
        latex = generate_ablation_table(
            "task_dim",
            results_dir=input_dir,
            output_path=output_path,
        )

        assert "\\begin{table}" in latex
        assert "\\end{table}" in latex
        assert "0.7600" in latex or "0.76" in latex
        assert os.path.exists(output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Ablation Framework Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAblationFramework:
    """Tests for ablation study infrastructure."""

    def test_import(self):
        from scripts.run_ablations import ABLATION_CONFIGS
        assert "task_dim" in ABLATION_CONFIGS
        assert "num_layers" in ABLATION_CONFIGS
        assert "hidden_dim" in ABLATION_CONFIGS
        assert "strategy" in ABLATION_CONFIGS

    def test_ablation_configs_valid(self):
        from scripts.run_ablations import ABLATION_CONFIGS

        for name, config in ABLATION_CONFIGS.items():
            assert "description" in config, f"Missing description in {name}"
            assert "variable" in config, f"Missing variable in {name}"
            assert "values" in config, f"Missing values in {name}"
            assert "fixed" in config, f"Missing fixed in {name}"
            assert len(config["values"]) >= 2, f"Need ≥2 values in {name}"

    def test_create_loaders(self):
        from scripts.run_ablations import create_loaders

        train_ds, train_loader, val_loaders, test_loaders = create_loaders(
            ["bbbp"], batch_size=32
        )

        assert len(train_ds) > 0
        assert len(val_loaders) > 0
        assert len(test_loaders) > 0

    def test_build_model_task_conditioned(self):
        from scripts.run_ablations import create_loaders, build_model
        from data.multitask_dataset import get_num_tasks

        train_ds, _, _, _ = create_loaders(["bbbp"], batch_size=32)

        model, num_params = build_model(
            train_ds, "task_conditioned",
            hidden_dim=64, num_layers=2,
            task_dim=32, num_tasks=get_num_tasks(),
        )

        assert model is not None
        assert num_params > 0

    def test_build_model_hard_sharing(self):
        from scripts.run_ablations import create_loaders, build_model
        from data.multitask_dataset import get_num_tasks

        train_ds, _, _, _ = create_loaders(["bace"], batch_size=32)

        model, num_params = build_model(
            train_ds, "hard_sharing",
            hidden_dim=64, num_layers=2,
            task_dim=0, num_tasks=get_num_tasks(),
        )

        assert model is not None
        assert num_params > 0

    def test_build_model_zero_task_dim(self):
        """task_dim=0 should fallback to hard sharing."""
        from scripts.run_ablations import create_loaders, build_model
        from data.multitask_dataset import get_num_tasks
        from models.task_conditioned_egnn import HardSharingClassifier

        train_ds, _, _, _ = create_loaders(["bbbp"], batch_size=32)

        model, _ = build_model(
            train_ds, "task_conditioned",
            hidden_dim=64, num_layers=2,
            task_dim=0, num_tasks=get_num_tasks(),
        )

        assert isinstance(model, HardSharingClassifier)


# ─────────────────────────────────────────────────────────────────────────────
# Integration Test
# ─────────────────────────────────────────────────────────────────────────────


class TestPhase4Integration:
    """End-to-end Phase 4 integration tests."""

    def test_full_ablation_single_value(self):
        """Run minimal ablation with one value and one seed."""
        from scripts.run_ablations import (
            create_loaders,
            run_single_experiment,
        )
        from training.trainer import set_seed
        from data.multitask_dataset import get_num_tasks

        set_seed(42)

        train_ds, train_loader, val_loaders, test_loaders = create_loaders(
            ["bbbp"], batch_size=32
        )

        result = run_single_experiment(
            train_ds=train_ds,
            train_loader=train_loader,
            val_loaders=val_loaders,
            test_loaders=test_loaders,
            model_type="task_conditioned",
            hidden_dim=32,
            num_layers=2,
            task_dim=16,
            use_pcgrad=False,
            seed=42,
            device="cpu",
            ckpt_tag="test_ablation",
        )

        assert "test_auc_avg" in result
        assert 0 <= result["test_auc_avg"] <= 1
        assert result["num_params"] > 0
        assert result["training_time"] > 0

        # Cleanup
        ckpt_path = "checkpoints/ablation_test_ablation_seed42.pt"
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def plt_close(fig):
    """Close matplotlib figure to free memory."""
    import matplotlib.pyplot as plt
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Run all tests
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])