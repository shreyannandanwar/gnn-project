"""
scripts/generate_figures.py
Generate all publication-ready figures.

Usage:
    python scripts/generate_figures.py
"""
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, ".")

from evaluation.visualizations import (
    extract_task_embeddings,
    plot_ablation_curve,
    plot_improvement_delta,
    plot_per_task_comparison,
    plot_strategy_comparison,
    plot_task_embedding_heatmap,
    plot_task_embedding_tsne,
    plot_training_curves,
)
from evaluation.significance import run_significance_tests
from evaluation.latex_tables import (
    generate_ablation_table,
    generate_main_results_table,
)

FIGURES_DIR = "results/figures"
TABLES_DIR = "results/tables"


def ensure_dirs():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)


def fig1_strategy_comparison():
    """Figure 1: Multi-task strategy comparison bar chart."""
    print("\n[Fig 1] Strategy comparison...")

    # Load results
    strategies = []
    means = []
    stds = []

    for model_tag in ["hard_sharing", "task_conditioned", "task_conditioned_pcgrad"]:
        aucs = []
        for seed in range(5):
            path = f"results/multitask_{model_tag}_seed{seed}.json"
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                aucs.append(data["test_auc_avg"])

        if aucs:
            strategies.append(model_tag)
            means.append(np.mean(aucs))
            stds.append(np.std(aucs))

    if strategies:
        plot_strategy_comparison(
            strategies,
            means,
            stds,
            save_path=f"{FIGURES_DIR}/fig1_strategy_comparison.png",
        )
    else:
        print("  No results found. Skipping.")


def fig2_per_task_comparison():
    """Figure 2: Per-task grouped bar chart."""
    print("\n[Fig 2] Per-task comparison...")

    model_results = {}

    for model_tag in ["hard_sharing", "task_conditioned", "task_conditioned_pcgrad"]:
        # Average per-task results across seeds
        task_aucs_all = {}

        for seed in range(5):
            path = f"results/multitask_{model_tag}_seed{seed}.json"
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)

                for task, auc in data.get("test_auc_per_task", {}).items():
                    if task not in task_aucs_all:
                        task_aucs_all[task] = []
                    task_aucs_all[task].append(auc)

        if task_aucs_all:
            model_results[model_tag] = {
                t: np.mean(aucs) for t, aucs in task_aucs_all.items()
            }

    if model_results:
        task_names = list(next(iter(model_results.values())).keys())
        plot_per_task_comparison(
            task_names,
            model_results,
            save_path=f"{FIGURES_DIR}/fig2_per_task_comparison.png",
        )
    else:
        print("  No results found. Skipping.")


def fig3_improvement_delta():
    """Figure 3: Per-task improvement delta chart."""
    print("\n[Fig 3] Improvement delta...")

    baseline_aucs = {}
    improved_aucs = {}

    for model_tag, target in [
        ("hard_sharing", baseline_aucs),
        ("task_conditioned_pcgrad", improved_aucs),
    ]:
        task_aucs_all = {}
        for seed in range(5):
            path = f"results/multitask_{model_tag}_seed{seed}.json"
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                for task, auc in data.get("test_auc_per_task", {}).items():
                    if task not in task_aucs_all:
                        task_aucs_all[task] = []
                    task_aucs_all[task].append(auc)

        for t, aucs in task_aucs_all.items():
            target[t] = np.mean(aucs)

    if baseline_aucs and improved_aucs:
        task_names = sorted(baseline_aucs.keys())
        plot_improvement_delta(
            task_names,
            baseline_aucs,
            improved_aucs,
            save_path=f"{FIGURES_DIR}/fig3_improvement_delta.png",
        )
    else:
        print("  No results found. Skipping.")


def fig4_task_embeddings():
    """Figure 4: Task embedding t-SNE and heatmap."""
    print("\n[Fig 4] Task embeddings...")

    # Find best task-conditioned checkpoint
    ckpt_path = "checkpoints/multitask_task_conditioned_seed0.pt"

    if not os.path.exists(ckpt_path):
        # Try PCGrad variant
        ckpt_path = "checkpoints/multitask_task_conditioned_pcgrad_seed0.pt"

    if not os.path.exists(ckpt_path):
        print("  No checkpoint found. Skipping.")
        return

    # Load model
    from data.multitask_dataset import MultiTaskDataset, get_num_tasks
    from models.task_conditioned_egnn import MultiTaskClassifier

    train_ds = MultiTaskDataset(["bbbp"], "train")
    sample = train_ds[0]

    model = MultiTaskClassifier(
        node_dim=sample.x.size(1),
        edge_dim=sample.edge_attr.size(1),
        hidden_dim=128,
        num_layers=4,
        num_tasks=get_num_tasks(),
        task_dim=64,
    )

    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    embeddings, task_names = extract_task_embeddings(model)

    # t-SNE
    plot_task_embedding_tsne(
        embeddings,
        task_names,
        save_path=f"{FIGURES_DIR}/fig4a_task_embedding_tsne.png",
    )

    # Heatmap
    plot_task_embedding_heatmap(
        embeddings,
        task_names,
        save_path=f"{FIGURES_DIR}/fig4b_task_embedding_similarity.png",
    )


def fig5_ablation_curves():
    """Figure 5: Ablation study curves."""
    print("\n[Fig 5] Ablation curves...")

    ablation_configs = {
        "task_dim": ("Task Embedding Dimension", "Task Dim"),
        "num_layers": ("Number of GNN Layers", "Layers"),
        "hidden_dim": ("Hidden Dimension", "Hidden Dim"),
    }

    for abl_name, (title, xlabel) in ablation_configs.items():
        path = f"results/ablations/ablation_{abl_name}.json"

        if not os.path.exists(path):
            print(f"  No results for {abl_name}. Skipping.")
            continue

        with open(path) as f:
            data = json.load(f)

        results = data["results"]
        values = [r["value"] for r in results.values()]
        means = [r["mean_auc"] for r in results.values()]
        stds = [r["std_auc"] for r in results.values()]

        plot_ablation_curve(
            values,
            means,
            stds,
            xlabel=xlabel,
            title=f"Ablation: {title}",
            save_path=f"{FIGURES_DIR}/fig5_{abl_name}_ablation.png",
        )


def fig6_gradient_analysis():
    """Figure 6: Gradient similarity matrix."""
    print("\n[Fig 6] Gradient analysis...")

    path = "results/gradient_analysis/gradient_analysis.pkl"

    if not os.path.exists(path):
        print("  No gradient analysis found. Skipping.")
        return

    with open(path, "rb") as f:
        data = pickle.load(f)

    from evaluation.gradient_analysis import plot_gradient_similarity_matrix

    plot_gradient_similarity_matrix(
        data["similarity_matrix"],
        data["task_labels"],
        save_path=f"{FIGURES_DIR}/fig6_gradient_similarity.png",
        title="Task Gradient Cosine Similarity",
    )


def generate_tables():
    """Generate all LaTeX tables."""
    print("\n[Tables] Generating LaTeX tables...")

    try:
        generate_main_results_table(output_path=f"{TABLES_DIR}/main_results.tex")
    except Exception as e:
        print(f"  Main results table: {e}")

    for abl_name in ["task_dim", "num_layers", "hidden_dim", "strategy"]:
        try:
            generate_ablation_table(
                abl_name,
                output_path=f"{TABLES_DIR}/ablation_{abl_name}.tex",
            )
        except Exception as e:
            print(f"  Ablation {abl_name}: {e}")


def run_stats():
    """Run significance tests."""
    print("\n[Stats] Running significance tests...")

    try:
        run_significance_tests()
    except Exception as e:
        print(f"  Significance tests: {e}")


def main():
    print("=" * 80)
    print("GENERATING PUBLICATION FIGURES & TABLES")
    print("=" * 80)

    ensure_dirs()

    fig1_strategy_comparison()
    fig2_per_task_comparison()
    fig3_improvement_delta()
    fig4_task_embeddings()
    fig5_ablation_curves()
    fig6_gradient_analysis()
    generate_tables()
    run_stats()

    # Summary
    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)

    print(f"\nFigures:")
    if os.path.exists(FIGURES_DIR):
        for f in sorted(os.listdir(FIGURES_DIR)):
            print(f"  {FIGURES_DIR}/{f}")

    print(f"\nTables:")
    if os.path.exists(TABLES_DIR):
        for f in sorted(os.listdir(TABLES_DIR)):
            print(f"  {TABLES_DIR}/{f}")


if __name__ == "__main__":
    main()