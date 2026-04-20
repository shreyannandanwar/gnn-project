"""
evaluation/visualizations.py
Publication-quality visualization tools.
"""
import json
import os
import pickle
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

matplotlib.rcParams.update(
    {
        "font.size": 12,
        "font.family": "serif",
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


# ─────────────────────────────────────────────────────────────────────────────
# Ablation Plots
# ─────────────────────────────────────────────────────────────────────────────


def plot_ablation_curve(
    values: List,
    means: List[float],
    stds: List[float],
    xlabel: str,
    ylabel: str = "Test ROC-AUC",
    title: str = "",
    save_path: Optional[str] = None,
    figsize: Tuple = (8, 5),
    color: str = "#2196F3",
    highlight_best: bool = True,
) -> plt.Figure:
    """
    Plot ablation study curve with error bars.
    """
    fig, ax = plt.subplots(figsize=figsize)

    means = np.array(means)
    stds = np.array(stds)
    x_pos = np.arange(len(values))

    # Plot with error bars
    ax.errorbar(
        x_pos,
        means,
        yerr=stds,
        fmt="o-",
        color=color,
        capsize=5,
        capthick=1.5,
        linewidth=2,
        markersize=8,
        elinewidth=1.5,
    )

    # Fill confidence band
    ax.fill_between(
        x_pos,
        means - stds,
        means + stds,
        alpha=0.15,
        color=color,
    )

    # Highlight best
    if highlight_best:
        best_idx = np.argmax(means)
        ax.scatter(
            [x_pos[best_idx]],
            [means[best_idx]],
            s=200,
            color="red",
            zorder=5,
            marker="*",
            label=f"Best: {means[best_idx]:.4f}",
        )
        ax.legend()

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(v) for v in values])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    return fig


def plot_strategy_comparison(
    strategies: List[str],
    means: List[float],
    stds: List[float],
    save_path: Optional[str] = None,
    figsize: Tuple = (8, 5),
) -> plt.Figure:
    """
    Bar chart comparing multi-task strategies.
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
    x_pos = np.arange(len(strategies))

    bars = ax.bar(
        x_pos,
        means,
        yerr=stds,
        capsize=5,
        color=colors[: len(strategies)],
        edgecolor="black",
        linewidth=0.5,
        alpha=0.85,
    )

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.003,
            f"{mean:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [s.replace("_", "\n") for s in strategies],
        fontsize=11,
    )
    ax.set_ylabel("Test ROC-AUC")
    ax.set_title("Multi-Task Strategy Comparison")
    ax.grid(axis="y", alpha=0.3)

    # Set y-axis to show differences clearly
    all_vals = np.array(means)
    y_min = max(0, min(all_vals) - 0.05)
    y_max = max(all_vals) + max(stds) + 0.02
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Per-Task Comparison
# ─────────────────────────────────────────────────────────────────────────────


def plot_per_task_comparison(
    task_names: List[str],
    model_results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    figsize: Optional[Tuple] = None,
) -> plt.Figure:
    """
    Grouped bar chart comparing per-task performance across models.

    Parameters
    ----------
    task_names : list of str
    model_results : dict mapping model_name -> {task_name: auc}
    """
    n_tasks = len(task_names)
    n_models = len(model_results)

    if figsize is None:
        figsize = (max(12, n_tasks * 1.5), 6)

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(n_tasks)
    width = 0.8 / n_models
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]

    for i, (model_name, task_aucs) in enumerate(model_results.items()):
        values = [task_aucs.get(t, 0) for t in task_names]
        offset = (i - n_models / 2 + 0.5) * width

        ax.bar(
            x + offset,
            values,
            width,
            label=model_name.replace("_", " ").title(),
            color=colors[i % len(colors)],
            edgecolor="black",
            linewidth=0.3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [t.replace("_", "\n") for t in task_names],
        rotation=45,
        ha="right",
        fontsize=9,
    )
    ax.set_ylabel("Test ROC-AUC")
    ax.set_title("Per-Task Performance Comparison")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Task Embedding Visualization
# ─────────────────────────────────────────────────────────────────────────────


def extract_task_embeddings(model) -> Tuple[np.ndarray, List[str]]:
    """
    Extract learned task embeddings from a trained model.

    Returns
    -------
    embeddings : np.ndarray [num_tasks, task_dim]
    task_names : list of str
    """
    from data.multitask_dataset import get_all_task_names

    if hasattr(model, "encoder") and hasattr(model.encoder, "task_embeddings"):
        emb = model.encoder.task_embeddings.weight.detach().cpu().numpy()
    elif hasattr(model, "task_embeddings"):
        emb = model.task_embeddings.weight.detach().cpu().numpy()
    else:
        raise ValueError("Model does not have task embeddings")

    task_names = get_all_task_names()

    return emb, task_names


def plot_task_embedding_tsne(
    embeddings: np.ndarray,
    task_names: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple = (10, 8),
    perplexity: float = 5.0,
) -> plt.Figure:
    """
    t-SNE visualization of task embeddings.
    """
    from sklearn.manifold import TSNE

    # Run t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(embeddings) - 1),
        random_state=42,
        n_iter=1000,
    )
    emb_2d = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=figsize)

    # Color by dataset
    dataset_colors = {
        "bbbp": "#FF6B6B",
        "bace": "#4ECDC4",
        "hiv": "#45B7D1",
        "clintox": "#96CEB4",
        "tox21": "#FFEAA7",
    }

    for i, name in enumerate(task_names):
        dataset = name.split("_")[0]
        color = dataset_colors.get(dataset, "#999999")

        ax.scatter(
            emb_2d[i, 0],
            emb_2d[i, 1],
            c=color,
            s=120,
            edgecolors="black",
            linewidth=0.5,
            zorder=3,
        )

        # Add label
        short_name = "_".join(name.split("_")[1:])
        ax.annotate(
            short_name,
            (emb_2d[i, 0], emb_2d[i, 1]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    # Legend
    for dataset, color in dataset_colors.items():
        ax.scatter([], [], c=color, s=80, label=dataset.upper(), edgecolors="black")

    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.set_title("Task Embedding Space (t-SNE)")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    return fig


def plot_task_embedding_heatmap(
    embeddings: np.ndarray,
    task_names: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple = (12, 8),
) -> plt.Figure:
    """
    Heatmap of task embedding cosine similarities.
    """
    # Compute cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    normalized = embeddings / norms
    similarity = normalized @ normalized.T

    fig, ax = plt.subplots(figsize=figsize)

    short_names = ["_".join(n.split("_")[1:]) for n in task_names]

    sns.heatmap(
        similarity,
        xticklabels=short_names,
        yticklabels=short_names,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        annot_kws={"fontsize": 7},
    )

    ax.set_title("Task Embedding Cosine Similarity")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Training Curves
# ─────────────────────────────────────────────────────────────────────────────


def plot_training_curves(
    histories: Dict[str, Dict],
    save_path: Optional[str] = None,
    figsize: Tuple = (14, 5),
) -> plt.Figure:
    """
    Plot training loss and validation AUC curves for multiple models.

    Parameters
    ----------
    histories : dict mapping model_name -> {train_loss: [...], val_auc_avg: [...]}
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

    for i, (name, hist) in enumerate(histories.items()):
        color = colors[i % len(colors)]
        label = name.replace("_", " ").title()

        epochs = range(1, len(hist["train_loss"]) + 1)

        # Training loss
        ax1.plot(epochs, hist["train_loss"], color=color, label=label, linewidth=1.5)

        # Validation AUC
        ax2.plot(epochs, hist["val_auc_avg"], color=color, label=label, linewidth=1.5)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation ROC-AUC")
    ax2.set_title("Validation Performance Curves")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Improvement Delta Chart
# ─────────────────────────────────────────────────────────────────────────────


def plot_improvement_delta(
    task_names: List[str],
    baseline_aucs: Dict[str, float],
    improved_aucs: Dict[str, float],
    baseline_label: str = "Hard Sharing",
    improved_label: str = "Task-Cond. + PCGrad",
    save_path: Optional[str] = None,
    figsize: Optional[Tuple] = None,
) -> plt.Figure:
    """
    Plot per-task AUC improvement (delta) over baseline.
    """
    if figsize is None:
        figsize = (max(10, len(task_names) * 0.8), 5)

    fig, ax = plt.subplots(figsize=figsize)

    deltas = []
    names = []

    for task in task_names:
        if task in baseline_aucs and task in improved_aucs:
            delta = improved_aucs[task] - baseline_aucs[task]
            deltas.append(delta)
            names.append("_".join(task.split("_")[1:]))

    x_pos = np.arange(len(names))
    colors = ["#4ECDC4" if d >= 0 else "#FF6B6B" for d in deltas]

    bars = ax.bar(
        x_pos,
        deltas,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )

    # Add value labels
    for bar, delta in zip(bars, deltas):
        y_pos = bar.get_height()
        va = "bottom" if delta >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            f"{delta:+.3f}",
            ha="center",
            va=va,
            fontsize=8,
            fontweight="bold",
        )

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Δ ROC-AUC")
    ax.set_title(f"Per-Task Improvement: {improved_label} vs {baseline_label}")
    ax.grid(axis="y", alpha=0.3)

    # Count improvements
    n_improved = sum(1 for d in deltas if d > 0)
    n_total = len(deltas)
    ax.text(
        0.02,
        0.98,
        f"Improved: {n_improved}/{n_total} tasks",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    return fig