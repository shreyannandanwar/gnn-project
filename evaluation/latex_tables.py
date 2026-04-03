"""
evaluation/latex_tables.py
Generate publication-ready LaTeX tables from results.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import pickle
import numpy as np


def generate_main_results_table(
    results_dir: str = "results",
    output_path: str = "results/tables/main_results.tex",
) -> str:
    """
    Generate main results table comparing all models.

    Format:
    | Dataset | Task | Single-Task | Hard Sharing | Task-Cond. | Task-Cond.+PCGrad |
    """
    # Collect results
    results_path = Path(results_dir)

    models = {
        "single_task": {},
        "hard_sharing": {},
        "task_conditioned": {},
        "task_conditioned_pcgrad": {},
    }

    # Load single-task results
    for f in results_path.glob("*_egnn_single.json"):
        with open(f) as fp:
            data = json.load(fp)
        dataset = data["dataset"]
        models["single_task"][dataset] = {
            "mean": data["mean_auc"],
            "std": data["std_auc"],
        }

    # Load multi-task results
    for f in results_path.glob("multitask_*.json"):
        with open(f) as fp:
            data = json.load(fp)

        model_name = data["model"]
        if model_name not in models:
            continue

        # Per-task results
        for task, auc in data.get("test_auc_per_task", {}).items():
            if task not in models[model_name]:
                models[model_name][task] = []
            models[model_name][task].append(auc)

    # Build LaTeX table
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Multi-task model comparison on MoleculeNet benchmarks (ROC-AUC).}")
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Dataset} & \textbf{Task} & "
        r"\textbf{Single-Task} & \textbf{Hard Sharing} & "
        r"\textbf{Task-Cond.} & \textbf{Task-Cond.+PCGrad} \\"
    )
    lines.append(r"\midrule")

    # Aggregate per-task statistics
    all_means = {m: [] for m in models}

    task_order = [
        "bbbp_p_np",
        "bace_Class",
        "hiv_HIV_active",
        "clintox_FDA_APPROVED",
        "clintox_CT_TOX",
    ]

    for task in task_order:
        dataset = task.split("_")[0]
        task_short = "_".join(task.split("_")[1:])

        row = f"{dataset.upper()} & {task_short}"

        for model_name in models:
            if model_name == "single_task":
                key = dataset
                if key in models[model_name]:
                    m = models[model_name][key]["mean"]
                    s = models[model_name][key]["std"]
                    row += f" & {m:.4f}$\\pm${s:.4f}"
                    all_means[model_name].append(m)
                else:
                    row += " & --"
            else:
                if task in models[model_name]:
                    aucs = models[model_name][task]
                    m = np.mean(aucs)
                    s = np.std(aucs) if len(aucs) > 1 else 0.0
                    row += f" & {m:.4f}$\\pm${s:.4f}"
                    all_means[model_name].append(m)
                else:
                    row += " & --"

        row += r" \\"
        lines.append(row)

    # Average row
    lines.append(r"\midrule")
    avg_row = r"\textbf{Average} & "
    for model_name in models:
        if all_means[model_name]:
            m = np.mean(all_means[model_name])
            avg_row += f" & \\textbf{{{m:.4f}}}"
        else:
            avg_row += " & --"
    avg_row += r" \\"
    lines.append(avg_row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(latex)
    print(f"Saved: {output_path}")

    return latex


def generate_ablation_table(
    ablation_name: str,
    results_dir: str = "results/ablations",
    output_path: Optional[str] = None,
) -> str:
    """
    Generate LaTeX table for a specific ablation study.
    """
    input_path = os.path.join(results_dir, f"ablation_{ablation_name}.json")

    with open(input_path) as f:
        data = json.load(f)

    variable = data["variable"]
    results = data["results"]

    if output_path is None:
        output_path = f"results/tables/ablation_{ablation_name}.tex"

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        rf"\caption{{Ablation: Effect of {variable.replace('_', ' ')}.}}"
    )
    lines.append(rf"\label{{tab:ablation_{ablation_name}}}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(
        rf"\textbf{{{variable.replace('_', ' ').title()}}} & "
        r"\textbf{ROC-AUC} & \textbf{Params} & \textbf{Time (s)} \\"
    )
    lines.append(r"\midrule")

    best_auc = max(r["mean_auc"] for r in results.values())

    for key, result in results.items():
        m = result["mean_auc"]
        s = result["std_auc"]
        params = result["num_params"]
        time_s = result["avg_training_time"]

        # Bold best
        if abs(m - best_auc) < 1e-6:
            auc_str = rf"\textbf{{{m:.4f}$\pm${s:.4f}}}"
        else:
            auc_str = f"{m:.4f}$\\pm${s:.4f}"

        lines.append(
            f"{key} & {auc_str} & {params:,} & {time_s:.0f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(latex)
    print(f"Saved: {output_path}")

    return latex


def generate_gradient_analysis_table(
    analysis_path: str = "results/gradient_analysis/gradient_analysis.pkl",
    output_path: str = "results/tables/gradient_analysis.tex",
) -> str:
    """
    Generate table summarizing gradient conflict statistics.
    """
    with open(analysis_path, "rb") as f:
        data = pickle.load(f)

    stats = data["statistics"]
    conflicts = data.get("conflicts", [])

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Gradient conflict statistics across tasks.}")
    lines.append(r"\label{tab:gradient_stats}")
    lines.append(r"\begin{tabular}{lc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Metric} & \textbf{Value} \\")
    lines.append(r"\midrule")

    metrics = [
        ("Mean Cosine Similarity", f"{stats['mean_similarity']:.4f}"),
        ("Std Cosine Similarity", f"{stats['std_similarity']:.4f}"),
        ("Min Cosine Similarity", f"{stats['min_similarity']:.4f}"),
        ("Max Cosine Similarity", f"{stats['max_similarity']:.4f}"),
        ("Conflict Ratio", f"{stats['conflict_ratio']:.2%}"),
        ("Mean Gradient Magnitude", f"{stats['mean_magnitude']:.4f}"),
        ("Num. Conflicting Pairs", f"{len(conflicts)}"),
    ]

    for name, value in metrics:
        lines.append(f"{name} & {value} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(latex)
    print(f"Saved: {output_path}")

    return latex