"""
scripts/compare_multitask_results.py
Compare multi-task model performance.
"""
import json
import os
from pathlib import Path
import pandas as pd
import numpy as np


def load_results():
    """Load all multi-task result files."""
    results_dir = Path("results")
    
    experiments = {
        "hard_sharing": [],
        "task_conditioned": [],
        "task_conditioned_pcgrad": [],
    }
    
    for json_file in results_dir.glob("multitask_*.json"):
        with open(json_file) as f:
            data = json.load(f)
        
        model = data["model"]
        if model in experiments:
            experiments[model].append(data)
    
    return experiments


def compute_statistics(results_list):
    """Compute mean ± std across seeds."""
    if not results_list:
        return {}
    
    # Collect test AUCs
    test_aucs = [r["test_auc_avg"] for r in results_list]
    
    # Per-task AUCs
    task_aucs = {}
    for task in results_list[0]["test_auc_per_task"].keys():
        aucs = [r["test_auc_per_task"][task] for r in results_list 
                if task in r["test_auc_per_task"]]
        task_aucs[task] = {
            "mean": np.mean(aucs),
            "std": np.std(aucs),
        }
    
    return {
        "avg_auc_mean": np.mean(test_aucs),
        "avg_auc_std": np.std(test_aucs),
        "num_seeds": len(test_aucs),
        "task_aucs": task_aucs,
    }


def print_comparison_table(experiments):
    """Print formatted comparison table."""
    print("\n" + "="*80)
    print("MULTI-TASK MODEL COMPARISON")
    print("="*80)
    
    for model_name, results in experiments.items():
        stats = compute_statistics(results)
        
        if not stats:
            continue
        
        print(f"\n{model_name.upper().replace('_', ' ')}")
        print(f"  Seeds: {stats['num_seeds']}")
        print(f"  Average ROC-AUC: {stats['avg_auc_mean']:.4f} ± {stats['avg_auc_std']:.4f}")
        
        print(f"\n  Per-Task Results:")
        for task, auc_stats in sorted(stats['task_aucs'].items()):
            print(f"    {task:30s}: {auc_stats['mean']:.4f} ± {auc_stats['std']:.4f}")
    
    print("\n" + "="*80)
    
    # Improvement analysis
    if len(experiments) >= 2:
        print("\nIMPROVEMENTS vs HARD SHARING BASELINE")
        print("="*80)
        
        baseline = compute_statistics(experiments.get("hard_sharing", []))
        
        for model_name in ["task_conditioned", "task_conditioned_pcgrad"]:
            if model_name not in experiments:
                continue
            
            model_stats = compute_statistics(experiments[model_name])
            
            if baseline and model_stats:
                delta = model_stats["avg_auc_mean"] - baseline["avg_auc_mean"]
                pct = 100 * delta / baseline["avg_auc_mean"]
                
                print(f"\n{model_name.upper().replace('_', ' ')}")
                print(f"  Δ AUC: {delta:+.4f} ({pct:+.2f}%)")
                
                # Count improved tasks
                improved = 0
                for task in baseline["task_aucs"].keys():
                    if task in model_stats["task_aucs"]:
                        if model_stats["task_aucs"][task]["mean"] > baseline["task_aucs"][task]["mean"]:
                            improved += 1
                
                total = len(baseline["task_aucs"])
                print(f"  Tasks improved: {improved}/{total} ({100*improved/total:.1f}%)")
        
        print("="*80)


def main():
    experiments = load_results()
    
    if not any(experiments.values()):
        print("No results found. Please train models first.")
        return
    
    print_comparison_table(experiments)
    
    # Save summary
    summary = {}
    for model_name, results in experiments.items():
        summary[model_name] = compute_statistics(results)
    
    with open("results/multitask_comparison.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print("\nSummary saved to: results/multitask_comparison.json")


if __name__ == "__main__":
    main()