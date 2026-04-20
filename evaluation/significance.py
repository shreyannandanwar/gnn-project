"""
evaluation/significance.py
Statistical significance testing for model comparisons.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats as scipy_stats


def paired_t_test(
    scores_a: List[float],
    scores_b: List[float],
    alpha: float = 0.05,
) -> Dict:
    """
    Paired t-test between two models run on the same seeds.

    Parameters
    ----------
    scores_a : list of float — per-seed AUC for model A
    scores_b : list of float — per-seed AUC for model B
    alpha : float — significance level

    Returns
    -------
    dict with: t_stat, p_value, significant, mean_diff, ci_lower, ci_upper
    """
    a = np.array(scores_a)
    b = np.array(scores_b)
    diffs = b - a

    n = len(diffs)
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)

    if std_diff < 1e-10:
        return {
            "t_stat": float("inf") if mean_diff > 0 else float("-inf"),
            "p_value": 0.0,
            "significant": abs(mean_diff) > 1e-10,
            "mean_diff": float(mean_diff),
            "ci_lower": float(mean_diff),
            "ci_upper": float(mean_diff),
        }

    t_stat, p_value = scipy_stats.ttest_rel(b, a)

    # Confidence interval
    t_crit = scipy_stats.t.ppf(1 - alpha / 2, df=n - 1)
    se = std_diff / np.sqrt(n)
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se

    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "significant": p_value < alpha,
        "mean_diff": float(mean_diff),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
    }


def run_significance_tests(
    results_dir: str = "results",
) -> Dict:
    """
    Run pairwise significance tests between all model pairs.

    Returns
    -------
    dict mapping (model_a, model_b) -> test result
    """
    results_path = Path(results_dir)

    # Collect per-seed results for each model
    model_seeds = {}

    for f in results_path.glob("multitask_*.json"):
        with open(f) as fp:
            data = json.load(fp)

        model = data["model"]
        seed = data["seed"]
        auc = data["test_auc_avg"]

        if model not in model_seeds:
            model_seeds[model] = {}
        model_seeds[model][seed] = auc

    # Sort by seed for alignment
    models = sorted(model_seeds.keys())
    common_seeds = None

    for model in models:
        seeds = set(model_seeds[model].keys())
        if common_seeds is None:
            common_seeds = seeds
        else:
            common_seeds &= seeds

    common_seeds = sorted(common_seeds)

    if len(common_seeds) < 2:
        print("Not enough common seeds for significance testing.")
        return {}

    # Pairwise tests
    test_results = {}

    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE TESTS (Paired t-test)")
    print("=" * 80)

    for i, model_a in enumerate(models):
        for j, model_b in enumerate(models):
            if i >= j:
                continue

            scores_a = [model_seeds[model_a][s] for s in common_seeds]
            scores_b = [model_seeds[model_b][s] for s in common_seeds]

            result = paired_t_test(scores_a, scores_b)
            test_results[(model_a, model_b)] = result

            sig_marker = "✓" if result["significant"] else "✗"

            print(f"\n{model_a} vs {model_b}:")
            print(f"  Mean diff:  {result['mean_diff']:+.4f}")
            print(f"  t-stat:     {result['t_stat']:.4f}")
            print(f"  p-value:    {result['p_value']:.6f}")
            print(f"  95% CI:     [{result['ci_lower']:+.4f}, {result['ci_upper']:+.4f}]")
            print(f"  Significant (α=0.05): {sig_marker}")

    # Save
    output = {
        f"{a}_vs_{b}": r for (a, b), r in test_results.items()
    }

    output_path = os.path.join(results_dir, "significance_tests.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {output_path}")

    return test_results