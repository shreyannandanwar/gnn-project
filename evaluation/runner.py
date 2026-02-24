"""
evaluation/runner.py
Multi-seed experiment runner and result persistence.
"""
import json
import os
from typing import List, Optional

import numpy as np

from training.trainer import Trainer, set_seed


def run_multi_seed(
    build_fn,
    dataset: str,
    cfg: dict,
    model_tag: str = "model",
    seeds: Optional[List[int]] = None,
    device: str = "cuda",
) -> dict:
    """
    Run the same experiment across multiple random seeds.

    Parameters
    ----------
    build_fn  : callable(seed) -> (model, train_loader, val_loader, test_loader)
    dataset   : str   – dataset name, used for checkpoint/result filenames
    cfg       : dict  – hyperparameter dict passed to Trainer
    model_tag : str   – short label for the model variant (e.g. 'egnn_single')
    seeds     : list  – default [0, 1, 2, 3, 4]
    device    : str   – 'cuda' or 'cpu'

    Returns
    -------
    dict with keys: dataset, model, seeds, per_seed_auc, mean_auc, std_auc
    """
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    test_aucs = []
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  Dataset: {dataset.upper()}  |  Model: {model_tag}  |  Seed: {seed}")
        print(f"{'='*60}")

        set_seed(seed)
        model, train_l, val_l, test_l = build_fn(seed)

        ckpt = os.path.join("checkpoints", f"{dataset}_{model_tag}_seed{seed}.pt")
        trainer = Trainer(model, train_l, val_l, cfg, device, ckpt)
        trainer.run()

        test_auc = trainer.eval_epoch(test_l)
        test_aucs.append(test_auc)
        print(f"  → Test ROC-AUC (seed {seed}): {test_auc:.4f}")

    results = {
        "dataset":      dataset,
        "model":        model_tag,
        "seeds":        seeds,
        "per_seed_auc": test_aucs,
        "mean_auc":     float(np.mean(test_aucs)),
        "std_auc":      float(np.std(test_aucs)),
    }

    print(f"\n  ✓ {dataset.upper()} [{model_tag}]  "
          f"mean={results['mean_auc']:.4f} ± {results['std_auc']:.4f}")

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", f"{dataset}_{model_tag}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved → {out_path}")

    return results