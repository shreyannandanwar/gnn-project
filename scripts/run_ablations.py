"""
scripts/run_ablations.py
Systematic ablation study over key hyperparameters.

Ablations:
  A1: Task embedding dimension (0, 16, 32, 64, 128)
  A2: Number of GNN layers (2, 3, 4, 5, 6)
  A3: Hidden dimension (32, 64, 128, 256)
  A4: Multi-task strategy (hard_sharing, task_conditioned, task_conditioned+pcgrad)
  A5: PCGrad reduction method (mean, sum)

Usage:
    python scripts/run_ablations.py --ablation task_dim --device cuda
    python scripts/run_ablations.py --ablation all --device cuda
"""
import argparse
import json
import os
import sys
import time
from typing import Dict, List

import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, ".")

from data.multitask_dataset import (
    MultiTaskDataset,
    get_all_task_names,
    get_num_tasks,
)
from models.task_conditioned_egnn import (
    MultiTaskClassifier,
    HardSharingClassifier,
)
from training.multitask_trainer import MultiTaskTrainer
from training.trainer import set_seed


# ─────────────────────────────────────────────────────────────────────────────
# Ablation Configurations
# ─────────────────────────────────────────────────────────────────────────────

ABLATION_CONFIGS = {
    "task_dim": {
        "description": "Effect of task embedding dimension",
        "variable": "task_dim",
        "values": [0, 16, 32, 64, 128],
        "fixed": {
            "model": "task_conditioned",
            "hidden_dim": 128,
            "num_layers": 4,
            "pcgrad": False,
        },
    },
    "num_layers": {
        "description": "Effect of number of GNN layers",
        "variable": "num_layers",
        "values": [2, 3, 4, 5, 6],
        "fixed": {
            "model": "task_conditioned",
            "hidden_dim": 128,
            "task_dim": 64,
            "pcgrad": False,
        },
    },
    "hidden_dim": {
        "description": "Effect of hidden dimension",
        "variable": "hidden_dim",
        "values": [32, 64, 128, 256],
        "fixed": {
            "model": "task_conditioned",
            "num_layers": 4,
            "task_dim": 64,
            "pcgrad": False,
        },
    },
    "strategy": {
        "description": "Multi-task strategy comparison",
        "variable": "strategy",
        "values": ["hard_sharing", "task_conditioned", "task_conditioned_pcgrad"],
        "fixed": {
            "hidden_dim": 128,
            "num_layers": 4,
            "task_dim": 64,
        },
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def create_loaders(datasets, batch_size=64):
    """Create train, val, test loaders."""
    train_ds = MultiTaskDataset(datasets, "train")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    val_loaders = {}
    test_loaders = {}

    for ds_name in datasets:
        for split, target in [("valid", val_loaders), ("test", test_loaders)]:
            split_ds = MultiTaskDataset([ds_name], split)
            for task_id, indices in split_ds.task_to_samples.items():
                graphs = [split_ds.samples[i][0] for i in indices]
                if len(graphs) > 0:
                    target[task_id] = DataLoader(
                        graphs, batch_size=batch_size, shuffle=False
                    )

    return train_ds, train_loader, val_loaders, test_loaders


def build_model(train_ds, model_type, hidden_dim, num_layers, task_dim, num_tasks):
    """Build model based on configuration."""
    sample = train_ds[0]
    node_dim = sample.x.size(1)
    edge_dim = sample.edge_attr.size(1)

    if model_type == "hard_sharing" or task_dim == 0:
        model = HardSharingClassifier(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_tasks=num_tasks,
            dropout=0.1,
        )
    else:
        model = MultiTaskClassifier(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_tasks=num_tasks,
            task_dim=task_dim,
            dropout=0.1,
        )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, num_params


def run_single_experiment(
    train_ds,
    train_loader,
    val_loaders,
    test_loaders,
    model_type,
    hidden_dim,
    num_layers,
    task_dim,
    use_pcgrad,
    seed,
    device,
    ckpt_tag,
):
    """Run single training experiment and return results."""
    set_seed(seed)

    num_tasks = get_num_tasks()
    model, num_params = build_model(
        train_ds, model_type, hidden_dim, num_layers, task_dim, num_tasks
    )

    cfg = {"lr": 0.001, "wd": 1e-5, "patience": 30, "epochs": 200}
    ckpt_path = f"checkpoints/ablation_{ckpt_tag}_seed{seed}.pt"

    trainer = MultiTaskTrainer(
        model=model,
        train_loader=train_loader,
        val_loaders=val_loaders,
        cfg=cfg,
        device=device,
        ckpt_path=ckpt_path,
        use_pcgrad=use_pcgrad,
    )

    start_time = time.time()
    results = trainer.run()
    elapsed = time.time() - start_time

    # Test evaluation
    all_task_names = get_all_task_names()
    test_aucs = {}
    for task_id, loader in test_loaders.items():
        auc = trainer.eval_task(task_id, loader)
        test_aucs[all_task_names[task_id]] = auc

    valid_aucs = [v for v in test_aucs.values() if v == v]
    avg_test_auc = sum(valid_aucs) / len(valid_aucs) if valid_aucs else 0.0

    return {
        "best_val_auc": results["best_avg_auc"],
        "test_auc_avg": avg_test_auc,
        "test_auc_per_task": test_aucs,
        "num_params": num_params,
        "training_time": elapsed,
        "epochs_trained": len(results["history"]["train_loss"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main Ablation Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(ablation_name, datasets, seeds, device):
    """Run a complete ablation study."""
    config = ABLATION_CONFIGS[ablation_name]

    print("\n" + "=" * 80)
    print(f"ABLATION STUDY: {config['description']}")
    print(f"Variable: {config['variable']}")
    print(f"Values: {config['values']}")
    print(f"Seeds: {seeds}")
    print("=" * 80)

    # Create data loaders (shared across experiments)
    train_ds, train_loader, val_loaders, test_loaders = create_loaders(datasets)

    all_results = {}

    for value in config["values"]:
        print(f"\n{'─' * 60}")
        print(f"{config['variable']} = {value}")
        print(f"{'─' * 60}")

        # Determine experiment parameters
        fixed = config["fixed"].copy()
        variable = config["variable"]

        if variable == "strategy":
            if value == "hard_sharing":
                model_type = "hard_sharing"
                use_pcgrad = False
                task_dim = fixed.get("task_dim", 64)
            elif value == "task_conditioned":
                model_type = "task_conditioned"
                use_pcgrad = False
                task_dim = fixed.get("task_dim", 64)
            else:  # task_conditioned_pcgrad
                model_type = "task_conditioned"
                use_pcgrad = True
                task_dim = fixed.get("task_dim", 64)

            hidden_dim = fixed["hidden_dim"]
            num_layers = fixed["num_layers"]
        else:
            model_type = fixed.get("model", "task_conditioned")
            use_pcgrad = fixed.get("pcgrad", False)
            hidden_dim = value if variable == "hidden_dim" else fixed.get("hidden_dim", 128)
            num_layers = value if variable == "num_layers" else fixed.get("num_layers", 4)
            task_dim = value if variable == "task_dim" else fixed.get("task_dim", 64)

        seed_results = []
        for seed in seeds:
            ckpt_tag = f"{ablation_name}_{value}"

            result = run_single_experiment(
                train_ds=train_ds,
                train_loader=train_loader,
                val_loaders=val_loaders,
                test_loaders=test_loaders,
                model_type=model_type,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                task_dim=task_dim,
                use_pcgrad=use_pcgrad,
                seed=seed,
                device=device,
                ckpt_tag=ckpt_tag,
            )

            seed_results.append(result)
            print(f"  Seed {seed}: Test AUC = {result['test_auc_avg']:.4f}")

        # Aggregate across seeds
        import numpy as np

        test_aucs = [r["test_auc_avg"] for r in seed_results]

        all_results[str(value)] = {
            "value": value,
            "mean_auc": float(np.mean(test_aucs)),
            "std_auc": float(np.std(test_aucs)),
            "per_seed": seed_results,
            "num_params": seed_results[0]["num_params"],
            "avg_training_time": float(np.mean([r["training_time"] for r in seed_results])),
        }

        print(f"  → Mean: {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}")

    # Save results
    output_dir = "results/ablations"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"ablation_{ablation_name}.json")

    with open(output_path, "w") as f:
        json.dump(
            {
                "ablation": ablation_name,
                "description": config["description"],
                "variable": config["variable"],
                "fixed": config["fixed"],
                "results": all_results,
            },
            f,
            indent=2,
        )

    print(f"\nResults saved: {output_path}")
    return all_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablation",
        type=str,
        default="all",
        choices=list(ABLATION_CONFIGS.keys()) + ["all"],
        help="Which ablation to run",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["bbbp", "bace", "hiv", "clintox", "tox21"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.ablation == "all":
        ablations = list(ABLATION_CONFIGS.keys())
    else:
        ablations = [args.ablation]

    for ablation_name in ablations:
        run_ablation(ablation_name, args.datasets, args.seeds, args.device)

    print("\n" + "=" * 80)
    print("ALL ABLATIONS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()