"""
scripts/train_single_task.py
Train single-task baselines for all datasets.

Usage:
    python scripts/train_single_task.py --dataset bace --seed 0
    python scripts/train_single_task.py --dataset tox21 --task NR-AR --seed 0
"""
import argparse
import os
import sys
import yaml

import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, ".")

from data.dataset import MoleculeDataset, get_task_names
from models.egnn import EGNNClassifier
from evaluation.runner import run_multi_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["tox21", "clintox", "bbbp", "bace", "hiv"],
                        help="Dataset name")
    parser.add_argument("--task", type=str, default=None,
                        help="Task name (for multi-task datasets)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML (default: configs/{dataset}.yaml)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="Random seeds for multi-seed evaluation")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    return parser.parse_args()


def build_model_and_loaders(dataset_name, task_name, cfg, seed):
    """
    Build model and data loaders for a single seed.
    
    Returns
    -------
    model, train_loader, val_loader, test_loader
    """
    # Load datasets
    train_ds = MoleculeDataset(dataset_name, "train", task_name=task_name)
    val_ds = MoleculeDataset(dataset_name, "valid", task_name=task_name)
    test_ds = MoleculeDataset(dataset_name, "test", task_name=task_name)
    
    print(f"  {train_ds}")
    print(f"  {val_ds}")
    print(f"  {test_ds}")
    
    # Data loaders
    batch_size = cfg["training"]["batch"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Model
    # Get feature dimensions from a sample
    sample = train_ds[0]
    node_dim = sample.x.size(1)
    edge_dim = sample.edge_attr.size(1)
    
    model = EGNNClassifier(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["n_layers"],
        dropout=cfg["model"]["dropout"],
    )
    
    return model, train_loader, val_loader, test_loader


def main():
    args = parse_args()
    
    # Load config
    if args.config is None:
        config_path = f"configs/{args.dataset}.yaml"
    else:
        config_path = args.config
    
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # For multi-task datasets, determine task
    tasks = get_task_names(args.dataset)
    
    if len(tasks) == 1:
        # Single-task dataset
        task_name = tasks[0]
        print(f"\nSingle-task dataset: {args.dataset}")
        print(f"Task: {task_name}")
        
        # Run multi-seed evaluation
        results = run_multi_seed(
            build_fn=lambda seed: build_model_and_loaders(args.dataset, task_name, cfg, seed),
            dataset=args.dataset,
            cfg=cfg["training"],
            model_tag="egnn_single",
            seeds=args.seeds,
            device=args.device,
        )
    
    else:
        # Multi-task dataset
        if args.task is None:
            print(f"\nMulti-task dataset: {args.dataset}")
            print(f"Available tasks: {tasks}")
            print(f"\nPlease specify a task with --task <task_name>")
            print(f"Example: --task {tasks[0]}")
            return
        
        if args.task not in tasks:
            raise ValueError(f"Task {args.task} not in {tasks}")
        
        task_name = args.task
        print(f"\nMulti-task dataset: {args.dataset}")
        print(f"Training task: {task_name}")
        
        # Run multi-seed evaluation
        results = run_multi_seed(
            build_fn=lambda seed: build_model_and_loaders(args.dataset, task_name, cfg, seed),
            dataset=f"{args.dataset}_{task_name}",
            cfg=cfg["training"],
            model_tag="egnn_single",
            seeds=args.seeds,
            device=args.device,
        )
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Task: {task_name}")
    print(f"Mean ROC-AUC: {results['mean_auc']:.4f} ± {results['std_auc']:.4f}")


if __name__ == "__main__":
    main()