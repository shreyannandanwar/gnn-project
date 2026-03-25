"""
scripts/train_multitask.py
Train multi-task model with optional PCGrad.

Usage:
    python scripts/train_multitask.py --model hard_sharing
    python scripts/train_multitask.py --model task_conditioned --pcgrad
"""
import argparse
import json
import os
import sys
from typing import Dict

import torch
import yaml
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="task_conditioned",
        choices=["hard_sharing", "task_conditioned"],
        help="Model architecture",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["bbbp", "bace", "hiv", "clintox", "tox21"],
        help="Datasets to include",
    )
    parser.add_argument("--pcgrad", action="store_true", help="Use PCGrad")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--task_dim", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def create_val_loaders(
    datasets: list,
    batch_size: int,
) -> Dict[int, DataLoader]:
    """Create per-task validation loaders."""
    val_loaders = {}
    
    for dataset_name in datasets:
        # Load full validation set
        val_ds = MultiTaskDataset([dataset_name], "valid")
        
        # Group by task
        for task_id, indices in val_ds.task_to_samples.items():
            graphs = [val_ds.samples[i][0] for i in indices]
            if len(graphs) > 0:
                val_loaders[task_id] = DataLoader(
                    graphs, batch_size=batch_size, shuffle=False
                )
    
    return val_loaders


def create_test_loaders(
    datasets: list,
    batch_size: int,
) -> Dict[int, DataLoader]:
    """Create per-task test loaders."""
    test_loaders = {}
    
    for dataset_name in datasets:
        test_ds = MultiTaskDataset([dataset_name], "test")
        
        for task_id, indices in test_ds.task_to_samples.items():
            graphs = [test_ds.samples[i][0] for i in indices]
            if len(graphs) > 0:
                test_loaders[task_id] = DataLoader(
                    graphs, batch_size=batch_size, shuffle=False
                )
    
    return test_loaders


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Model tag
    model_tag = args.model
    if args.pcgrad:
        model_tag += "_pcgrad"
    
    print("="*80)
    print(f"MULTI-TASK TRAINING: {model_tag.upper()}")
    print("="*80)
    print(f"Datasets: {args.datasets}")
    print(f"Model: {args.model}")
    print(f"PCGrad: {args.pcgrad}")
    print(f"Seed: {args.seed}")
    
    # Load data
    print("\nLoading datasets...")
    train_ds = MultiTaskDataset(args.datasets, "train")
    print(train_ds)
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    
    val_loaders = create_val_loaders(args.datasets, args.batch_size)
    test_loaders = create_test_loaders(args.datasets, args.batch_size)
    
    print(f"Validation tasks: {len(val_loaders)}")
    print(f"Test tasks: {len(test_loaders)}")
    
    # Initialize model
    sample = train_ds[0]
    node_dim = sample.x.size(1)
    edge_dim = sample.edge_attr.size(1)
    num_tasks = get_num_tasks()
    
    print(f"\nModel config:")
    print(f"  Node dim: {node_dim}")
    print(f"  Edge dim: {edge_dim}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Num layers: {args.num_layers}")
    print(f"  Num tasks: {num_tasks}")
    
    if args.model == "task_conditioned":
        model = MultiTaskClassifier(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_tasks=num_tasks,
            task_dim=args.task_dim,
            dropout=0.1,
        )
    else:
        model = HardSharingClassifier(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_tasks=num_tasks,
            dropout=0.1,
        )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {num_params:,}")
    
    # Training config
    cfg = {
        "lr": args.lr,
        "wd": 1e-5,
        "patience": 30,
        "epochs": args.epochs,
    }
    
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    ckpt_path = f"checkpoints/multitask_{model_tag}_seed{args.seed}.pt"
    
    # Train
    trainer = MultiTaskTrainer(
        model=model,
        train_loader=train_loader,
        val_loaders=val_loaders,
        cfg=cfg,
        device=device,
        ckpt_path=ckpt_path,
        use_pcgrad=args.pcgrad,
    )
    
    results = trainer.run(epochs=args.epochs)
    
    # Test evaluation
    print("\n" + "="*80)
    print("TEST EVALUATION")
    print("="*80)
    
    all_task_names = get_all_task_names()
    test_aucs = {}
    
    for task_id, loader in test_loaders.items():
        auc = trainer.eval_task(task_id, loader)
        test_aucs[task_id] = auc
        task_name = all_task_names[task_id]
        print(f"  {task_name}: {auc:.4f}")
    
    valid_aucs = [v for v in test_aucs.values() if v == v]  # Filter NaN
    avg_test_auc = sum(valid_aucs) / len(valid_aucs) if valid_aucs else 0.0
    
    print(f"\n  Average Test AUC: {avg_test_auc:.4f}")
    
    # Save results
    os.makedirs("results", exist_ok=True)
    
    output = {
        "model": model_tag,
        "datasets": args.datasets,
        "seed": args.seed,
        "pcgrad": args.pcgrad,
        "best_val_auc": results["best_avg_auc"],
        "test_auc_avg": avg_test_auc,
        "test_auc_per_task": {
            all_task_names[tid]: auc 
            for tid, auc in test_aucs.items()
        },
        "config": {
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "task_dim": args.task_dim,
            "lr": args.lr,
            "batch_size": args.batch_size,
        },
    }
    
    output_path = f"results/multitask_{model_tag}_seed{args.seed}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print("="*80)


if __name__ == "__main__":
    main()