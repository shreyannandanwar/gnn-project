"""
scripts/analyze_gradients.py
Analyze gradient conflicts between tasks.

Usage:
    python scripts/analyze_gradients.py --datasets bbbp bace hiv
"""
import argparse
import os
import sys
import pickle

import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, ".")

from data.multitask_dataset import (
    MultiTaskDataset,
    get_all_task_names,
    get_num_tasks,
)
from models.task_conditioned_egnn import HardSharingClassifier
from evaluation.gradient_analysis import (
    compute_task_gradients,
    compute_gradient_similarity_matrix,
    compute_gradient_statistics,
    plot_gradient_similarity_matrix,
    identify_conflicting_pairs,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["bbbp", "bace", "hiv", "clintox", "tox21"],
        help="Datasets to analyze",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_batches", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="results/gradient_analysis")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*80)
    print("GRADIENT CONFLICT ANALYSIS")
    print("="*80)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load multi-task dataset
    print(f"\nLoading datasets: {args.datasets}")
    train_ds = MultiTaskDataset(args.datasets, "train")
    print(train_ds)
    
    # Create per-task loaders
    print("\nCreating per-task data loaders...")
    task_loaders = {}
    
    for task_id, indices in train_ds.task_to_samples.items():
        # Create subset dataset
        subset_graphs = [train_ds.samples[i][0] for i in indices[:500]]  # Limit size
        
        if len(subset_graphs) > 0:
            loader = DataLoader(subset_graphs, batch_size=args.batch_size, shuffle=True)
            task_loaders[task_id] = loader
            print(f"  Task {task_id}: {len(subset_graphs)} samples")
    
    # Initialize model
    print("\nInitializing model...")
    sample = train_ds[0]
    node_dim = sample.x.size(1)
    edge_dim = sample.edge_attr.size(1)
    num_tasks = get_num_tasks()
    
    model = HardSharingClassifier(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=128,
        num_layers=4,
        num_tasks=num_tasks,
        dropout=0.1,
    )
    
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Compute task gradients
    print(f"\nComputing gradients ({args.num_batches} batches per task)...")
    task_gradients = compute_task_gradients(
        model=model,
        data_loaders=task_loaders,
        device=device,
        num_batches=args.num_batches,
    )
    
    print(f"Computed gradients for {len(task_gradients)} tasks")
    
    # Get task names
    all_task_names = get_all_task_names()
    task_names = {tid: all_task_names[tid] for tid in task_gradients.keys()}
    
    # Compute similarity matrix
    print("\nComputing gradient similarity matrix...")
    similarity_matrix, task_labels = compute_gradient_similarity_matrix(
        task_gradients,
        task_names=[task_names[tid] for tid in sorted(task_gradients.keys())],
    )
    
    # Statistics
    stats = compute_gradient_statistics(task_gradients)
    
    print("\n" + "-"*60)
    print("GRADIENT STATISTICS")
    print("-"*60)
    print(f"  Mean similarity:    {stats['mean_similarity']:.4f}")
    print(f"  Std similarity:     {stats['std_similarity']:.4f}")
    print(f"  Min similarity:     {stats['min_similarity']:.4f}")
    print(f"  Max similarity:     {stats['max_similarity']:.4f}")
    print(f"  Conflict ratio:     {stats['conflict_ratio']:.2%}")
    print(f"  Mean magnitude:     {stats['mean_magnitude']:.4f}")
    
    # Identify conflicts
    conflicts = identify_conflicting_pairs(
        similarity_matrix, task_labels, threshold=0.0
    )
    
    if conflicts:
        print("\n" + "-"*60)
        print("CONFLICTING TASK PAIRS (similarity < 0)")
        print("-"*60)
        for t1, t2, sim in conflicts[:10]:  # Top 10
            print(f"  {t1} ↔ {t2}: {sim:.4f}")
    else:
        print("\nNo conflicting task pairs found.")
    
    # Plot and save
    print("\nGenerating visualizations...")
    
    fig = plot_gradient_similarity_matrix(
        similarity_matrix,
        task_labels,
        save_path=os.path.join(args.output_dir, "gradient_similarity_matrix.png"),
        title="Task Gradient Cosine Similarity (Hard Sharing)",
    )
    
    # Save results
    results = {
        "similarity_matrix": similarity_matrix,
        "task_labels": task_labels,
        "statistics": stats,
        "conflicts": conflicts,
    }
    
    with open(os.path.join(args.output_dir, "gradient_analysis.pkl"), "wb") as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to: {args.output_dir}/")
    print("="*80)


if __name__ == "__main__":
    main()