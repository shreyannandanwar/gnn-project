"""
scripts/summarize_results.py
Generate results table from saved JSON files.
"""
import json
import os
from pathlib import Path


def main():
    results_dir = Path("results")
    
    if not results_dir.exists():
        print("No results found. Please train models first.")
        return
    
    # Collect all results
    results = []
    for json_file in sorted(results_dir.glob("*.json")):
        with open(json_file) as f:
            data = json.load(f)
            results.append(data)
    
    # Print table
    print("\n" + "="*80)
    print("SINGLE-TASK BASELINE RESULTS")
    print("="*80)
    print(f"{'Dataset':<20} {'Task':<20} {'Model':<15} {'ROC-AUC':<20}")
    print("-"*80)
    
    for r in results:
        dataset = r["dataset"]
        model = r["model"]
        mean = r["mean_auc"]
        std = r["std_auc"]
        
        # Extract task name if present
        if "_" in dataset:
            ds, task = dataset.split("_", 1)
        else:
            ds = dataset
            task = "-"
        
        print(f"{ds:<20} {task:<20} {model:<15} {mean:.4f} ± {std:.4f}")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    main()