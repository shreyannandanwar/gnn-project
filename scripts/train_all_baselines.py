"""
scripts/train_all_baselines.py
Train all single-task baselines sequentially.
"""
import subprocess
import sys

EXPERIMENTS = [
    # Single-task datasets
    ("bbbp", None),
    ("bace", None),
    ("hiv", None),
    
    # Multi-task datasets (select representative tasks)
    ("tox21", "NR-AR"),
    ("tox21", "NR-ER"),
    ("clintox", "FDA_APPROVED"),
    ("clintox", "CT_TOX"),
]


def main():
    device = "cuda" if len(sys.argv) < 2 else sys.argv[1]
    
    for dataset, task in EXPERIMENTS:
        print(f"\n{'='*80}")
        print(f"Training: {dataset}" + (f" ({task})" if task else ""))
        print(f"{'='*80}\n")
        
        cmd = ["python", "scripts/train_single_task.py", "--dataset", dataset, "--device", device]
        
        if task is not None:
            cmd.extend(["--task", task])
        
        subprocess.run(cmd, check=True)
    
    print("\n" + "="*80)
    print("ALL BASELINES COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()