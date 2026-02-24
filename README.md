# Task-Conditioned Multi-Task 3D GNN for Molecular Property Benchmark Optimization

**Feature 1** of a research-oriented AI for Drug Discovery project.

## Project Structure

```
project/
  data/           # raw MoleculeNet downloads & processed PyG datasets
  models/         # GNN architecture definitions
  training/       # trainer, losses, optimizers
  evaluation/     # metrics, result tables
  checkpoints/    # saved model states (git-ignored)
  results/        # JSON/CSV experiment outputs
  notebooks/      # exploratory analysis
  scripts/        # entry points (train.py, evaluate.py, ablate.py)
  configs/        # YAML hyperparameter files
  tests/          # unit tests
```

## Setup

### 1. Create conda environment
```bash
conda create -n mol3d python=3.9 -y
conda activate mol3d
```

### 2. Install PyTorch (CUDA 11.8)
```bash
pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install PyG extensions
```bash
pip install torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install torch-geometric==2.4.0
```

### 4. Install remaining dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

### 5. Verify
```bash
python -m pytest tests/test_phase0.py -v
```

## Phases

| Phase | Description | Members |
|-------|-------------|---------|
| 0 | Infrastructure & Setup | C, D |
| 1 | Dataset Processing & 3D Pipeline | C, D |
| 2 | Strong Single-Task 3D Baselines | C, D |
| 3 | Gradient Analysis & Hard Sharing | C, D |
| 4 | Task Conditioning + PCGrad | C, D |
| 5 | Ablations & Paper Writing | C, D |

## Datasets (MoleculeNet — Classification)

| Dataset | Tasks | Molecules | Metric |
|---------|-------|-----------|--------|
| Tox21   | 12    | ~7,831    | ROC-AUC |
| ClinTox | 2     | ~1,478    | ROC-AUC |
| BBBP    | 1     | ~2,039    | ROC-AUC |
| BACE    | 1     | ~1,513    | ROC-AUC |
| HIV     | 1     | ~41,127   | ROC-AUC |

All splits use **scaffold splitting** for fair generalization evaluation.
