# Task-Conditioned Multi-Task 3D GNN for Molecular Property Benchmark Optimization

Research code for multi-task molecular property prediction using **3D geometric graph neural networks** with **task-conditioned message passing** and optional **PCGrad** for gradient conflict mitigation.

## Project Highlights
- **3D molecular graphs** from RDKit conformers + geometric features
- **E(n)-equivariant GNN (EGNN)** encoder (translation/rotation equivariant)
- **Task conditioning** via learnable task embeddings that modulate message passing
- **Multi-task optimization** with optional **PCGrad**
- **Benchmarks**: MoleculeNet classification datasets with scaffold splits

---

## Repository Structure

```text
project/
├── data/
│   ├── __init__.py
│   ├── moleculenet.py              # MoleculeNet dataset loaders
│   ├── scaffold_analysis.py        # Scaffold split verification
│   ├── featurizer.py               # 3D molecular graph generation
│   ├── dataset.py                  # Single-task PyG datasets
│   └── multitask_dataset.py        # Multi-task combined dataset
│
├── models/
│   ├── __init__.py
│   ├── egnn.py                     # E(n)-equivariant GNN encoder
│   └── task_conditioned_egnn.py    # Task-conditioned multi-task models
│
├── training/
│   ├── __init__.py
│   ├── trainer.py                  # Single-task trainer
│   ├── multitask_trainer.py        # Multi-task trainer
│   └── pcgrad.py                   # PCGrad optimizer
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                  # ROC-AUC computation
│   ├── runner.py                   # Multi-seed experiment runner
│   └── gradient_analysis.py        # Gradient conflict analysis
│
├── scripts/
│   ├── __init__.py
│   ├── preprocess_phase1.py        # Phase 1: Data preprocessing
│   ├── train_single_task.py        # Phase 2: Single-task baselines
│   ├── train_all_baselines.py      # Batch single-task training
│   ├── analyze_gradients.py        # Phase 3: Gradient analysis
│   ├── train_multitask.py          # Phase 3: Multi-task training
│   └── compare_multitask_results.py# Results comparison
│
├── tests/
│   ├── __init__.py
│   ├── test_phase0.py
│   ├── test_phase1.py
│   ├── test_phase2.py
│   └── test_phase3.py
│
├── configs/
│   ├── default.yaml
│   ├── bace.yaml
│   ├── bbbp.yaml
│   ├── clintox.yaml
│   ├── hiv.yaml
│   └── tox21.yaml
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## Datasets (MoleculeNet — Classification)

All splits use **scaffold splitting** for realistic generalization evaluation.

| Dataset  | Tasks | Molecules | Metric  | Split     |
|---------:|------:|----------:|--------:|----------:|
| Tox21    | 12    | ~7,831    | ROC-AUC | Scaffold  |
| ClinTox  | 2     | ~1,478    | ROC-AUC | Scaffold  |
| BBBP     | 1     | ~2,039    | ROC-AUC | Scaffold  |
| BACE     | 1     | ~1,513    | ROC-AUC | Scaffold  |
| HIV      | 1     | ~41,127   | ROC-AUC | Scaffold  |

**Total:** 17 binary classification tasks

---

## Quick Start

### 1) Environment Setup

```bash
# Create conda env
conda create -n mol3d python=3.9 -y
conda activate mol3d

# Install PyTorch (CUDA 11.8 example)
pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu118

# Install PyG extensions (match torch/cuda versions)
pip install torch-scatter torch-sparse torch-cluster \
  -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install torch-geometric==2.4.0

# Remaining deps
pip install -r requirements.txt

# Editable install
pip install -e .
```

### 2) Verify Installation

```bash
python -m pytest tests/test_phase0.py -v
```

---

## Execution Pipeline

### Phase 1 — Data Preprocessing
**Goal:** load datasets, verify scaffold splits, and generate 3D molecular graphs.

```bash
python scripts/preprocess_phase1.py
python -m pytest tests/test_phase1.py -v
```

**Outputs (examples):**
- `data/processed/{dataset}_raw.pkl`
- `data/processed/{dataset}_graphs_3d.pkl`
- `data/processed/{dataset}_scaffold_stats.pkl`

---

### Phase 2 — Single-Task Baselines
**Goal:** establish strong single-task performance.

```bash
# BBBP
python scripts/train_single_task.py --dataset bbbp --device cuda

# BACE
python scripts/train_single_task.py --dataset bace --device cuda

# Example: single task within a multi-task dataset (Tox21)
python scripts/train_single_task.py --dataset tox21 --task NR-AR --device cuda

# Batch run
python scripts/train_all_baselines.py cuda

# Verify
python -m pytest tests/test_phase2.py -v
```

---

### Phase 3 — Multi-Task Learning + Analysis

#### 3a) Gradient Conflict Analysis
```bash
python scripts/analyze_gradients.py --device cuda

# Faster debug run
python scripts/analyze_gradients.py --datasets bbbp bace --device cuda
```

**Outputs:**
- `results/gradient_analysis/gradient_similarity_matrix.png`
- `results/gradient_analysis/gradient_analysis.pkl`

#### 3b) Hard Sharing Baseline
```bash
python scripts/train_multitask.py \
  --model hard_sharing \
  --datasets bbbp bace hiv clintox tox21 \
  --seed 0 \
  --device cuda
```

#### 3c) Task-Conditioned Model
```bash
python scripts/train_multitask.py \
  --model task_conditioned \
  --datasets bbbp bace hiv clintox tox21 \
  --task_dim 64 \
  --seed 0 \
  --device cuda
```

#### 3d) Task-Conditioned + PCGrad
```bash
python scripts/train_multitask.py \
  --model task_conditioned \
  --pcgrad \
  --datasets bbbp bace hiv clintox tox21 \
  --seed 0 \
  --device cuda
```

#### 3e) Compare Results
```bash
python scripts/compare_multitask_results.py
```

#### 3f) Run Tests
```bash
python -m pytest tests/test_phase3.py -v
```

---

## Configuration

Default hyperparameters live in `configs/default.yaml`. Dataset-specific overrides are in `configs/{dataset}.yaml`.

Example (illustrative):

```yaml
model:
  hidden_dim: 128
  n_layers: 4
  task_dim: 64
  dropout: 0.1

training:
  lr: 0.001
  wd: 0.00001
  batch: 64
  epochs: 200
  patience: 30
  seeds: [0, 1, 2, 3, 4]
  device: cuda

data:
  split: scaffold
  missing_label: -1
```

---

## Results Artifacts

```text
results/
├── gradient_analysis/
│   ├── gradient_similarity_matrix.png
│   └── gradient_analysis.pkl
├── {dataset}_{model}_seed{N}.json
├── multitask_{model}_seed{N}.json
└── multitask_comparison.json
```

---

## Troubleshooting

### Out of Memory
```bash
python scripts/train_multitask.py --batch_size 32
python scripts/train_multitask.py --hidden_dim 64 --num_layers 3
```

### Slow training / quick CPU sanity checks
```bash
python scripts/train_multitask.py --device cpu
python scripts/train_multitask.py --datasets bbbp bace
python scripts/train_multitask.py --epochs 50
```

---

## References
- **EGNN:** Satorras et al., *E(n) Equivariant Graph Neural Networks*, ICML 2021
- **PCGrad:** Yu et al., *Gradient Surgery for Multi-Task Learning*, NeurIPS 2020
- **FiLM / Task Conditioning:** Perez et al., *FiLM*, AAAI 2018
- **MoleculeNet:** Wu et al., *MoleculeNet*, Chemical Science 2018

---

## Contributing
This is a research project—PRs are welcome. Good contribution targets:
- Additional encoders (SchNet, PaiNN, GemNet)
- Alternative MTL methods (GradNorm, uncertainty weighting)
- More datasets (PCBA, MUV, ToxCast)
- Performance improvements (AMP / mixed precision, distributed training)

---

## Citation

Replace placeholders before publishing:

```bibtex
@software{task_conditioned_3d_gnn,
  title={Task-Conditioned Multi-Task 3D GNN for Molecular Property Prediction},
  author={Your Name},
  year={2026},
  url={https://github.com/shreyannandanwar/gnn-project}
}
```

## License
MIT License — see `LICENSE`.
