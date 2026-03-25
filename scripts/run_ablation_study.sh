#!/bin/bash

# Run all ablations with 5 seeds

DATASETS="bbbp bace hiv clintox tox21"
DEVICE="cuda"

for SEED in 0 1 2 3 4; do
    echo "======================================"
    echo "Seed $SEED"
    echo "======================================"
    
    # Hard sharing baseline
    python scripts/train_multitask.py \
        --model hard_sharing \
        --datasets $DATASETS \
        --seed $SEED \
        --device $DEVICE
    
    # Task-conditioned
    python scripts/train_multitask.py \
        --model task_conditioned \
        --datasets $DATASETS \
        --seed $SEED \
        --device $DEVICE
    
    # Task-conditioned + PCGrad
    python scripts/train_multitask.py \
        --model task_conditioned \
        --pcgrad \
        --datasets $DATASETS \
        --seed $SEED \
        --device $DEVICE
    
done

echo "All experiments complete!"