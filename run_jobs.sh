#!/bin/bash

# Array of seeds (42, 43, 44, and two randoms represented by -1)
SEEDS=(42 43 44 -1 -1)

# Run 5 Leiden Jobs
for i in "${!SEEDS[@]}"; do
    SEED=${SEEDS[$i]}
    OUTDIR="exp_leiden_run_$i"
    echo "Starting Leiden Run $i (Seed: $SEED)..."
    nohup python -u run_exp.py --seed $SEED --cluster leiden --outdir $OUTDIR > "${OUTDIR}_log.txt" 2>&1 &
done

# Run 5 Argmax Jobs
for i in "${!SEEDS[@]}"; do
    SEED=${SEEDS[$i]}
    OUTDIR="exp_argmax_run_$i"
    echo "Starting Argmax Run $i (Seed: $SEED)..."
    nohup python -u run_exp.py --seed $SEED --cluster argmax --outdir $OUTDIR > "${OUTDIR}_log.txt" 2>&1 &
done

echo "All 10 jobs submitted to background. Run 'top' or 'htop' to monitor."
