#!/bin/bash
# Run the full OOD grid: 9 conditions, train + 2 evals each.
# Resume-safe: skips any condition whose results already exist.
#
# Usage:
#   bash run_ood_grid.sh
#
# Estimated time: ~4-5 hours on H100. Each condition ~25-35 min.

set +e  # let the loop continue if any single condition fails

# 9 conditions in order: cheapest first so we get fast feedback.
CONDITIONS=(
    "swap_N4_alphabetical"
    "swap_N4_name_length"
    "swap_N4_hash"
    "swap_N6_alphabetical"
    "swap_N6_name_length"
    "swap_N6_hash"
    "lehmer_N4_alphabetical"
    "lehmer_N4_name_length"
    "lehmer_N4_hash"
)

# Eval cap. Keeps Lehmer (which has 480-600 test examples) manageable.
# Statistically sufficient: at BER=0.5, std error on 200 samples is ~0.035.
MAX_EVAL=200

mkdir -p results_ood_grid

for i in "${!CONDITIONS[@]}"; do
    cond="${CONDITIONS[$i]}"
    n_done=$((i + 1))
    n_total=${#CONDITIONS[@]}

    echo ""
    echo "======================================================"
    echo "[${n_done}/${n_total}] $cond"
    echo "======================================================"

    # Resume: skip if both result files already exist.
    if [ -f "results_ood_grid/${cond}_id.json" ] \
       && [ -f "results_ood_grid/${cond}_ood.json" ]; then
        echo "Already done, skipping."
        continue
    fi

    # Train (skip if adapter already exists from a previous attempt).
    if [ ! -f "/dev/shm/${cond}-lora/adapter_config.json" ]; then
        python train_permutation.py train \
            --data "datasets_ood_grid/${cond}/train.jsonl" \
            --output-dir "/dev/shm/${cond}-lora" \
            --epochs 3 --batch-size 1 --gradient-accumulation 8
        if [ $? -ne 0 ]; then
            echo "Train failed for $cond, moving on."
            continue
        fi
    else
        echo "Adapter exists, skipping train."
    fi

    # Eval id_test
    if [ ! -f "results_ood_grid/${cond}_id.json" ]; then
        python train_permutation.py evaluate \
            --adapter-dir "/dev/shm/${cond}-lora" \
            --eval-file "datasets_ood_grid/${cond}/id_test.jsonl" \
            --output "results_ood_grid/${cond}_id.json" \
            --max-examples $MAX_EVAL
        if [ $? -ne 0 ]; then
            echo "id_test eval failed for $cond, moving on."
            continue
        fi
    fi

    # Eval ood_test
    if [ ! -f "results_ood_grid/${cond}_ood.json" ]; then
        python train_permutation.py evaluate \
            --adapter-dir "/dev/shm/${cond}-lora" \
            --eval-file "datasets_ood_grid/${cond}/ood_test.jsonl" \
            --output "results_ood_grid/${cond}_ood.json" \
            --max-examples $MAX_EVAL
        if [ $? -ne 0 ]; then
            echo "ood_test eval failed for $cond, moving on."
            continue
        fi
    fi

    echo "[${n_done}/${n_total}] $cond done."
done

echo ""
echo "======================================================"
echo "ALL GRID DONE"
echo "======================================================"
ls -la results_ood_grid/
