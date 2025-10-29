#!/bin/bash

function process_snapshot() {
    local snapshot=$1
    python scripts/extract-en/english_pld_ablation.py \
        --snapshot "$snapshot" \
        --num-workers 64 \
        --train-start 0 \
        --train-end 500 \
        --valid-start 0 \
        --valid-end 5   
}

# For utilizing multiple nodes for processing, you may distribute the workload per snapshot across your cluster.
# No dependency between snapshots.

snapshot_file=pattern_aware_filtering/utils/snapshots.txt
while IFS= read -r snapshot; do
    process_snapshot "$snapshot"
done < "$snapshot_file"
