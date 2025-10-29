#!/bin/bash

function process_snapshot() {
    local snapshot=$1
    python scripts/extract-ko/korean_pld_ablation.py \
        --snapshot "$snapshot" \
        --num-workers 64
}

# For utilizing multiple nodes for processing, you may distribute the workload per snapshot across your cluster.
# No dependency between snapshots.

snapshot_file=pattern_aware_filtering/utils/snapshots.txt
while IFS= read -r snapshot; do
    process_snapshot "$snapshot"
done < "$snapshot_file"
