#!/bin/bash

function process_snapshot() {
    local snapshot=$1
    # Set the number of workers accordingly to the size of system RAM.
    python scripts/count_info/english_extract_count_info.py \
        --snapshot "$snapshot" \
        --num-workers 8 \
        --train-start 0 \
        --train-end 500 \
        --valid-start 0 \
        --valid-end 5
}

# For utilizing multiple nodes for processing, you may distribute the workload per snapshot across your cluster.
# No dependency between snapshots.
snapshots_file=pattern_aware_filtering/utils/snapshots.txt
while IFS= read -r snapshot; do
    process_snapshot "$snapshot"
done < "$snapshots_file"