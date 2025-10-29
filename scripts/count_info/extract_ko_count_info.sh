#!/bin/bash

function process_snapshot() {
    local snapshot=$1
    # Set the number of workers accordingly to the size of system RAM.
    python scripts/count_info/korean_extract_count_info.py \
        --snapshot "$snapshot" \
        
}

# For utilizing multiple nodes for processing, you may distribute the workload per snapshot across your cluster.
# No dependency between snapshots.
snapshots_file=pattern_aware_filtering/utils/snapshots.txt
while IFS= read -r snapshot; do
    process_snapshot "$snapshot"
done < "$snapshots_file"