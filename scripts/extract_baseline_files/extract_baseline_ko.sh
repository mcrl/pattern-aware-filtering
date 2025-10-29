#!/bin/bash

# For utilizing multiple nodes for processing, you may distribute the workload per snapshot across your cluster.

snapshot_file=pattern_aware_filtering/utils/snapshots.txt

while IFS= read -r snapshot; do
    python scripts/extract_baseline_files/extract_baseline_ko.py --snapshot "$snapshot"
done < "$snapshot_file"