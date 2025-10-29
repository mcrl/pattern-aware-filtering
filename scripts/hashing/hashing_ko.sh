#!/bin/bash

# For utilizing multiple nodes for processing, you may distribute the workload per snapshot across your cluster.
# No dependency between snapshots.
snapshot_file=pattern_aware_filtering/utils/snapshots.txt
while IFS= read -r snapshot; do
    python scripts/hashing/sharded_hashing_ko.py --snapshot "$snapshot"
done < "$snapshot_file"

# This script merges all hashbins for all snapshots using multiprocessing
python scripts/hashing/merge_each_shard_ko.py