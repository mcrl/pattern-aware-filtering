#!/bin/bash

# For utilizing multiple nodes for processing, you may distribute the workload per index across your cluster.


for i in `seq 0 499`; do
    python scripts/extract_baseline_files/extract_baseline_en.py --index $i --mode train
done
for i in `seq 0 4`; do
    python scripts/extract_baseline_files/extract_baseline_en.py --index $i --mode valid
done