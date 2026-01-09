#!/bin/bash
#
# RefinedWeb Heuristic Filtering
#
# Applies the full RefinedWeb heuristic filters (except URL filters) from DCLM.
# This is typically Stage 2 of the pipeline, after PLD/PTF extraction.
#
# Pipeline Order: [PLD/PTF Extraction] -> [RefinedWeb Heuristics] -> [BFF Dedup] -> [FastText Quality]
#
# Usage:
#   bash run_refinedweb.sh INPUT_DIR OUTPUT_DIR
#

set -e

echo "============================================================"
echo "RefinedWeb Heuristic Filtering"
echo "============================================================"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "CPUs: $(nproc)"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "============================================================"

# Configuration - UPDATE THESE PATHS
INPUT_DIR="${1:-/path/to/input}"
OUTPUT_DIR="${2:-/path/to/output}"
CONDA_ENV="${CONDA_ENV:-korcc-dev}"
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname $(dirname $(dirname $(realpath $0))))}"

# Parallelization settings
SHARD_ID="${SHARD_ID:-0}"
TOTAL_SHARDS="${TOTAL_SHARDS:-1}"
WORKERS="${WORKERS:-64}"

# Environment setup
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
fi

# Checks
if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Input directory not found: $INPUT_DIR"
    echo "Usage: bash run_refinedweb.sh INPUT_DIR OUTPUT_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
mkdir -p logs

echo ""
echo "Configuration:"
echo "  Input:        $INPUT_DIR"
echo "  Output:       $OUTPUT_DIR"
echo "  Shard:        $SHARD_ID / $TOTAL_SHARDS"
echo "  Workers:      $WORKERS"
echo "  Project Root: $PROJECT_ROOT"
echo ""

# Count input files
input_count=$(find "$INPUT_DIR" -name "*.json.gz" -o -name "*.jsonl" | wc -l)
echo "Input files: $input_count"

# Run the filtering
python -m pattern_aware_filtering.filtering.refinedweb \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --shard_id "$SHARD_ID" \
    --total_shards "$TOTAL_SHARDS" \
    --workers "$WORKERS"

# Count output
output_count=$(find "$OUTPUT_DIR" -name "*.json.gz" -o -name "*.jsonl" | wc -l)
echo ""
echo "Output files: $output_count"

echo ""
echo "============================================================"
echo "RefinedWeb Heuristic Filtering Completed"
echo "Time: $(date)"
echo "============================================================"
