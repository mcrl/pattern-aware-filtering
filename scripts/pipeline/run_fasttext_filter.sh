#!/bin/bash
#
# FastText Quality Filtering
#
# Applies DCLM fastText classifier (OH2.5 + ELI5) to filter documents by quality.
# This is typically Stage 4 (final stage) of the pipeline.
#
# Pipeline Order: [PLD/PTF Extraction] -> [RefinedWeb Heuristics] -> [BFF Dedup] -> [FastText Quality]
#
# Model:
#   - DCLM fastText classifier: mlfoundations/fasttext-oh-eli5
#   - Download: huggingface-cli download mlfoundations/fasttext-oh-eli5 --local-dir ./models
#
# Default threshold: 0.018112 (from DCLM paper, ~27% kept ratio)
#
# Usage:
#   bash run_fasttext_filter.sh INPUT_DIR OUTPUT_DIR MODEL_PATH
#

set -e

echo "============================================================"
echo "FastText Quality Filtering"
echo "============================================================"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "============================================================"

# Configuration - UPDATE THESE PATHS
INPUT_DIR="${1:-/path/to/input}"
OUTPUT_DIR="${2:-/path/to/output}"
MODEL_PATH="${3:-/path/to/dclm_fasttext.bin}"
CONDA_ENV="${CONDA_ENV:-korcc-dev}"

# DCLM default threshold (~27% kept ratio)
THRESHOLD="${THRESHOLD:-0.018112}"
WORKERS="${WORKERS:-64}"

# Environment setup
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
fi

# Check input
if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Input directory not found: $INPUT_DIR"
    echo "Usage: bash run_fasttext_filter.sh INPUT_DIR OUTPUT_DIR MODEL_PATH"
    exit 1
fi

# Check model
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: FastText model not found: $MODEL_PATH"
    echo "Download from: huggingface-cli download mlfoundations/fasttext-oh-eli5"
    exit 1
fi

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

echo ""
echo "Configuration:"
echo "  Input:     $INPUT_DIR"
echo "  Output:    $OUTPUT_DIR"
echo "  Model:     $MODEL_PATH"
echo "  Threshold: $THRESHOLD"
echo "  Workers:   $WORKERS"
echo ""

# If input is json.gz, convert to JSONL first
input_format=$(find "$INPUT_DIR" -maxdepth 1 -name "*.json.gz" | head -1)
if [ -n "$input_format" ]; then
    echo "[Step 1] Converting json.gz to JSONL..."
    JSONL_DIR="${OUTPUT_DIR}_jsonl_tmp"
    mkdir -p "$JSONL_DIR"

    # Find all json.gz files and convert
    find "$INPUT_DIR" -name "*.json.gz" | while read gz_file; do
        rel_path="${gz_file#$INPUT_DIR/}"
        flat_name=$(echo "$rel_path" | tr '/' '_' | sed 's/.json.gz$/.jsonl/')
        output_file="$JSONL_DIR/$flat_name"

        if [ ! -f "$output_file" ]; then
            zcat "$gz_file" >> "$output_file"
        fi
    done

    jsonl_count=$(ls "$JSONL_DIR"/*.jsonl 2>/dev/null | wc -l)
    echo "JSONL files created: $jsonl_count"
    FILTER_INPUT="$JSONL_DIR"
else
    FILTER_INPUT="$INPUT_DIR"
fi

# Run FastText filtering
echo ""
echo "[Step 2] Running FastText filtering..."

python -m pattern_aware_filtering.filtering.quality_classifier \
    --input_dir "$FILTER_INPUT" \
    --output_dir "$OUTPUT_DIR" \
    --model_path "$MODEL_PATH" \
    --threshold "$THRESHOLD" \
    --workers "$WORKERS"

# Cleanup temp JSONL directory
if [ -n "$input_format" ] && [ -d "$JSONL_DIR" ]; then
    echo "Cleaning up temporary JSONL directory..."
    rm -rf "$JSONL_DIR"
fi

# Verify output
echo ""
echo "============================================================"
echo "FastText Filtering Completed"
echo "============================================================"
output_count=$(ls "$OUTPUT_DIR"/*.jsonl 2>/dev/null | wc -l)
echo "Output files: $output_count"
echo "Time: $(date)"
echo "============================================================"
