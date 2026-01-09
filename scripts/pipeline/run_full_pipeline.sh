#!/bin/bash
#
# Full Preprocessing Pipeline
#
# Runs the complete data preprocessing pipeline:
# 1. RefinedWeb Heuristic Filtering
# 2. BFF Deduplication
# 3. FastText Quality Filtering
#
# Note: This assumes PLD/PTF extraction has already been completed.
#       For PLD/PTF extraction, see scripts/extract-en/ or scripts/extract-ko/
#
# Usage:
#   bash run_full_pipeline.sh INPUT_DIR OUTPUT_BASE_DIR MODEL_PATH
#
# Example:
#   bash run_full_pipeline.sh /data/pld_ptf_extracted /data/processed /models/dclm_fasttext.bin
#

set -e

# Configuration
INPUT_DIR="${1:?Usage: $0 INPUT_DIR OUTPUT_BASE_DIR MODEL_PATH}"
OUTPUT_BASE="${2:?Usage: $0 INPUT_DIR OUTPUT_BASE_DIR MODEL_PATH}"
MODEL_PATH="${3:?Usage: $0 INPUT_DIR OUTPUT_BASE_DIR MODEL_PATH}"

# Environment
BFF_PATH="${BFF_PATH:-/path/to/dclm/dedup/bff}"
CONDA_ENV="${CONDA_ENV:-korcc-dev}"

# Output directories
RW_OUTPUT="${OUTPUT_BASE}/01_refinedweb"
BFF_OUTPUT="${OUTPUT_BASE}/02_bff_deduped"
FASTTEXT_OUTPUT="${OUTPUT_BASE}/03_fasttext_filtered"

# Create log directory
LOG_DIR="${OUTPUT_BASE}/logs"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Full Preprocessing Pipeline"
echo "============================================================"
echo "Input:           $INPUT_DIR"
echo "RefinedWeb Out:  $RW_OUTPUT"
echo "BFF Output:      $BFF_OUTPUT"
echo "FastText Output: $FASTTEXT_OUTPUT"
echo "Model:           $MODEL_PATH"
echo "============================================================"
echo ""

# Get script directory
SCRIPT_DIR="$(dirname $(realpath $0))"

# Step 1: RefinedWeb Heuristic Filtering
echo "[1/3] Running RefinedWeb filtering..."
bash "${SCRIPT_DIR}/run_refinedweb.sh" "$INPUT_DIR" "$RW_OUTPUT" \
    2>&1 | tee "${LOG_DIR}/refinedweb.log"
echo "      RefinedWeb filtering completed."
echo ""

# Step 2: BFF Deduplication
echo "[2/3] Running BFF deduplication..."
BFF_PATH="$BFF_PATH" bash "${SCRIPT_DIR}/run_bff_dedup.sh" "$RW_OUTPUT" "$BFF_OUTPUT" \
    2>&1 | tee "${LOG_DIR}/bff_dedup.log"
echo "      BFF deduplication completed."
echo ""

# Step 3: FastText Quality Filtering
echo "[3/3] Running FastText filtering..."
bash "${SCRIPT_DIR}/run_fasttext_filter.sh" "$BFF_OUTPUT" "$FASTTEXT_OUTPUT" "$MODEL_PATH" \
    2>&1 | tee "${LOG_DIR}/fasttext.log"
echo "      FastText filtering completed."
echo ""

echo "============================================================"
echo "Pipeline Completed"
echo "============================================================"
echo "Output directories:"
echo "  [1] RefinedWeb: $RW_OUTPUT"
echo "  [2] BFF Dedup:  $BFF_OUTPUT"
echo "  [3] FastText:   $FASTTEXT_OUTPUT"
echo ""
echo "Logs in: $LOG_DIR"
echo "============================================================"
