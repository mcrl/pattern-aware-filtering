#!/bin/bash
#
# BFF (Bloom Filter Fuzzy) Deduplication
#
# Uses DCLM's BFF tool for fuzzy deduplication at both paragraph and document level.
# This is typically Stage 3 of the pipeline, after RefinedWeb heuristic filtering.
#
# Pipeline Order: [PLD/PTF Extraction] -> [RefinedWeb Heuristics] -> [BFF Dedup] -> [FastText Quality]
#
# Prerequisites:
#   - BFF tool compiled: https://github.com/mlfoundations/dclm/tree/main/dedup/bff
#   - Build with: cd dclm/dedup/bff && cargo build --release
#
# Usage:
#   bash run_bff_dedup.sh INPUT_DIR OUTPUT_DIR
#
# BFF Parameters (DCLM defaults):
#   - Expected N-grams: 100B (100,000,000,000)
#   - False positive rate: 1%
#   - N-gram size: 13
#   - Filtering threshold: 80% overlap
#   - Remove type: old-both (paragraph + document level)
#

set -e

echo "============================================================"
echo "BFF Deduplication"
echo "============================================================"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "CPUs: $(nproc)"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "============================================================"

# Configuration - UPDATE THESE PATHS
INPUT_DIR="${1:-/path/to/input}"
OUTPUT_DIR="${2:-/path/to/output}"
BFF_PATH="${BFF_PATH:-/path/to/dclm/dedup/bff}"

# BFF Parameters (matching DCLM paper recommendations)
EXPECTED_NGRAM_COUNT="${EXPECTED_NGRAM_COUNT:-100000000000}"  # 100B n-grams
FP_RATE="${FP_RATE:-0.01}"                                    # 1% false positive rate
MIN_NGRAM_SIZE="${MIN_NGRAM_SIZE:-13}"                        # Minimum n-gram size
MAX_NGRAM_SIZE="${MAX_NGRAM_SIZE:-13}"                        # Maximum n-gram size
FILTERING_THRESHOLD="${FILTERING_THRESHOLD:-0.8}"             # 80% overlap threshold
REMOVE_TYPE="${REMOVE_TYPE:-old-both}"                        # Remove paragraph + document duplicates
THREADS="${THREADS:-64}"                                      # Number of threads

# Derived paths
BLOOM_FILTER_FILE="${OUTPUT_DIR}/bloom_filter.bin"

# Check input
if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Input directory not found: $INPUT_DIR"
    echo "Usage: bash run_bff_dedup.sh INPUT_DIR OUTPUT_DIR"
    exit 1
fi

# Check BFF binary
if [ ! -f "$BFF_PATH/target/release/bff" ]; then
    echo "ERROR: BFF binary not found at $BFF_PATH/target/release/bff"
    echo "Please build BFF first: cd $BFF_PATH && cargo build --release"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname $BLOOM_FILTER_FILE)"
mkdir -p logs

# Count input files
input_count=$(find "$INPUT_DIR" -name "*.json.gz" | wc -l)
echo "Input files: $input_count"

if [ "$input_count" -eq 0 ]; then
    echo "ERROR: No json.gz files found in $INPUT_DIR"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Input:              $INPUT_DIR"
echo "  Output:             $OUTPUT_DIR"
echo "  Bloom Filter:       $BLOOM_FILTER_FILE"
echo "  Expected N-grams:   $EXPECTED_NGRAM_COUNT"
echo "  FP Rate:            $FP_RATE"
echo "  N-gram Size:        $MIN_NGRAM_SIZE-$MAX_NGRAM_SIZE"
echo "  Filtering Threshold:$FILTERING_THRESHOLD"
echo "  Remove Type:        $REMOVE_TYPE"
echo "  Threads:            $THREADS"
echo ""

# Run BFF
cd "$BFF_PATH"
echo "Starting BFF deduplication..."

./target/release/bff bff \
    --inputs "$INPUT_DIR" \
    --output-directory "$OUTPUT_DIR" \
    --bloom-filter-file "$BLOOM_FILTER_FILE" \
    --expected-ngram-count "$EXPECTED_NGRAM_COUNT" \
    --fp-rate "$FP_RATE" \
    --min-ngram-size "$MIN_NGRAM_SIZE" \
    --max-ngram-size "$MAX_NGRAM_SIZE" \
    --filtering-threshold "$FILTERING_THRESHOLD" \
    --remove-type "$REMOVE_TYPE" \
    --threads "$THREADS" \
    --no-progress-bar

# Verify output
echo ""
echo "============================================================"
echo "BFF Deduplication Completed"
echo "============================================================"
output_count=$(find "$OUTPUT_DIR" -name "*.json.gz" | wc -l)
echo "Input files:  $input_count"
echo "Output files: $output_count"
echo "Time: $(date)"
echo "============================================================"
