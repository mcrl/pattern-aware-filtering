#!/usr/bin/env python3
"""
DCLM FastText Quality Filter

Applies the DCLM fastText classifier (OH2.5 + ELI5) to filter documents.
The model classifies between __label__hq (high-quality) and __label__cc (low-quality).

Usage:
    python -m pattern_aware_filtering.filtering.quality_classifier --input_dir INPUT --output_dir OUTPUT [--threshold 0.018112]

Reference:
    - Model: mlfoundations/fasttext-oh-eli5
    - Default threshold: 0.018112 (from DCLM paper)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Generator
import multiprocessing as mp
from functools import partial

try:
    import fasttext
except ImportError:
    fasttext = None
    print("WARNING: fasttext not installed. Install with: pip install fasttext")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# Default threshold from DCLM paper
DEFAULT_THRESHOLD = 0.018112

# Default model path (can be overridden)
DEFAULT_MODEL_PATH = None


def load_model(model_path: str):
    """Load the fastText model."""
    if fasttext is None:
        raise ImportError("fasttext is required. Install with: pip install fasttext")
    print(f"Loading fastText model from {model_path}...")
    model = fasttext.load_model(model_path)
    print("Model loaded successfully.")
    return model


def preprocess_text(text: str) -> str:
    """Preprocess text for fastText prediction."""
    # Replace newlines with spaces (fastText expects single-line input)
    return text.replace('\n', ' ').replace('\r', ' ')


def predict_quality(model, text: str) -> float:
    """
    Predict quality score for a document.
    Returns the probability of __label__hq (high-quality).
    """
    processed = preprocess_text(text)
    predictions = model.predict(processed, k=2)
    labels, scores = predictions

    # Find the score for __label__hq
    for label, score in zip(labels, scores):
        if label == '__label__hq':
            return score

    # If __label__hq not in top-2, score is effectively 0
    return 0.0


def process_file(
    input_path: str,
    output_path: str,
    model_path: str,
    threshold: float,
    verbose: bool = False
) -> dict:
    """
    Process a single JSONL file and filter documents.
    Returns statistics about the filtering.
    """
    model = load_model(model_path)

    stats = {
        'input_file': input_path,
        'total': 0,
        'kept': 0,
        'filtered': 0,
        'avg_score_kept': 0.0,
        'avg_score_filtered': 0.0
    }

    kept_scores = []
    filtered_scores = []

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        for line in infile:
            stats['total'] += 1

            try:
                doc = json.loads(line.strip())
                text = doc.get('text', '')

                if not text:
                    stats['filtered'] += 1
                    continue

                score = predict_quality(model, text)

                if score >= threshold:
                    # Add score to document for reference
                    doc['fasttext_score'] = score
                    outfile.write(json.dumps(doc, ensure_ascii=False) + '\n')
                    stats['kept'] += 1
                    kept_scores.append(score)
                else:
                    stats['filtered'] += 1
                    filtered_scores.append(score)

            except json.JSONDecodeError:
                stats['filtered'] += 1
                continue

    # Calculate average scores
    if kept_scores:
        stats['avg_score_kept'] = sum(kept_scores) / len(kept_scores)
    if filtered_scores:
        stats['avg_score_filtered'] = sum(filtered_scores) / len(filtered_scores)

    return stats


def process_file_wrapper(args):
    """Wrapper for multiprocessing."""
    input_path, output_path, model_path, threshold = args
    return process_file(input_path, output_path, model_path, threshold)


def process_directory(
    input_dir: str,
    output_dir: str,
    model_path: str,
    threshold: float,
    num_workers: int = 1,
    file_pattern: str = "*.jsonl"
) -> dict:
    """
    Process all JSONL files in a directory.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all input files
    input_files = list(input_path.glob(file_pattern))
    print(f"Found {len(input_files)} files to process")

    if not input_files:
        print(f"No files matching {file_pattern} found in {input_dir}")
        return {}

    # Prepare arguments for processing
    args_list = []
    for infile in input_files:
        outfile = output_path / infile.name
        args_list.append((str(infile), str(outfile), model_path, threshold))

    # Process files
    all_stats = []

    if num_workers == 1:
        # Single process
        for args in tqdm(args_list, desc="Processing files"):
            stats = process_file(*args)
            all_stats.append(stats)
    else:
        # Multiprocessing
        with mp.Pool(num_workers) as pool:
            all_stats = list(tqdm(
                pool.imap_unordered(process_file_wrapper, args_list),
                total=len(args_list),
                desc="Processing files"
            ))

    # Aggregate statistics
    total_stats = {
        'total_files': len(all_stats),
        'total_docs': sum(s['total'] for s in all_stats),
        'total_kept': sum(s['kept'] for s in all_stats),
        'total_filtered': sum(s['filtered'] for s in all_stats),
        'keep_rate': 0.0
    }

    if total_stats['total_docs'] > 0:
        total_stats['keep_rate'] = total_stats['total_kept'] / total_stats['total_docs']

    return total_stats


def main():
    parser = argparse.ArgumentParser(
        description="DCLM FastText Quality Filter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--input_dir', '-i',
        required=True,
        help='Input directory containing JSONL files'
    )
    parser.add_argument(
        '--output_dir', '-o',
        required=True,
        help='Output directory for filtered JSONL files'
    )
    parser.add_argument(
        '--model_path', '-m',
        required=True,
        help='Path to fastText model (e.g., dclm_fasttext.bin)'
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f'Quality score threshold (default: {DEFAULT_THRESHOLD})'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=1,
        help='Number of worker processes (default: 1)'
    )
    parser.add_argument(
        '--pattern', '-p',
        default='*.jsonl',
        help='File pattern to match (default: *.jsonl)'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("DCLM FastText Quality Filter")
    print("=" * 60)
    print(f"Input directory:  {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model path:       {args.model_path}")
    print(f"Threshold:        {args.threshold}")
    print(f"Workers:          {args.workers}")
    print("=" * 60)

    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model not found at {args.model_path}")
        sys.exit(1)

    # Process
    stats = process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        threshold=args.threshold,
        num_workers=args.workers,
        file_pattern=args.pattern
    )

    # Print results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Total files processed: {stats.get('total_files', 0)}")
    print(f"Total documents:       {stats.get('total_docs', 0)}")
    print(f"Documents kept:        {stats.get('total_kept', 0)}")
    print(f"Documents filtered:    {stats.get('total_filtered', 0)}")
    print(f"Keep rate:             {stats.get('keep_rate', 0):.2%}")
    print("=" * 60)


if __name__ == '__main__':
    main()
