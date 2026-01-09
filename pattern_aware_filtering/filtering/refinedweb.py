#!/usr/bin/env python3
"""
RefinedWeb-style heuristic filtering for JSONL/JSON.GZ corpora.

This module applies the RefinedWeb heuristic filters (EXCEPT URL-based filters)
from the DCLM baselines pipeline. It is designed for multi-node parallel processing
without using Ray, using file-level sharding.

Applied filters (from refinedweb.yaml):
1. page_length_filter: 50-100k words
2. word_length_filter: avg 3-10 chars
3. symbol_ratio_filter: max 0.1 symbol-to-word ratio
4. bullet_count_filter: max 0.9 bullet line ratio
5. ellipsis_count_filter: max 0.3 ellipsis line ratio
6. alphabetic_word_ratio_filter: max 0.2 non-alphabetic word ratio
7. stop_word_filter: min 2 stop words
8. massive_web_repetition_filters: various n-gram and line/paragraph repetition filters

Applied modifiers (text cleaning):
1. newline_removal_modifier: max 2 consecutive newlines
2. uppercase_ratio_line_modifier: remove lines with >0.5 uppercase ratio
3. numeric_ratio_line_modifier: remove lines with >0.999999 numeric ratio
4. counter_line_modifier: remove lines with counter patterns
5. line_length_modifier: remove lines shorter than 2 chars
6. substring_line_modifier: remove boilerplate phrases

Final filter:
- word_removal_ratio_filter: if >5% words removed, discard document

Usage:
    # Single file processing
    python -m pattern_aware_filtering.filtering.refinedweb --input_dir /path/to/input --output_dir /path/to/output

    # Multi-node sharding (for SLURM array jobs)
    python -m pattern_aware_filtering.filtering.refinedweb --input_dir /path/to/input --output_dir /path/to/output \\
        --shard_id 0 --total_shards 16

Note: URL-based filters are intentionally excluded since we're working with
already filtered PLD+PTF / LD+TF data.
"""

import argparse
import gzip
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp


# ==============================================================================
# Core utilities (adapted from dclm/baselines/mappers/core_utils.py)
# ==============================================================================

def split_words(text: str, model: str = 'fasttext', ignore_punctuation: bool = False,
                ignore_whitespace: bool = True) -> List[str]:
    """
    Split text into words.

    Args:
        text: The text to split.
        model: The tokenizer model to use. Options: 'fasttext', 'uniseg', 'split'.
               Default is 'fasttext' to match DCLM.
        ignore_punctuation: Whether to ignore punctuation tokens.
        ignore_whitespace: Whether to ignore whitespace tokens.

    Returns:
        List of word tokens.
    """
    if model == 'fasttext':
        try:
            import fasttext
            tokens = fasttext.FastText.tokenize(text)
        except ImportError:
            # Fallback to uniseg if fasttext not available
            try:
                import uniseg.wordbreak
                tokens = list(uniseg.wordbreak.words(text))
            except ImportError:
                tokens = text.split()
    elif model == 'uniseg':
        try:
            import uniseg.wordbreak
            tokens = list(uniseg.wordbreak.words(text))
        except ImportError:
            tokens = text.split()
    elif model == 'split':
        tokens = text.split()
    else:
        raise ValueError(f"Unknown word tokenizer: {model}")

    # Filter tokens based on options (matching DCLM's logic)
    if ignore_punctuation and ignore_whitespace:
        return [w for w in tokens if w[0].isalnum()]
    elif ignore_punctuation:
        return [w for w in tokens if w[0].isalnum() or w[0].isspace()]
    elif ignore_whitespace:
        return [w for w in tokens if w.strip()]
    else:
        return list(tokens)


def split_paragraphs(text: str, paragraph_end: str = '\n\n', remove_empty: bool = True) -> List[str]:
    """Split text into paragraphs."""
    paragraphs = text.split(paragraph_end)
    if remove_empty:
        paragraphs = [p for p in paragraphs if p.strip()]
    return paragraphs


# ==============================================================================
# Filters (adapted from dclm/baselines/mappers/filters/content_filters.py)
# ==============================================================================

def page_length_filter(text: str, min_length: int = 50, max_length: int = 100000,
                       ignore_punctuation: bool = True) -> bool:
    """
    Filter based on word count.

    Uses fasttext tokenizer by default to match DCLM's behavior.
    YAML config passes: length_type=word, min_length=50, max_length=100000, ignore_punctuation=True
    """
    words = split_words(text, model='fasttext', ignore_punctuation=ignore_punctuation)
    word_count = len(words)
    return min_length <= word_count <= max_length


def word_length_filter(text: str, min_length: float = 3, max_length: float = 10) -> bool:
    """Filter based on average word length."""
    words = text.split()
    if not words:
        return False
    avg_word_length = sum(len(w) for w in words) / len(words)
    return min_length <= avg_word_length <= max_length


def symbol_ratio_filter(text: str, max_symbol_to_word_ratio: float = 0.1) -> bool:
    """Filter based on symbol-to-word ratio."""
    SYMBOLS = ["#", "...", ". . .", "\u2026"]
    number_of_symbols = sum(text.count(sym) for sym in SYMBOLS)
    number_of_words = len(text.split())
    if number_of_words == 0:
        return False
    return number_of_symbols / number_of_words <= max_symbol_to_word_ratio


def bullet_count_filter(text: str, max_bullet_start_ratio: float = 0.9) -> bool:
    """
    Filter based on ratio of lines starting with bullets.

    DCLM uses split_paragraphs(paragraph_end='\n') and checks line.startswith(bullet)
    without stripping whitespace.
    """
    lines = split_paragraphs(text, paragraph_end='\n', remove_empty=False)
    if not lines:
        return True
    max_bullet_count = max_bullet_start_ratio * len(lines)
    bullet_count = sum(1 for line in lines if any(line.startswith(b) for b in ['●', '•', '*', '-']))
    return bullet_count <= max_bullet_count


def ellipsis_count_filter(text: str, max_ellipsis_end_ratio: float = 0.3) -> bool:
    """
    Filter based on ratio of lines ending with ellipsis.

    DCLM uses split_paragraphs(paragraph_end='\n') and checks line.endswith(ellipsis)
    without stripping whitespace.
    """
    lines = split_paragraphs(text, paragraph_end='\n', remove_empty=False)
    if not lines:
        return True
    max_ellipsis_count = max_ellipsis_end_ratio * len(lines)
    ellipsis_count = sum(1 for line in lines if any(line.endswith(e) for e in ['...', '. . .', '\u2026']))
    return ellipsis_count <= max_ellipsis_count


def alphabetic_word_ratio_filter(text: str, max_ratio: float = 0.2) -> bool:
    """Filter based on ratio of non-alphabetic words."""
    words = text.split()
    if not words:
        return False
    non_alpha_count = sum(1 for w in words if not any(c.isalpha() for c in w))
    return non_alpha_count / len(words) <= max_ratio


def stop_word_filter(text: str, min_stop_word: int = 2, count_unique: bool = False) -> bool:
    """Filter based on minimum number of stop words."""
    stop_words = {'the', 'be', 'to', 'of', 'and', 'that', 'have', 'with'}
    words = text.lower().split()

    if count_unique:
        found = set(w for w in words if w in stop_words)
        return len(found) >= min_stop_word
    else:
        count = sum(1 for w in words if w in stop_words)
        return count >= min_stop_word


def repetition_filter(text: str, granularity: Union[str, int], max_fraction: float,
                      count_characters: bool = True, cache: Optional[Dict] = None) -> bool:
    """Filter based on repetition at various granularities."""
    if not text:
        return False

    if cache is None:
        cache = {}

    if isinstance(granularity, str):
        sep = '\n\n' if granularity == 'paragraph' else '\n'

        if granularity not in cache:
            cache[granularity] = segments = split_paragraphs(text, paragraph_end=sep, remove_empty=True)
        else:
            segments = cache[granularity]

        if len(segments) <= 1:
            return True

        if granularity + '/count' not in cache:
            cache[granularity + '/chars'] = total_chars = sum(len(s) for s in segments)
            cache[granularity + '/count'] = segment_counts = Counter(segments)
        else:
            total_chars = cache[granularity + '/chars']
            segment_counts = cache[granularity + '/count']

        if count_characters:
            repeated_fraction = sum(len(seg) * cnt for seg, cnt in segment_counts.items() if cnt > 1) / max(total_chars, 1)
        else:
            repeated_fraction = sum(cnt for cnt in segment_counts.values() if cnt > 1) / len(segments)

        return repeated_fraction <= max_fraction

    elif isinstance(granularity, int):
        if 'words' not in cache:
            cache['words'] = words = split_words(text, ignore_punctuation=True, model='uniseg')
            cache['words/chars'] = total_chars = sum(len(w) for w in words)
        else:
            words = cache['words']
            total_chars = cache['words/chars']

        if len(words) < granularity:
            return True

        # Create n-grams
        n_grams = [tuple(words[i:i+granularity]) for i in range(len(words) - granularity + 1)]
        if not n_grams:
            return True

        ngram_counts = Counter(n_grams)
        ordered_counts = ngram_counts.most_common()
        most_common_ngram, most_common_count = ordered_counts[0]

        if most_common_count == 1:
            return True

        # For n-grams 2-4, use most_common; for 5-10, use all repeated
        if granularity in {2, 3, 4}:
            most_common_length = sum(len(w) for w in most_common_ngram)
            for ngram, count in ordered_counts:
                if count != most_common_count:
                    break
                ngram_length = sum(len(w) for w in ngram)
                most_common_length = max(ngram_length, most_common_length)
            repeated_fraction = (most_common_length * most_common_count) / max(total_chars, 1)
        else:
            repeated_word_indices = set()
            for idx, ngram in enumerate(n_grams):
                if ngram_counts[ngram] > 1:
                    repeated_word_indices.update(range(idx, idx + granularity))
            repeated_char_count = sum(len(words[i]) for i in repeated_word_indices if i < len(words))
            repeated_fraction = repeated_char_count / max(total_chars, 1)

        return repeated_fraction <= max_fraction

    return True


def massive_web_repetition_filters(text: str, skip_paragraph: bool = False) -> bool:
    """Apply the full set of Gopher-style repetition filters."""
    cache = {}

    # Line and paragraph duplicate filters
    if not repetition_filter(text, "line", 0.3, count_characters=False, cache=cache):
        return False
    if not skip_paragraph and not repetition_filter(text, "paragraph", 0.3, count_characters=False, cache=cache):
        return False
    if not repetition_filter(text, "line", 0.2, count_characters=True, cache=cache):
        return False
    if not skip_paragraph and not repetition_filter(text, "paragraph", 0.2, count_characters=True, cache=cache):
        return False

    # N-gram repetition filters
    ngram_thresholds = {2: 0.2, 3: 0.18, 4: 0.16, 5: 0.15, 6: 0.14, 7: 0.13, 8: 0.12, 9: 0.11, 10: 0.10}
    for n, threshold in ngram_thresholds.items():
        if not repetition_filter(text, n, threshold, cache=cache):
            return False

    return True


# ==============================================================================
# Modifiers (adapted from dclm/baselines/mappers/modifiers.py)
# ==============================================================================

def newline_removal_modifier(text: str, max_consecutive: int = 2) -> str:
    """Remove excessive consecutive newlines."""
    pattern = r'\n{' + str(max_consecutive + 1) + r',}'
    return re.sub(pattern, '\n' * max_consecutive, text)


def uppercase_ratio_line_modifier(text: str, max_ratio: float = 0.5) -> str:
    """
    Remove lines where uppercase characters exceed a certain ratio.

    DCLM uses: num_uppercase / len(line) (all characters, not just alphabetic)
    """
    lines = text.split('\n')
    filtered_lines = []
    for line in lines:
        if not line:
            filtered_lines.append(line)
            continue
        num_uppercase = sum(1 for c in line if c.isupper())
        if num_uppercase / len(line) <= max_ratio:
            filtered_lines.append(line)
    return '\n'.join(filtered_lines)


def numeric_ratio_line_modifier(text: str, max_ratio: float = 0.999999) -> str:
    """
    Remove lines if numerical characters exceed a certain ratio.

    DCLM uses: num_numeric / len(line) (all characters)
    RefinedWeb removes lines which contain 100% numerical characters.
    """
    lines = text.split('\n')
    filtered_lines = []
    for line in lines:
        if not line:
            filtered_lines.append(line)
            continue
        num_numeric = sum(1 for c in line if c.isdigit())
        if num_numeric / len(line) <= max_ratio:
            filtered_lines.append(line)
    return '\n'.join(filtered_lines)


def counter_line_modifier(text: str) -> str:
    """
    Remove lines if it is a counter (e.g. '3 likes', '1.2K shares').

    DCLM pattern matches social media engagement counters like:
    - "3 likes"
    - "1.2K shares"
    - "100 comments"
    """
    # DCLM's counter regex for social media counters
    counter_pattern = re.compile(
        r'^\W*\d(?:,|\.|\d)*(?:K|k|M|m|B|b)?\s+(?:likes|shares|comments|retweets|reposts|quotes|bookmarks|upvotes|downvotes|downloads|views|followers)\W*$'
    )

    lines = text.split('\n')
    filtered_lines = [line for line in lines if not counter_pattern.search(line.lower())]
    return '\n'.join(filtered_lines)


def line_length_modifier(text: str, min_length: int = 2, max_length: float = float('inf')) -> str:
    """
    Remove lines with word counts outside accepted range.

    DCLM counts words (len(line.split())), not characters.
    """
    lines = text.split('\n')
    filtered_lines = [
        line for line in lines
        if (min_length <= len(line.split()) <= max_length) or not line
    ]
    return '\n'.join(filtered_lines)


def substring_line_modifier(text: str, banlist: Union[str, List[str]],
                            max_length: int = 10, location: str = 'any',
                            remove_substring_only: bool = True,
                            case_sensitive: bool = False) -> str:
    """
    Remove lines that contain the given substring.

    DCLM uses word count for max_length check: len(line.split()) <= max_length
    """
    if isinstance(banlist, str):
        banlist = [banlist]

    # Build regex pattern (matching DCLM)
    banlist_lower = banlist if case_sensitive else [b.lower() for b in banlist]
    pattern_str = f"(?:{'|'.join(re.escape(b) for b in banlist_lower)})"

    if location == 'prefix':
        pattern_str = rf"^{pattern_str}\s?"
    elif location == 'suffix':
        pattern_str = rf"\s?{pattern_str}$"
    else:
        pattern_str = rf"\s?{pattern_str}"

    pattern = re.compile(pattern_str) if case_sensitive else re.compile(pattern_str, re.I)

    lines = text.split('\n')
    filtered_lines = []

    for line in lines:
        # DCLM uses word count for max_length check
        if max_length is None or len(line.split()) <= max_length:
            if remove_substring_only:
                modified_line = pattern.sub("", line)
                # If line becomes empty after removal, skip it
                if line and (not modified_line or modified_line.isspace()):
                    continue
                line = modified_line
            elif pattern.search(line if case_sensitive else line.lower()):
                continue

        filtered_lines.append(line)

    return '\n'.join(filtered_lines)


# ==============================================================================
# Main processing logic
# ==============================================================================

def apply_refinedweb_heuristics(doc: Dict) -> Optional[Dict]:
    """
    Apply RefinedWeb heuristic filters and modifiers to a document.

    Returns the modified document if it passes all filters, None otherwise.
    """
    text = doc.get('text', '')
    if not text or not isinstance(text, str):
        return None

    # ============= FILTERS (applied first to avoid unnecessary processing) =============

    # 1. page_length_filter: 50-100k words
    if not page_length_filter(text, min_length=50, max_length=100000):
        return None

    # 2. word_length_filter: avg 3-10 chars
    if not word_length_filter(text, min_length=3, max_length=10):
        return None

    # 3. symbol_ratio_filter: max 0.1
    if not symbol_ratio_filter(text, max_symbol_to_word_ratio=0.1):
        return None

    # 4. bullet_count_filter: max 0.9
    if not bullet_count_filter(text, max_bullet_start_ratio=0.9):
        return None

    # 5. ellipsis_count_filter: max 0.3
    if not ellipsis_count_filter(text, max_ellipsis_end_ratio=0.3):
        return None

    # 6. alphabetic_word_ratio_filter: max 0.2
    if not alphabetic_word_ratio_filter(text, max_ratio=0.2):
        return None

    # 7. stop_word_filter: min 2 stop words
    if not stop_word_filter(text, min_stop_word=2):
        return None

    # 8. massive_web_repetition_filters
    if not massive_web_repetition_filters(text, skip_paragraph=False):
        return None

    # ============= MODIFIERS (text cleaning) =============

    # Count words before modification for removal ratio check
    prev_word_count = len(split_words(text, ignore_punctuation=True))

    # Apply modifiers in sequence
    text = newline_removal_modifier(text, max_consecutive=2)
    text = uppercase_ratio_line_modifier(text, max_ratio=0.5)
    text = numeric_ratio_line_modifier(text, max_ratio=0.999999)
    text = counter_line_modifier(text)
    text = line_length_modifier(text, min_length=2)

    # Substring line modifiers for common boilerplate
    text = substring_line_modifier(text, banlist="items in cart", max_length=10,
                                   remove_substring_only=True)
    text = substring_line_modifier(text, banlist="Read more...", max_length=10,
                                   location='suffix', remove_substring_only=True)
    text = substring_line_modifier(text, banlist="Sign-in", max_length=10,
                                   location='prefix', remove_substring_only=True)

    # ============= WORD REMOVAL RATIO FILTER =============

    new_word_count = len(split_words(text, ignore_punctuation=True))
    if prev_word_count > 0:
        removal_ratio = (prev_word_count - new_word_count) / prev_word_count
        if removal_ratio > 0.05:  # max_removed_ratio
            return None

    # Check if text is still valid after modifications
    if not text.strip():
        return None

    # Update the document
    doc = doc.copy()
    doc['text'] = text

    return doc


def iter_documents(path: Path):
    """Yield parsed JSON objects from a JSONL or JSON.GZ file."""
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def process_file(input_path: Path, output_path: Path) -> Tuple[int, int, int]:
    """
    Process a single file, applying RefinedWeb heuristics.

    Returns: (kept_count, filtered_count, total_count)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    opener = gzip.open if output_path.suffix == '.gz' else open

    kept = 0
    filtered = 0
    total = 0

    with opener(output_path, 'wt', encoding='utf-8') as out_f:
        for doc in iter_documents(input_path):
            total += 1
            result = apply_refinedweb_heuristics(doc)
            if result is not None:
                out_f.write(json.dumps(result, ensure_ascii=False) + '\n')
                kept += 1
            else:
                filtered += 1

    return kept, filtered, total


def get_files_for_shard(all_files: List[Path], shard_id: int, total_shards: int) -> List[Path]:
    """Get the subset of files for this shard."""
    return [f for i, f in enumerate(all_files) if i % total_shards == shard_id]


def main():
    parser = argparse.ArgumentParser(
        description="Apply RefinedWeb heuristic filters (except URL filters) to a corpus."
    )
    parser.add_argument("--input_dir", required=True, type=Path,
                        help="Directory containing input JSONL/JSON.GZ files")
    parser.add_argument("--output_dir", required=True, type=Path,
                        help="Directory to write filtered files")
    parser.add_argument("--file_pattern", default="*.json.gz",
                        help="Glob pattern to match files (default: *.json.gz)")
    parser.add_argument("--shard_id", type=int, default=0,
                        help="Current shard ID (0-indexed) for multi-node processing")
    parser.add_argument("--total_shards", type=int, default=1,
                        help="Total number of shards for multi-node processing")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: number of CPUs)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip files that already exist in output directory")

    args = parser.parse_args()

    # Find all input files
    all_files = sorted(args.input_dir.rglob(args.file_pattern))
    if not all_files:
        # Also try .jsonl files
        all_files = sorted(args.input_dir.rglob("*.jsonl"))

    if not all_files:
        print(f"ERROR: No input files found in {args.input_dir}")
        return 1

    print(f"Found {len(all_files)} total files")

    # Get files for this shard
    shard_files = get_files_for_shard(all_files, args.shard_id, args.total_shards)
    print(f"Processing shard {args.shard_id}/{args.total_shards}: {len(shard_files)} files")

    if not shard_files:
        print("No files to process for this shard")
        return 0

    # Prepare output paths
    file_pairs = []
    for input_path in shard_files:
        rel_path = input_path.relative_to(args.input_dir)
        output_path = args.output_dir / rel_path

        if args.skip_existing and output_path.exists():
            continue

        file_pairs.append((input_path, output_path))

    if not file_pairs:
        print("All files already processed (skip_existing=True)")
        return 0

    print(f"Will process {len(file_pairs)} files")

    # Process files
    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_kept = 0
    total_filtered = 0
    total_docs = 0

    num_workers = args.workers or mp.cpu_count()

    if num_workers > 1 and len(file_pairs) > 1:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(process_file, inp, out): (inp, out)
                for inp, out in file_pairs
            }

            for i, future in enumerate(as_completed(futures), 1):
                inp, out = futures[future]
                try:
                    kept, filtered, total = future.result()
                    total_kept += kept
                    total_filtered += filtered
                    total_docs += total

                    keep_rate = kept / max(total, 1) * 100
                    print(f"[{i}/{len(file_pairs)}] {inp.name}: kept {kept}/{total} ({keep_rate:.1f}%)")
                except Exception as e:
                    print(f"ERROR processing {inp}: {e}")
    else:
        # Sequential processing
        for i, (inp, out) in enumerate(file_pairs, 1):
            try:
                kept, filtered, total = process_file(inp, out)
                total_kept += kept
                total_filtered += filtered
                total_docs += total

                keep_rate = kept / max(total, 1) * 100
                print(f"[{i}/{len(file_pairs)}] {inp.name}: kept {kept}/{total} ({keep_rate:.1f}%)")
            except Exception as e:
                print(f"ERROR processing {inp}: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("RefinedWeb Heuristic Filtering Complete")
    print("=" * 60)
    print(f"Shard: {args.shard_id}/{args.total_shards}")
    print(f"Files processed: {len(file_pairs)}")
    print(f"Total documents: {total_docs}")
    print(f"Kept documents: {total_kept}")
    print(f"Filtered documents: {total_filtered}")
    print(f"Keep rate: {total_kept / max(total_docs, 1) * 100:.2f}%")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
