import os
import json
import re
import multiprocessing as mp
from io import BufferedReader, BufferedWriter
from argparse import ArgumentParser
from random import random
from tqdm import tqdm
from warcio.archiveiterator import ArchiveIterator
import gzip

from pattern_aware_filtering.utils.langdetect import detect_en
from pattern_aware_filtering.utils.normalizer import normalize_data
from pattern_aware_filtering.utils.constants import CCWET_PATH, EN_CCWET_PATH

# ------------------------------------------------------------------------------------
# Constants and Patterns
# ------------------------------------------------------------------------------------
CHUNK_SIZE = 1024 * 1024 * 1024


VARIOUS_WHITESPACES = {
    " ",
    "	",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    "　",
    "​",
    "‌",
    "‍",
    "⁠",
    "￼",
    "",
}
punctuation_unicode = {
    "，": ",",
    "。": ".",
    "、": ",",
    "„": '"',
    "”": '"',
    "“": '"',
    "«": '"',
    "»": '"',
    "１": '"',
    "」": '"',
    "「": '"',
    "《": '"',
    "》": '"',
    "´": "'",
    "∶": ":",
    "：": ":",
    "？": "?",
    "！": "!",
    "（": "(",
    "）": ")",
    "；": ";",
    "–": "-",
    "—": " - ",
    "．": ". ",
    "～": "~",
    "’": "'",
    "…": "...",
    "━": "-",
    "〈": "<",
    "〉": ">",
    "【": "[",
    "】": "]",
    "％": "%",
    "►": "-",
}
wsp_pttn = re.compile("|".join(VARIOUS_WHITESPACES))


one_char_map = {k: v for k, v in punctuation_unicode.items() if len(v) == 1}
multi_char_map = {k: v for k, v in punctuation_unicode.items() if len(v) != 1}
translation = str.maketrans(one_char_map)
    

# ------------------------------------------------------------------------------------
# Main Extraction Function
# ------------------------------------------------------------------------------------
def save_english_only(args):
    """
    Reads a WET file with warcio, filters English text, and writes out JSON lines.
    Each line contains the WARC headers and the extracted text.
    """
    wet_path, save_path = args
    cnt = 0  # number of documents that passed the filter
    ttl = 0  # total conversion records

    if os.path.exists(save_path):
        print(f"File {save_path} already exists. Skipping.")
        return 0, 0

    # we need to maintain information: in a csv file, we log wether the document is English or not
    csv_path = save_path.replace(".json.gz", ".csv")
    csv_temp_path = csv_path.replace(".csv", ".csv.tmp")
    temp_path = save_path.replace(".json.gz", ".tmp")
    with gzip.open(wet_path, "r") as stream, gzip.open(temp_path, "wt") as fout, open(csv_temp_path, "w") as csv_out:
        csv_out.write("idx,english\n")
        for record in ArchiveIterator(stream):
            # Only process conversion records
            if record.rec_type != "conversion":
                continue
            ttl += 1

            # Read the content
            content = record.content_stream().read().decode("utf-8")

            # Apply language filter
            if not detect_en(content):
                csv_out.write(f"{ttl},0\n")
                continue
            csv_out.write(f"{ttl},1\n")

            # Remove weird whitespaces / unify punctuation
            content = normalize_data(content)
            
            headers = record.rec_headers.headers
            # Collect WARC headers
            entry = {}
            for k, v in headers:
                if k == "WARC-Identified-Content-Language":
                    continue
                entry[k] = v

            # Insert text
            entry["text"] = content

            # Write as JSON line
            l = json.dumps(entry, ensure_ascii=False) + "\n"
            fout.write(l)
            cnt += 1
    # rename temp file to final file
    if os.path.exists(save_path):
        os.remove(save_path)
    os.rename(temp_path, save_path)
    
    if os.path.exists(csv_path):
        os.remove(csv_path)
    os.rename(csv_temp_path, csv_path)
    return ttl, cnt


# ------------------------------------------------------------------------------------
# Main Script
# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Example: read a list of WET files, produce parallel extraction
    # -------------------------------------------------------------------------
    # Suppose you have a text file with lines of .warc.wet.gz paths
    # that you want to process, like `sampled_english_shards.txt`
    parser = ArgumentParser()
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=mp.cpu_count())
    parser.add_argument("--mode", type=str, default="train", choices=["train", "valid"])
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()

    print(f"Processing mode: {args.mode}, index: {args.index}, limit: {args.limit}")
    print(f"Using {args.num_workers} workers for processing.")
    
    
    json_path = "wet_paths_en.json"
    if not os.path.exists(json_path):
        print("Did you run `scripts/prepare-ccnet/sample_splits.py` first?")
        raise FileNotFoundError(f"{json_path} not found.")
    with open(json_path, "r") as f:
        entries = json.load(f)
    
    lines = entries[args.mode][args.index]
    if args.limit > 0:
        lines = lines[:args.limit]
    
    
    # Convert these lines into actual local paths
    # Example: replacing "crawl-data" with your local directory
    srcs = [l.replace("crawl-data", CCWET_PATH) for l in lines]
    targets = [
        l.replace("crawl-data", EN_CCWET_PATH).replace(".warc.wet.gz", ".json.gz")
        for l in lines
    ]

    # Ensure output directories exist
    dirs = set(os.path.dirname(t) for t in targets)
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    # -------------------------------------------------------------------------
    # Parallel processing
    # -------------------------------------------------------------------------
    total_records = 0
    saved_records = 0

    # Use all CPU cores
    with mp.Pool(args.num_workers) as pool:
        # Run save_english_only in parallel
        res = pool.imap_unordered(save_english_only, zip(srcs, targets))

        for r in tqdm(res, total=len(srcs), desc="Processing WET files"):
            if r is not None:
                ttl, cnt = r
                total_records += ttl
                saved_records += cnt

    print(f"\nProcessed {total_records} conversion records total.")
    print(f"Saved {saved_records} English-only records.")