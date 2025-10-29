import os
from tqdm import tqdm
from argparse import ArgumentParser
from glob import glob
import multiprocessing as mp
import json
import gzip
import numpy as np
import re
import io
import time
from functools import partial

from pattern_aware_filtering.utils.constants import PROJECT_ROOT, EN_EXTRACTED_PATH
from pattern_aware_filtering.extraction.extractor import pld_extractor

io.DEFAULT_BUFFER_SIZE = 1024 * 1024 * 1024 # Set buffer size to 1GiB

EN_BASE = EN_EXTRACTED_PATH
reds = [1000]
greens = [1]

def naming(red, green):
    return f"pld-r{red}-g{green}"


def reconstruct_path(path, base_dir):
    return path.replace("crawl-data", base_dir).replace(".warc.wet.gz", ".json.gz")

def restore_path(path, base_dir):
    return path.replace(base_dir, "crawl-data").replace(".json.gz", ".warc.wet.gz")


def get_files_to_process(snapshot, allowed_filepaths, train_start, train_end, valid_start, valid_end, limit=-1):
    info_path = f"{PROJECT_ROOT}/../wet_paths_en.json"
    with open(info_path, "r") as f:
        info = json.load(f)
    train_files = info["train"][train_start:train_end]
    val_files = info["valid"][valid_start:valid_end]
    files = []
    for file_shard in train_files + val_files:
        if isinstance(file_shard, str):
            file_shard = [file_shard]
        files.extend(file_shard)
    files = [f for f in files if f in allowed_filepaths]
    if limit > 0:
        files = files[:limit]
    return files


def file_routine(fpath, src_dir):
    names = []
    thresholds = []
    for red in reds:
        for green in greens:
            names.append(naming(red, green))
            thresholds.append((red, green))

    src_fpath = reconstruct_path(fpath, src_dir)
    tgt_fpaths = [reconstruct_path(fpath, os.path.join(EN_BASE, name)) for name in names]
    docs = [[] for _ in names]
    for name, tgt_fpath in zip(names, tgt_fpaths):
        os.makedirs(os.path.dirname(tgt_fpath), exist_ok=True)



    with gzip.open(src_fpath, 'rt', encoding='utf-8') as fin:
        cnt = 0
        ttl = 0
        for line in fin:
            ttl += 1
            entry = json.loads(line)
            text = entry.get("text", "")
            if not text:
                continue

            # Extract counts
            count = entry.get("count", None)
            if count is None:
                raise ValueError(f"Missing count in entry: {entry}")
            if not isinstance(count, list):
                raise ValueError(f"Invalid count format in entry: {entry}")

            # pop count
            entry.pop("count", None)
                

            # Process text for CP and CP-PP
            lines = text.split("\n")
            for i, (red, green) in enumerate(thresholds):
                # Apply pattern-based remover
                cp_lines = pld_extractor(lines, count, red=red, green=green)
                
                if cp_lines:
                    entry["text"] = "\n".join(cp_lines)

                    docs[i].append(json.dumps(entry, ensure_ascii=False))
            cnt += 1
        print(f"Processed {cnt} lines out of {ttl} in {src_fpath}", flush=True)
    # with gzip.open(tgt_fpath, "wt", encoding='utf-8') as fout_pp, \
    #         gzip.open(tgt_cp_fpath, "wt", encoding='utf-8') as fout_cp:
    for i, (name, tgt_fpath) in enumerate(zip(names, tgt_fpaths)):
        with gzip.open(tgt_fpath, "wt", encoding='utf-8') as fout_pp:
            for line in docs[i]:
                fout_pp.write(line)
                fout_pp.write("\n")

def main():
    parser = ArgumentParser()
    parser.add_argument("--snapshot", type=str, default="2019-04")
    parser.add_argument("--train-start", type=int, default=0)
    parser.add_argument("--train-end", type=int, default=500)
    parser.add_argument("--valid-start", type=int, default=0)
    parser.add_argument("--valid-end", type=int, default=5)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--num-workers", type=int, default=64)
    
    args = parser.parse_args()


    # src dataset : count_info
    # get the count info files of the snapshot
    count_info_dir = os.path.join(EN_BASE, "count_info")
    count_info_pttn = os.path.join(count_info_dir, "CC-MAIN-" + args.snapshot, "**/*.json.gz")
    count_info_files = glob(count_info_pttn, recursive=True)
    print(f"Found {len(count_info_files)} count info files in snapshot {args.snapshot}", flush=True)
    
    # only keep the files that are in the range of the train/valid shard
    count_info_allowed_filepaths = set([restore_path(f, count_info_dir) for f in count_info_files])
    files_to_process = get_files_to_process(args.snapshot, count_info_allowed_filepaths, args.train_start, args.train_end, args.valid_start, args.valid_end)
    print(f"Found {len(files_to_process)} files to process", flush=True)

    if args.limit > 0:
        print(f"Limiting to {args.limit} files", flush=True)
        files_to_process = files_to_process[:args.limit]

    process_func = partial(file_routine, src_dir=count_info_dir)
    with mp.Pool(args.num_workers) as pool:
        # map the file routine to the files to process
        res = pool.imap_unordered(process_func, files_to_process)
        for _ in tqdm(res, total=len(files_to_process), desc="Processing files"):
            pass
        

if __name__ == "__main__":
    main()
