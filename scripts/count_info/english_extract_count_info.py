import os
from tqdm import tqdm
from argparse import ArgumentParser
from glob import glob
import multiprocessing as mp
import json
import gzip
import re
import io
import time

from pattern_aware_filtering.hashing.flat_hash_set import FlatHashSet
from pattern_aware_filtering.utils.ccpath import get_wet_paths_file
from pattern_aware_filtering.hashing.hash_functions import compute_hashes
from pattern_aware_filtering.utils.constants import PROJECT_ROOT, EN_EXTRACTED_PATH, EN_MERGED_HASH_DIR

io.DEFAULT_BUFFER_SIZE = 1024 * 1024 * 1024 # Set buffer size to 1GiB

EN_BASE = EN_EXTRACTED_PATH
baseline = os.path.join(EN_BASE, "baseline")
count_info = os.path.join(EN_BASE, "count_info")

def reconstruct_path(path, base_dir):
    return path.replace("crawl-data", base_dir).replace(".warc.wet.gz", ".json.gz")


def get_allowed_filepaths(snapshot, shard_idx):
    wet_paths_files = get_wet_paths_file(snapshot)
    start_idx = 1000 * shard_idx
    end_idx = 1000 * (shard_idx + 1)
    wet_paths_files = wet_paths_files[start_idx:end_idx]
    return set(wet_paths_files)

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


def flush(doclist, fout):
    buf = io.StringIO()
    if doclist:
        for doc in doclist:
            buf.write(doc)
            buf.write("\n")
    fout.write(buf.getvalue())
    buf.close()
    doclist.clear()
        

def file_routine(filepath, hashmap):
    src_path = reconstruct_path(filepath, baseline)
    cb_info_path = reconstruct_path(filepath, count_info)
    print("Source path:", src_path, flush=True)
    os.makedirs(os.path.dirname(cb_info_path), exist_ok=True)

    # check if the src path exists
    if not os.path.exists(src_path):
        print(f"Source file does not exist: {src_path}", flush=True)
        return

    cb_info_list = []
    doc_count = 0

    start = time.time()
    
    with gzip.open(src_path, "rt", encoding="utf-8") as fin:
        with gzip.open(cb_info_path, "wt", encoding="utf-8") as fout_info:
            for i, line in enumerate(fin, start=1):
                try:
                    entry = json.loads(line.strip())
                    text = entry.get("text")
                    if not text:
                        print(f"Skipping entry with no text: {entry}", flush=True)
                        continue
                    lines = text.split("\n")
                    lines = lines[1:]  # skip the first line which the title
                    if len(lines) == 0:
                        continue
                    hashes = compute_hashes(lines)
                    count = hashmap[hashes]

                    cb_info = {
                        "text": "\n".join(lines),
                        "count": count.tolist(),
                    }
                    cb_info_list.append(json.dumps(cb_info, ensure_ascii=False))

                    doc_count += 1

                    if doc_count % 1000 == 0:
                        flush(cb_info_list, fout_info)
                        curr = time.time()
                        elapsed = curr - start
                        docs_per_sec = doc_count / elapsed
                        print(f"Processed {doc_count} documents in {elapsed:.2f} seconds ({docs_per_sec:.2f} docs/sec)", flush=True)

                except Exception as e:
                    print(f"Error processing line {i} in {src_path}: {e}", flush=True)
                    continue

            flush(cb_info_list, fout_info)
            elapsed = time.time() - start
            print(f"Finished processing {doc_count} documents in {src_path}", flush=True)
            print(f"Total time taken: {elapsed / 60:.2f} minutes", flush=True)
                    


def shard_routine(snapshot, shard_idx, sharded_hashbin_dir, train_start, train_end, valid_start, valid_end, limit=-1):
    bin_path = os.path.join(sharded_hashbin_dir, snapshot, f"merged_{str(shard_idx).zfill(3)}.bin")
    allowed_filepaths = get_allowed_filepaths(snapshot, shard_idx)
    files_to_process = get_files_to_process(snapshot, allowed_filepaths, train_start, train_end, valid_start, valid_end, limit=limit)
    print(len(files_to_process), "files to process in shard", shard_idx, flush=True)
    hashmap = FlatHashSet()
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Hash bin file not found: {bin_path}")
    hashmap.load(bin_path)
    for filepath in files_to_process:
        print(f"Processing file: {filepath} in shard {shard_idx}", flush=True)
        file_routine(filepath, hashmap)
        print(f"Processed file: {filepath} in shard {shard_idx}", flush=True)


def worker_routine(snapshot, shard_idx, sharded_hashbin_dir, work_queue, limit=-1):
    bin_path = os.path.join(sharded_hashbin_dir, snapshot, f"merged_{str(shard_idx).zfill(3)}.bin")
    hashmap = FlatHashSet()
    hashmap.load(bin_path)
    while True:
        filepath = work_queue.get()
        if filepath is None:  # Sentinel value to stop the worker
            break
        try:
            print(f"Worker processing file: {filepath} in shard {shard_idx}", flush=True)
            file_routine(filepath, hashmap)
            print(f"Worker finished processing file: {filepath} in shard {shard_idx}", flush=True)
        except Exception as e:
            print(f"Error processing file {filepath} in shard {shard_idx}: {e}", flush=True)


def shard_wrapper(args):
    shard_routine(*args)

def main():
    parser = ArgumentParser()
    parser.add_argument("--snapshot", type=str, default="2019-04")
    parser.add_argument("--train-start", type=int, default=0)
    parser.add_argument("--train-end", type=int, default=500)
    parser.add_argument("--valid-start", type=int, default=0)
    parser.add_argument("--valid-end", type=int, default=5)
    parser.add_argument("--merged-hashbin-dir",
        type=str,
        default=EN_MERGED_HASH_DIR,
    )
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--num-workers", type=int, default=8)
    
    

    args = parser.parse_args()

    hash_dir = os.path.join(args.merged_hashbin_dir, args.snapshot)
    hash_files = glob(os.path.join(hash_dir, "*.bin"))
    
    # get indexes from hash files
    shard_idx = [int(os.path.basename(f).split("_")[1].split(".")[0]) for f in hash_files]
    print(f"Found {len(shard_idx)} shards in {args.merged_hashbin_dir} for snapshot {args.snapshot}", flush=True)
    
    with mp.Pool(args.num_workers) as pool:
        res = pool.imap_unordered(
            shard_wrapper,
            [(args.snapshot, idx, args.merged_hashbin_dir, args.train_start, args.train_end, args.valid_start, args.valid_end, args.limit) for idx in shard_idx]
        )
        for _ in tqdm(res, total=len(shard_idx), desc="Processing shards"):
            pass
    

        

if __name__ == "__main__":
    main()
