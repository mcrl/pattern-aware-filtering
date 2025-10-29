import os
from glob import glob
from argparse import ArgumentParser
from tqdm import tqdm
import multiprocessing as mp
import sys

from pattern_aware_filtering.hashing.flat_hash_set import FlatHashSet
from pattern_aware_filtering.utils.constants import EN_SHARDED_HASH_DIR, EN_MERGED_HASH_DIR

def get_idx_from_filepath(filepath):
    basename = os.path.basename(filepath)
    idx = basename.split("_")[-1].split(".")[0]
    return int(idx)

def get_sharded_bin_files(snapshot):
    dirbase = EN_SHARDED_HASH_DIR
    dirpath = os.path.join(dirbase, snapshot)
    pttn = os.path.join(dirpath, "*.bin")
    filepaths = glob(pttn)
    filepaths = sorted(filepaths, key=get_idx_from_filepath)
    return filepaths

def merge_shard(args):
    shard, idx, merged_dir, limit = args
    if limit > 0:
        shard = shard[:limit]
    print(f"Merging shard {idx} with {len(shard)} files.", flush=True)
    
    merged_set = FlatHashSet()
    for filepath in shard:
        try:
            print(f"Loading {filepath}", flush=True)
            merged_set.load(filepath)
        except Exception as e:
            print(f"Error loading {filepath}: {e}", flush=True)
            continue
    
    try:
        merged_filepath = os.path.join(merged_dir, f"merged_{str(idx).zfill(3)}.bin")
        print(f"Saving merged set to {merged_filepath}", flush=True)
        merged_set.dump(merged_filepath)
    except Exception as e:
        print(f"Error saving merged set: {e}",flush=True)
        print(f"shard: {shard}, idx: {idx}, merged_dir: {merged_dir}, limit: {limit}",flush=True)
        return None
    

def main():
    parser = ArgumentParser()
    
    parser.add_argument("--sharded-dir", type=str, default=EN_SHARDED_HASH_DIR)
    parser.add_argument("--merged-dir", type=str, default=EN_MERGED_HASH_DIR)
    parser.add_argument("--merge-count", type=int, default=50)
    parser.add_argument("--snapshot", type=str, default="2019-04")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--num-workers", type=int, default=64)
    args = parser.parse_args()

    # redirect stdout and stderr to a log file
    log_file = os.path.join("logs", "merging", f"merge_{args.snapshot}.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    err_file = os.path.join("logs", "merging", f"merge_{args.snapshot}_err.log")
    os.makedirs(os.path.dirname(err_file), exist_ok=True)

    sys.stdout = open(log_file, "w")
    sys.stderr = open(err_file, "w")
    print(f"Redirecting stdout and stderr to {log_file} and {err_file}", flush=True)
    files = get_sharded_bin_files(args.snapshot)
    print(f"Number of files to merge: {len(files)}", flush=True)
    print(f"Merge count: {args.merge_count}", flush=True)
    print(f"Limit: {args.limit}", flush=True)
    print(f"Number of workers: {args.num_workers}", flush=True)
    print(f"Snapshot: {args.snapshot}", flush=True)
    
    merged_dir = os.path.join(args.merged_dir, args.snapshot)
    os.makedirs(merged_dir, exist_ok=True)

    shards = [files[i:i+args.merge_count] for i in range(0, len(files), args.merge_count)]

    shards = [(shard, idx, merged_dir, args.limit) for idx, shard in enumerate(shards)]
    with mp.Pool(args.num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(merge_shard, shards), total=len(shards)))
    
    sys.stdout.close()
    sys.stderr.close()

if __name__ == "__main__":
    main()

    
