import json
import os
import glob
import math
from argparse import ArgumentParser
import multiprocessing as mp
import gzip
from tqdm import tqdm
from warcio.archiveiterator import ArchiveIterator

from pattern_aware_filtering.utils import ccpath
from pattern_aware_filtering.hashing.flat_hash_set import FlatHashSet
from pattern_aware_filtering.hashing.hash_functions import compute_hashes
from pattern_aware_filtering.utils.normalizer import normalize_data
from pattern_aware_filtering.utils.langdetect import detect_en
from pattern_aware_filtering.utils.constants import PROJECT_ROOT, CCWET_PATH, EN_EXTRACTED_PATH, EN_SHARDED_HASH_DIR


def make_shard(filepaths, shard_size):
    # calculate the number of files per shard
    num_files = len(filepaths)

    shards = [filepaths[i:i+shard_size] for i in range(0, len(filepaths), shard_size)]
    
    # print(shards)
    return shards

def bin_path(dirname, shard_num):
    return f"{dirname}/shard_{str(shard_num).zfill(3)}.bin"

def convert_wet_path_to_json_path(wet_path):
    converted = wet_path.replace("crawl-data", EN_EXTRACTED_PATH).replace(".warc.wet.gz", ".json.gz")
    return converted

def convert_wet_path_to_local_wet_path(wet_path):
    converted = wet_path.replace("crawl-data", CCWET_PATH)
    
    return converted

def json_gz_routine(filepath, hashmap, limit=-1):
    cnt = 0
    print(f"Processing {filepath}")
    with gzip.open(filepath, "rt") as f:
        for line in f:
            entry = json.loads(line)
            text = entry["text"]
            hashed = compute_hashes(text)
            hashmap.add(hashed)
            cnt += 1
            if limit == cnt:
                break

def wet_gz_routine(filepath, wet_path, hashmap, limit=-1):
    print(f"Processing {filepath}")
    json_gz_path = convert_wet_path_to_json_path(wet_path)
    csv_path = json_gz_path.replace(".json.gz", ".csv")
    # make base directory
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    ttl = 0
    
    with gzip.open(filepath, "r") as stream, open(csv_path, "w") as csv_out:
        csv_out.write("idx,english\n")
        for record in ArchiveIterator(stream):
            if limit == ttl:
                break
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
            hashed = compute_hashes(content)
            hashmap.add(hashed)
            

def check_json_gz(filepath):
    converted = convert_wet_path_to_json_path(filepath)
    if os.path.exists(converted):
        return True, converted
    return False, convert_wet_path_to_local_wet_path(filepath)

def get_shard_cache_path(snapshot, workder_index, shard_index):
    base_dir = f"{PROJECT_ROOT}/hashing_checkpoints"
    cache_path = os.path.join(base_dir, snapshot,  f"worker_{workder_index}", f"{str(shard_index).zfill(3)}.cache")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    return cache_path


def shard_routine(snapshot, worker_index, shard, shard_num, target_dirctory, limit):
    hashmap = FlatHashSet()
    cache_path = get_shard_cache_path(snapshot, worker_index, shard_num)
    binary_path = bin_path(target_dirctory, shard_num)
    temp_bin_path = binary_path + ".tmp"
    done_idx = 0
    if os.path.exists(cache_path):
        if os.path.exists(temp_bin_path):
            # something went wrong. We do not trust cached file
            os.remove(temp_bin_path)
        elif os.path.exists(binary_path):
            with open(cache_path, "r") as f:
                done_idx = int(f.read().strip())
            if done_idx >= len(shard) - 1:
                print(f"Shard {shard_num} is already done. Skipping.")
                return
            hashmap.load(binary_path)            
    else:
        done_idx = 0

    for i, filepath in enumerate(shard):
        if i <= done_idx:
            continue
        res, converted = check_json_gz(filepath)
        if res:
            json_gz_routine(converted, hashmap, limit=limit)
        else:
            wet_gz_routine(converted, filepath, hashmap, limit=limit)
        
        hashmap.dump(temp_bin_path)
        with open(cache_path, "w") as f:
            f.write(str(i) + "\n")
            f.flush()
        os.rename(temp_bin_path, binary_path)
        
    
def shard_routine_wrapper(args):
    shard_routine(*args)


def get_filepaths(snapshot, limit=-1):
    wet_paths = ccpath.get_wet_paths_file(snapshot, limit=limit)
    print(f"Number of files: {len(wet_paths)}")
    return wet_paths

def main():
    parser = ArgumentParser()
    parser.add_argument("--snapshot", type=str, default="2021-04")
    parser.add_argument("--shard-size", type=int, default=20)
    parser.add_argument("--index-start", type=int, default=0)
    parser.add_argument("--index-end", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--num-workers", type=int, default=64)
    args = parser.parse_args()

    files = get_filepaths(args.snapshot)
    target_dir = EN_SHARDED_HASH_DIR
    target_dir = os.path.join(target_dir, args.snapshot)
    os.makedirs(target_dir, exist_ok=True)

    shards = make_shard(files, args.shard_size)
    print(f"Number of shards: {len(shards)}")
    shards_to_process = shards[args.index_start:args.index_end]
    with mp.Pool(args.num_workers) as pool:
        res = pool.imap_unordered(shard_routine_wrapper, [(args.snapshot, args.index_start, shard, i, target_dir, args.limit) for i, shard in enumerate(shards_to_process, start=args.index_start)])
        for _ in tqdm(res, total=len(shards_to_process), desc="Processing shards"):
            pass


if __name__ == "__main__":
    main()


