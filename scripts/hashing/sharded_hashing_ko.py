import json
import os
import glob
import math
from argparse import ArgumentParser
import multiprocessing as mp
import gzip
import tqdm

from pattern_aware_filtering.utils import ccpath
from pattern_aware_filtering.hashing.flat_hash_set import FlatHashSet
from pattern_aware_filtering.hashing.hash_functions import compute_hashes
from pattern_aware_filtering.utils.constants import KO_SHARDED_HASH_DIR

def make_shard(filepaths, num_shard):
    # filepaths is a list of filepaths
    # num_shard is the number of shards to create
    # return a list of list of filepaths where each list of filepaths is a shard

    # sort the filepaths
    filepaths.sort()

    # calculate the number of files per shard
    num_files = len(filepaths)
    files_per_shard = math.ceil(len(filepaths) / num_shard)

    # create the shards
    if num_files >= num_shard:
        return [[filepath] for filepath in filepaths]
    shards = [filepaths[i:i+files_per_shard] for i in range(0, len(filepaths), files_per_shard)]
    # print(shards)
    return shards

def bin_path(dirname, shard_num):
    return f"{dirname}/shard_{str(shard_num).zfill(3)}.bin"

def shard_routine(shard, shard_num, target_dirctory):
    hashmap = FlatHashSet()
    for filepath in shard:
        with gzip.open(filepath, "rt") as f:
            for line in f:
                entry = json.loads(line)
                text = entry["text"]
                hashed = compute_hashes(text)
                hashmap.add(hashed)
    hashmap.dump(bin_path(target_dirctory, shard_num))
    
def shard_routine_wrapper(args):
    shard_routine(*args)


def get_filepaths(snapshot, limit=-1):
    wet_paths = ccpath.get_wet_paths_file(snapshot, limit=limit)
    filepaths = [ccpath.make_korean_extracted_path(wet_path) for wet_path in wet_paths]
    print(f"Number of files: {len(filepaths)}")
    return filepaths

def main():
    parser = ArgumentParser()
    parser.add_argument("--snapshot", type=str, default="2021-04")
    parser.add_argument("--num-shards", type=int, default=10000000)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--num-workers", type=int, default=64)
    args = parser.parse_args()
    files = get_filepaths(args.snapshot, limit=args.limit)

    target_dir = KO_SHARDED_HASH_DIR
    target_dir = os.path.join(target_dir, args.snapshot)
    os.makedirs(target_dir, exist_ok=True)

    shards = make_shard(files, args.num_shards)
    with mp.Pool(args.num_workers) as pool:
        res = pool.imap_unordered(shard_routine_wrapper, [(shard, i, target_dir) for i, shard in enumerate(shards)])
        for _ in tqdm.tqdm(res, total=len(shards), desc="Processing shards"):
            pass


if __name__ == "__main__":
    main()


