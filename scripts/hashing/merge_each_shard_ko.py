import os
from glob import glob
from argparse import ArgumentParser
from tqdm import tqdm
import multiprocessing as mp
from pattern_aware_filtering.hashing.flat_hash_set import FlatHashSet
from pattern_aware_filtering.utils.constants import KO_SHARDED_HASH_DIR, KO_MERGED_HASH_DIR
from pattern_aware_filtering.utils.snapshots import get_snapshots

def get_filepaths(sharded_dir, snapshot, limit=-1):
    pttn = os.path.join(sharded_dir, snapshot, "*.bin")
    filepaths = glob(pttn)
    filepaths.sort()
    if limit > 0:
        filepaths = filepaths[:limit]
    return filepaths


def merge_one_snapshot(snapshot, sharded_dir, merged_dir):
    shard_filepaths = get_filepaths(sharded_dir, snapshot)
    hashmap = FlatHashSet()
    for filepath in tqdm(shard_filepaths):
        hashmap.load(filepath)
    save_destination = os.path.join(merged_dir, f"{snapshot}.bin")
    os.makedirs(merged_dir, exist_ok=True)
    hashmap.dump(save_destination)

def main():
    parser = ArgumentParser()
    
    parser.add_argument("--sharded-dir", type=str, default=KO_SHARDED_HASH_DIR)
    parser.add_argument("--merged-dir", type=str, default=KO_MERGED_HASH_DIR)
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()

    snapshots = get_snapshots()
    with mp.Pool() as pool:
        pool.starmap(merge_one_snapshot, [(snapshot, args.sharded_dir, args.merged_dir) for snapshot in snapshots])


if __name__ == "__main__":
    main()

    
