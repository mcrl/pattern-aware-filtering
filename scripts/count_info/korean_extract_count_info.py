import os
import json
import gzip
from tqdm import tqdm
import multiprocessing as mp
from argparse import ArgumentParser
from functools import partial
from glob import glob

from pattern_aware_filtering.hashing.flat_hash_set import FlatHashSet
from pattern_aware_filtering.hashing.hash_functions import compute_hashes
from pattern_aware_filtering.utils.constants import PROJECT_ROOT, KO_EXTRACTED_PATH, KO_MERGED_HASH_DIR, KO_CCWET_PATH

def get_save_path_from_src(fpath, src_dir, dest_dir):
    return fpath.replace(src_dir, dest_dir)

def worker_routine(task_queue,  hash_path, src_dir, save_dir):
    # load hashmap
    hashmap = FlatHashSet()
    hashmap.load(hash_path)
    print(f"Hashmap loaded")

    while True:
        task = task_queue.get()
        if task is None:
            break
        
        print(f"Processing {task}")
        # task is a filepath
        dest_path = get_save_path_from_src(task, src_dir, save_dir)
        docs = []
        with gzip.open(task, "rt") as fin:
            for line in fin:
                document = json.loads(line)
                text = document.pop("text")
                lines = text.split("\n")
                lines = lines[1:]
                if len(lines) == 0:
                    continue
                hashes = compute_hashes(lines)
                count = hashmap[hashes]
                count = count.tolist()

                new_doc = {
                    "text": "\n".join(lines),
                    "count": count,
                }
                docs.append(new_doc)
        # create destination directory if not exists
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)        
        with gzip.open(dest_path, "wt") as fout:
            for doc in docs:
                fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
                
    print(f"Worker terminated")

def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--hashbin-dir", type=str, default=KO_MERGED_HASH_DIR)
    parser.add_argument("--snapshot", type=str, required=True)
    
    # Filepath
    parser.add_argument("--file-extension", type=str, default=".json.gz")
    
    # limit
    parser.add_argument("--limit", type=int, default=-1)

    args = parser.parse_args()
    
    return args


def main():
    args = parse_args()

    num_workers = 8
    workers = []

    hash_path = os.path.join(args.hashbin_dir, f"{args.snapshot}.bin")
    input_dir = os.path.join(KO_CCWET_PATH, f"CC-MAIN-{args.snapshot}")
    save_base = KO_EXTRACTED_PATH
    save_dir = os.path.join(save_base, "count_info", f"CC-MAIN-{args.snapshot}")

    task_queue = mp.Queue(maxsize=100)
    for _ in range(num_workers):
        worker = mp.Process(target=worker_routine, args=(task_queue, hash_path, input_dir, save_dir))
        worker.start()
        workers.append(worker)
    print(f"Master: Workers started")

    glob_pttn = os.path.join(input_dir, f"**/*{args.file_extension}")
    print(f"glob pattern: {glob_pttn}")
    task_list = glob(glob_pttn, recursive=True)
    task_list.sort()
    print(f"Task list created")

    if args.limit > 0:
        task_list = task_list[:args.limit]
    for task in tqdm(task_list):
        task_queue.put(task)
    print(f"Task queue filled")
    for _ in range(num_workers):
        task_queue.put(None)
    for worker in workers:
        worker.join()

    print(f"Finished")
if __name__ == "__main__":
    main()
    
