import subprocess as sp
import multiprocessing as mp
import os
from time import sleep, time
from datetime import datetime
from argparse import ArgumentParser
import traceback
import socket

from pattern_aware_filtering.utils.snapshots import get_snapshots
from pattern_aware_filtering.utils.constants import CCWET_PATH



JOBS = 10
NO_LIMIT = ""


def get_limit(full_load=False):
    # you may implement your own policy for speed limit
    return NO_LIMIT
    

def update_full_load():
    # update this function to specify whether the network is in full load or not
    # you may implement your own policy
    return True 


def setup_dir(index_name):
    # cache dir
    cache_dir = f"cache/progress"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    cache_dir = f"cache/wet_paths"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    # save dir
    save_dir = f"{CCWET_PATH}/{index_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


def download_wet_paths(index_name):
    web_url = f"https://data.commoncrawl.org/crawl-data/{index_name}/wet.paths.gz"
    target_path = f"cache/wet_paths/{index_name}.wet.paths"


    if os.path.exists(target_path):
        print(f"Already downloaded {target_path}")
        return None

    if os.path.exists(target_path + ".gz"):
        cmd = f"gunzip {target_path}.gz"
        print(f"Unzip {target_path}.gz")
        sp.run(cmd.split(" "), check=True)
        return None
    
    dirname = os.path.dirname(target_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    print(f"Download {web_url} to {target_path}.gz")
    cmd = f"wget {web_url} -O {target_path}.gz"
    sp.run(cmd.split(" "), check=True)
    cmd = f"gunzip {target_path}.gz"
    sp.run(cmd.split(" "), check=True)
    return None


def listup_wet_paths(index_name, force=False):
    if not os.path.exists("cache"):
        os.makedirs("cache")
    double_check_file = "cache/double_check.txt"
    if not os.path.exists(double_check_file):
        with open(double_check_file, "w") as f:
            f.write("")
            f.flush()
    with open("cache/double_check.txt") as f:
        checked = [line.strip() for line in f.readlines()]
    if index_name in checked and not force:
        return []
    target_path = f"cache/wet_paths/{index_name}.wet.paths"
    if not os.path.exists(target_path):
        download_wet_paths(index_name)

    with open(target_path, "r") as f:
        paths = [line.strip() for line in f.readlines()]

    basedirs = [os.path.dirname(path) for path in paths]
    dirset = set(basedirs)
    dirset = [dir.replace("crawl-data/", "") for dir in dirset]
    for dir in dirset:
        tgt_dir = os.path.join(CCWET_PATH, dir)
        os.makedirs(tgt_dir, exist_ok=True)

    if force:
        return paths

    # check caches
    cache_file = f"cache/progress/{index_name}.done"
    if not os.path.exists(cache_file):
        return paths

    with open(cache_file, "r") as f:
        done_paths = [line.strip() for line in f.readlines()]
    targets = [path for path in paths if path not in done_paths]
    if len(targets) == 0:
        with open("cache/double_check.txt", "a") as f:
            f.write(f"{index_name}\n")
            f.flush()
    return targets


def download_path(wet_path, logger_queue, verbose=False, limit=None):
    uri = f"https://data.commoncrawl.org/{wet_path}"
    save_path = wet_path.replace("crawl-data", CCWET_PATH)
    if limit is None:
        limit = get_limit()

    cmd = (
        f"wget -c -t 0 --retry-on-http-error=503 --waitretry={JOBS} {uri} -O {save_path}"
        + limit
    )
    # print(cmd)
    if verbose:
        sp.run(cmd.split(" "), check=True)
    else:
        sp.run(cmd.split(" "), check=True, stdout=sp.DEVNULL, stderr=sp.DEVNULL)

    logger_queue.put(wet_path)
    print(f"Done {wet_path}")


def downloader_routine(wet_paths_queue, logger_queue, verbose=False):
    count = 0
    full_load = update_full_load()
    limit = get_limit(full_load=full_load)
    wet_path = None
    while True:
        try:
            if wet_path is None:
                wet_path = wet_paths_queue.get()
            if wet_path is None:
                break
            download_path(wet_path, logger_queue, limit=limit, verbose=verbose)
            count += 1
            if count % 20 == 0:
                limit = get_limit(full_load=full_load)
            if count % 400 == 0:
                full_load = update_full_load()
            wet_path = None
        except Exception as e:
            sleep(60)


def log_wet_path(wet_path):
    index_name = wet_path.split("/")[1]
    cache_file = f"cache/progress/{index_name}.done"
    with open(cache_file, "a") as f:
        f.write(f"{wet_path}\n")
        f.flush()


def download_logger(log_queue):
    while True:
        log = log_queue.get()
        if log is None:
            break
        log_wet_path(log)
        


def download_index(index_name, wet_queue, devel=0, force=False):
    target_paths = listup_wet_paths(index_name, force=force)
    if len(target_paths) == 0:
        return
    setup_dir(index_name)
    if devel > 1:
        target_paths = target_paths[:devel]
    for path in target_paths:
        wet_queue.put(path)
        sleep(0.5)


def downloader(index_names, num_workers=JOBS, verbose=False, force=False):
    workers = []
    wet_queue = mp.Queue(maxsize=num_workers * 3)
    logger_queue = mp.Queue()

    try:
        logger = mp.Process(target=download_logger, args=(logger_queue,))
        logger.start()
    except Exception as e:
        logger.terminate()
        return

    try:
        for _ in range(num_workers):
            p = mp.Process(
                target=downloader_routine,
                args=(wet_queue, logger_queue),
                kwargs={"verbose": verbose},
            )
            p.start()
            workers.append(p)
    except Exception as e:
        print(e)
        for worker in workers:
            worker.terminate()
        return

    try:
        for index_name in index_names:
            print(f"Start downloading {index_name}")
            download_index(index_name, wet_queue, force=force)

        for _ in range(num_workers):
            wet_queue.put(None)

        for worker in workers:
            worker.join()
    except Exception as e:
        print(e)

        for worker in workers:
            worker.terminate()
        

    logger_queue.put(None)
    logger.join()
    


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--ignore-cache", action="store_true")
    args = parser.parse_args()

    snapshots = get_snapshots()
    index_names = [f"CC-MAIN-{snapshot}" for snapshot in snapshots]
    downloader(index_names, verbose=args.verbose, force=args.ignore_cache)
