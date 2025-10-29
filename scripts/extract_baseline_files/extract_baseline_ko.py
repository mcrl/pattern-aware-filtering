import warcio
import os
from io import BufferedReader, BufferedWriter
from argparse import ArgumentParser
import json
from random import random
from tqdm import tqdm
import multiprocessing as mp
import re
from functools import partial
import gzip

from pattern_aware_filtering.utils.constants import CCWET_PATH, KO_CCWET_PATH
from pattern_aware_filtering.utils.ccpath import get_wet_paths_file

korean_pttn = re.compile("[가-힣]")

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
wsp_pttn = re.compile("|".join(VARIOUS_WHITESPACES))

# set file read chunk size to 1GiB
CHUNK_SIZE = 16 * 1024 * 1024

# CONFIGURATION: Modify the following functions to match the data path and format
def make_src_path(wet_path):
    base_dir = CCWET_PATH
    replaced = wet_path.replace("crawl-data", base_dir)
    # check if file exists
    if os.path.exists(replaced):
        return replaced
    raise FileNotFoundError(f"File not found: {replaced}")
    

def make_target_path(wet_path):
    base_dir = KO_CCWET_PATH
    replaced = wet_path.replace("crawl-data", base_dir)
    fpath = replaced.replace(".warc.wet.gz", ".json.gz")
    # check if the base directory exists. If not, create it.
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    return fpath
    


def target_sanity_check(target_path):
    try:
        with open(target_path, "r") as f:
            for line in f:
                json.loads(line)
    except Exception as e:
        print(f"Error: {target_path}")
        print(e)
        return None
    return True


def save_korean_only(wet_path, save_path, snapshot=None):
    cnt = 0
    ttl = 0
    total_char_len = 0
    saved_char_len = 0

    try:
        with open(wet_path, "rb") as fin, gzip.open(save_path, "wt") as fout:
            istream = BufferedReader(fin, CHUNK_SIZE)

            for record in warcio.archiveiterator.ArchiveIterator(fin):
                if record.rec_type != "conversion":
                    continue
                content = record.content_stream().read().decode("utf-8")
                content_len = len(content)

                ttl += 1
                total_char_len += content_len

                # at least 10% of the content should be Korean
                if len(korean_pttn.findall(content)) / content_len < 0.1:
                    continue

                entry = {}
                headers = record.rec_headers.headers
                for k, v in headers:
                    if k == "WARC-Identified-Content-Language":
                        continue
                    entry[k] = v
                entry["text"] = content
                line = json.dumps(entry, ensure_ascii=False) + "\n"
                fout.write(line)

                saved_char_len += content_len
                cnt += 1

            istream.close()
    except Exception as e:
        with open(f"logs/{snapshot}.err", "a") as f:
            f.write(f"{wet_path}\n{e}\n")
    return wet_path, ttl, cnt, total_char_len, saved_char_len

def file_routine(p, snapshot=None):
    src_path = make_src_path(p)
    target_path = make_target_path(p)
    return save_korean_only(src_path, target_path, snapshot)

def main():
    parser = ArgumentParser()
    parser.add_argument("--snapshot", type=str, default="2024-30")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--num-workers", type=int, default=mp.cpu_count())
    args = parser.parse_args()

    if "CC-MAIN-" not in args.snapshot:
        args.snapshot = f"CC-MAIN-{args.snapshot}"

    wet_paths = get_wet_paths_file(args.snapshot, args.limit)
    if not wet_paths:
        print(f"snapshot {args.snapshot} not found.")
    print(f"Processing {len(wet_paths)} WET files.")
    
    sum_ttl = 0
    sum_cnt = 0
    sum_char_len = 0
    sum_saved_len = 0

    with mp.Pool(args.num_workers) as pool, open(f"logs/{args.snapshot}.csv", "w") as f:
        f.write("wet_path,total,count,total_chars,saved_chars\n")
        results = pool.imap_unordered(file_routine, wet_paths)
        for r in tqdm(results, total=len(wet_paths)):
            sum_ttl += r[1]
            sum_cnt += r[2]
            sum_char_len += r[3]
            sum_saved_len += r[4]

            f.write(f"{r[0]},{r[1]},{r[2]}, {r[3]}, {r[4]}\n")

    print(f"Total: {sum_ttl} / Korean: {sum_cnt} ({sum_cnt/sum_ttl :.2%})")
    print(f"Total characters: {sum_char_len} / Saved characters: {sum_saved_len} ({sum_saved_len/sum_char_len :.2%})")

if __name__ == "__main__":
    main()
