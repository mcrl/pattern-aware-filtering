import os
import json
import gzip
from tqdm import tqdm
import multiprocessing as mp
from argparse import ArgumentParser
from functools import partial
from glob import glob
import io

from pattern_aware_filtering.extraction.extractor import pld_extractor
from pattern_aware_filtering.utils.constants import KO_EXTRACTED_PATH

ablation_red = [50, 100, 500, 1000, 2500, 5000]
ablation_green = [1, 3, 5]

io.DEFAULT_BUFFER_SIZE = 1024 * 1024 * 1024 # Set buffer size to 1GiB

src_base = KO_EXTRACTED_PATH
def replace_filename(fpath, src_dir, red, green):
    exp_name = f"pld-r{red}-g{green}"
    tgt_dir = os.path.join(src_base, exp_name)
    return fpath.replace(src_dir, tgt_dir)

def file_routine(fpath, src_dir):
    # 2d list: [red][green]
    data = [[[] for _ in range(len(ablation_green))] for _ in range(len(ablation_red))]

    with gzip.open(fpath, 'rt', encoding='utf-8') as fin:
        for line in fin:
            item = json.loads(line)
            text= item["text"]
            count = item["count"]
            original_lines = text.split("\n")
            if not original_lines:
                continue
            
            for i, red in enumerate(ablation_red):
                for j, green in enumerate(ablation_green):
            
                    lines = pld_extractor(original_lines, count, red=red, green=green)
                    if not lines:
                        continue
                    data[i][j].append({
                        "text": "\n".join(lines),
                    })
    
    for i, red in enumerate(ablation_red):
        for j, green in enumerate(ablation_green):
            save_path = replace_filename(fpath, src_dir, red, green)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            buf = io.StringIO()
            for item in data[i][j]:
                buf.write(json.dumps(item, ensure_ascii=False))
                buf.write("\n")
            with gzip.open(save_path, "wt", encoding='utf-8') as fout:
                fout.write(buf.getvalue())


def main():
    parser = ArgumentParser(description="Process files with multiprocessing.")
    parser.add_argument("--snapshot", type=str, default="2019-04", help="Snapshot date to process")
    parser.add_argument("--num-workers", type=int, default=64, help="Number of worker processes")
    parser.add_argument("--limit", type=int, default=-1, help="Limit the number of files to process (for testing purposes)")
    args = parser.parse_args()

    snapshot = args.snapshot
    
    src_base_dir = os.path.join(src_base, "count_info")
    src_dir = os.path.join(src_base_dir, "CC-MAIN-" + snapshot)

    files = glob(os.path.join(src_dir, "**/*.json.gz"), recursive=True)
    if args.limit > 0:
        files = files[:args.limit]
    print(f"Found {len(files)} files in {src_dir}")

    with mp.Pool(processes=args.num_workers) as pool:
        res = pool.imap_unordered(partial(file_routine, src_dir=src_base_dir), files)
        for _ in tqdm(res, total=len(files), desc="Processing results"):
            pass


if __name__ == "__main__":
    main()