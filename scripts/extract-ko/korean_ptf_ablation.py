import os
import json
import gzip
from tqdm import tqdm
import multiprocessing as mp
from argparse import ArgumentParser
from functools import partial
from glob import glob
import re
import io

from pattern_aware_filtering.extraction.extractor import pld_extractor, ptf_extractor
from pattern_aware_filtering.utils.constants import KO_EXTRACTED_PATH

ablation_red = [50]
ablation_green = [3]
ks = [1, 3, 5, 7, 10, 15]

io.DEFAULT_BUFFER_SIZE = 1024 * 1024 * 1024 # Set buffer size to 1GiB


punc_pattern_main_text_pttn = [r"g+",  r"g+([yr]{0,5}g+)+"]
punc_patterns = []
for k in ks:
    p_replaced = [p.replace("5}", f"{k}" + "}") for p in punc_pattern_main_text_pttn]
    compiled = [re.compile(pttn) for pttn in p_replaced]
    punc_patterns.append(compiled)


src_base = KO_EXTRACTED_PATH
def replace_filename(fpath, src_dir, red, green, k):
    exp_name = f"pld-r{red}-g{green}-ptf-{k}"
    tgt_dir = os.path.join(src_base, exp_name)
    return fpath.replace(src_dir, tgt_dir)

def file_routine(fpath, src_dir):
    # 3d list: [red][green][k]
    # data = [[[] for _ in range(len(ablation_green))] for _ in range(len(ablation_red))]
    data = [[[[] for _ in range(len(ks))] for _ in range(len(ablation_green))] for _ in range(len(ablation_red))]

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
                    lines = pld_extractor(original_lines, count, red=red, green=green, use_heuristic=False)
                    for k_idx, k in enumerate(ks):
                        compiled = punc_patterns[k_idx]
                        cp_lines = ptf_extractor(lines, compiled)
                        if not cp_lines:
                            continue
                        data[i][j][k_idx].append({
                            "text": "\n".join(cp_lines),
                        })
    
    for i, red in enumerate(ablation_red):
        for j, green in enumerate(ablation_green):
            for k_idx, k in enumerate(ks):
                save_path = replace_filename(fpath, src_dir, red, green, k)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                buf = io.StringIO()
                for item in data[i][j][k_idx]:
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