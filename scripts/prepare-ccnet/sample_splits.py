import json
import random

from pattern_aware_filtering.utils.ccpath import get_all_wet_paths
from argparse import ArgumentParser


def ceil(num, divisor):
    return (num + divisor - 1) // divisor

# enlgish:
def english_routine(wet_paths):
    num_shards = 1001
    shard_size = ceil(len(wet_paths), num_shards)
    random.shuffle(wet_paths)

    shard_paths = [wet_paths[i*shard_size:(i+1)*shard_size] for i in range(num_shards - 1)]
    shard_paths.append(wet_paths[(num_shards - 1)*shard_size:])

    # for each shard, split into 5 pieces. This makes it easier to handle the files.
    # example: shard 0 -> [shard_0_0, shard_0_1, shard_0_2, shard_0_3, shard_0_4]
    shard_paths = [shard_paths[i][j*ceil(len(shard_paths[i]), 5):(j+1)*ceil(len(shard_paths[i]), 5)] for i in range(num_shards) for j in range(5)]

    data = {
        "train" : [shard_paths[i] for i in range(0, (num_shards - 1)*5)],
        "valid" : shard_paths[(num_shards - 1)*5:],
    }
    print(f"Number of shards: {len(shard_paths)}")   
    print("Number of paths in each shard:", shard_size)
    print("Number of paths in the last shard:", len(shard_paths[-1]))
    print("Total number of paths:", len(wet_paths))

    with open("wet_paths_en.json", "w") as f:
        json.dump(data, f, indent=4)

# korean:
def korean_routine(wet_paths):
    num_shards = 101
    shard_size = ceil(len(wet_paths), num_shards)
    random.shuffle(wet_paths)
    shard_paths = [wet_paths[i*shard_size:(i+1)*shard_size] for i in range(num_shards - 1)]
    shard_paths.append(wet_paths[(num_shards - 1)*shard_size:])
    print(f"Number of shards: {len(shard_paths)}")
    print("Number of shards:", num_shards)
    print("Number of paths in each shard:", shard_size)
    print("Number of paths in the last shard:", len(shard_paths[-1]))
    print("Total number of paths:", len(wet_paths)) 
    data = {
        "train" : [shard_paths[i] for i in range(0, num_shards - 1)],
        "valid" : shard_paths[num_shards - 1:],
    }
    with open("wet_paths_ko.json", "w") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("mode", type=str, choices=["en", "ko"], help="Mode: 'en' for English, 'ko' for Korean")
    args = parser.parse_args()

    wet_paths = get_all_wet_paths()

    if args.mode == "en":
        english_routine(wet_paths)
    elif args.mode == "ko":
        korean_routine(wet_paths)