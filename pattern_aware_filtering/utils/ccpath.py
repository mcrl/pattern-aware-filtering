import os
from pattern_aware_filtering.utils.constants import CCWET_PATH, KO_CCWET_PATH, EN_CCWET_PATH, PROJECT_ROOT

def get_wet_paths_file(snapshot, limit=-1):
    """Note that sorting the result might result in different order defined in the wet.paths file, because of numbering."""
    fpath = f"{PROJECT_ROOT}/../cache/wet_paths/CC-MAIN-{snapshot}.wet.paths"
    with open(fpath, "r") as f:
        paths = f.readlines()
    if limit > 0:
        paths = paths[:limit]
    return [p.strip() for p in paths]

def make_raw_wet_path(wet_path):
    replaced = wet_path.replace("crawl-data", CCWET_PATH)
    # check if file exists
    if os.path.exists(replaced):
        return replaced
    raise FileNotFoundError(f"File not found: {replaced}")
    

def make_korean_extracted_path(wet_path):
    replaced = wet_path.replace("crawl-data", KO_CCWET_PATH)
    fpath = replaced.replace(".warc.wet.gz", ".json.gz")
    return fpath

def make_english_extracted_path(wet_path):
    replaced = wet_path.replace("crawl-data", EN_CCWET_PATH)
    fpath = replaced.replace(".warc.wet.gz", ".json.gz")
    return fpath


def get_target_snapshots():
    fpath = f"{PROJECT_ROOT}/utils/snapshots.txt"
    with open(fpath, "r") as f:
        snapshots = f.readlines()
    return [snapshot.strip() for snapshot in snapshots]
    
def get_all_wet_paths():
    snapshots = get_target_snapshots()
    all_wet_paths = []
    for snapshot in snapshots:
        wet_paths = get_wet_paths_file(snapshot)
        all_wet_paths.extend(wet_paths)
    all_wet_paths.sort()
    return all_wet_paths