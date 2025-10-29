import os
import subprocess as sp

from pattern_aware_filtering.utils.snapshots import get_snapshots
from pattern_aware_filtering.utils.constants import CCWET_PATHFILE_DIR

def download_wet_paths(index_name):
    web_url = f"https://data.commoncrawl.org/crawl-data/{index_name}/wet.paths.gz"
    target_path = os.path.join(CCWET_PATHFILE_DIR, f"{index_name}.wet.paths")
    

    # make sure the target directory exists
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    if os.path.exists(target_path):
        print(f"Already downloaded {target_path}")
        return None

    if os.path.exists(target_path + ".gz"):
        cmd = f"gunzip {target_path}.gz"
        print(f"Unzip {target_path}.gz")
        sp.run(cmd.split(" "), check=True)
        return None

    print(f"Download {web_url} to {target_path}.gz")
    cmd = f"wget {web_url} -O {target_path}.gz"
    sp.run(cmd.split(" "), check=True)
    cmd = f"gunzip {target_path}.gz"
    sp.run(cmd.split(" "), check=True)
    return None


snapshots = get_snapshots()

for snapshot in snapshots:
    snapshot = snapshot.strip()
    download_wet_paths("CC-MAIN-" + snapshot)