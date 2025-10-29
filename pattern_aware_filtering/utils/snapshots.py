import os

from pattern_aware_filtering.utils.constants import PROJECT_ROOT

def get_snapshots():
    snapshot_path = os.path.join(PROJECT_ROOT, "utils", "snapshots.txt")

    with open(snapshot_path, "r") as f:
        snapshots = f.readlines()
    snapshots = [snapshot.strip() for snapshot in snapshots]
    return snapshots