import hashlib
from typing import Optional

import numpy as np

from pattern_aware_filtering.hashing.flat_hash_set import HASH_TYPE
from pattern_aware_filtering.utils.normalizer import normalize_for_dedup

BYTE_ORDER = "little"
HASH_SIZE = HASH_TYPE(0).nbytes

def compute_hashes(content, span=1) -> Optional[np.ndarray]:
    def _split_lines(lines, span=1):
        lines = [normalize_for_dedup(l) for l in lines]
        if span == 1:
            return lines
        return ["\n".join(lines[i:i + span]) for i in range(0, len(lines) - span + 1)]
    if not content:
        return None
        
    if isinstance(content, str):
        lines = content.split("\n")
    elif isinstance(content, list):
        lines = content
    else:
        raise ValueError(f"Unknown type: {type(content)}")
    # save hashes as bytes but reinterpret them as uint64.
    hashes = np.fromiter(
        (
            hashlib.sha1(bytes(l, encoding="utf-8")).digest()[
                :HASH_SIZE
            ]
            for l in _split_lines(lines, span)
        ),
        dtype=np.dtype((bytes, HASH_SIZE)),
        count=len(lines) - span + 1,
    )
    return np.ndarray(dtype=HASH_TYPE, buffer=hashes.data, shape=hashes.shape)