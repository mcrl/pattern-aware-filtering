# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# Modified by Chanwoo Park

import numpy as np
from collections import Counter
from typing import Iterable, Iterator, Sequence, Sized, Tuple, Type

HASH_TYPE: Type[np.uint64] = np.uint64
RECORD_TYPE: Type[np.uint16] = np.uint16

def truncate_sum(
    arr1, arr2, arr1_transform=True, sum_dtype=np.uint32, result_dtype=np.uint8
):
    """Perform element-wise sum of two arrays with truncation to avoid overflow.

    Args:
        arr1 (np.ndarray): First input array.
        arr2 (np.ndarray): Second input array.
        arr1_transform (bool): Whether to transform `arr1` to `sum_dtype`.
        sum_dtype (np.dtype): Data type for the summation.
        result_dtype (np.dtype): Data type for the result.

    Returns:
        np.ndarray: Resultant array after summation and truncation.
    """
    arr1_truncated = arr1.astype(sum_dtype) if arr1_transform else arr1
    arr2_truncated = arr2.astype(sum_dtype)
    arr3 = arr1_truncated + arr2_truncated
    return np.where(
        arr3 > np.iinfo(result_dtype).max, np.iinfo(result_dtype).max, arr3
    ).astype(result_dtype)

class AbstractDedupHashSet(Sized, Iterable[np.uint64]):
    """Abstract base class for a deduplication hash set with batch operations."""

    dtype: Type[np.uint64] = HASH_TYPE

    def __repr__(self):
        implementation = type(self).__name__
        return f"[{implementation}, len: {len(self)}]"

    def __len__(self) -> int:
        ...

    def __contains__(self, values: Sequence[np.uint64]) -> np.ndarray:
        ...

    def __getitem__(self, values) -> np.ndarray:
        ...

    def __setitem__(self, keys, values) -> None:
        ...

    def items(self) -> Iterable[Tuple[np.uint64, RECORD_TYPE]]:
        ...

    def keys(self) -> Iterable[np.uint64]:
        ...

    def __iter__(self) -> Iterator[np.uint64]:
        return iter(self.keys())

    def add(self, h, contains=None):
        """Add keys to the hash set, updating values for duplicate entries."""
        if not isinstance(h, np.ndarray):
            h = np.array(h, dtype=HASH_TYPE)
        if contains is None:
            contains = self.__contains__(h)
        self.__setitem__(h, contains)
        return contains

    def merge(self, keys, values):
        """Merge given keys and values into the hash set."""
        contains = self.__contains__(keys)
        self.__setitem__(keys, contains | values)

    def dump(self, filename):
        """Dump hash set data to a file."""
        self.dump_np(filename)

    def load(self, filename):
        """Load hash set data from a file."""
        self.load_np(filename)

    def dump_np(self, filename):
        """Dump hash set as a NumPy structured array."""
        kv_type = np.dtype([("k", HASH_TYPE), ("v", RECORD_TYPE)])
        items = np.fromiter(self.items(), dtype=kv_type, count=len(self))
        with open(filename, "wb") as f:
            np.save(f, items)

    def load_np(self, filename):
        """Load hash set from a NumPy structured array."""
        items = np.load(str(filename))
        keys = items["k"].copy()
        values = items["v"].copy().astype(RECORD_TYPE)
        self.merge(keys, values)

class NaiveHashSet(dict, AbstractDedupHashSet):
    """Pure Python implementation of AbstractDedupHashSet using a dictionary."""

    def __init__(self, iterable=None):
        super().__init__()

    def __contains__(self, values):
        """Check if values are in the hash set."""
        contains_point = super().__contains__
        return np.fromiter(
            map(contains_point, values), count=len(values), dtype=RECORD_TYPE
        )

    def __getitem__(self, values):
        """Get values corresponding to the keys in the hash set."""
        get_point = super().get
        return np.fromiter(
            map(lambda x: get_point(x, False), values),
            count=len(values),
            dtype=RECORD_TYPE,
        )

    def __setitem__(self, keys, values):
        assert len(keys) == len(values)
        for k, v in zip(keys, values):
            dict.__setitem__(self, k, v)

class MultipleHashSet(Counter, AbstractDedupHashSet):
    """Hash set that counts occurrences of each key."""

    def __init__(self, iterable=None, count_crit=1):
        super().__init__()
        self.count_crit = count_crit

    def __contains__(self, values):
        """Check if keys have been added more than `count_crit` times."""
        get_count = super().__getitem__
        return np.fromiter(
            (get_count(x) > self.count_crit for x in values),
            count=len(values),
            dtype=RECORD_TYPE,
        )

    def __getitem__(self, values):
        """Get occurrence counts for the keys."""
        get_point = super().get
        return np.fromiter(
            (get_point(x, 0) for x in values), count=len(values), dtype=RECORD_TYPE
        )

    def __setitem__(self, keys, values):
        assert len(keys) == len(values)
        for k, v in zip(keys, values):
            dict.__setitem__(self, k, v)

    def add(self, h, contains=None):
        """Add keys, incrementing their occurrence counts."""
        if not isinstance(h, np.ndarray):
            h = np.array(h, dtype=HASH_TYPE)
        if contains is None:
            contains = self.__getitem__(h)
        contains += 1
        self.__setitem__(h, contains)
        return contains

    def merge(self, keys, values):
        """Merge keys and values into the hash set with truncation."""
        contains = self.__getitem__(keys)
        res = truncate_sum(
            contains, values, sum_dtype=np.uint32, result_dtype=RECORD_TYPE, arr1_transform=True
        )
        self.__setitem__(keys, res)
    
    def remove_all_less_than(self, count):
        """Remove all entries whose value is less than the given count."""
        keys = list(self.keys())
        for k in keys:
            v = self.get(k, 0)
            if v < count:
                self.pop(k)


class CountHashSet(MultipleHashSet):
    """CountHashSet accurately counts all occurrences of each key.

    Unlike MultipleHashSet, this implementation increments by the number of occurrences added.
    """

    def __getitem__(self, values):
        """Get occurrence counts for the keys."""
        get_point = super().get
        return np.fromiter(
            (get_point(x, 0) for x in values), count=len(values), dtype=RECORD_TYPE
        )

    def add(self, h, contains=None):
        """Add keys, incrementing occurrence counts accurately."""
        if not isinstance(h, np.ndarray):
            h = np.array(h, dtype=HASH_TYPE)
        
        unique, counts = np.unique(h, return_counts=True)
        counts = counts.astype(RECORD_TYPE)
        if contains is None:
            contains = self.__getitem__(unique)
        contains = truncate_sum(contains, counts, sum_dtype=np.uint32, result_dtype=RECORD_TYPE)
        self.__setitem__(unique, contains)
        contains = self.__getitem__(h)
        return contains


FlatHashSet = MultipleHashSet
