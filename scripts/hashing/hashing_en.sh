#!/bin/bash
function process_snapshot {
    local snapshot=$1
    
    # get line count from 'cache/wet_paths/CC-MAIN-$snapshot.wet.paths'
    lc=$(wc -l < "cache/wet_paths/CC-MAIN-$snapshot.wet.paths")
    # i range: 0 ~ (lc/1000), ceilting
    end_idx=`expr $lc / 1000`
    if [ `expr $lc % 1000` -ne 0 ]; then
        end_idx=`expr $end_idx + 1`
    fi

    # This loop can be parallelized. No dependency between non-overlapping indexes. 
    # Parallelization is left as an option for the user.
    for (( i=0; i<$end_idx; i++ )); do
        start_idx=`expr $i \* 1000`
        end_idx=`expr $start_idx + 1000`
        python scripts/hashing/sharded_hashing_en.py \
            --snapshot "$snapshot"\
            --index-start $start_idx \
            --index-end $end_idx
    done

    # Then we merge all the hash files using multiprocessing
    python scripts/hashing/merge_each_shard_en.py --snapshot "$snapshot"
}

# For utilizing multiple nodes for processing, you may distribute the workload per snapshot across your cluster.
# No dependency between snapshots.
snapshot_file=pattern_aware_filtering/utils/snapshots.txt
while IFS= read -r snapshot; do
    process_snapshot "$snapshot"
done < "$snapshot_file"
