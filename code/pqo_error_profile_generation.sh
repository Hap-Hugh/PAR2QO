#!/bin/bash

# CMD="python import_raw_data.py"
# echo "Executing: $CMD"
# eval $CMD

# q=18
t=0
qs=(3)
n_samples=( 50 400 )
workloads=( csv kepler cardinality )

for q in "${qs[@]}"; do
############### Convert raw data to sample.csv
    for workload in "${workloads[@]}"; do
        for n in "${n_samples[@]}"; do
            CMD="python ./data/imdb-new/trans_pqo_combination_to_csv.py --q=${q} --t=${t} --n=${n} --workload=${workload}"
            echo "Executing: $CMD"
            eval $CMD
        done
    done
# # # python ./data/imdb-new/trans_pqo_combination_to_csv.py --q=7 --t=0 --n=50 --workload=csv

############### Generate the error profile
    for workload in "${workloads[@]}"; do
        for n in "${n_samples[@]}"; do
            CMD="python ./gen_real_error_pqo.py --q=${q} --t=${t} --n=${n} --workload=${workload}"
            echo "Executing: $CMD"
            eval $CMD
        done
    done
# # # python ./gen_real_error_pqo.py --q=7 --t=0 --n=50 --workload=csv
done