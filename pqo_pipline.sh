#!/bin/bash

t=0
n_samples=( 50 400 )
workloads=( csv cardinality kepler )

qs=( 18 )
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

############### Generate pqo anchors
    for workload in "${workloads[@]}"; do
        for n in "${n_samples[@]}"; do
            CMD="python robustness.py --query_id=${q}a --find_anchors --template_id=${t} --n=${n} --workload=${workload}"
            echo "Executing: $CMD"
            eval $CMD
        done
    done
# # # python robustness.py --query_id=7a --find_anchors --template_id=0 --n=50 --workload=csv


############### Generate rqo sen_dim and cached robust plan
    # Directory containing the SQL files
    # DIRECTORY="query/join-order-benchmark/rank-by-prob/"
    for workload in "${workloads[@]}"; do  
        DIRECTORY="query/join-order-benchmark/on-demand/${workload}_workload/"
        for n in "${n_samples[@]}"; do
            for FILE in "$DIRECTORY"*.sql; do
                BASENAME=$(basename "$FILE")
                # query_id (first number in the file name)
                QUERY_ID=$(echo "$BASENAME" | sed -n 's/^q\([0-9]\+\)-.*/\1/p')
                # template_id
                TEMPLATE_ID=$(echo "$BASENAME" | sed -n 's/^.*-t\([0-9]\+\)-.*/\1/p')
                # N
                N=$(echo "$BASENAME" | sed -n 's/^.*-t[0-9]\+-\([0-9]\+\)-.*/\1/p')

                if [[ "$QUERY_ID" == "$q" ]] && [[ "$TEMPLATE_ID" == "$t" ]] && [[ "$N" == "$n" ]]; then
                    CMD="python robustness.py --query_id=${q}a --rqo_query='$FILE' --template_id=${t} --n=${n} --workload=${workload} --cal_sen=sobol"
                    echo "Executing: $CMD"
                    eval $CMD
                    # CMD="python robustness.py --query_id=${q}a --rqo_query='$FILE' --template_id=${t} --n=${n} --workload=${workload}"
                    # echo "Executing: $CMD"
                    # eval $CMD
                    CMD="python robustness.py --query_id=${q}a --rqo_query='$FILE' --template_id=${TEMPLATE_ID} --n=${N} --workload=${workloads} --exe" # for debug purposes
                    echo "Executing: $CMD"
                    eval $CMD
                fi
            done
        done
    done
# # # python robustness.py --query_id=7a --rqo_query='query/join-order-benchmark/on-demand/csv_workload/q7-t1-50-0-b0.5.sql' --template_id=1 --n=50 --workload=csv --cal_sen=sobol
# # # python robustness.py --query_id=7a --rqo_query='query/join-order-benchmark/on-demand/csv_workload/q7-t1-50-0-b0.5.sql' --template_id=1 --n=50 --workload=csv

############### Generate robust plan with pqo
    for workload in "${workloads[@]}"; do
        for n in "${n_samples[@]}"; do
            CMD="python robustness.py --query_id=${q}a --pqo --template_id=${t} --n=${n} --workload=${workload}"
            echo "Executing: $CMD"
            eval $CMD
        done
    done

############### Generate robust plan with pqo
    for workload in "${workloads[@]}"; do
        for n in "${n_samples[@]}"; do
            CMD="python ratio_calculate.py --qid=${q} --tid=${t} --n=${n} --workload=${workload}"
            echo "Executing: $CMD"
            eval $CMD
        done
    done
done