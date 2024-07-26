#!/bin/bash

POPULATIONS = (10 100 250 500 1000 10000)
MUTATION_RATES = (0.01 0.1 0.25 0.5 0.75 1.0)
MAX_GEN = (1000)
RUNS = 100
ITERATIONS = $((${#POPULATIONS[@]} * ${#MUTATION_RATES[@]} * ${#MAX_GEN[@]} * $RUNS))

for pop in "${POPULATIONS[@]}"
do
    for mut in "${MUTATION_RATES[@]}"
    do
        for gen in "${MAX_GEN[@]}"
        do
            for run in $(seq 1 $RUNS)
            do
                echo "Run $run of $ITERATIONS"
                python3 main.py -p $pop -m $mut -g $gen -P -D
            done
        done
    done
done
