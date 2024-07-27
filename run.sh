#!/bin/bash

# Lorenzo D. Moon
# Run 36 Experiments, 100 times, with the eight_queens.py program

PROGRAM="eight_queens.py"
POPULATIONS=(100 250 500 1000 10000)
MUTATION_RATES=(1 10 25 50 75 100)
MAX_GEN=(1000)
RUNS=100
CUR_ITERATION=0
ITERATIONS=$((${#POPULATIONS[@]} * ${#MUTATION_RATES[@]} * ${#MAX_GEN[@]} * $RUNS))

for pop in "${POPULATIONS[@]}"; do
	for mut in "${MUTATION_RATES[@]}"; do
		for gen in "${MAX_GEN[@]}"; do
			for run in $(seq 1 $RUNS); do
                CUR_ITERATION=$((CUR_ITERATION+1))
				echo "Run $CUR_ITERATION of $ITERATIONS: python $PROGRAM -p $pop -m $mut -g $gen -P -D"
                python $PROGRAM -p $pop -m $mut -g $gen -P -D
                if [ $? -ne 0 ]; then
                    echo "Error in run $run of $ITERATIONS"
                    exit 1
                fi
			done
		done
	done
done
