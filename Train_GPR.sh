#!/bin/bash

# Load environment variables from .env file
source .env

# Total number of jobs
TOTAL_JOBS=100
FILE_PATH="data/input_data_n_1000_q_sample_max_0_9_q_star_max_0_90_csv"

# Directory for logs
LOGS_DIR="./logs"
mkdir -p $LOGS_DIR
rm -rf $LOGS_DIR/*

# Generate a random seed for each job
SEED=$RANDOM

# Submit jobs as an array
sbatch --parsable \
    $SBATCH_OPTIONS \
    --array=1-$TOTAL_JOBS \
    --job-name="train GPR" \
    --output="$LOGS_DIR/output_%A_%a.txt" \
    --error="$LOGS_DIR/error_%A_%a.txt" \
    --wrap="$SBATCH_MODULES; python Train_GPR.py $FILE_PATH --seed \$((SEED + SLURM_ARRAY_TASK_ID)) --n_jobs 1"
