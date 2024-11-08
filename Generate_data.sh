#!/bin/bash

# Load environment variables from .env file
source .env

# Number of samples per job
N_SAMPLES=10

# Total number of jobs
TOTAL_JOBS=$((1000 / N_SAMPLES))

# Directory for logs
LOGS_DIR="./logs"
mkdir -p $LOGS_DIR
rm -rf $LOGS_DIR/*

# Generate a random seed for each job
SEED=$RANDOM

# Parameters
N=1000
REPETITIONS=500
Q_SAMPLE_MAX=1.0
Q_STAR_MAX=1.0

# Submit jobs as an array
sbatch --parsable \
    $SBATCH_OPTIONS \
    --array=1-$TOTAL_JOBS \
    --job-name="generate_data" \
    --output="$LOGS_DIR/output_%A_%a.txt" \
    --error="$LOGS_DIR/error_%A_%a.txt" \
    --wrap="$SBATCH_MODULES; python Generate_data.py --seed \$((SEED + SLURM_ARRAY_TASK_ID)) --n_samples $N_SAMPLES --n $N --repetitions $REPETITIONS --q_sample_max $Q_SAMPLE_MAX --q_star_max $Q_STAR_MAX"