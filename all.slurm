#!/bin/bash
#SBATCH -J distilbert_ag_news
#SBATCH -o logs/distilbert_ag_news_%j.out
#SBATCH -e logs/distilbert_ag_news_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --mem=128G

# Activate your environment
source .venv/bin/activate
# source ~/.bashrc
# conda activate ml_approx

# Define your list of methods
methods=(
    "weighted_rand_sparse_0"
    "weighted_rand_sparse_1"
    "weighted_rand_sparse_2"
    "exact_sparse"
    "rand_svd_0"
    "rand_svd_1"
    "rand_svd_2"
    "low_rank"
    "quantize"
)

NUM_GPUS=4 # Modify this if not using slurm

for idx in "${!methods[@]}"
do
    method="${methods[$idx]}"
    gpu_id=$(( idx % NUM_GPUS ))
    echo "Running method: $method on GPU $gpu_id"

    CUDA_VISIBLE_DEVICES=$gpu_id python distilbert_ag_news.py --method "$method" --metric acc > "logs/${method}.log" 2>&1 &
done

wait
echo "All methods completed."
