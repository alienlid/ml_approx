#!/bin/bash
#SBATCH -J distilbert_ag_news
#SBATCH -o logs/distilbert_ag_news_%j.out
#SBATCH -e logs/distilbert_ag_news_%j.err

#SBATCH --mail-user=ddl@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --exclusive

## Activate WMLCE virtual environment 
source ~/.bashrc
conda activate ml_approx
ulimit -s unlimited

## Creating SLURM nodes list
export NODELIST=nodelist.$
srun -l bash -c 'hostname' |  sort -k 2 -u | awk -vORS=, '{print $2":4"}' | sed 's/,$//' > $NODELIST

## Number of total processes 
echo " "
echo " Nodelist:= " $SLURM_JOB_NODELIST
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " NGPUs per node:= " $SLURM_GPUS_PER_NODE 
echo " Ntasks per node:= "  $SLURM_NTASKS_PER_NOD


####    Use MPI for communication with Horovod - this can be hard-coded during installation as well.
export HOROVOD_GPU_ALLREDUCE=MPI
export HOROVOD_GPU_ALLGATHER=MPI
export HOROVOD_GPU_BROADCAST=MPI
export NCCL_DEBUG=DEBUG

echo " Running on multiple nodes and GPU devices"
echo ""
echo " Run started at:- "
date

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

## Horovod execution
for method in "${methods[@]}"; do
    horovodrun -np $SLURM_NTASKS -H `cat $NODELIST` python distilbert_ag_news.py --method "$method" --metric acc > "logs/${method}.log" 2>&1
done

echo "Run completed at:- "
date