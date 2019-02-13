#!/bin/sh --login

#SCW_TPN_OVERRIDE=1 #overide warning about 1 node
# BASH Environment Variable	           SBATCH Field Code	Description
# $SLURM_JOB_ID	                        %J	                Job identifier
# $SLURM_ARRAY_JOB_ID	                %A	                Array parent job identifier
# $SLURM_ARRAY_TASK_ID	                %a	                Array job iteration index

#SBATCH --gres=gpu:1
#SBATCH -p gpu

#SBATCH --job-name=pip_only
#SBATCH -o output-%J.o
#SBATCH -n 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1

module load CUDA/9.1
module load pip 
python -m pip install tensorflow-gpu --user



MODEL_NAME="wrn"
CHECKPOINT_DIR="$(pwd)/checkpoints_$1"
DATA_PATH="/home/c.c0919382/datasets/cifar-10-batches-py"
DATASET="cifar10"

python test_without_flags.py $DATA_PATH