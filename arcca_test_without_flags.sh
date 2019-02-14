#!/bin/sh --login

#SCW_TPN_OVERRIDE=1 #overide warning about 1 node
# BASH Environment Variable	           SBATCH Field Code	Description
# $SLURM_JOB_ID	                        %J	                Job identifier
# $SLURM_ARRAY_JOB_ID	                %A	                Array parent job identifier
# $SLURM_ARRAY_TASK_ID	                %a	                Array job iteration index

#SBATCH --gres=gpu:1
#SBATCH -p gpu

#SBATCH --job-name=eval_child_model
#SBATCH -o output-%J.o
#SBATCH -n 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1

echo eval_child_model

module load CUDA/9.1
module load tensorflow
# module load pip 
# echo now load TF
# module load tensorflow
# echo tf loaded

echo 1 - "$1"mo

MODEL_NAME="wrn"
CHECKPOINT_DIR="$(pwd)/checkpoints/checkpoints_$1"
DATA_PATH="/home/c.c0919382/datasets/cifar-10-batches-py"
DATASET="cifar10"

echo datapath - "$DATA_PATH"

python2 test_without_flags.py $DATA_PATH