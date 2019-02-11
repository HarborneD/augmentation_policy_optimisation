#!/bin/sh --login

#SCW_TPN_OVERRIDE=1 #overide warning about 1 node
# BASH Environment Variable	           SBATCH Field Code	Description
# $SLURM_JOB_ID	                        %J	                Job identifier
# $SLURM_ARRAY_JOB_ID	                %A	                Array parent job identifier
# $SLURM_ARRAY_TASK_ID	                %a	                Array job iteration index

#SBATCH --gres=gpu:2
#SBATCH -p gpu

#SBATCH --job-name=eval_child_model
#SBATCH -o output-%J.o
#SBATCH -n 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1

module load CUDA/9.1
module load pip
python -m pip install --upgrade --force-reinstall pip --user
module load tensorflow


MODEL_NAME="wrn"
CHECKPOINT_DIR="$(pwd)/checkpoints_$1"
DATA_PATH="/home/c.c0919382/datasets/cifar-10-batches-py"
DATASET="cifar10"

python -m evaluator.evaluate_policies.py \
    --model_name=$MODEL_NAME \
    --checkpoint_dir=$CHECKPOINT_DIR \
    --data_path $DATA_PATH \
    --dataset=$DATASET \
    --use_cpu=0 \
    --policy_id=$1 \
    --num_epochs=$2

