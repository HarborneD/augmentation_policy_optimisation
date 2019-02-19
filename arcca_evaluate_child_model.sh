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

POLICY_ID="$1"
echo policy_id: "$POLICY_ID"

NUM_EPOCHS="$2"
echo num_epochs: "$NUM_EPOCHS"

if [ $3 -eq 0 ]; then
    NUM_TRAINING_IMAGES=4000
else
    NUM_TRAINING_IMAGES=$3
fi
echo num_training_images: "$NUM_TRAINING_IMAGES"

MODEL_NAME="wrn"
CHECKPOINT_DIR="$(pwd)/checkpoints/checkpoints_$1"
DATA_PATH="/home/c.c0919382/datasets/cifar-10-batches-py"
DATASET="cifar10"
USE_CPU=0

echo datapath - "$DATA_PATH"

python evaluate_child_model.py --data_path $DATA_PATH --policy_id $POLICY_ID --num_epochs $NUM_EPOCHS --model_name $MODEL_NAME --dataset $DATASET --use_cpu $USE_CPU --num_training_images $NUM_TRAINING_IMAGES
      