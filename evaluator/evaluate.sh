#!/bin/sh

MODEL_NAME="wrn"
CHECKPOINT_DIR="$(pwd)/checkpoints_$1"
DATA_PATH="/media/harborned/ShutUpN/datasets/cifar/cifar-10-batches-py"
DATASET="cifar10"

python2 -m evaluator.evaluate_policies.py \
    --model_name=$MODEL_NAME \
    --checkpoint_dir=$CHECKPOINT_DIR \
    --data_path $DATA_PATH \
    --dataset=$DATASET \
    --use_cpu=0 \
    --policy_id=$1 \
    --num_epochs=$2

