POLICY_ID="000001"
echo policy_id: "$POLICY_ID"

NUM_EPOCHS="2"
echo num_epochs: "$NUM_EPOCHS"

MODEL_NAME="wrn"
CHECKPOINT_DIR="$(pwd)/checkpoints/checkpoints_$1"
DATA_PATH="/media/harborned/ShutUpN/datasets/cifar/cifar-10-batches-py"
DATASET="cifar10"
USE_CPU=0

echo datapath - "$DATA_PATH"

python evaluate_child_model.py --data_path $DATA_PATH --policy_id $POLICY_ID --num_epochs $NUM_EPOCHS --model_name $MODEL_NAME --dataset $DATASET --use_cpu $USE_CPU
      