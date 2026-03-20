#!/bin/bash
# Distributed training launch script for CryTransformer
#
# Usage:
#   ./train_ddp.sh --config configs/model_medium.yaml --train_list audio_list/train.json
#
# The script automatically detects the number of GPUs and launches training with DDP.

set -e

# Default values
CONFIG="configs/model_medium.yaml"
TRAIN_LIST=""
VAL_LIST=""
EPOCHS=""
BATCH_SIZE=""
LR=""
NGPU=""
LOG_FILE=""
SEED=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --train_list)
            TRAIN_LIST="$2"
            shift 2
            ;;
        --val_list)
            VAL_LIST="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --ngpu)
            NGPU="$2"
            shift 2
            ;;
        --log_file)
            LOG_FILE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config CONFIG       Path to config file (default: configs/model_medium.yaml)"
            echo "  --train_list PATH     Path to training data list JSON (required)"
            echo "  --val_list PATH       Path to validation data list JSON (optional)"
            echo "  --epochs N            Number of epochs (overrides config)"
            echo "  --batch_size N        Batch size per GPU (overrides config)"
            echo "  --lr LR               Learning rate (overrides config)"
            echo "  --ngpu N              Number of GPUs to use (default: auto-detect)"
            echo "  --log_file PATH       Path to log file (optional, logs to console only if not set)"
            echo "  --seed N              Random seed for reproducibility (default: 42)"
            echo ""
            echo "Examples:"
            echo "  # Single GPU training (automatic)"
            echo "  $0 --train_list audio_list/train.json --val_list audio_list/val.json"
            echo ""
            echo "  # Multi-GPU training on 4 GPUs"
            echo "  $0 --ngpu 4 --train_list audio_list/train.json"
            echo ""
            echo "  # Training with custom config"
            echo "  $0 --config configs/model_large.yaml --train_list audio_list/train.json"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$TRAIN_LIST" ]; then
    echo "Error: --train_list is required"
    echo "Use --help for usage information"
    exit 1
fi

# Auto-detect number of GPUs if not specified
if [ -z "$NGPU" ]; then
    if command -v nvidia-smi &> /dev/null; then
        NGPU=$(nvidia-smi -L | wc -l)
    else
        NGPU=1
    fi
fi

echo "=============================================="
echo "CryTransformer Distributed Training"
echo "=============================================="
echo "Config:       $CONFIG"
echo "Train list:   $TRAIN_LIST"
echo "Val list:     ${VAL_LIST:-None}"
echo "GPUs:         $NGPU"
echo "Log file:     ${LOG_FILE:-None}"
echo "Seed:         ${SEED:-42}"
echo "=============================================="

# Build command
CMD="train.py --config $CONFIG --train_list $TRAIN_LIST"

if [ -n "$VAL_LIST" ]; then
    CMD="$CMD --val_list $VAL_LIST"
fi

if [ -n "$EPOCHS" ]; then
    CMD="$CMD --epochs $EPOCHS"
fi

if [ -n "$BATCH_SIZE" ]; then
    CMD="$CMD --batch_size $BATCH_SIZE"
fi

if [ -n "$LR" ]; then
    CMD="$CMD --lr $LR"
fi

if [ -n "$LOG_FILE" ]; then
    CMD="$CMD --log_file $LOG_FILE"
fi

if [ -n "$SEED" ]; then
    CMD="$CMD --seed $SEED"
fi

# Launch training
if [ "$NGPU" -gt 1 ]; then
    echo "Launching distributed training with $NGPU GPUs..."
    echo ""

    # Try torchrun first (PyTorch >= 1.9)
    if command -v torchrun &> /dev/null; then
        echo "Using torchrun for distributed launch"
        torchrun \
            --standalone \
            --nnodes=1 \
            --nproc_per_node=$NGPU \
            $CMD
    else
        # Fall back to deprecated torch.distributed.launch
        echo "torchrun not found, using torch.distributed.launch"
        python -m torch.distributed.launch \
            --nproc_per_node=$NGPU \
            --use_env \
            $CMD
    fi
else
    echo "Launching single GPU training..."
    echo ""
    python $CMD
fi

echo ""
echo "Training completed!"
