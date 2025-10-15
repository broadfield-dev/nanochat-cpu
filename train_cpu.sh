#!/bin/bash
set -e

# This script is a modified version of speedrun.sh for a CPU-only environment.
# It trains a very small model for a few steps to demonstrate functionality.
# It will be very slow.

# Set up directories in the user's home directory, which is always writable.
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
echo "Using base directory: $NANOCHAT_BASE_DIR"

# No wandb for this demo
WANDB_RUN=dummy

# Reset report
echo "Resetting report..."
python -m nanochat.report reset

# --- Tokenizer ---
# The Dockerfile has already built the tokenizer. We just need the data.

# Download a small amount of data (8 shards = ~800MB)
echo "Downloading dataset (8 shards)..."
python -m nanochat.dataset -n 8

# Train tokenizer on a small subset of data (100M chars) and a smaller vocab
echo "Training tokenizer..."
python -m scripts.tok_train --max_chars=100000000 --vocab_size=8192
echo "Evaluating tokenizer..."
python -m scripts.tok_eval

# --- Base model (pretraining) ---
# We will train a tiny model for just a few steps.

# Download the eval_bundle for CORE metric
EVAL_BUNDLE_URL=https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
if [ ! -d "$NANOCHAT_BASE_DIR/eval_bundle" ]; then
    echo "Downloading eval_bundle..."
    curl -L -o eval_bundle.zip $EVAL_BUNDLE_URL
    unzip -q eval_bundle.zip
    rm eval_bundle.zip
    mv eval_bundle $NANOCHAT_BASE_DIR
fi

# Pretrain a tiny d4 model for 20 steps
echo "Starting base model pre-training (20 steps)..."
python -m scripts.base_train \
    --depth=4 \
    --device_batch_size=4 \
    --total_batch_size=8192 \
    --num_iterations=20 \
    --eval_every=10 \
    --core_metric_every=15 \
    --sample_every=15 \
    --run=$WANDB_RUN

# Evaluate the tiny model
echo "Evaluating base model loss..."
python -m scripts.base_loss --device_batch_size=4 --split_tokens=32768
echo "Evaluating base model CORE metric..."
python -m scripts.base_eval

# --- Final Report ---
echo "Generating final report..."
python -m nanochat.report generate

echo "CPU training script finished successfully!"
