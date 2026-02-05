#!/bin/bash
# Cartridges Baseline for MTOB (Machine Translation) Domain
#
# This script runs the complete Cartridges baseline pipeline:
# 1. Install cartridges library
# 2. Synthesize training data using self-study
# 3. Train a cartridge (learned KV cache)
# 4. Evaluate on translation task
#
# Hyperparameters match the paper's settings

set -e  # Exit on error

# Configuration - matching paper settings
export CARTRIDGES_MODEL_NAME="Qwen/Qwen3-4B"  # Use 4B for synthesis
export CARTRIDGES_SGLANG_URL="http://localhost:8000"
export CARTRIDGES_NUM_SAMPLES=65536  # Paper uses 65K samples
export CARTRIDGES_PROB_THINKING=0    # No thinking mode
export CARTRIDGES_OUTPUT_DIR="./cartridges_output"
export MTOB_SETUP="latex_and_sentences"  # Use full LaTeX grammar book + sentences

# Training configuration
export MODEL="qwen8b"           # Train on 8B model
export NUM_TOKENS=4096          # KV cache size (must be multiple of 16)

echo "=== Cartridges Baseline for MTOB Domain ==="
echo "Synthesis Model: $CARTRIDGES_MODEL_NAME"
echo "Training Model: Qwen/Qwen3-8B"
echo "Num Samples: $CARTRIDGES_NUM_SAMPLES"
echo "KV Cache Tokens: $NUM_TOKENS"
echo "Setup: $MTOB_SETUP"
echo "Output directory: $CARTRIDGES_OUTPUT_DIR"
echo ""

# Step 0: Install cartridges library
echo "[0/3] Installing cartridges library..."
cd baselines/cartridges
pip install -e .
cd ../..
echo ""

# Step 1: Synthesize training data
echo "[1/3] Synthesizing training data..."
echo "This requires an SGLang server running at $CARTRIDGES_SGLANG_URL"
echo "Expected: Qwen/Qwen3-4B on 8 GPUs (--tp 8)"
echo ""
python -m mtob.cartridges_baseline.cartridges_synthesize

# The synthesized data will be saved to $CARTRIDGES_OUTPUT_DIR/mtob_synthesize_*/dataset.parquet
# Find the most recent output directory
SYNTH_DATA_DIR=$(find $CARTRIDGES_OUTPUT_DIR -name "mtob_synthesize_*" -type d | sort -r | head -1)
export DATA_SOURCES="$SYNTH_DATA_DIR/dataset.parquet"

echo ""
echo "Synthesized data saved to: $DATA_SOURCES"
echo ""

# Step 2: Train cartridge
echo "[2/3] Training cartridge..."
echo "Using torchrun for distributed training (8 GPUs)"
torchrun --standalone --nproc_per_node=8 \
  mtob/cartridges_baseline/cartridges_train.py

# Step 3: Evaluate
echo "[3/3] Evaluating cartridge..."
python -m mtob.cartridges_baseline.cartridges_eval

echo ""
echo "=== Cartridges baseline complete! ==="
