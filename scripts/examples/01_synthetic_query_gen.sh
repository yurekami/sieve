#!/bin/bash
# SIEVE Example: Synthetic Query Generation
# Generates synthetic queries from natural language feedback using base + instruction models

export BASE_MODEL="Qwen/Qwen3-8B-Base"
export INSTRUCTION_MODEL="Qwen/Qwen3-8B"
export N_EXAMPLES=16384
export MAX_MODEL_LEN=32768
export TENSOR_PARALLEL_SIZE=4
export TEMPERATURE_BASE=1.0
export TEMPERATURE_INSTRUCTION=0.7
export MAX_WORKERS=50
export OUTPUT_PATH="synthetic_data.parquet"

python -m retail.synthetic_data_gen \
  --start_servers \
  --base_model_path "${BASE_MODEL}" \
  --instruction_model_path "${INSTRUCTION_MODEL}" \
  --base_model_gpus "0,1,2,3" \
  --instruction_model_gpus "4,5,6,7" \
  --base_model_port 8000 \
  --instruction_model_port 8001 \
  --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
  --max_model_len ${MAX_MODEL_LEN} \
  --temperature_base ${TEMPERATURE_BASE} \
  --temperature_instruction ${TEMPERATURE_INSTRUCTION} \
  --n_examples ${N_EXAMPLES} \
  --max_workers ${MAX_WORKERS} \
  --output_path "${OUTPUT_PATH}" \
  --include_feedback \
  --use_base_model \
  --verify_feedback
