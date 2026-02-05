#!/bin/bash
# SIEVE Example: Soft Distillation Data Generation
# Generates teacher token distributions for soft distillation training

export TEACHER_MODEL="Qwen/Qwen3-8B"
export STUDENT_TOKENIZER="Qwen/Qwen3-8B"
export INPUT_DATASET="synthetic_data.parquet"
export SAMPLES_PER_INPUT=1
export K_TOKENS=100
export MAX_MODEL_LEN=32768
export TENSOR_PARALLEL_SIZE=8
export DATA_PARALLEL_SIZE=1
export TEMPERATURE=0.7
export MAX_WORKERS=100
export OUTPUT_PATH="soft_distill_data.parquet"

python -m sieve.soft_distillation_data \
  --input_dataset "${INPUT_DATASET}" \
  --teacher_model "${TEACHER_MODEL}" \
  --student_tokenizer_path "${STUDENT_TOKENIZER}" \
  --samples_per_input ${SAMPLES_PER_INPUT} \
  --k_tokens ${K_TOKENS} \
  --max_model_len ${MAX_MODEL_LEN} \
  --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
  --data_parallel_size ${DATA_PARALLEL_SIZE} \
  --temperature ${TEMPERATURE} \
  --max_workers ${MAX_WORKERS} \
  --output_path "${OUTPUT_PATH}"
