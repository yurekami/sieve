#!/bin/bash
# SIEVE Example: Soft Distillation Training
# Trains student model via soft distillation with DeepSpeed ZeRO-3

export STUDENT_MODEL="Qwen/Qwen3-8B"
export TRAIN_DATA="soft_distill_data.parquet"
export RUN_ID="sieve_example_run"
export NUM_EPOCHS=2
export LEARNING_RATE=1e-5
export TEMPERATURE=1.0
export MAX_LENGTH=16384
export TRAIN_SIZE=32768
export TOPK=100

accelerate launch --config-file configs/zero3.yaml --num-processes 8 \
  sieve/soft_distillation_trainer.py \
    --run_identifier "${RUN_ID}" \
    --num_train_epochs ${NUM_EPOCHS} \
    --data_path "${TRAIN_DATA}" \
    --model_to_train "${STUDENT_MODEL}" \
    --temperature ${TEMPERATURE} \
    --learning_rate ${LEARNING_RATE} \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_length ${MAX_LENGTH} \
    --warmup_steps 50 \
    --save_steps 500 \
    --logging_steps 10 \
    --train_size ${TRAIN_SIZE} \
    --topk ${TOPK} \
    --remove_feedback
