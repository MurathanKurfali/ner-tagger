#!/bin/bash

system_type=$1

# The name of the model to be fine-tuned.
bert_name=$2

# A flag to activate weighted loss during training. Set to 'true' or '1' to enable.
use_weighted_loss=$3

# Specifies the number of distinct fine-tuning executions to perform.
# Each execution is initiated with a different seed to account for fluctuations in results.
run_count=4

for (( i = 0; i < run_count; i++ )); do
  seed=$(( RANDOM % 100001 ))
  echo "Run number $((i+1)) of $run_count"
  echo "Seed: ${seed} ----"
  python -u scripts/run_ner.py \
        --model_name_or_path "${bert_name}" \
        --task_name "ner" \
        --train_file "data/system_${system_type}/train.json" \
        --validation_file "data/system_${system_type}/validation.json" \
        --test_file "data/system_${system_type}/test.json" \
        --output_dir "saved_models/system_${system_type}/${bert_name}_${seed}" \
        --text_column_name "tokens" \
        --label_column_name "ner_tags" \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 64 \
        --do_train \
        --use_weighted_loss "${use_weighted_loss}"\
        --seed ${seed} \
        --do_eval \
        --do_predict \
        --overwrite_cache \
        --overwrite_output_dir \
        --return_entity_level_metrics \
        --save_total_limit 2 \
        --greater_is_better "True" \
        --metric_for_best_model "eval_overall_f1" \
        --load_best_model_at_end \
        --save_strategy steps \
        --evaluation_strategy steps \
        --save_steps 1000 \
        --eval_steps 1000
done