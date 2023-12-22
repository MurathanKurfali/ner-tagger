#!/bin/bash

system_type=$1

# The name of the model to be fine-tuned.
bert_name=$2

# A flag to activate weighted loss during training. Set to 'true' or '1' to enable.
use_weighted_loss=$3

if [ "$#" -eq 5 ]; then
    data_dir=$4
    # Specifies the number of distinct fine-tuning executions to perform.
    # Each execution is initiated with a different seed to account for fluctuations in results.
    run_count=$5
elif [ "$#" -eq 4 ]; then
    data_dir=$4
    run_count=4
elif [ "$#" -eq 3 ]; then
    data_dir="data"
    run_count=4
else
    echo "Error: Invalid number of arguments."
    echo "Usage: $0 <system-name> <target-model> <loss_choice> [data_dir] [run_count]"
    exit 1
fi

for (( i = 0; i < run_count; i++ )); do
  seed=$(( RANDOM % 100001 ))
  echo "Run number $((i+1)) of $run_count"
  echo "Seed: ${seed} ----"
  python -u scripts/run_ner.py \
        --model_name_or_path "${bert_name}" \
        --task_name "ner" \
        --train_file "${data_dir}/system_${system_type}/train.json" \
        --validation_file "${data_dir}/system_${system_type}/validation.json" \
        --test_file "${data_dir}/system_${system_type}/test.json" \
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


echo "Training is done. The average performance across ${run_count} will be calculated next.
        The results will be saved in system-${system_type}_${bert_name}_${run_count}.md"

python scripts/evaluate_predictions.py "saved_models" ${bert_name}

