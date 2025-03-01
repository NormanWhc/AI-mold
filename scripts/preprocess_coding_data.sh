#! /bin/bash
model_name_or_path="~/Qwen/Qwen2.5-3B-Instruct"
template="Qwen2.5"

work_time=$(date +%Y%m%d%H%M%S)
model="Qwen2.5"
root="~/AI-mold/mold"
output_dir="$root/$work_time"

echo "output_dir: $output_dir"

array=("coding")

# 遍历数组中的每个元素
for dataset in "${array[@]}"
do
    cache_path="~/AI-mold/mold/cache"
    echo "cache_path: $cache_path"
    python ~/AI-mold/src/preprocess_dataset.py \
        --stage sft \
        --do_train \
        --model_name_or_path "$model_name_or_path" \
        --dataset "$dataset" \
        --template "$template" \
        --cutoff_len 2048 \
        --finetuning_type full \
        --output_dir "$output_dir" \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 8 \
        --lr_scheduler_type cosine \
        --logging_steps 10 \
        --save_steps 10000 \
        --learning_rate 2e-5 \
        --num_train_epochs 2.0 \
        --plot_loss \
        --bf16 \
        --report_to tensorboard \
        --val_size 1000 \
        --tokenized_path "$cache_path" \
        --packing true

    # 将$dataset写入cache_path/dataset.txt
    echo "$dataset" > "$cache_path/dataset.txt"
done
