#! /bin/bash
model_name_or_path="/data/zhaoyuhang/data/pretrain_model/gm_llm_v2.0_cpt_data_1T/20240418224659-pretrain-gm-llm-v2.0/ckpt_step_14000/hf"
template="llama2_zh"

work_time=$(date +%Y%m%d%H%M%S)
model="llama2_7b_zh"
root="/defaultShare/zhaoyuhang/data/sft_model/$model"
output_dir="$root/$work_time"

echo "output_dir: $output_dir"

work_dir=/data/zhaoyuhang/develop/LLaMA-Factory-main
cd $work_dir || exit

array=("one2two" "one2five" "one2ten")

# 遍历数组中的每个元素
for dataset in "${array[@]}"
do
    cache_path="/data/zhaoyuhang/data/sft/gm-llm-v2.0/$dataset/cache"
    echo "cache_path: $cache_path"
    python $work_dir/src/preprocess_dataset.py \
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


