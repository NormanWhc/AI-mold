#! /bin/bash
# model
model_name_or_path="/zhaoyuhangT/data/pretrain_model/gm_llm_v2.0_cpt_data_1T/20240418224659-pretrain-gm-llm-v2.0/ckpt_step_34000/hf"

# method
stage="sft"
do_train=true
finetuning_type="full"

# dataset
dataset="coding"
template="llama2_zh"
cutoff_len=4096
tokenized_path="/zhaoyuhangT/data/sft/gm-llm-v2.0/$dataset/cache"

# output
work_time=$(date +%Y%m%d%H%M%S)
model="gmllm-v2.0"
root="/data/sft_model/$model"

output_dir="$root/$work_time"
logging_steps=5
plot_loss=true
save_strategy="epoch"
report_to="tensorboard"

mkdir -p "$output_dir"

# train
per_device_train_batch_size=16
gradient_accumulation_steps=4
learning_rate=2e-5
num_train_epochs=5.0
lr_scheduler_type="cosine"
warmup_ratio=0.1
bf16=true
flash_attn="fa2"
save_safetensors=false

# eval
val_size=1000
per_device_eval_batch_size=8
eval_strategy="epoch"

work_dir=/zhaoyuhangT/develop/LLM_dev/GM-LLM/SFT/LLaMA-Factory-main
cd $work_dir || exit

pip install -r requirements.txt

ds_config="$work_dir/config/ds_z2_config.json"

TORCH_EXTENSIONS_DIR="$work_dir"/torch-extensions deepspeed $work_dir/src/train.py \
    --deepspeed "$ds_config" \
    --model_name_or_path "$model_name_or_path" \
    --stage $stage \
    --do_train $do_train\
    --finetuning_type $finetuning_type\
    --dataset $dataset \
    --template $template \
    --cutoff_len $cutoff_len \
    --tokenized_path "$tokenized_path" \
    --output_dir "$output_dir" \
    --logging_steps $logging_steps \
    --plot_loss $plot_loss \
    --save_strategy $save_strategy \
    --report_to $report_to \
    --per_device_train_batch_size $per_device_train_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --learning_rate $learning_rate \
    --num_train_epochs $num_train_epochs \
    --lr_scheduler_type $lr_scheduler_type \
    --warmup_ratio $warmup_ratio \
    --bf16 $bf16 \
    --flash_attn $flash_attn \
    --save_safetensors $save_safetensors \
    --val_size $val_size \
    --per_device_eval_batch_size $per_device_eval_batch_size \
    --eval_strategy $eval_strategy
