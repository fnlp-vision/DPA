set -e

cd ${WORK_DIR}/reps/DPA
NUM_GPUS=8
echo "NUM_GPUS: ${NUM_GPUS}"

LLM_VERSION=${WORK_DIR}/models/Llama-3.1-8B-Instruct

############### Pretrain ################

PROMPT_VERSION="plain"

lr=4e-5
warmup_ratio=0.03
train_batch_size=256
num_train_epochs=1
per_device_train_batch_size=16
gradient_accumulation_steps=$((train_batch_size / (per_device_train_batch_size * NUM_GPUS)))
lora_r=256
lora_alpha=$((lora_r * 2))
lora_dropout=0.0

BASE_RUN_NAME="$(basename "$LLM_VERSION")"
BASE_RUN_NAME="${BASE_RUN_NAME}-pretrain-lora_r${lora_r}_alpha${lora_alpha}-blip558k"
DATA_PATH=${WORK_DIR}/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k_text_only.json

echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

WANDB_MODE=offline WANDB_PROJECT=midtune torchrun --nproc_per_node="${NUM_GPUS}" --master_port=20001 \
    llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path=$DATA_PATH \
    --bf16 True \
    --output_dir "./checkpoints/llm/${BASE_RUN_NAME}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --evaluation_strategy "no" \
    --eval_steps 1 \
    --save_strategy "no" \
    --save_steps 30000 \
    --learning_rate ${lr} \
    --weight_decay 0. \
    --warmup_ratio ${warmup_ratio} \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --run_name $BASE_RUN_NAME \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --lora_enable True \
    --lora_r ${lora_r} \
    --lora_alpha ${lora_alpha} \
    --lora_dropout ${lora_dropout}
    # --attn_implementation sdpa

# You can delete the sdpa attn_implementation if you want to use flash attn
