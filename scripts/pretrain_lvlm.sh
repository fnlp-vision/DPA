set -e

cd ${WORK_DIR}/reps/DPA

NUM_GPUS=8
echo "NUM_GPUS: ${NUM_GPUS}"

LLM_VERSION=${WORK_DIR}/models/Llama-3.1-8B-Instruct
VISION_MODEL_VERSION=${WORK_DIR}/models/clip-vit-large-patch14-336
LORA_WEIGHT_PATH=${WORK_DIR}/reps/DPA/checkpoints/llm/Llama-3.1-8B-Instruct-pretrain-lora_r256_alpha512-blip558k

############### Pretrain ################

PROMPT_VERSION="plain"

lr=2e-3
warmup_ratio=0.03
train_batch_size=256
num_train_epochs=1
per_device_train_batch_size=32
gradient_accumulation_steps=$((train_batch_size / (per_device_train_batch_size * NUM_GPUS)))

delta_probs_max=0.5
delta_probs_min=0.05

# Determine the suffix based on tunable parts
SUFFIX=""
MM_TUNABLE_PARTS="mm_mlp_adapter"

[[ $MM_TUNABLE_PARTS == *"mm_vision_tower"* ]] && SUFFIX="${SUFFIX}_vt"
[[ $MM_TUNABLE_PARTS == *"mm_mlp_adapter"* ]] && SUFFIX="${SUFFIX}_mlp"
[[ $MM_TUNABLE_PARTS == *"mm_language_model"* ]] && SUFFIX="${SUFFIX}_lm"

BASE_RUN_NAME="llavanext-$(basename "$VISION_MODEL_VERSION")-$(basename "$LORA_WEIGHT_PATH")"
BASE_RUN_NAME="${BASE_RUN_NAME}-pretrain-pma${SUFFIX}-blip558k"
DATA_PATH=${WORK_DIR}/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json
image_folder=${WORK_DIR}/data/LLaVA-Pretrain/images

echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

WANDB_MODE=offline WANDB_PROJECT=midtune torchrun --nproc_per_node="${NUM_GPUS}" --master_port=20001 \
    llava/train/train_dpa.py \
    --delta_probs_max ${delta_probs_max} \
    --delta_probs_min ${delta_probs_min} \
    --deepspeed scripts/zero2.json \
    --model_name_or_path ${LLM_VERSION} \
    --lora_weight_path ${LORA_WEIGHT_PATH} \
    --version ${PROMPT_VERSION} \
    --data_path=$DATA_PATH \
    --image_folder $image_folder \
    --mm_tunable_parts=$MM_TUNABLE_PARTS \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir "./checkpoints/projectors/${BASE_RUN_NAME}" \
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
    --report_to tensorboard \
    --run_name $BASE_RUN_NAME \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    # --attn_implementation sdpa

# You can delete the sdpa attn_implementation if you want to use flash attn
