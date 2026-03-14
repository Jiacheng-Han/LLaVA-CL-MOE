#!/bin/bash

cd /media/AI4MED1/hanjiacheng/LLaVA-CL-MOE

# 原始基座模型参数路径
MODEL_PATH="/media/AI4MED1/hanjiacheng/LLaVA-CL-MOE/models/llava-med-v1.5-mistral-7b"

# 指向上一任务 LoRA 权重的路径
PRETRAIN_MOE_LORA_PATH="/media/AI4MED1/hanjiacheng/LLaVA-CL-MOE/checkpoints/1-IL-EV17/llava-med-v1.5-moe-lora-3.11"

DATA_PATH="/media/AI4MED1/hanjiacheng/Surgical-VQACL-Data/IS-EV17/instrument_state_ev17_train.json"
IMAGE_FOLDER="/media/AI4MED1/hanjiacheng/data/EndoVis-17-VQLA/left_frames"
VISION_TOWER_PATH="/media/AI4MED1/hanjiacheng/LLaVA-CL-MOE/models/clip-vit-large-patch14-336"
PRETRAIN_PROJECTOR_PATH="/media/AI4MED1/hanjiacheng/LLaVA/checkpoints/upper-bound/5data/llava-med-v1.5-lora-1.29/non_lora_trainables.bin"

# 原始输出目录前缀
OUTPUT_DIR_PREFIX="/media/AI4MED1/hanjiacheng/LLaVA-CL-MOE/checkpoints/2-IS-EV17/llava-med-v1.5-moe-lora-3.11-2"

# 要循环的 router_loss_alpha 值
ALPHA_LIST=(1.0 10.0)

for ALPHA in "${ALPHA_LIST[@]}"; do
    # 为每个 alpha 创建独立输出目录
    OUTPUT_DIR="${OUTPUT_DIR_PREFIX}-alpha-${ALPHA}"

    echo "Starting training with router_loss_alpha = ${ALPHA}"
    echo "Output directory: ${OUTPUT_DIR}"

    deepspeed --include localhost:2,3 \
        --master_port=29701 \
        llava/train/train_mem.py \
        --deepspeed /media/AI4MED1/hanjiacheng/LLaVA-CL-MOE/script/zero2.json \
        --model_name_or_path $MODEL_PATH \
        --pretrained_moe_lora_path $PRETRAIN_MOE_LORA_PATH \
        --version v1 \
        --data_path $DATA_PATH \
        --image_folder $IMAGE_FOLDER \
        --vision_tower $VISION_TOWER_PATH \
        --pretrain_mm_mlp_adapter $PRETRAIN_PROJECTOR_PATH \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio pad \
        --group_by_modality_length True \
        --bf16 True \
        --output_dir $OUTPUT_DIR \
        --num_train_epochs 30 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy "no" \
        --save_strategy "epoch" \
        --save_total_limit 100 \
        --learning_rate 2e-4 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --lazy_preprocess True \
        --report_to tensorboard \
        --seed 42 \
        --lora_enable True \
        --lora_r 64 \
        --lora_alpha 128 \
        --freeze_mm_mlp_adapter True \
        --tune_mm_mlp_adapter False \
        --task_id 1 \
        --router_temperature 1.0 \
        --router_loss_alpha $ALPHA \
        --save_full_model_at_end False

    echo "Finished training for router_loss_alpha = ${ALPHA}"
done