#!/bin/bash

cd /media/AI4MED1/hanjiacheng/LLaVA-CL-MOE

# 1. 基座模型保持不变
MODEL_PATH="/media/AI4MED1/hanjiacheng/LLaVA-CL-MOE/models/llava-med-v1.5-mistral-7b"

# 2. 【核心修改】：指向上一个任务（任务二 IS-EV17）训好的 checkpoint 路径
PRETRAIN_MOE_LORA_PATH="/media/AI4MED1/hanjiacheng/LLaVA-CL-MOE/checkpoints/2-IS-EV17/llava-med-v1.5-moe-lora-3.8"

# 3. 任务三的数据和图片路径 (IL-EV18)
DATA_PATH="/media/AI4MED1/hanjiacheng/Surgical-VQACL-Data/IL-EV18/instrument_location_ev18_train.json"
IMAGE_FOLDER="/media/AI4MED1/hanjiacheng/data/EndoVis-18-VQLA"
VISION_TOWER_PATH="/media/AI4MED1/hanjiacheng/LLaVA-CL-MOE/models/clip-vit-large-patch14-336"

# 通用视觉映射MLP层（继续冻结使用）
PRETRAIN_PROJECTOR_PATH="/media/AI4MED1/hanjiacheng/LLaVA/checkpoints/upper-bound/5data/llava-med-v1.5-lora-1.29/non_lora_trainables.bin"

# 4. 任务三的输出路径
OUTPUT_DIR="/media/AI4MED1/hanjiacheng/LLaVA-CL-MOE/checkpoints/3-IL-EV18/llava-med-v1.5-moe-lora-3.8"

deepspeed --include localhost:0,1 llava/train/train_mem.py \
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
    --num_train_epochs 100 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
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
    --task_id 2 \
    --router_temperature 1.0 \
    --router_loss_alpha 1.0