#!/bin/bash

cd /media/AI4MED1/hanjiacheng/LLaVA-CL-MOE

# 【恢复】：必须使用原始的基座模型来加载结构和 Tokenizer
MODEL_PATH="/media/AI4MED1/hanjiacheng/LLaVA-CL-MOE/models/llava-med-v1.5-mistral-7b"

# 【新增】：专门指向上一任务 LoRA 权重的路径
PRETRAIN_MOE_LORA_PATH="/media/AI4MED1/hanjiacheng/LLaVA-CL-MOE/checkpoints/1-IL-EV17/llava-med-v1.5-moe-lora-2.28-2"

DATA_PATH="/media/AI4MED1/hanjiacheng/Surgical-VQACL-Data/IS-EV17/instrument_state_ev17_train.json"
IMAGE_FOLDER="/media/AI4MED1/hanjiacheng/data/EndoVis-17-VQLA/left_frames"
VISION_TOWER_PATH="/media/AI4MED1/hanjiacheng/LLaVA-CL-MOE/models/clip-vit-large-patch14-336"
PRETRAIN_PROJECTOR_PATH="/media/AI4MED1/hanjiacheng/LLaVA/checkpoints/upper-bound/5data/llava-med-v1.5-lora-1.29/non_lora_trainables.bin"

OUTPUT_DIR="/media/AI4MED1/hanjiacheng/LLaVA-CL-MOE/checkpoints/2-IS-EV17/llava-med-v1.5-moe-lora-3.1"

deepspeed --include localhost:0,1 \
    --master_port=29401 \
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
    --num_train_epochs 100 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
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
    --lora_r 128 \
    --lora_alpha 256 \
    --freeze_mm_mlp_adapter True \
    --tune_mm_mlp_adapter False \
    --task_id 1 \
    --router_temperature 1.0 \
    --router_loss_alpha 1.0