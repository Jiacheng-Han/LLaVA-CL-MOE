#!/bin/bash

# --- 1. 显卡设置 ---
# 指定使用哪张卡运行推理
export CUDA_VISIBLE_DEVICES=1

# --- 2. 路径配置 ---
# 切换到你修改后的工程根目录
cd /media/AI4MED1/hanjiacheng/LLaVA-CL-MOE

# 底座模型
MODEL_BASE="/media/AI4MED1/hanjiacheng/LLaVA-CL-MOE/models/llava-med-v1.5-mistral-7b"

# 指向刚训练好的路径
MODEL_PATH="/media/AI4MED1/hanjiacheng/LLaVA-CL-MOE/checkpoints/2-IS-EV17/llava-med-v1.5-moe-lora-epoch50"

# question 文件 
QUESTION_FILE="/media/AI4MED1/hanjiacheng/Surgical-VQACL-Data/IL-EV17/instrument_location_ev17_test.jsonl"

# 图片文件夹 
IMAGE_FOLDER="/media/AI4MED1/hanjiacheng/data/EndoVis-17-VQLA/left_frames"

# 结果保存文件 (修改为当前工程的 output)
ANSWERS_FILE="/media/AI4MED1/hanjiacheng/LLaVA-CL-MOE/output/2-IS-EV17/3.1-epoch50/IL-EV17.jsonl"

# --- 3. 运行推理 ---  
python -m llava.eval.model_vqa \
    --model-path $MODEL_PATH \
    --model-base $MODEL_BASE \
    --question-file $QUESTION_FILE \
    --image-folder $IMAGE_FOLDER \
    --answers-file $ANSWERS_FILE \
    --conv-mode vicuna_v1 \
    --temperature 0 \
    --top_p 1.0 \
    --num_beams 1