#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0

# 1. 基础路径配置
BASE_DIR="/media/AI4MED1/hanjiacheng/LLaVA-CL-MOE"
cd "${BASE_DIR}"

MODEL_BASE="${BASE_DIR}/models/llava-med-v1.5-mistral-7b"
TASK_DIR="${BASE_DIR}/checkpoints/2-IS-EV17/llava-med-v1.5-moe-lora-3.11-2-alpha-0.5"

QUESTION_FILE="/media/AI4MED1/hanjiacheng/Surgical-VQACL-Data/IS-EV17/instrument_state_ev17_test.jsonl"
IMAGE_FOLDER="/media/AI4MED1/hanjiacheng/data/EndoVis-17-VQLA/left_frames"
OUTPUT_DIR="${BASE_DIR}/output/2-IS-EV17/3.11-2-alpha-0.5/IS-EV17"

mkdir -p "${OUTPUT_DIR}"

# 2. 遍历所有 checkpoint 文件夹
for CKPT_PATH in $(ls -d ${TASK_DIR}/checkpoint-* | sort -V); do
    
    CKPT_NAME=$(basename "${CKPT_PATH}")
    echo "========================================================"
    echo "Processing ${CKPT_NAME} ..."
    
    # 3. 配置文件补全 (使用软链接，既省空间又避开 cp 报错)
    # config.json 必须从底座模型拿
    ln -sf "${MODEL_BASE}/config.json" "${CKPT_PATH}/config.json"
    
    # adapter_config.json 和权重可能在父目录
    [ -f "${TASK_DIR}/adapter_config.json" ] && ln -sf "${TASK_DIR}/adapter_config.json" "${CKPT_PATH}/adapter_config.json"
    [ -f "${TASK_DIR}/non_lora_trainables.bin" ] && ln -sf "${TASK_DIR}/non_lora_trainables.bin" "${CKPT_PATH}/non_lora_trainables.bin"

    # 4. 执行推理
    ANSWERS_FILE="${OUTPUT_DIR}/mid_${CKPT_NAME}.jsonl"

    # 如果该 checkpoint 已经跑过了，可以跳过（可选，取消下面注释即可）
    # if [ -f "$ANSWERS_FILE" ]; then echo "Skip ${CKPT_NAME}, output exists."; continue; fi

    python -m llava.eval.model_vqa \
        --model-path "${CKPT_PATH}" \
        --model-base "${MODEL_BASE}" \
        --question-file "${QUESTION_FILE}" \
        --image-folder "${IMAGE_FOLDER}" \
        --answers-file "${ANSWERS_FILE}" \
        --conv-mode vicuna_v1 \
        --temperature 0 \
        --top_p 1.0 \
        --num_beams 1

    echo "Finished evaluation for ${CKPT_NAME}."
done

echo "Done! All outputs are in ${OUTPUT_DIR}"