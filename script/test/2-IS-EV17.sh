#!/bin/bash

# --- 测试1-IL-EV17 ---
echo "--- 测试1-IL-EV17 ---"
output="/media/AI4MED1/hanjiacheng/LLaVA-CL-MOE/output/2-IS-EV17/3.8/IL-EV17-top1.jsonl"
ground_truth="/media/AI4MED1/hanjiacheng/Surgical-VQACL-Data/IL-EV17/instrument_location_ev17_test.json"

python /media/AI4MED1/hanjiacheng/LLaVA-CL-MOE/metric/eval_recall.py --pred_file $output --gt_file $ground_truth

# --- 测试2-IS-EV17 ---
echo " --- 测试2-IS-EV17 ---"
output="/media/AI4MED1/hanjiacheng/LLaVA-CL-MOE/output/2-IS-EV17/3.8/IS-EV17-top1.jsonl"
ground_truth="/media/AI4MED1/hanjiacheng/Surgical-VQACL-Data/IS-EV17/instrument_state_ev17_test.json"

python /media/AI4MED1/hanjiacheng/LLaVA-CL-MOE/metric/eval_recall.py --pred_file $output --gt_file $ground_truth