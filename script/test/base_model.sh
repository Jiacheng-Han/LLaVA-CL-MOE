#!/bin/bash

# --- 测试1-IL-EV17 ---
echo "--- 测试1-IL-EV17 ---"
output="/media/AI4MED1/hanjiacheng/LLaVA-CL-MOE/output/base_model/IL-EV17.jsonl"
ground_truth="/media/AI4MED1/hanjiacheng/Surgical-VQACL-Data/IL-EV17/instrument_location_ev17_test.json"

python /media/AI4MED1/hanjiacheng/LLaVA-CL-MOE/metric/eval_recall.py --pred_file $output --gt_file $ground_truth

# --- 测试2-IS-EV17 ---
echo " --- 测试2-IS-EV17 ---"
output="/media/AI4MED1/hanjiacheng/LLaVA-CL-MOE/output/base_model/IS-EV17.jsonl"
ground_truth="/media/AI4MED1/hanjiacheng/Surgical-VQACL-Data/IS-EV17/instrument_state_ev17_test.json"

python /media/AI4MED1/hanjiacheng/LLaVA-CL-MOE/metric/eval_recall.py --pred_file $output --gt_file $ground_truth

# --- 测试3-IL-EV18 ---
echo " --- 测试3-IL-EV18 ---"
output="/media/AI4MED1/hanjiacheng/LLaVA-CL-MOE/output/base_model/IL-EV18.jsonl"
ground_truth="/media/AI4MED1/hanjiacheng/Surgical-VQACL-Data/IL-EV18/instrument_location_ev18_test.json"

python /media/AI4MED1/hanjiacheng/LLaVA-CL-MOE/metric/eval_recall.py --pred_file $output --gt_file $ground_truth

# --- 测试4-IS-EV18 ---
echo " --- 测试4-IS-EV18 ---"
output="/media/AI4MED1/hanjiacheng/LLaVA-CL-MOE/output/base_model/IS-EV18.jsonl"
ground_truth="/media/AI4MED1/hanjiacheng/Surgical-VQACL-Data/IS-EV18/instrument_state_ev18_test.json"

python /media/AI4MED1/hanjiacheng/LLaVA-CL-MOE/metric/eval_recall.py --pred_file $output --gt_file $ground_truth

# --- 测试5-IS-DV ---
echo " --- 测试5-IS-DV ---"
output="/media/AI4MED1/hanjiacheng/LLaVA-CL-MOE/output/base_model/IS-DV.jsonl"
ground_truth="/media/AI4MED1/hanjiacheng/Surgical-VQACL-Data/IS-DV/instrument_state_dv_test.json"

python /media/AI4MED1/hanjiacheng/LLaVA-CL-MOE/metric/eval_recall.py --pred_file $output --gt_file $ground_truth


