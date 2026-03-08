#!/bin/bash

# --- 测试1-IL-EV17 ---
output="/media/AI4MED1/hanjiacheng/LLaVA-CL-MOE/output/1-IL-EV17/3.8-2.jsonl"
ground_truth="/media/AI4MED1/hanjiacheng/Surgical-VQACL-Data/IL-EV17/instrument_location_ev17_test.json"

python /media/AI4MED1/hanjiacheng/LLaVA-CL-MOE/metric/eval_recall.py --pred_file $output --gt_file $ground_truth