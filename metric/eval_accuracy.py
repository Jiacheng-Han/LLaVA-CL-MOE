import json

def calculate_accuracy(pred_file, gt_file):
    # 1. 读取预测结果 (强制转换 ID 为 string)
    preds = {}
    print(f"正在读取预测文件: {pred_file} ...")
    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            # 关键修改：str(item["question_id"])
            preds[str(item["question_id"])] = item["text"].lower().strip()

    # 2. 读取标准答案 (强制转换 ID 为 string)
    gt = {}
    print(f"正在读取标准答案文件: {gt_file} ...")
    with open(gt_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            for conv in item['conversations']:
                if conv['from'] == 'gpt':
                    # 关键修改：str(item['id'])
                    gt[str(item['id'])] = conv['value'].lower().strip()

    # 3. 计算准确率
    correct = 0
    total = 0
    mismatches = []

    for q_id, pred_ans in preds.items():
        if q_id in gt:
            total += 1
            gt_ans = gt[q_id]
            
            # 判定逻辑：去除标点后比较
            clean_pred = pred_ans.replace('.', '')
            clean_gt = gt_ans.replace('.', '')
            
            if clean_pred == clean_gt:
                correct += 1
            else:
                mismatches.append((q_id, pred_ans, gt_ans))

    accuracy = correct / total
    print("-" * 30)
    print(f"评估完成！")
    print(f"总样本数: {total}")
    print(f"正确数量: {correct}")
    print(f"准确率 (Accuracy): {accuracy:.2%}")
    print("-" * 30)
    
    if mismatches:
        print("\n[错误样例展示 (前5个)]:")
        for idx, pred, real in mismatches[:5]:
            print(f"ID: {idx} | 预测: {pred} | 真实: {real}")

if __name__ == "__main__":
    PRED_FILE = "/media/AI4MED1/hanjiacheng/LLaVA/output/FT/ordered/1-IL-EV17/IL-EV17/12.28.jsonl"
    GT_FILE = "/media/AI4MED1/hanjiacheng/Surgical-VQACL-Data/IL-EV17/instrument_location_ev17_test.json"
    
    calculate_accuracy(PRED_FILE, GT_FILE)