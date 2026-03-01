import sys
import os
import json

# 1. 显式添加包含 'rrg_eval' 内部包的目录到环境变量
# 根据你的报错路径: /media/AI4MED1/hanjiacheng/LLaVA/eval/rrg_eval
sys.path.append("/media/AI4MED1/hanjiacheng/LLaVA/eval/rrg_eval")
# 2. 显式添加项目根目录，确保能找到 eval 模块本身
sys.path.append("/media/AI4MED1/hanjiacheng/LLaVA")

# 3. 然后再 import
from eval.rrg_eval.run import ReportGenerationEvaluator


def read_aligned(pred_file, gt_file):
    print(f"正在读取预测文件: {pred_file} ...")
    # 使用字典存储，key 为 id，value 为文本
    pred_dict = {}
    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            # 注意：将 id 转为字符串以统一格式
            qid = str(item.get("question_id", item.get("id"))) 
            pred_dict[qid] = item["text"].lower().strip()

    print(f"正在读取标准答案文件: {gt_file} ...")
    gt_dict = {}
    with open(gt_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            # 这里的 id 应该是唯一的
            qid = str(item["id"])
            for conv in item['conversations']:
                # 提取 GPT (Assistant) 的回答作为 Ground Truth
                if conv['from'] == 'gpt':
                    gt_dict[qid] = conv['value'].lower().strip()
                    break # 找到答案后跳出当前对话循环

    # === 对齐步骤 ===
    aligned_preds = []
    aligned_gt = []
    
    # 找出两个文件中都存在的 ID
    common_ids = sorted(list(set(pred_dict.keys()) & set(gt_dict.keys())))
    
    if len(common_ids) < len(gt_dict):
        print(f"警告: 标注文件有 {len(gt_dict)} 条，但只匹配到 {len(common_ids)} 条预测结果。")
    
    for qid in common_ids:
        aligned_preds.append(pred_dict[qid])
        aligned_gt.append(gt_dict[qid])
        
    print(f"成功对齐 {len(aligned_preds)} 条数据用于评估。")
    return aligned_preds, aligned_gt

def test_custom_metrics(preds, gt):
    if not preds:
        print("错误: 没有数据可评估。")
        return

    # 2. 指定要使用的自然语言生成指标 (NLG Metrics)
    nlg_scorers = [
        "BLEU-1",
        "BLEU-4",
        "ROUGE-L",
        "ROUGE-2"
    ]

    print(f"正在使用以下指标进行评估: {nlg_scorers}")

    # 3. 初始化评估器
    evaluator = ReportGenerationEvaluator(scorers=nlg_scorers, bootstrap_ci=False)

    # 4. 运行评估
    results = evaluator.evaluate(preds, gt)

    # 5. 打印结果
    print("\n========== 评估结果 ==========")
    for metric, score in results.items():
        print(f"{metric}: {score:.2%}")

if __name__ == "__main__": 
    # 建议使用绝对路径，避免找不到文件
    PRED_FILE = "/media/AI4MED1/hanjiacheng/LLaVA/output/slake/12.18/open_predictions.jsonl"
    GT_FILE = "/media/AI4MED1/hanjiacheng/LLaVA/data/slake/train/slake_llava_open.json"
    
    if os.path.exists(PRED_FILE) and os.path.exists(GT_FILE):
        preds, gt = read_aligned(PRED_FILE, GT_FILE)
        test_custom_metrics(preds, gt)
    else:
        print("错误：无法找到指定的文件路径，请检查路径是否正确。")