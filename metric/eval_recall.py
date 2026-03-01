import os
import json
import string
import argparse
from collections import Counter

def get_tokens(text):
    # 移除所有标点符号，替换为空格防止连词 (比如 "hello.world" -> "hello world" 而不是 "helloworld")
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    text = text.translate(translator)
    return text.split()


def calculate_word_recall(pred_file, gt_file):
    # 1. 读取预测结果 (JSONL 格式)
    preds = {}
    print(f"正在读取预测文件: {pred_file} ...")
    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            # 统一 ID 为字符串，转换为小写并去首尾空格
            preds[str(item["question_id"])] = item["text"].lower().strip()

    # 2. 读取标准答案 (JSON 格式)
    gt = {}
    print(f"正在读取标准答案文件: {gt_file} ...")
    with open(gt_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            for conv in item['conversations']:
                if conv['from'] == 'gpt':
                    gt[str(item['id'])] = conv['value'].lower().strip()
                    break  # 假设只有一轮 VQA，取到第一个 GPT 回答后直接跳出，防止多轮覆盖

    # 3. 计算 Recall
    total_recall = 0
    sample_count = 0
    
    # 以预测集为基准进行遍历
    for q_id, pred_text in preds.items():
        if q_id in gt:
            sample_count += 1
            gt_text = gt[q_id]
            
            # 分词
            pred_tokens = get_tokens(pred_text)
            gt_tokens = get_tokens(gt_text)
            
            # 如果标准答案为空，跳过以防除零错误
            if not gt_tokens:
                continue
            
            # 计算单词重合数 (考虑词频)
            # Counter 会统计每个词出现的次数，& 操作符取两个集合的交集（取频率最小值）
            common_counts = Counter(gt_tokens) & Counter(pred_tokens)
            overlap_count = sum(common_counts.values())
            
            # 计算 Recall = (预测对的词) / (标准答案总词数)
            recall_score = overlap_count / len(gt_tokens)
            total_recall += recall_score

    # 4. 输出结果
    if sample_count == 0:
        print("未匹配到任何 ID，请检查预测文件与答案文件中的 ID 是否对应。")
        return 0.0

    mean_recall = total_recall / sample_count
    print("-" * 30)
    print(f"评估完成！")
    print(f"匹配到的样本数: {sample_count}")
    print(f"单词级平均召回率 (Mean Word-level Recall): {mean_recall:.2%}")
    return mean_recall

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="计算模型推理结果的 Word-level Recall")
    # parser.add_argument("--pred_file", type=str, required=True, help="推理生成的 .jsonl 文件路径")
    # parser.add_argument("--gt_file", type=str, required=True, help="包含 Ground Truth 的 .json 文件路径")
    
    # args = parser.parse_args()

    # 建议使用绝对路径，避免找不到文件
    PRED_FILE = "/media/AI4MED1/hanjiacheng/LLaVA-CL-MOE/output/2-IS-EV17/3.1/IS-EV17.jsonl"
    GT_FILE = "/media/AI4MED1/hanjiacheng/Surgical-VQACL-Data/IS-EV17/instrument_state_ev17_test.json"
    
    if os.path.exists(PRED_FILE) and os.path.exists(GT_FILE):
        score = calculate_word_recall(PRED_FILE, GT_FILE)
    else:
        print("错误：请检查文件路径是否正确！其中有一个或多个文件不存在。")