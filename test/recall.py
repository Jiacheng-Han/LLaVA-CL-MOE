import json
import string
import argparse
import os
import re
from collections import Counter
import matplotlib.pyplot as plt


def get_tokens(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.split()


def calculate_word_recall(pred_file, gt_file):
    preds = {}
    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            preds[str(item["question_id"])] = item["text"].lower().strip()

    gt = {}
    with open(gt_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            for conv in item['conversations']:
                if conv['from'] == 'gpt':
                    gt[str(item['id'])] = conv['value'].lower().strip()

    total_recall = 0
    sample_count = 0

    for q_id, pred_text in preds.items():
        if q_id in gt:
            sample_count += 1
            gt_text = gt[q_id]

            pred_tokens = get_tokens(pred_text)
            gt_tokens = get_tokens(gt_text)

            if not gt_tokens:
                continue

            common_counts = Counter(gt_tokens) & Counter(pred_tokens)
            overlap_count = sum(common_counts.values())

            recall_score = overlap_count / len(gt_tokens)
            total_recall += recall_score

    if sample_count == 0:
        return 0.0

    return total_recall / sample_count


def get_checkpoints_and_scores(pred_dir, gt_file):
    steps = []
    scores = []

    pattern = re.compile(r"mid_checkpoint-(\d+)\.jsonl")

    for filename in os.listdir(pred_dir):
        match = pattern.match(filename)
        if match:
            step = int(match.group(1))
            pred_file = os.path.join(pred_dir, filename)

            score = calculate_word_recall(pred_file, gt_file)

            steps.append(step)
            scores.append(score)

            print(f"{os.path.basename(pred_dir)} | Checkpoint {step} | Recall {score:.4f}")

    sorted_pairs = sorted(zip(steps, scores))
    if not sorted_pairs:
        return [], []

    sorted_steps, sorted_scores = zip(*sorted_pairs)
    return list(sorted_steps), list(sorted_scores)


def plot_curves(task1_steps, task1_scores, task2_steps, task2_scores, output_image_path, alpha):
    plt.figure(figsize=(10, 6))

    if task1_steps:
        plt.plot(
            task1_steps,
            task1_scores,
            marker='o',
            linestyle='-',
            color='blue',
            label='Task 1 (IL-EV17)',
            linewidth=2
        )

    if task2_steps:
        plt.plot(
            task2_steps,
            task2_scores,
            marker='s',
            linestyle='-',
            color='red',
            label='Task 2 (IS-EV17)',
            linewidth=2
        )

    plt.title(f'Recall Curve (router_loss_alpha = {alpha})', fontsize=14)
    plt.xlabel('Checkpoint Step', fontsize=12)
    plt.ylabel('Mean Word-level Recall', fontsize=12)

    plt.ylim(0, 1.05)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    print(f"✅ 保存图表: {output_image_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--alphas", nargs="+", required=True)

    parser.add_argument("--t1_pred_dirs", nargs="+", required=True)
    parser.add_argument("--t2_pred_dirs", nargs="+", required=True)

    parser.add_argument("--t1_gt_file", required=True)
    parser.add_argument("--t2_gt_file", required=True)

    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for alpha, t1_dir, t2_dir in zip(args.alphas, args.t1_pred_dirs, args.t2_pred_dirs):

        print(f"\n===== Processing alpha = {alpha} =====")

        t1_steps, t1_scores = get_checkpoints_and_scores(t1_dir, args.t1_gt_file)

        t2_steps, t2_scores = get_checkpoints_and_scores(t2_dir, args.t2_gt_file)

        output_img = os.path.join(args.output_dir, f"recall_alpha_{alpha}.png")

        plot_curves(
            t1_steps,
            t1_scores,
            t2_steps,
            t2_scores,
            output_img,
            alpha
        )

'''
python /media/AI4MED1/hanjiacheng/LLaVA-CL-MOE/test/recall.py \
--alphas 0.0 0.05 0.5 \
--t1_pred_dirs \
/media/AI4MED1/hanjiacheng/LLaVA-CL-MOE/output/2-IS-EV17/3.11-2-alpha-0.0/IL-EV17 \
/media/AI4MED1/hanjiacheng/LLaVA-CL-MOE/output/2-IS-EV17/3.11-2-alpha-0.05/IL-EV17 \
/media/AI4MED1/hanjiacheng/LLaVA-CL-MOE/output/2-IS-EV17/3.11-2-alpha-0.5/IL-EV17 \
--t2_pred_dirs \
/media/AI4MED1/hanjiacheng/LLaVA-CL-MOE/output/2-IS-EV17/3.11-2-alpha-0.0/IS-EV17 \
/media/AI4MED1/hanjiacheng/LLaVA-CL-MOE/output/2-IS-EV17/3.11-2-alpha-0.05/IS-EV17 \
/media/AI4MED1/hanjiacheng/LLaVA-CL-MOE/output/2-IS-EV17/3.11-2-alpha-0.5/IS-EV17 \
--t1_gt_file /media/AI4MED1/hanjiacheng/Surgical-VQACL-Data/IL-EV17/instrument_location_ev17_test.json \
--t2_gt_file /media/AI4MED1/hanjiacheng/Surgical-VQACL-Data/IS-EV17/instrument_state_ev17_test.json \
--output_dir /media/AI4MED1/hanjiacheng/LLaVA-CL-MOE/test/recall_figures
'''