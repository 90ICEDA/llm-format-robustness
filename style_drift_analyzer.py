import json
import pandas as pd
import os

from 第一周.jsonl_write import task_type

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
KEYWORDS_FILE = os.path.join(BASE_DIR, 'refusal_keywords.txt')
INPUT_JSONL = os.path.join(BASE_DIR, 'mutated_outputs.jsonl')
OUTPUT_STYLE_CSV = os.path.join(BASE_DIR, 'style_drift.csv')


def check_relevance(prompt, output):
    """重叠度检查"""
    prompt_chars = set(prompt.replace(" ", "").replace("\n", ""))
    output_chars = set(output.replace(" ", "").replace("\n", ""))
    if not prompt_chars or not output_chars:
        return False
    overlap = prompt_chars.intersection(output_chars)
    return len(overlap) > 2


def main():
    # 1. 加载关键词
    if not os.path.exists(KEYWORDS_FILE):
        raise FileNotFoundError(f"找不到关键词文件: {KEYWORDS_FILE}")
    with open(KEYWORDS_FILE, 'r', encoding='utf-8') as f:
        keywords = [line.strip() for line in f if line.strip()]

    records = []

    print("正在扫描数据并分离出拒答组与正常组...")
    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line.strip())

            output = data.get('output', '')
            mutation_type = data.get('mutation_type')
            prompt = data.get('mutated_prompt', data.get('original_prompt', ''))
            task_type = data.get('task_type')

            contains_kw = 1 if any(kw in output for kw in keywords) else 0
            is_short = len(output) < 30
            has_relevance = check_relevance(prompt, output)
            output_too_short = 1 if (is_short and not has_relevance) else 0
            if task_type == "问答" or "翻译":
                output_too_short = 0
            refusal_flag = 1 if (contains_kw or output_too_short) else 0
            # ---------------------------

            # 记录每条回答的长度
            records.append({
                'mutation_type': mutation_type,
                'refusal_flag': refusal_flag,
                'output_len': len(output)
            })

    df = pd.DataFrame(records)

    # 2. 筛选出正常回答组 (refusal_flag == 0)
    df_normal = df[df['refusal_flag'] == 0]

    # 3. 算正常回答数和输出长度中位数
    style_summary = df_normal.groupby('mutation_type').agg(
        normal_count=('output_len', 'count'),
        median_len_normal=('output_len', 'median')
    ).reset_index()

    # 4. 获取 plain的中位数
    plain_row = style_summary[style_summary['mutation_type'] == 'plain']
    if plain_row.empty:
        print("错误：正常回答组里竟然没有 plain ")
        return

    plain_median = plain_row['median_len_normal'].values[0]

    # 5. 计算对照字段
    style_summary['plain_median_len'] = plain_median
    style_summary['len_ratio'] = style_summary['median_len_normal'] / style_summary['plain_median_len']
    style_summary['len_ratio'] = style_summary['len_ratio'].round(4)

    # 6. 保存结果
    style_summary.to_csv(OUTPUT_STYLE_CSV, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 15 + " 风格漂移分析结果 " + "=" * 15)
    print(style_summary.to_string(index=False))
    print(f"\n报告已成功导出至: {OUTPUT_STYLE_CSV}")
    print("=" * 50)


if __name__ == '__main__':
    main()
