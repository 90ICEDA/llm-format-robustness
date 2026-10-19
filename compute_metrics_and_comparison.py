import pandas as pd
import numpy as np
import os


def compute_metrics():
    
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'result' in f]
    if not csv_files:
        print("❌ 找不到结果 CSV 文件！")
        return

    # 优先读取最新的评测结果，或者指定读取目标文件
    target_file = 'revised_results' if os.path.exists('revised_results') else csv_files[0]

    # 如果是目录，找里面的csv
    if os.path.isdir(target_file):
        sub_files = [os.path.join(target_file, f) for f in os.listdir(target_file) if f.endswith('.csv')]
        if sub_files:
            target_file = sub_files[0]
        else:
            target_file = csv_files[0]

    print(f"正在分析数据文件: {target_file}")
    df = pd.read_csv(target_file)
    print(f"当前 CSV 的所有列名: {df.columns.tolist()}")

    # 兼容性适配：自动识别对错列和置信度差值列
    correct_col = None
    for col in ['is_correct', 'stage1_correct']:
        if col in df.columns:
            correct_col = col
            break

    score_col = None
    for col in ['commit_minus_decline', 'two_stage_decline_logit_diff']:
        if col in df.columns:
            score_col = col
            break

    if not correct_col or not score_col:
        print(f"无法在 CSV 中找到匹配的对错列或置信度列！现有列名: {df.columns.tolist()}")
        return

    print(f"-> 自动识别判定列: [{correct_col}], 置信度排序列: [{score_col}]")

    # 统一将对错列转为布尔/整型
    df['__is_correct_bool'] = df[correct_col].apply(lambda x: True if str(x).lower() in ['true', '1', 'yes'] else False)

    # 1. 整体 AURC 计算函数
    def calculate_aurc(sub_df):
        sub_df = sub_df.sort_values(by=score_col, ascending=False).reset_index(drop=True)
        total = len(sub_df)
        if total == 0:
            return 0.0
        risks = []
        for i in range(1, total + 1):
            subset = sub_df.head(i)
            incorrect = (subset['__is_correct_bool'] == False)
            risks.append(incorrect.mean())
        return np.mean(risks)

    print("\n================ 1. 整体与格式对比指标 (AURC) ================")
    if 'format' in df.columns:
        for fmt in df['format'].unique():
            fmt_df = df[df['format'] == fmt]
            aurc = calculate_aurc(fmt_df)
            full_risk = (fmt_df['__is_correct_bool'] == False).mean()
            print(f"格式 [{fmt}]: 全覆盖错误率 (Risk@1.0) = {full_risk:.4f} | AURC = {aurc:.4f}")
    else:
        aurc = calculate_aurc(df)
        full_risk = (df['__is_correct_bool'] == False).mean()
        print(f"整体: 全覆盖错误率 (Risk@1.0) = {full_risk:.4f} | AURC = {aurc:.4f}")

    print("\n================ 2. 题目类别对比 (Logic vs Factual) ================")
    if 'category' in df.columns:
        for cat in df['category'].unique():

            cat_df = df[df['category'] == cat]
            aurc = calculate_aurc(cat_df)
            full_risk = (cat_df['__is_correct_bool'] == False).mean()
            print(f"类别 [{cat}]: 全覆盖错误率 (Risk@1.0) = {full_risk:.4f} | AURC = {aurc:.4f}")
    else:
        print("（当前数据不包含 category 字段）")

    print("\n================ 3. 排序能力检验 (Spearman 相关性) ================")
    correct_numeric = df['__is_correct_bool'].astype(int)
    corr = df[score_col].corr(correct_numeric, method='spearman')
    print(f"置信度差值与正确性的 Spearman 秩相关系数: {corr:.4f}")
    print("（注：相关系数显著大于 0，说明模型的置信度排序具备良好的区分能力）")


if __name__ == "__main__":
    compute_metrics()
