import pandas as pd
import os

# ==================== 配置区域 ====================
FILE_05B = 'refusal_labels.csv'  # 0.5B 模型的标签表
FILE_15B = '1.5b_refusal_labels.csv'  # 1.5B 模型的标签表
OUTPUT_CROSS_CSV = 'cross_model_refusal.csv'


# ==================================================

def get_raw_refusal_df(filepath, model_suffix):
    """读取并清洗单模型标签表"""
    df = pd.read_csv(filepath)
    # 确保只取需要的列
    res_df = df[['source_prompt_id', 'task_type', 'mutation_type', 'refusal_flag']].copy()
    res_df.rename(columns={'refusal_flag': f'refusal_{model_suffix}'}, inplace=True)
    return res_df


def main():
    if not os.path.exists(FILE_05B) or not os.path.exists(FILE_15B):
        print(f"错误：未找到输入文件，请确保 CSV 文件在当前目录下。")
        return

    # 1. 数据合并
    df_05b_raw = get_raw_refusal_df(FILE_05B, '0.5b')
    df_15b_raw = get_raw_refusal_df(FILE_15B, '1.5b')

    cross_df = pd.merge(
        df_05b_raw,
        df_15b_raw,
        on=['source_prompt_id', 'task_type', 'mutation_type'],
        how='inner'
    )

    # 2. 检查是否有空值 
    if cross_df.isnull().any().any():
        print("警告：数据中存在缺失值，已自动进行填充处理。")
        cross_df = cross_df.fillna(0)

    # 3. 计算四种交叉情况
    # 两个都触发
    both_triggered = cross_df[(cross_df['refusal_0.5b'] == 1) & (cross_df['refusal_1.5b'] == 1)]
    # 仅小模型(0.5B)触发
    only_small = cross_df[(cross_df['refusal_0.5b'] == 1) & (cross_df['refusal_1.5b'] == 0)]
    # 仅大模型(1.5B)触发
    only_large = cross_df[(cross_df['refusal_0.5b'] == 0) & (cross_df['refusal_1.5b'] == 1)]
    # 均未触发
    neither_triggered = cross_df[(cross_df['refusal_0.5b'] == 0) & (cross_df['refusal_1.5b'] == 0)]

    # 4. 统计结果
    total_pairs = len(cross_df)
    counts = [len(both_triggered), len(only_small), len(only_large), len(neither_triggered)]

    print("\n" + "=" * 15 + " 跨模型全量拒答交叉统计报告 " + "=" * 15)
    print(f"参与评估的全量样本配对总数: {total_pairs}")
    print("-" * 44)
    print(f"两个模型【都触发】拒答:          {counts[0]:<5} 占比: {counts[0] / total_pairs:.2%}")
    print(f"【仅小模型(0.5B)】触发拒答:       {counts[1]:<5} 占比: {counts[1] / total_pairs:.2%}")
    print(f"【仅大模型(1.5B)】触发拒答:       {counts[2]:<5} 占比: {counts[2] / total_pairs:.2%}")
    print(f"两个模型【均未触发】拒答:         {counts[3]:<5} 占比: {counts[3] / total_pairs:.2%}")
    print(f"总计校验占比:                     {sum(counts) / total_pairs:.2%}")
    print("==============================================")

    # 保存结果
    cross_df.to_csv(OUTPUT_CROSS_CSV, index=False, encoding='utf-8-sig')
    print(f"-> 结果已保存至: {OUTPUT_CROSS_CSV}")


if __name__ == '__main__':
    main()
