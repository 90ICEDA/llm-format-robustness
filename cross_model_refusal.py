import pandas as pd
import os

# ==================== 配置区域 ====================
FILE_05B = 'refusal_labels.csv'  # 0.5B 模型的标签表
FILE_15B = '1.5b_refusal_labels.csv'  # 1.5B 模型的标签表
OUTPUT_CROSS_CSV = 'cross_model_refusal.csv'


# ==================================================

def get_raw_refusal_df(filepath, model_suffix):
    """读取单模型标签表"""
    df = pd.read_csv(filepath)

    res_df = df[['source_prompt_id', 'task_type', 'mutation_type', 'refusal_flag']].copy()

    res_df.rename(columns={'refusal_flag': f'refusal_{model_suffix}'}, inplace=True)
    return res_df


def main():
    if not os.path.exists(FILE_05B) or not os.path.exists(FILE_15B):
        print(f"错误：未能在当前目录下找到对应的输入文件，请检查文件名！")
        return

    print("正在处理 0.5B 模型的数据...")
    df_05b_raw = get_raw_refusal_df(FILE_05B, '0.5b')

    print("正在处理 1.5B 模型的数据...")
    df_15b_raw = get_raw_refusal_df(FILE_15B, '1.5b')

    # 连表合并
    cross_df = pd.merge(
        df_05b_raw,
        df_15b_raw,
        on=['source_prompt_id', 'task_type', 'mutation_type'],
        how='inner'
    )

    cross_df.to_csv(OUTPUT_CROSS_CSV, index=False, encoding='utf-8-sig')
    print(f"-> 跨模型全量对比明细表已保存至: {OUTPUT_CROSS_CSV}")

    # ==================== 核心统计分析 ====================
    total_pairs = len(cross_df)

    # 1. 两个模型都拒答
    both_triggered = cross_df[(cross_df['refusal_0.5b'] == 1) & (cross_df['refusal_1.5b'] == 1)]
    count_both = len(both_triggered)

    # 2. 仅在小模型拒答
    only_small = cross_df[(cross_df['refusal_0.5b'] == 1) & (cross_df['refusal_0.5b'] == 0)]
    count_only_small = len(only_small)

    # 3. 仅在大模型拒答
    only_large = cross_df[(cross_df['refusal_0.5b'] == 0) & (cross_df['refusal_1.5b'] == 1)]
    count_only_large = len(only_large)

    # 4. 两个模型都没拒答
    neither_triggered = cross_df[(cross_df['refusal_0.5b'] == 0) & (cross_df['refusal_1.5b'] == 0)]
    count_neither = len(neither_triggered)

    print("\n" + "=" * 15 + " 跨模型全量拒答交叉统计报告 " + "=" * 15)
    print(f"参与评估的全量样本配对总数: {total_pairs}")
    print("-" * 44)
    print(f"两个模型【都触发】拒答:          {count_both:<5} 占比: {count_both / total_pairs:.2%}")
    print(f"【仅小模型(0.5B)】触发拒答:       {count_only_small:<5} 占比: {count_only_small / total_pairs:.2%}")
    print(f"【仅大模型(1.5B)】触发拒答:       {count_only_large:<5} 占比: {count_only_large / total_pairs:.2%}")
    print(f"两个模型【均未触发】拒答:         {count_neither:<5} 占比: {count_neither / total_pairs:.2%}")
    print("==============================================")


if __name__ == '__main__':
    main()
