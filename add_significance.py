import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
import statsmodels.stats.multitest as smm

input_file_strict = "strict_flip_matrix.csv"
input_file_loose = "loose_flip_matrix.csv"
output_file_strict = "strict_flip_matrix_add_significance.csv"
output_file_loose = "loose_flip_matrix_add_significance.csv"

ALPHA = 0.05
TOTAL_SAMPLES = 300  # 真实总样本300


def percent_to_float(percent_str):
    return float(str(percent_str).replace("%", "")) / 100


def calculate_mcnemar_p(b_count: int, c_count: int) -> float:
    table = [[0, b_count], [c_count, 0]]
    result = mcnemar(table, exact=True)
    return result.pvalue


def process_original_csv(input_path: str, output_path: str):
    df_original = pd.read_csv(input_path, encoding="utf-8-sig")
    print(f"开始处理文件：{input_path}，原始列：{list(df_original.columns)}")

    percent_columns = ["accuracy", "right_turn_false_rate", "false_turn_right_rate",
                       "total_flip_rate", "net_change"]
    for col in percent_columns:
        df_original[col] = df_original[col].apply(percent_to_float)

    # 新增：四格表四列 + 原有统计三列
    new_cols = ["n11", "n10", "n01", "n00", "raw_p", "holm_adj_p", "is_significant"]
    for col in new_cols:
        df_original[col] = None

    model_groups = df_original["model_name"].unique()

    for model in model_groups:
        df_model = df_original[df_original["model_name"] == model]
        row_plain = df_model[df_model["mutation_type"] == "plain"].iloc[0]
        acc_plain = row_plain["accuracy"]
        N1 = acc_plain * TOTAL_SAMPLES  # plain正确总样本
        N0 = TOTAL_SAMPLES - N1         # plain错误总样本

        df_disturb = df_model[df_model["mutation_type"] != "plain"]
        p_value_list = []
        index_mapping = []

        for idx, row in df_disturb.iterrows():
            br = row["right_turn_false_rate"]
            cr = row["false_turn_right_rate"]

            # 计算四格表频数
            n10 = int(br * TOTAL_SAMPLES)
            n01 = int(cr * TOTAL_SAMPLES)
            n11 = int(N1 - n10)
            n00 = int(N0 - n01)

            # 写入四格表数值
            df_original.at[idx, "n11"] = n11
            df_original.at[idx, "n10"] = n10
            df_original.at[idx, "n01"] = n01
            df_original.at[idx, "n00"] = n00

            # 计算McNemar原始p
            p_val = calculate_mcnemar_p(n10, n01)
            df_original.at[idx, "raw_p"] = p_val
            p_value_list.append(p_val)
            index_mapping.append(idx)

        # Holm校正
        reject_array, p_adjusted, _, _ = smm.multipletests(
            p_value_list, alpha=ALPHA, method="holm"
        )

        for pos, table_index in enumerate(index_mapping):
            df_original.at[table_index, "holm_adj_p"] = p_adjusted[pos]
            df_original.at[table_index, "is_significant"] = reject_array[pos]

    # 关闭科学计数法，保留8位小数
    pd.set_option('display.float_format', '{:.8f}'.format)
    for p_col in ["raw_p", "holm_adj_p"]:
        df_original[p_col] = df_original[p_col].apply(
            lambda val: "{0:.8f}".format(val) if pd.notnull(val) else val)

    df_original.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"文件处理完成！输出：{output_path}")
    print(f"新增字段：n11,n10,n01,n00,raw_p,holm_adj_p,is_significant\n")


if __name__ == "__main__":
    process_original_csv(input_file_strict, output_file_strict)
    process_original_csv(input_file_loose, output_file_loose)
    print("两个文件全部处理完毕！")
