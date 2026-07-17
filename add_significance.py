import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
import statsmodels.stats.multitest as smm

input_file_strict = "strict_flip_matrix.csv"
input_file_loose = "loose_flip_matrix.csv"
output_file_strict = "strict_flip_matrix_add_significance.csv"
output_file_loose = "loose_flip_matrix_add_significance.csv"

ALPHA = 0.05
VIRTUAL_TOTAL = 10000  # 虚拟样本数，仅用于比例转频数，不影响统计结论


def percent_to_float(percent_str):
    return float(str(percent_str).replace("%", "")) / 100


def calculate_mcnemar_p(b_rate: float, c_rate: float) -> float:
    b_count = int(b_rate * VIRTUAL_TOTAL)
    c_count = int(c_rate * VIRTUAL_TOTAL)
    table = [[0, b_count], [c_count, 0]]
    result = mcnemar(table, exact=True)
    return result.pvalue


def process_original_csv(input_path: str, output_path: str):
    df_original = pd.read_csv(input_path, encoding="utf-8-sig")
    print(f"开始处理文件：{input_path}，原始列：{list(df_original.columns)}")

    # 百分比字段转为数值型
    percent_columns = ["accuracy", "right_turn_false_rate", "false_turn_right_rate",
                       "total_flip_rate", "net_change"]
    for col in percent_columns:
        df_original[col] = df_original[col].apply(percent_to_float)

    # 新增三个空列，后续填充数据
    df_original["raw_p"] = None
    df_original["holm_adj_p"] = None
    df_original["is_significant"] = None

    model_groups = df_original["model_name"].unique()

    for model in model_groups:
        df_model = df_original[df_original["model_name"] == model]
        # 基准行：plain原始无扰动
        row_plain = df_model[df_model["mutation_type"] == "plain"].iloc[0]
        # 筛选所有扰动格式（排除plain基准）
        df_disturb = df_model[df_model["mutation_type"] != "plain"]

        p_value_list = []
        index_mapping = []

        # 批量计算全部McNemar原始P值
        for idx, row in df_disturb.iterrows():
            br = row["right_turn_false_rate"]
            cr = row["false_turn_right_rate"]
            p_val = calculate_mcnemar_p(br, cr)

            df_original.at[idx, "raw_p"] = p_val
            p_value_list.append(p_val)
            index_mapping.append(idx)

        # 对本组全部P值执行Holm校正
        reject_array, p_adjusted, _, _ = smm.multipletests(
            p_value_list, alpha=ALPHA, method="holm"
        )

        # 把校正结果回填到原DataFrame
        for pos, table_index in enumerate(index_mapping):
            df_original.at[table_index, "holm_adj_p"] = p_adjusted[pos]
            df_original.at[table_index, "is_significant"] = reject_array[pos]

    # 关闭科学计数法
    pd.set_option('display.float_format', '{:.8f}'.format)
    # 格式化两列P值，固定8位小数，彻底取消E指数格式
    df_original['raw_p'] = df_original['raw_p'].apply(lambda val: "{0:.8f}".format(val) if pd.notnull(val) else val)
    df_original['holm_adj_p'] = df_original['holm_adj_p'].apply(
        lambda val: "{0:.8f}".format(val) if pd.notnull(val) else val)

    df_original.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"文件处理完成！输出：{output_path}\n新增字段：raw_p、holm_adj_p、is_significant")


if __name__ == "__main__":
    process_original_csv(input_file_strict, output_file_strict)
    process_original_csv(input_file_loose, output_file_loose)
    print("两个文件全部处理完毕！")
