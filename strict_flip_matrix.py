import csv
from collections import defaultdict


def generate_accuracy_matrix(input_csv_path, output_csv_path):
    # 三层结构：stats[model_name][mutation_type] = 统计字典
    stats = defaultdict(lambda: defaultdict(lambda: {
        "right_id": [],
        "false_id": [],
        "accuracy": 0.0,
        "right_turn_false": 0.0,  # plain对的，当前变体错的 翻转率
        "false_turn_right": 0.0   # plain错的，当前变体对的 翻转率
    }))
    print(f"正在读取 {input_csv_path} 进行效能对齐...")

    # 1. 读取csv，按模型+变异类型归集id
    with open(input_csv_path, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_name = row["model_name"]
            mutation_type = row["mutation_type"]
            correct_flag = int(row["correct_flag"])

            qid = int(row["question_id"])
            if correct_flag == 1:
                stats[model_name][mutation_type]["right_id"].append(qid)
            else :
                stats[model_name][mutation_type]["false_id"].append(qid)
    # 2. 遍历每个模型，计算每个变异类型指标
    for model_name, mut_dict in stats.items():
        # 先获取plain基准数据
        plain_data = mut_dict.get("plain", {"right_id": [], "false_id": []})
        plain_right = set(plain_data["right_id"])
        plain_false = set(plain_data["false_id"])
        plain_all = plain_right | plain_false
        len_plain_all = len(plain_all)

        # 遍历该模型下所有变异类型（plain/markdown/json...）逐个计算
        for mut_type, data in mut_dict.items():
            curr_right = set(data["right_id"])
            curr_false = set(data["false_id"])
            curr_all = curr_right | curr_false
            len_curr_all = len(curr_all)

            # 计算当前变体准确率，防除0
            if len_curr_all == 0:
                acc = 0.0
            else:
                acc = round((len(curr_right) / len_curr_all) * 100, 2)
            data["accuracy"] = acc

            # 无plain样本时翻转率全部置0
            if len_plain_all == 0:
                data["right_turn_false"] = 0.0
                data["false_turn_right"] = 0.0
                continue

            # 翻转集合：plain对→当前错；plain错→当前对
            flip_right2false = plain_right & curr_false
            flip_false2right = plain_false & curr_right

            # 计算翻转百分比，基于plain总题目数
            data["right_turn_false"] = round((len(flip_right2false) / len_plain_all) * 100, 2)
            data["false_turn_right"] = round((len(flip_false2right) / len_plain_all) * 100, 2)

    # 输出顺序定义
    mutation_order = ["plain", "markdown", "json", "code_block", "zh_en_mix", "multi_turn"]
    fieldnames = [
        "model_name", "mutation_type", "accuracy",
        "right_turn_false_rate", "false_turn_right_rate",
        "total_flip_rate", "net_change"
    ]

    # 3. 写入结果csv
    with open(output_csv_path, mode='w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # 按模型名排序输出
        for model_name in sorted(stats.keys()):
            mut_dict = stats[model_name]
            available_muts = list(mut_dict.keys())
            # 按预设顺序排序变异类型
            ordered_muts = [m for m in mutation_order if m in available_muts]
            ordered_muts += [m for m in available_muts if m not in mutation_order]

            for mut_type in ordered_muts:
                data = mut_dict[mut_type]
                acc = data["accuracy"]
                rt_f = data["right_turn_false"]
                ft_r = data["false_turn_right"]
                total_flip = round(rt_f + ft_r,2)
                net = round(ft_r - rt_f,2)

                writer.writerow({
                    "model_name": model_name,
                    "mutation_type": mut_type,
                    "accuracy": f"{acc}%",
                    "right_turn_false_rate": f"{rt_f}%",
                    "false_turn_right_rate": f"{ft_r}%",
                    "total_flip_rate": f"{total_flip}%",
                    "net_change": f"{net}%"
                })

    print(f"矩阵计算完毕！统计结果已保存至: {output_csv_path}")


if __name__ == "__main__":
    input_csv = "new_qa_correctness.csv"
    output_csv = "strict_flip_matrix.csv"
    generate_accuracy_matrix(input_csv, output_csv)
