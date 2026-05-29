import json


def mutate_prompt(original_prompt):

    mutations = {}

    # 变体 1: plain (原始问题保持不变)
    mutations["plain"] = original_prompt

    # 变体 2: markdown (用Markdown格式包装)
    mutations["markdown"] = (
        f"**问题**: {original_prompt}\n"
    )

    # 变体 3: json (把问题包装成 JSON 格式)
    json_structure = {"request": "query_task", "content": original_prompt}
    mutations["json"] = json.dumps(json_structure, ensure_ascii=False)

    # 变体 4: code_block (把问题放进代码块里)
    mutations["code_block"] = f"```text\n{original_prompt}\n```"

    # 变体 5: zh_en_mix (中英混合表达)
    mutations["zh_en_mix"] = f"Hello! 关于下面这个问题，请 Give me a detailed answer: '{original_prompt}'. Thank you!"

    # 变体 6: multi_turn (改成多轮对话形式)
    mutations["multi_turn"] = (
        f"User: 你好，请问我可以向你请教问题吗？\n"
        f"Assistant: 您好！当然可以。\n"
        f"User: 太好了，我的具体问题是：{original_prompt}"
    )

    return mutations


def main():
    input_file = "prompts.jsonl"
    output_file = "mutated_prompts.jsonl"

    print("变体生成启动")

    with open(input_file, "r", encoding="utf-8") as f_in, \
            open(output_file, "w", encoding="utf-8") as f_out:

        for line in f_in:
            total_mutations = 0
            line = line.strip()
            # 1. 流式读取上周的JSON对象
            data = json.loads(line)

            # 2. 提取最核心的基础字段
            original_id = data["prompt_id"]
            task_type = data["task_type"]
            original_prompt = data["prompt"]

            # 3. 激活变体
            all_mutations = mutate_prompt(original_prompt)

            # 4. 将每一条变体在输出文件里单独占一行
            for mutation_type, mutated_text in all_mutations.items():
                mutated_object = {
                    "source_prompt_id": original_id,  # 原始编号
                    "mutation_id": total_mutations + 1,  # 变体编号
                    "task_type": task_type,  # 任务类型
                    "mutation_type": mutation_type,  # 变体类型
                    "original_prompt": original_prompt,  # 原始问题
                    "mutated_prompt": mutated_text  # 变体
                }
                total_mutations+=1
                f_out.write(json.dumps(mutated_object, ensure_ascii=False) + "\n")

    print("成功生成变体")


if __name__ == "__main__":
    main()
