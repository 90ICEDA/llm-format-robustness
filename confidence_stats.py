import os
import json
import math
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict

# 纯CPU运行
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ===================== 全局配置（一键切换参数） =====================
# 模型选择：测试用0.5B，正式用7B
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
# MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

INPUT_JSONL = "mutated_prompts.jsonl"
OUTPUT_JSONL = "model_generation_logprob.jsonl" 
OUTPUT_CSV = "confidence_stats.csv"

# 测试模式开关：TEST_NUM=20 仅跑前20条；TEST_NUM=None 全量运行
TEST_NUM = 20
# TEST_NUM = None

# 生成参数
MAX_NEW_TOKENS = 256
DO_SAMPLE = False
DEVICE_MAP = "cpu"  # 强制CPU

# 随机种子固定
SEED = 42
torch.manual_seed(SEED)
# 下面两行CUDA相关种子代码，无GPU可注释掉，不影响CPU运行
# torch.cuda.manual_seed_all(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

def calculate_confidence_metrics(generation_logprobs: List[Dict]) -> dict:
    """
    根据生成token的logprob列表计算4个指标
    generation_logprobs: 每个元素 {token_id, token_str, logprob, top2_logprob}
    """
    token_count = len(generation_logprobs)
    if token_count == 0:
        return {
            "first_generated_token_margin": None,
            "avg_answer_logprob": None,
            "perplexity": None,
            "generated_token_count": 0
        }

    first_token = generation_logprobs[0]
    first_margin = first_token["logprob"] - first_token["top2_logprob"]

    total_logp = sum([item["logprob"] for item in generation_logprobs])
    avg_logp = total_logp / token_count

    ppl = math.exp(-avg_logp)

    return {
        "first_generated_token_margin": round(first_margin, 6),
        "avg_answer_logprob": round(avg_logp, 6),
        "perplexity": round(ppl, 4),
        "generated_token_count": token_count
    }

# 主函数
def run_full_inference():
    print(f"=== 加载模型 {MODEL_NAME} ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map=DEVICE_MAP
    )
    model.eval()

    # 读取jsonl
    all_input_lines = []
    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_input_lines.append(json.loads(line))
    print(f"原始数据集总条数：{len(all_input_lines)}")

    # 测试模式截断前N条
    if TEST_NUM is not None:
        run_lines = all_input_lines[:TEST_NUM]
        print(f"【测试模式】仅运行前 {TEST_NUM} 条数据")
    else:
        run_lines = all_input_lines
        print("【全量模式】运行全部数据")

    # 存储中间jsonl结果、csv统计结果
    jsonl_records = []
    csv_records = []

    # 逐样本推理
    for idx, data in enumerate(run_lines):
        source_id = data["source_prompt_id"]
        mut_id = data["mutation_id"]
        mut_type = data["mutation_type"]
        raw_prompt = data["original_prompt"]
        mutated_prompt = data["mutated_prompt"]
        task_type = data["task_type"]

        # 构造chat模板输入
        messages = [
            {"role": "system", "content": "你是一个严谨的助手。"},
            {"role": "user", "content": mutated_prompt}
        ]
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([chat_text], return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        # 生成+输出每一步token的log概率
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                return_dict_in_generate=True,
                output_scores=True  # 输出每一步token概率分布
            )

        # 拆分输入token与生成token
        full_ids = outputs.sequences[0]
        gen_ids = full_ids[input_len:]
        response_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # 解析每一步生成token的top1、top2 logprob
        gen_logprob_list = []
        for step_scores in outputs.scores:
            log_probs = torch.log_softmax(step_scores, dim=-1)[0]
            # 取top2概率值
            top2_vals, top2_idx = torch.topk(log_probs, k=2)
            top1_logp = float(top2_vals[0])
            top2_logp = float(top2_vals[1])
            top1_token_id = int(top2_idx[0])
            top1_token_str = tokenizer.decode([top1_token_id])

            gen_logprob_list.append({
                "token_id": top1_token_id,
                "token_str": top1_token_str,
                "logprob": top1_logp,
                "top2_logprob": top2_logp
            })

        # 计算置信度指标
        metrics = calculate_confidence_metrics(gen_logprob_list)

        # 1. 写入jsonl完整记录
        jsonl_item = {
            "model_name": MODEL_NAME,
            "source_prompt_id": source_id,
            "mutation_id": mut_id,
            "mutation_type": mut_type,
            "task_type": task_type,
            "original_prompt": raw_prompt,
            "mutated_prompt": mutated_prompt,
            "model_response": response_text,
            "generated_token_logprob_details": gen_logprob_list,
            **metrics
        }
        jsonl_records.append(jsonl_item)

        # 2. 写入csv统计行
        csv_item = {
            "model_name": MODEL_NAME,
            "source_prompt_id": source_id,
            "mutation_id": mut_id,
            "mutation_type": mut_type,
            "task_type": task_type,
            "first_generated_token_margin": metrics["first_generated_token_margin"],
            "avg_answer_logprob": metrics["avg_answer_logprob"],
            "perplexity": metrics["perplexity"],
            "generated_token_count": metrics["generated_token_count"]
        }
        csv_records.append(csv_item)

        print(f"进度 {idx+1}/{len(run_lines)} | source_id={source_id}, mut_type={mut_type} 完成")

    # 保存jsonl中间文件
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for item in jsonl_records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\n完整生成日志已保存至: {OUTPUT_JSONL}")

    # 输出最终confidence_stats.csv
    df_stats = pd.DataFrame(csv_records)
    df_stats.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"置信度统计表格已保存至: {OUTPUT_CSV}")

if __name__ == "__main__":
    run_full_inference()
