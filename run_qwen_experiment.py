import json
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_token_logit(logits, tokenizer, text):
    token_id = tokenizer.encode(text, add_special_tokens=False)[-1]
    return float(logits[token_id].item())


def render_prompt(stem, options, format_type):
    opt_str = "\n".join([f"{k}. {v}" for k, v in options.items()])
    if format_type == "plain":
        return f"{stem}\n{opt_str}"
    elif format_type == "markdown":
        return f"**Question:**\n{stem}\n\n**Options:**\n{opt_str}"
    else:
        raise ValueError(f"Unknown format: {format_type}")


def run_qwen():
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"加载 Qwen 模型: {model_name} ...")

    # 恢复你之前最顺手、最标准的加载方式
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    model.eval()
    print("模型加载成功！")

    json_file = "independent_questions_30.json"
    with open(json_file, "r", encoding="utf-8") as f:
        bank_data = json.load(f)
    all_questions = bank_data["logic_bank"] + bank_data["factual_bank"]

    results = []
    for idx, item in enumerate(all_questions):
        qid, cat, stem, opts, gold = item["qid"], item["category"], item["stem"], item["options"], item["gold"]

        for fmt in ["plain", "markdown"]:
            rendered_base = render_prompt(stem, opts, fmt)

            # 1. 两阶段实验
            for label_swap in [False, True]:
                sub_label, dec_label = ("A", "B") if not label_swap else ("B", "A")
                sub_text, dec_text = ("提交", "不提交，因为没有足够把握") if not label_swap else (
                "不提交，因为没有足够把握", "提交")
                prompt_s2 = f"{rendered_base}\n\n你刚才针对本题给出了回答。请问你是否愿意提交？\n{sub_label}. {sub_text}\n{dec_label}. {dec_text}\n请选择："

                messages = [{"role": "user", "content": prompt_s2}]
                chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer([chat_text], return_tensors="pt").to(DEVICE)

                with torch.no_grad():
                    outputs = model(**inputs)
                logits = outputs.logits[0, -1, :]

                results.append({
                    "qid": qid, "category": cat, "format": fmt, "model": model_name,
                    "task_type": "two_stage", "config_id": f"s2_swap_{label_swap}",
                    "commit_logit": get_token_logit(logits, tokenizer, sub_label),
                    "decline_logit": get_token_logit(logits, tokenizer, dec_label),
                    "commit_minus_decline": get_token_logit(logits, tokenizer, sub_label) - get_token_logit(logits,
                                                                                                            tokenizer,
                                                                                                            dec_label),
                    "gold": gold
                })

            # 2. 单阶段对照组
            for method_type, abstain_word in [("unknown", "放弃回答"), ("control", "青金石")]:
                for pos in [0, 2, 4]:
                    base_keys = ["A", "B", "C", "D"]
                    base_keys.insert(pos, "E")
                    new_opts = {}
                    orig_keys_iter = iter(["A", "B", "C", "D"])
                    for k in base_keys:
                        new_opts[k] = abstain_word if k == "E" else opts[next(orig_keys_iter)]

                    single_prompt = f"{rendered_base}\n\n" + "\n".join(
                        [f"{k}. {v}" for k, v in new_opts.items()]) + "\n请选择正确选项："
                    messages = [{"role": "user", "content": single_prompt}]
                    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = tokenizer([chat_text], return_tensors="pt").to(DEVICE)

                    with torch.no_grad():
                        outputs = model(**inputs)
                    logits = outputs.logits[0, -1, :]

                    option_logits = {opt: get_token_logit(logits, tokenizer, opt) for opt in ["A", "B", "C", "D", "E"]}
                    logits_abcd = np.array(
                        [option_logits["A"], option_logits["B"], option_logits["C"], option_logits["D"]])
                    max_val = np.max(logits_abcd)
                    logsumexp_abcd = max_val + np.log(np.sum(np.exp(logits_abcd - max_val)))
                    sorted_vals = np.sort(logits_abcd)[::-1]
                    probs = np.exp(logits_abcd - max_val) / np.sum(np.exp(logits_abcd - max_val))

                    results.append({
                        "qid": qid, "category": cat, "format": fmt, "model": model_name,
                        "task_type": "single_stage_control", "method": method_type, "position": pos,
                        "margin": sorted_vals[0] - sorted_vals[1],
                        "entropy": -np.sum(probs * np.log(probs + 1e-9)),
                        "relative_preference": option_logits["E"] - logsumexp_abcd,
                        "gold": gold
                    })

        if (idx + 1) % 5 == 0:
            print(f"[Qwen] 进度: {idx + 1}/{len(all_questions)}")

    df = pd.DataFrame(results)
    df.to_csv("qwen_results.csv", index=False, encoding="utf-8-sig")
    print(" Qwen 实验跑完，已保存至 qwen_results.csv")


if __name__ == "__main__":
    run_qwen()
