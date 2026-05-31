import json
import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==================== 配置区域 ====================
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
INPUT_JSONL = "mutated_outputs.jsonl"
LABELS_CSV = "refusal_labels.csv"

OUTPUT_PT = "hidden_states.pt"
OUTPUT_META = "metadata.json"
OUTPUT_MD = "hidden_states_sanity.md"

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# ==================================================

def main():
    print(f"正在加载模型和分词器: {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto"
    )
    model.eval()  # 切换至评估模式

    # 1. 对齐数据
    print("正在对齐输入数据与拒答标签...")
    df_labels = pd.read_csv(LABELS_CSV)
    label_dict = {}
    for _, row in df_labels.iterrows():
        # 用source_prompt_id + mutation_type作为唯一Key
        key = (int(row['source_prompt_id']), str(row['mutation_type']))
        label_dict[key] = int(row['refusal_flag'])

    # 2. 依次读取 Prompt 并提取 Hidden States
    all_hidden_states = []
    metadata = []

    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        lines = [json.loads(line.strip()) for line in f if line.strip()]

    print(f"开始提取 Hidden States (共 {len(lines)} 条)...")
    for idx, data in enumerate(tqdm(lines)):
        sp_id = int(data['source_prompt_id'])
        mut_type = str(data['mutation_type'])
        task_type = str(data['task_type'])
        prompt = data['mutated_prompt']

        refusal_flag = label_dict.get((sp_id, mut_type))

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # 调用模型并要求输出hidden_states
        with torch.no_grad():
            outputs = model(**model_inputs, output_hidden_states=True,do_sample = False)

        # 取每层的最后一个token的向量
        layer_features = []
        for layer_hs in outputs.hidden_states:
            layer_features.append(layer_hs[0, -1, :].cpu())

        # 堆叠成 [25, 896] 的张量
        single_prompt_hs = torch.stack(layer_features)
        all_hidden_states.append(single_prompt_hs)

        # 记录元数据
        metadata.append({
            "index": idx,
            "source_prompt_id": sp_id,
            "mutation_type": mut_type,
            "task_type": task_type,
            "refusal_flag": refusal_flag
        })

    # 3. 堆叠成最终的 [600, 25, 896] 大矩阵
    final_tensor = torch.stack(all_hidden_states)
    print(f"成功构建张量，最终形状为: {list(final_tensor.shape)}")

    torch.save(final_tensor, OUTPUT_PT)
    with open(OUTPUT_META, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)
    print(f"-> 张量已保存至: {OUTPUT_PT}")
    print(f"-> 元数据已保存至: {OUTPUT_META}")

    # ==================== 4. SANITY CHECK  ====================
    print("\n正在进行Sanity Check")
    df_meta = pd.DataFrame(metadata)

    target_id = 66

    idx_plain = \
    df_meta[(df_meta['source_prompt_id'] == target_id) & (df_meta['mutation_type'] == 'plain')]['index'].values[0]
    idx_json = \
    df_meta[(df_meta['source_prompt_id'] == target_id) & (df_meta['mutation_type'] == 'json')]['index'].values[0]

    # 提取这两条数据的 [25, 896] 矩阵
    hs_plain = final_tensor[idx_plain]
    hs_json = final_tensor[idx_json]

    # 逐层计算余弦相似度
    cos_sim_list = []
    for layer in range(25):
        sim = torch.nn.functional.cosine_similarity(hs_plain[layer], hs_json[layer], dim=0)
        cos_sim_list.append((layer, sim.item()))

    # 写入 hidden_states_sanity.md
    with open(OUTPUT_MD, 'w', encoding='utf-8') as f:
        f.write(f"# Sanity Check\n\n")
        f.write(f"本报告抽取了 **source_prompt_id = {target_id}** 的 `plain` 组与 `json` 组进行对比。\n\n")
        f.write(f"### 逐层余弦相似度 (Cosine Similarity)\n\n")
        f.write(f"| 层级 (Layer) | 余弦相似度 |\n")
        f.write(f"| :--- | :--- |\n")
        for layer, sim_val in cos_sim_list:
            layer_name = "Layer 0 (Embedding 词嵌入层)" if layer == 0 else f"Layer {layer} (Transformer 层)"
            f.write(f"| {layer_name} | **{sim_val:.4f}** |\n")
    print(f"Sanity Check 报告已生成: {OUTPUT_MD}\n")


if __name__ == '__main__':
    main()
