# Sanity Check

本报告抽取了 **source_prompt_id = 66** 的 `plain` 组与 `json` 组进行对比。

### 逐层余弦相似度 (Cosine Similarity)

| 层级 (Layer) | 余弦相似度 |
| :--- | :--- |
| Layer 0 (Embedding 词嵌入层) | **1.0078** |
| Layer 1 (Transformer 层) | **1.0000** |
| Layer 2 (Transformer 层) | **0.9961** |
| Layer 3 (Transformer 层) | **0.9922** |
| Layer 4 (Transformer 层) | **0.9883** |
| Layer 5 (Transformer 层) | **0.9727** |
| Layer 6 (Transformer 层) | **0.9766** |
| Layer 7 (Transformer 层) | **0.9688** |
| Layer 8 (Transformer 层) | **0.9766** |
| Layer 9 (Transformer 层) | **0.9688** |
| Layer 10 (Transformer 层) | **0.9727** |
| Layer 11 (Transformer 层) | **0.9688** |
| Layer 12 (Transformer 层) | **0.9688** |
| Layer 13 (Transformer 层) | **0.9688** |
| Layer 14 (Transformer 层) | **0.9766** |
| Layer 15 (Transformer 层) | **0.9727** |
| Layer 16 (Transformer 层) | **0.9766** |
| Layer 17 (Transformer 层) | **0.9805** |
| Layer 18 (Transformer 层) | **0.9844** |
| Layer 19 (Transformer 层) | **0.9805** |
| Layer 20 (Transformer 层) | **0.9844** |
| Layer 21 (Transformer 层) | **0.9844** |
| Layer 22 (Transformer 层) | **0.9883** |
| Layer 23 (Transformer 层) | **0.9844** |
| Layer 24 (Transformer 层) | **0.9727** |
没有完全符合逐层下降
