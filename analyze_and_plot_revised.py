import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

def analyze_and_plot():
    # 查找 revised_results 目录下的 csv 文件
    results_dir = "revised_results"
    if not os.path.exists(results_dir):
        print(f"找不到结果文件夹 {results_dir}！请确保评测脚本已成功运行并保存数据。")
        return

    csv_files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"{results_dir} 目录下没有找到任何 CSV 结果文件！")
        return

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        model_name = df['model'].iloc[0] if 'model' in df.columns else "Model"

        # 筛选两阶段任务数据
        df_s2 = df[df.get('task_type', 'two_stage') == 'two_stage'].copy() if 'task_type' in df.columns else df.copy()

        # 按照 commit_minus_decline 从大到小排序（置信度从高到低）
        df_s2 = df_s2.sort_values(by='commit_minus_decline', ascending=False).reset_index(drop=True)

        total_items = len(df_s2)
        coverages = []
        risks = []
        ci_lower = []
        ci_upper = []

        # 动态滑动截断计算真实 Coverage 与 Risk (带 Bootstrap 置信区间)
        for i in range(1, total_items + 1):
            subset = df_s2.head(i)
            coverage = i / total_items

            # 真实错误率：在当前接受的子集中，预测错误（is_correct == False）的比例
            # 注意：is_correct 在 CSV 中是布尔值或整型
            incorrect_mask = (subset['is_correct'] == False) | (subset['is_correct'] == 0)
            risk = incorrect_mask.mean()

            coverages.append(coverage)
            risks.append(risk)

            # 使用 Bootstrap 计算 95% 置信区间
            if len(subset) > 5:
                boot_risks = []
                for _ in range(100):
                    sample = subset.sample(n=len(subset), replace=True)
                    b_inc = (sample['is_correct'] == False) | (sample['is_correct'] == 0)
                    boot_risks.append(b_inc.mean())
                ci_l = np.percentile(boot_risks, 2.5)
                ci_u = np.percentile(boot_risks, 97.5)
            else:
                ci_l, ci_u = risk, risk

            ci_lower.append(ci_l)
            ci_upper.append(ci_u)

        coverages = np.array(coverages)
        risks = np.array(risks)
        ci_lower = np.array(ci_lower)
        ci_upper = np.array(ci_upper)

        # 绘制真实波动曲线（保留波动，不强行平滑）
        line, = ax.plot(coverages, risks, marker='o', markersize=3, linewidth=2, label=model_name, alpha=0.9)
        color = line.get_color()

        # 填充置信区间阴影
        ax.fill_between(coverages, ci_lower, ci_upper, color=color, alpha=0.2)

    # 图表修饰
    ax.set_title("Selective Risk vs. Coverage 真实性能曲线 (含置信区间)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Coverage (覆盖率 / 系统接受样本比例)", fontsize=12, labelpad=10)
    ax.set_ylabel("Selective Risk (选择性风险 / 错误率)", fontsize=12, labelpad=10)

    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(-0.02, 1.05)

    ax.legend(frameon=True, facecolor='white', edgecolor='none', shadow=True, fontsize=11, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.5, color='#cccccc')

    output_fig = "revised_risk_coverage_with_ci.png"
    plt.tight_layout()
    plt.savefig(output_fig, dpi=300)
    print(f"真实带置信区间的 Risk-Coverage 曲线图已生成并保存至 {output_fig}")
    plt.show()


if __name__ == "__main__":
    analyze_and_plot()
