import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict

# ===================== 配置项 =====================
RESULT_FILE = "retrieval_results/result_top60.txt"
# 12个地点标签
LOCATIONS = ["fhy", "jx", "kx", "mh", "nm", "sjz", "sy", "tsg", "ty", "yf", "yk", "zx"]
# 目标保存文件夹名称
SAVE_DIR = "P@K图"
# K值列表
K_LIST = [20, 40, 60]

# ===================== 数据结构 =====================
# 存储每个地点所有查询的 P@20/P@40/P@60
loc_raw_data = defaultdict(lambda: {"p20": [], "p40": [], "p60": []})
# 存储每个地点的平均 P@20/P@40/P@60（用于绘图）
loc_avg_data = {}


# ===================== 工具函数 =====================
def extract_label(filename):
    """从文件名提取地点标签"""
    for loc in LOCATIONS:
        if filename.startswith(loc + "-"):
            return loc
    return None


def compute_precision(ret_list, target_loc, k):
    """计算单条查询的 Precision@K"""
    top_k = ret_list[:k]
    correct = sum(1 for name in top_k if extract_label(name) == target_loc)
    return correct / k


def get_y_ticks(min_val, max_val):
    """生成Y轴刻度"""
    step = 0.1
    ticks = []
    start = round(min_val - (min_val % step), 2)
    start = 0 if start < 0 else start
    while start <= max_val + step:
        ticks.append(round(start, 2))
        start += step
    return ticks


# ===================== 核心逻辑：计算准确率 =====================
def calculate_precision():
    try:
        with open(RESULT_FILE, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("错误：未找到 result_top60.txt，请检查文件路径！")
        return False

    idx = 0
    total_lines = len(lines)

    while idx < total_lines:
        line = lines[idx]
        # 识别查询行
        if line.startswith("查询图:"):
            # 提取查询图所属地点
            query_name = line.split(":", 1)[1].strip()
            query_loc = extract_label(query_name)
            if not query_loc:
                idx += 1
                continue

            idx += 1
            # 跳过 TOP60 结果标题行
            if idx < total_lines and lines[idx].startswith("TOP60"):
                idx += 1

            # 读取当前查询的60条检索结果
            ret_names = []
            while idx < total_lines and not lines[idx].startswith("-" * 60):
                match = re.match(r"\s*\d+\.\s+(.+)", lines[idx])
                if match:
                    ret_names.append(match.group(1))
                idx += 1

            # 仅处理完整的60条结果
            if len(ret_names) == 60:
                p20 = compute_precision(ret_names, query_loc, 20)
                p40 = compute_precision(ret_names, query_loc, 40)
                p60 = compute_precision(ret_names, query_loc, 60)
                loc_raw_data[query_loc]["p20"].append(p20)
                loc_raw_data[query_loc]["p40"].append(p40)
                loc_raw_data[query_loc]["p60"].append(p60)
        else:
            idx += 1

    # 计算每个地点的平均准确率
    for loc in LOCATIONS:
        data = loc_raw_data[loc]
        sample_num = len(data["p20"])
        if sample_num == 0:
            avg_p20 = avg_p40 = avg_p60 = 0.0
        else:
            avg_p20 = round(sum(data["p20"]) / sample_num, 4)
            avg_p40 = round(sum(data["p40"]) / sample_num, 4)
            avg_p60 = round(sum(data["p60"]) / sample_num, 4)
        loc_avg_data[loc] = [avg_p20, avg_p40, avg_p60]

    return True


# ===================== 核心逻辑：绘制图表 =====================
def plot_precision_charts():
    # 自动创建文件夹（不存在则新建，存在则跳过）
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 循环绘制12张独立图片
    for loc, precision in loc_avg_data.items():
        plt.figure(figsize=(6, 4.5), dpi=150)
        x = K_LIST
        y = precision

        # 绘制散点
        plt.scatter(x, y, color='#4a90e2', s=180, zorder=5)

        # 绘制灰色辅助虚线 + 标注数值
        for xi, yi in zip(x, y):
            plt.vlines(x=xi, ymin=0, ymax=yi, color='gray', linestyle='--', linewidth=1)
            plt.hlines(y=yi, xmin=0, xmax=xi, color='gray', linestyle='--', linewidth=1)
            plt.annotate(f"{yi:.4f}", (xi, yi), textcoords="offset points",
                         xytext=(0, 8), ha='center', fontsize=8)

        # 坐标轴、标题
        plt.xlabel("K", fontsize=11)
        plt.ylabel("Precision", fontsize=11)
        plt.title(f"{loc} Precision@K", fontsize=11)

        # 刻度设置
        plt.xticks(K_LIST)
        y_min = min(y)
        y_max = max(y)
        plt.yticks(get_y_ticks(y_min, y_max))
        plt.ylim(bottom=0)

        # 坐标轴样式
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')

        plt.tight_layout()
        # 拼接完整保存路径，图片存入 P@K图 文件夹
        save_path = os.path.join(SAVE_DIR, f"{loc}_precision.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    print(f"\n所有图表已成功保存至文件夹：{SAVE_DIR}")


# ===================== 输出统计结果 =====================
def print_statistics():
    print("=" * 70)
    print("                各地点检索平均准确率统计结果")
    print("=" * 70)
    print(f"{'地点':<4} | {'平均P@20':<10} | {'平均P@40':<10} | {'平均P@60':<10} | {'查询样本数'}")
    print("-" * 70)

    for loc in LOCATIONS:
        data = loc_raw_data[loc]
        sample_num = len(data["p20"])
        if sample_num == 0:
            print(f"{loc:<4} | 无数据     | 无数据     | 无数据     | 0")
            continue

        avg_p20 = round(sum(data["p20"]) / sample_num, 4)
        avg_p40 = round(sum(data["p40"]) / sample_num, 4)
        avg_p60 = round(sum(data["p60"]) / sample_num, 4)
        print(f"{loc:<4} | {avg_p20:<10} | {avg_p40:<10} | {avg_p60:<10} | {sample_num}")

    print("=" * 70)


# ===================== 主函数 =====================
def main():
    # 1. 计算准确率
    if not calculate_precision():
        return

    # 2. 输出统计结果
    print_statistics()

    # 3. 绘制并保存图表
    plot_precision_charts()


if __name__ == "__main__":
    main()