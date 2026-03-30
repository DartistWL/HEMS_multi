"""
参数扫描结果绘图
读取 run_parameter_sweep.py 生成的 CSV/JSON，绘制社区峰值、总成本、公平性等随 initial_soc / initial_credit 的变化。

用法:
  python multi_agent/plot_parameter_sweep.py --data_file multi_agent/sweep_results/sweep_mappo.csv
  python multi_agent/plot_parameter_sweep.py --data_file multi_agent/sweep_results/sweep_mappo.json --output_dir multi_agent/sweep_results/figures
"""
import sys
import os
import json
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def load_sweep_csv(csv_path):
    """从 sweep_mappo.csv 加载数据，返回 list of dict。"""
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            row = {}
            for k, v in r.items():
                try:
                    row[k] = float(v)
                except (ValueError, TypeError):
                    row[k] = v
            rows.append(row)
    return rows


def load_sweep_json(json_path):
    """从 sweep_mappo.json 加载汇总数据，转为与 CSV 相同的行格式。"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    rows = []
    for r in data.get('results', []):
        mc = r.get('mean_agent_costs', [0, 0, 0])
        rows.append({
            'initial_soc': r['initial_soc'],
            'initial_credit': r['initial_credit'],
            'mean_peak_load': r['mean_peak_load'],
            'std_peak_load': r['std_peak_load'],
            'mean_total_cost': r['mean_total_cost'],
            'std_total_cost': r['std_total_cost'],
            'mean_agent1_cost': mc[0] if len(mc) > 0 else 0,
            'mean_agent2_cost': mc[1] if len(mc) > 1 else 0,
            'mean_agent3_cost': mc[2] if len(mc) > 2 else 0,
            'mean_cost_std': r.get('mean_cost_std', 0),
        })
    return rows


def load_data(data_file):
    """根据扩展名加载 CSV 或 JSON。"""
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"数据文件不存在: {data_file}")
    ext = os.path.splitext(data_file)[1].lower()
    if ext == '.csv':
        return load_sweep_csv(data_file)
    if ext == '.json':
        return load_sweep_json(data_file)
    raise ValueError(f"不支持的文件格式: {data_file}，请使用 .csv 或 .json")


def get_unique_sorted(rows, key):
    return sorted(set(r[key] for r in rows))


def _add_subplot_label(ax, label, x_offset=-0.05):
    """在子图左上角标注小写字母（与 cost_comparison 格式一致）。x_offset 为 transAxes 下横坐标，负值越大越向左。"""
    ax.text(x_offset, 1.02, label, transform=ax.transAxes, fontsize=20, fontweight='bold',
            va='bottom', ha='left')


def plot_peak_vs_soc(rows, output_dir):
    """社区峰值 vs 初始 SOC，每条线一个 initial_credit。"""
    credits = get_unique_sorted(rows, 'initial_credit')
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for ic in credits:
        sub = [r for r in rows if r['initial_credit'] == ic]
        sub = sorted(sub, key=lambda x: x['initial_soc'])
        x = [r['initial_soc'] for r in sub]
        y = [r['mean_peak_load'] for r in sub]
        ye = [r['std_peak_load'] for r in sub]
        ax.plot(x, y, 'o-', label=f'初始积分={ic:.0f}')
        ax.fill_between(x, np.array(y) - np.array(ye), np.array(y) + np.array(ye), alpha=0.2)
    ax.set_xlabel('社区储能初始 SOC')
    ax.set_ylabel('社区净负荷峰值 (kW)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, 'peak_vs_initial_soc.png')
    fig.savefig(path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"已保存: {path}")


def plot_cost_vs_soc(rows, output_dir):
    """总成本 vs 初始 SOC，每条线一个 initial_credit。"""
    credits = get_unique_sorted(rows, 'initial_credit')
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for ic in credits:
        sub = [r for r in rows if r['initial_credit'] == ic]
        sub = sorted(sub, key=lambda x: x['initial_soc'])
        x = [r['initial_soc'] for r in sub]
        y = [r['mean_total_cost'] for r in sub]
        ye = [r['std_total_cost'] for r in sub]
        ax.plot(x, y, 'o-', label=f'初始积分={ic:.0f}')
        ax.fill_between(x, np.array(y) - np.array(ye), np.array(y) + np.array(ye), alpha=0.2)
    ax.set_xlabel('社区储能初始 SOC')
    ax.set_ylabel('总电网成本 (元/天)')
    ax.legend(ncol=5, loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, 'cost_vs_initial_soc.png')
    fig.savefig(path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"已保存: {path}")


def plot_peak_vs_credit(rows, output_dir):
    """社区峰值 vs 初始积分，每条线一个 initial_soc。"""
    socs = get_unique_sorted(rows, 'initial_soc')
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for soc in socs:
        sub = [r for r in rows if r['initial_soc'] == soc]
        sub = sorted(sub, key=lambda x: x['initial_credit'])
        x = [r['initial_credit'] for r in sub]
        y = [r['mean_peak_load'] for r in sub]
        ye = [r['std_peak_load'] for r in sub]
        ax.plot(x, y, 'o-', label=f'初始SOC={soc}')
        ax.fill_between(x, np.array(y) - np.array(ye), np.array(y) + np.array(ye), alpha=0.2)
    ax.set_xlabel('各家庭初始积分')
    ax.set_ylabel('社区净负荷峰值 (kW)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, 'peak_vs_initial_credit.png')
    fig.savefig(path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"已保存: {path}")


def plot_cost_vs_credit(rows, output_dir):
    """总成本 vs 初始积分，每条线一个 initial_soc。"""
    socs = get_unique_sorted(rows, 'initial_soc')
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for soc in socs:
        sub = [r for r in rows if r['initial_soc'] == soc]
        sub = sorted(sub, key=lambda x: x['initial_credit'])
        x = [r['initial_credit'] for r in sub]
        y = [r['mean_total_cost'] for r in sub]
        ye = [r['std_total_cost'] for r in sub]
        ax.plot(x, y, 'o-', label=f'初始SOC={soc}')
        ax.fill_between(x, np.array(y) - np.array(ye), np.array(y) + np.array(ye), alpha=0.2)
    ax.set_xlabel('各家庭初始积分')
    ax.set_ylabel('总电网成本 (元/天)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, 'cost_vs_initial_credit.png')
    fig.savefig(path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"已保存: {path}")


def plot_fairness_vs_soc(rows, output_dir):
    """公平性（成本标准差）vs 初始 SOC，每条线一个 initial_credit。"""
    credits = get_unique_sorted(rows, 'initial_credit')
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for ic in credits:
        sub = [r for r in rows if r['initial_credit'] == ic]
        sub = sorted(sub, key=lambda x: x['initial_soc'])
        x = [r['initial_soc'] for r in sub]
        y = [r['mean_cost_std'] for r in sub]
        ax.plot(x, y, 'o-', label=f'初始积分={ic:.0f}')
    ax.set_xlabel('社区储能初始 SOC')
    ax.set_ylabel('各家庭成本标准差 (元/天)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, 'fairness_vs_initial_soc.png')
    fig.savefig(path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"已保存: {path}")


def plot_fairness_vs_credit(rows, output_dir):
    """公平性 vs 初始积分，每条线一个 initial_soc。"""
    socs = get_unique_sorted(rows, 'initial_soc')
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for soc in socs:
        sub = [r for r in rows if r['initial_soc'] == soc]
        sub = sorted(sub, key=lambda x: x['initial_credit'])
        x = [r['initial_credit'] for r in sub]
        y = [r['mean_cost_std'] for r in sub]
        ax.plot(x, y, 'o-', label=f'初始SOC={soc}')
    ax.set_xlabel('各家庭初始积分')
    ax.set_ylabel('各家庭成本标准差 (元/天)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, 'fairness_vs_initial_credit.png')
    fig.savefig(path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"已保存: {path}")


def plot_heatmap_peak(rows, output_dir):
    """热力图：initial_soc x initial_credit -> 社区峰值。"""
    socs = get_unique_sorted(rows, 'initial_soc')
    credits = get_unique_sorted(rows, 'initial_credit')
    Z = np.nan * np.ones((len(credits), len(socs)))
    for r in rows:
        i = credits.index(r['initial_credit'])
        j = socs.index(r['initial_soc'])
        Z[i, j] = r['mean_peak_load']
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(Z, aspect='auto', origin='lower',
                   extent=[min(socs), max(socs), min(credits), max(credits)],
                   cmap='viridis')
    ax.set_xlabel('社区储能初始 SOC')
    ax.set_ylabel('各家庭初始积分')
    plt.colorbar(im, ax=ax, label='峰值 (kW)')
    fig.tight_layout()
    path = os.path.join(output_dir, 'heatmap_peak.png')
    fig.savefig(path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"已保存: {path}")


def plot_heatmap_cost(rows, output_dir):
    """热力图：initial_soc x initial_credit -> 总成本。"""
    socs = get_unique_sorted(rows, 'initial_soc')
    credits = get_unique_sorted(rows, 'initial_credit')
    Z = np.nan * np.ones((len(credits), len(socs)))
    for r in rows:
        i = credits.index(r['initial_credit'])
        j = socs.index(r['initial_soc'])
        Z[i, j] = r['mean_total_cost']
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(Z, aspect='auto', origin='lower',
                   extent=[min(socs), max(socs), min(credits), max(credits)],
                   cmap='plasma')
    ax.set_xlabel('社区储能初始 SOC')
    ax.set_ylabel('各家庭初始积分')
    plt.colorbar(im, ax=ax, label='成本 (元/天)')
    fig.tight_layout()
    path = os.path.join(output_dir, 'heatmap_cost.png')
    fig.savefig(path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"已保存: {path}")


def plot_group_heatmaps(rows, output_dir):
    """组图：两张热力图并排，(a) 峰值 (b) 成本，左上角标注 a、b。"""
    socs = get_unique_sorted(rows, 'initial_soc')
    credits = get_unique_sorted(rows, 'initial_credit')
    Z_peak = np.nan * np.ones((len(credits), len(socs)))
    Z_cost = np.nan * np.ones((len(credits), len(socs)))
    for r in rows:
        i = credits.index(r['initial_credit'])
        j = socs.index(r['initial_soc'])
        Z_peak[i, j] = r['mean_peak_load']
        Z_cost[i, j] = r['mean_total_cost']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im1 = ax1.imshow(Z_peak, aspect='auto', origin='lower',
                     extent=[min(socs), max(socs), min(credits), max(credits)],
                     cmap='viridis')
    ax1.set_xlabel('社区储能初始 SOC')
    ax1.set_ylabel('各家庭初始积分')
    plt.colorbar(im1, ax=ax1, label='峰值 (kW)')
    _add_subplot_label(ax1, 'a', x_offset=-0.12)
    im2 = ax2.imshow(Z_cost, aspect='auto', origin='lower',
                     extent=[min(socs), max(socs), min(credits), max(credits)],
                     cmap='plasma')
    ax2.set_xlabel('社区储能初始 SOC')
    ax2.set_ylabel('各家庭初始积分')
    plt.colorbar(im2, ax=ax2, label='成本 (元/天)')
    _add_subplot_label(ax2, 'b', x_offset=-0.12)
    fig.tight_layout()
    path = os.path.join(output_dir, 'sensitivity_heatmaps.png')
    fig.savefig(path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"已保存: {path}")


def plot_group_cost_curves(rows, output_dir):
    """组图：两张成本曲线并排，(a) 成本 vs SOC (b) 成本 vs 初始积分，左上角标注 a、b。"""
    credits = get_unique_sorted(rows, 'initial_credit')
    socs = get_unique_sorted(rows, 'initial_soc')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for ic in credits:
        sub = [r for r in rows if r['initial_credit'] == ic]
        sub = sorted(sub, key=lambda x: x['initial_soc'])
        x = [r['initial_soc'] for r in sub]
        y = [r['mean_total_cost'] for r in sub]
        ye = [r['std_total_cost'] for r in sub]
        ax1.plot(x, y, 'o-', label=f'初始积分={ic:.0f}')
        ax1.fill_between(x, np.array(y) - np.array(ye), np.array(y) + np.array(ye), alpha=0.2)
    ax1.set_xlabel('社区储能初始 SOC')
    ax1.set_ylabel('总电网成本 (元/天)')
    ax1.legend(ncol=2, loc='upper left', fontsize=7)
    ax1.grid(True, alpha=0.3)
    _add_subplot_label(ax1, 'a', x_offset=-0.08)
    for soc in socs:
        sub = [r for r in rows if r['initial_soc'] == soc]
        sub = sorted(sub, key=lambda x: x['initial_credit'])
        x = [r['initial_credit'] for r in sub]
        y = [r['mean_total_cost'] for r in sub]
        ye = [r['std_total_cost'] for r in sub]
        ax2.plot(x, y, 'o-', label=f'初始SOC={soc}')
        ax2.fill_between(x, np.array(y) - np.array(ye), np.array(y) + np.array(ye), alpha=0.2)
    ax2.set_xlabel('各家庭初始积分')
    ax2.set_ylabel('总电网成本 (元/天)')
    ax2.legend(ncol=2, loc='upper left', fontsize=7)
    ax2.grid(True, alpha=0.3)
    _add_subplot_label(ax2, 'b', x_offset=-0.08)
    fig.tight_layout(w_pad=0.5)
    path = os.path.join(output_dir, 'sensitivity_cost_curves.png')
    fig.savefig(path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"已保存: {path}")


def plot_group_peak_curves(rows, output_dir):
    """组图：两张峰值曲线并排，(a) 峰值 vs SOC (b) 峰值 vs 初始积分，左上角标注 a、b。"""
    credits = get_unique_sorted(rows, 'initial_credit')
    socs = get_unique_sorted(rows, 'initial_soc')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for ic in credits:
        sub = [r for r in rows if r['initial_credit'] == ic]
        sub = sorted(sub, key=lambda x: x['initial_soc'])
        x = [r['initial_soc'] for r in sub]
        y = [r['mean_peak_load'] for r in sub]
        ye = [r['std_peak_load'] for r in sub]
        ax1.plot(x, y, 'o-', label=f'初始积分={ic:.0f}')
        ax1.fill_between(x, np.array(y) - np.array(ye), np.array(y) + np.array(ye), alpha=0.2)
    ax1.set_xlabel('社区储能初始 SOC')
    ax1.set_ylabel('社区净负荷峰值 (kW)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    _add_subplot_label(ax1, 'a')
    for soc in socs:
        sub = [r for r in rows if r['initial_soc'] == soc]
        sub = sorted(sub, key=lambda x: x['initial_credit'])
        x = [r['initial_credit'] for r in sub]
        y = [r['mean_peak_load'] for r in sub]
        ye = [r['std_peak_load'] for r in sub]
        ax2.plot(x, y, 'o-', label=f'初始SOC={soc}')
        ax2.fill_between(x, np.array(y) - np.array(ye), np.array(y) + np.array(ye), alpha=0.2)
    ax2.set_xlabel('各家庭初始积分')
    ax2.set_ylabel('社区净负荷峰值 (kW)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    _add_subplot_label(ax2, 'b')
    fig.tight_layout()
    path = os.path.join(output_dir, 'sensitivity_peak_curves.png')
    fig.savefig(path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"已保存: {path}")


def plot_agent_costs_bar(rows, output_dir):
    """柱状图：选定参数组合下三户平均成本对比（取若干代表性组合）。"""
    socs = get_unique_sorted(rows, 'initial_soc')
    credits = get_unique_sorted(rows, 'initial_credit')
    # 选 3 个组合：低/中/高 SOC 或 低/中/高 credit
    n_soc, n_cred = len(socs), len(credits)
    indices = [
        (0, 0), (n_soc // 2, n_cred // 2), (n_soc - 1, n_cred - 1)
    ]  # (soc_idx, cred_idx)
    combos = [(socs[i], credits[j]) for i, j in indices]
    fig, axes = plt.subplots(1, len(combos), figsize=(4 * len(combos), 4), sharey=True)
    if len(combos) == 1:
        axes = [axes]
    for idx, (soc, cred) in enumerate(combos):
        r = next((x for x in rows if x['initial_soc'] == soc and x['initial_credit'] == cred), None)
        if r is None:
            continue
        ax = axes[idx]
        x = ['家庭1', '家庭2', '家庭3']
        y = [r['mean_agent1_cost'], r['mean_agent2_cost'], r['mean_agent3_cost']]
        ax.bar(x, y, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
        ax.set_ylabel('平均电网成本 (元/天)')
        ax.set_title(f'初始SOC={soc}, 初始积分={cred:.0f}')
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    fig.tight_layout()
    path = os.path.join(output_dir, 'agent_costs_bar.png')
    fig.savefig(path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"已保存: {path}")


def main():
    parser = argparse.ArgumentParser(description='参数扫描结果绘图（峰值、成本、公平性、热力图）')
    parser.add_argument('--data_file', type=str, default='multi_agent/sweep_results/sweep_mappo.csv',
                        help='扫描结果 CSV 或 JSON（默认 sweep_mappo.csv）')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='图保存目录，默认与 data_file 同目录')
    parser.add_argument('--no_heatmap', action='store_true', help='不绘制热力图')
    parser.add_argument('--no_bar', action='store_true', help='不绘制各家庭成本柱状图')
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.data_file), 'figures')
    os.makedirs(args.output_dir, exist_ok=True)
    rows = load_data(args.data_file)
    if not rows:
        print("没有数据，退出")
        return
    print(f"已加载 {len(rows)} 个参数组合")
    plot_peak_vs_soc(rows, args.output_dir)
    plot_cost_vs_soc(rows, args.output_dir)
    plot_peak_vs_credit(rows, args.output_dir)
    plot_cost_vs_credit(rows, args.output_dir)
    plot_fairness_vs_soc(rows, args.output_dir)
    plot_fairness_vs_credit(rows, args.output_dir)
    if not args.no_heatmap:
        plot_heatmap_peak(rows, args.output_dir)
        plot_heatmap_cost(rows, args.output_dir)
        plot_group_heatmaps(rows, args.output_dir)
    plot_group_cost_curves(rows, args.output_dir)
    plot_group_peak_curves(rows, args.output_dir)
    if not args.no_bar:
        plot_agent_costs_bar(rows, args.output_dir)
    print("绘图完成。")


if __name__ == '__main__':
    main()
