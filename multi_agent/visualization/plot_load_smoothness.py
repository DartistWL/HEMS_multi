"""
负荷平滑度对比
Plot load smoothness comparison
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)


def load_data_from_json(data_file):
    """从JSON文件加载数据"""
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"数据文件不存在: {data_file}")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def plot_load_smoothness(data, save_path=None, show_plot=True):
    """
    绘制负荷平滑度对比图
    
    Args:
        data: 数据字典（从comparison_data.json读取）
        save_path: 保存路径
        show_plot: 是否显示图表
    """
    # 提取数据
    independent_data = data.get('independent', {})
    rule_based_data = data.get('rule_based', {})
    mappo_data = data.get('mappo', {})
    
    # 准备数据
    methods_data = []
    labels = []
    
    if independent_data and independent_data.get('community_net_loads'):
        methods_data.append(np.array(independent_data['community_net_loads']))
        labels.append('独立学习基线')
    
    if rule_based_data and rule_based_data.get('community_net_loads'):
        methods_data.append(np.array(rule_based_data['community_net_loads']))
        labels.append('固定规则基线')
    
    if mappo_data and mappo_data.get('community_net_loads'):
        methods_data.append(np.array(mappo_data['community_net_loads']))
        labels.append('MAPPO算法')
    
    if not methods_data:
        raise ValueError("没有可用的负荷数据")
    
    # 创建图形（2个子图：箱线图 + 标准差对比）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ===== 子图1：箱线图 =====
    bp = ax1.boxplot(methods_data, labels=labels, patch_artist=True, 
                     showmeans=True, meanline=True)
    
    # 设置箱线图颜色
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_ylabel('社区净负荷 (kW)', fontsize=12)
    ax1.set_title('负荷分布对比（箱线图）', fontsize=14, pad=20)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # 添加统计信息标注
    for i, (loads, label) in enumerate(zip(methods_data, labels)):
        mean_val = np.mean(loads)
        median_val = np.median(loads)
        std_val = np.std(loads)
        ax1.text(i+1, mean_val + std_val * 0.5, 
                f'均值: {mean_val:.2f}\n中位数: {median_val:.2f}\n标准差: {std_val:.2f}',
                ha='center', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # ===== 子图2：标准差对比柱状图 =====
    std_values = [np.std(loads) for loads in methods_data]
    bars = ax2.bar(labels, std_values, color=colors[:len(labels)], alpha=0.7, 
                   edgecolor='black', linewidth=1.5)
    
    # 在柱状图上标注数值
    for bar, std_val in zip(bars, std_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(std_values) * 0.02,
                f'{std_val:.2f} kW',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_ylabel('负荷标准差 (kW)', fontsize=12)
    ax2.set_title('负荷平滑度对比（标准差越小越平滑）', fontsize=14, pad=20)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # 添加平滑度排名
    sorted_indices = np.argsort(std_values)
    for rank, idx in enumerate(sorted_indices):
        method_name = labels[idx]
        std_val = std_values[idx]
        improvement = (std_values[0] - std_val) / std_values[0] * 100 if std_values[0] > 0 else 0
        ax2.text(idx, std_val - max(std_values) * 0.05,
                f'排名 {rank+1}\n改进 {improvement:+.1f}%',
                ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"负荷平滑度对比图已保存到: {save_path}")
    
    # 显示图表
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='绘制负荷平滑度对比图')
    parser.add_argument('--data_file', type=str, required=True,
                       help='JSON数据文件路径（comparison_data.json）')
    parser.add_argument('--output_dir', type=str, default='multi_agent/visualization/output',
                       help='输出目录 (default: multi_agent/visualization/output)')
    parser.add_argument('--show_plot', action='store_true',
                       help='显示图表 (默认: False, 只保存不显示)')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    data = load_data_from_json(args.data_file)
    
    # 绘制图表
    output_path = os.path.join(args.output_dir, 'load_smoothness_comparison.png')
    plot_load_smoothness(data, save_path=output_path, show_plot=args.show_plot)
    
    print("\n" + "=" * 80)
    print("完成！")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断操作。")
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()
