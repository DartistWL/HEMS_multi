"""
社区峰值负荷对比图
Plot community peak load comparison between different methods
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import json
import argparse

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)


def load_data_from_json(data_file):
    """
    从JSON文件加载数据
    
    Args:
        data_file: JSON文件路径
    
    Returns:
        dict: 数据字典
    """
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"数据文件不存在: {data_file}")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def plot_peak_comparison(data, baseline_peak=None, save_path=None, show_plot=True):
    """
    绘制社区峰值负荷对比图
    
    Args:
        data: 数据字典，包含三种方法的数据（从comparison_data.json读取的格式）
            {
                'baseline_peak': float,
                'independent': {
                    'community_net_loads': [list of 48 values],
                    'peak_load': float,
                    ...
                },
                'rule_based': { ... },
                'mappo': { ... }
            }
        baseline_peak: 基准峰值（如果为None，从data中获取）
        save_path: 保存路径
        show_plot: 是否显示图表
    """
    # 从data中获取基准峰值
    if baseline_peak is None:
        baseline_peak = data.get('baseline_peak', None)
    
    # 提取数据（处理None情况）
    independent_data = data.get('independent', {})
    rule_based_data = data.get('rule_based', {})
    mappo_data = data.get('mappo', {})
    
    independent_loads = np.array(independent_data.get('community_net_loads', [])) if independent_data else np.array([])
    rule_based_loads = np.array(rule_based_data.get('community_net_loads', [])) if rule_based_data else np.array([])
    mappo_loads = np.array(mappo_data.get('community_net_loads', [])) if mappo_data else np.array([])
    
    # 获取峰值
    independent_peak = independent_data.get('peak_load', np.max(independent_loads) if len(independent_loads) > 0 else 0) if independent_data else 0
    rule_based_peak = rule_based_data.get('peak_load', np.max(rule_based_loads) if len(rule_based_loads) > 0 else 0) if rule_based_data else 0
    mappo_peak = mappo_data.get('peak_load', np.max(mappo_loads) if len(mappo_loads) > 0 else 0) if mappo_data else 0
    
    # 使用independent峰值作为基准峰值（如果没有提供）
    if baseline_peak is None:
        baseline_peak = independent_peak if independent_peak > 0 else 31.01
    
    # 时间步（转换为小时）
    time_steps = np.arange(len(independent_loads)) if len(independent_loads) > 0 else np.arange(48)
    hours = time_steps * 0.5  # 每个时间步30分钟
    
    # 创建图形（2个子图：时间序列对比 + 峰值柱状图）
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # ===== 子图1：时间序列对比 =====
    if len(independent_loads) > 0:
        ax1.plot(hours, independent_loads, label='独立学习基线', color='#FF6B6B', linewidth=2, alpha=0.8)
    if len(rule_based_loads) > 0:
        ax1.plot(hours, rule_based_loads, label='固定规则基线', color='#4ECDC4', linewidth=2, alpha=0.8)
    if len(mappo_loads) > 0:
        ax1.plot(hours, mappo_loads, label='MAPPO算法（我们的方法）', color='#45B7D1', linewidth=2.5, alpha=0.9)
    
    # 绘制基准峰值线
    ax1.axhline(y=baseline_peak, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label=f'基准峰值 ({baseline_peak:.2f} kW)')
    
    # 设置标签和标题
    ax1.set_xlabel('时间 (小时)', fontsize=12)
    ax1.set_ylabel('社区净负荷 (kW)', fontsize=12)
    ax1.set_title('社区峰值负荷对比图（24小时）', fontsize=14, pad=20)
    ax1.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0, 24)
    
    # 设置x轴刻度
    ax1.set_xticks(range(0, 25, 2))
    
    # ===== 子图2：峰值柱状图 =====
    methods = ['独立学习\n基线', '固定规则\n基线', 'MAPPO\n算法']
    peaks = [independent_peak, rule_based_peak, mappo_peak]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    bars = ax2.bar(methods, peaks, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # 绘制基准峰值线
    ax2.axhline(y=baseline_peak, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label=f'基准峰值 ({baseline_peak:.2f} kW)')
    
    # 在柱状图上标注数值和降低率
    for i, (bar, peak) in enumerate(zip(bars, peaks)):
        height = bar.get_height()
        # 标注峰值
        ax2.text(bar.get_x() + bar.get_width()/2., height + baseline_peak * 0.02,
                f'{peak:.2f} kW',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 标注峰值降低率（相对于基准峰值）
        if i > 0:  # 不对基准方法计算降低率
            reduction_rate = (baseline_peak - peak) / baseline_peak * 100
            ax2.text(bar.get_x() + bar.get_width()/2., height - baseline_peak * 0.05,
                    f'降低 {reduction_rate:.1f}%',
                    ha='center', va='top', fontsize=10, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # 设置标签和标题
    ax2.set_ylabel('峰值负荷 (kW)', fontsize=12)
    ax2.set_title('峰值负荷对比', fontsize=14, pad=20)
    ax2.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax2.set_ylim(0, max(peaks) * 1.15)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"峰值对比图已保存到: {save_path}")
    
    # 显示图表
    if show_plot:
        plt.show()
    else:
        plt.close()


def generate_summary_table(data, baseline_peak=None, save_path=None):
    """
    生成峰值降低效果汇总表
    
    Args:
        data: 数据字典（从comparison_data.json读取的格式）
        baseline_peak: 基准峰值（如果为None，从data中获取）
        save_path: 保存路径（可选，如果提供则保存为文本文件）
    
    Returns:
        str: 汇总表文本
    """
    # 从data中获取基准峰值
    if baseline_peak is None:
        baseline_peak = data.get('baseline_peak', None)
    
    # 提取数据
    independent_data = data.get('independent', {})
    rule_based_data = data.get('rule_based', {})
    mappo_data = data.get('mappo', {})
    
    independent_loads = np.array(independent_data.get('community_net_loads', []))
    rule_based_loads = np.array(rule_based_data.get('community_net_loads', []))
    mappo_loads = np.array(mappo_data.get('community_net_loads', []))
    
    # 计算统计指标
    def calculate_stats(loads, peak):
        if len(loads) > 0:
            return {
                'peak': peak,
                'avg': np.mean(loads),
                'std': np.std(loads),
                'min': np.min(loads),
                'max': peak
            }
        return {'peak': peak, 'avg': 0, 'std': 0, 'min': 0, 'max': peak}
    
    independent_stats = calculate_stats(independent_loads, independent_data.get('peak_load', 0))
    rule_based_stats = calculate_stats(rule_based_loads, rule_based_data.get('peak_load', 0))
    mappo_stats = calculate_stats(mappo_loads, mappo_data.get('peak_load', 0))
    
    if baseline_peak is None:
        baseline_peak = independent_stats['peak']
    
    # 计算峰值降低率
    def reduction_rate(peak):
        return (baseline_peak - peak) / baseline_peak * 100 if baseline_peak > 0 else 0
    
    rule_based_reduction = reduction_rate(rule_based_stats['peak'])
    mappo_reduction = reduction_rate(mappo_stats['peak'])
    
    # 获取成本和回报（如果有）
    independent_cost = independent_data.get('total_cost', 0)
    rule_based_cost = rule_based_data.get('total_cost', 0)
    mappo_cost = mappo_data.get('total_cost', 0)
    
    independent_return = independent_data.get('total_return', 0)
    rule_based_return = rule_based_data.get('total_return', 0)
    mappo_return = mappo_data.get('total_return', 0)
    
    # 生成表格文本
    table_text = "=" * 100 + "\n"
    table_text += "峰值降低效果汇总表\n"
    table_text += "=" * 100 + "\n\n"
    table_text += f"{'指标':<20} {'独立学习基线':<20} {'固定规则基线':<20} {'MAPPO算法（我们的）':<20}\n"
    table_text += "-" * 100 + "\n"
    table_text += f"{'峰值负荷 (kW)':<20} {independent_stats['peak']:<20.2f} {rule_based_stats['peak']:<20.2f} {mappo_stats['peak']:<20.2f}\n"
    table_text += f"{'峰值降低率 (%)':<20} {'0.00 (基准)':<20} {rule_based_reduction:<20.2f} {mappo_reduction:<20.2f}\n"
    table_text += f"{'平均负荷 (kW)':<20} {independent_stats['avg']:<20.2f} {rule_based_stats['avg']:<20.2f} {mappo_stats['avg']:<20.2f}\n"
    table_text += f"{'负荷标准差 (kW)':<20} {independent_stats['std']:<20.2f} {rule_based_stats['std']:<20.2f} {mappo_stats['std']:<20.2f}\n"
    table_text += f"{'最小负荷 (kW)':<20} {independent_stats['min']:<20.2f} {rule_based_stats['min']:<20.2f} {mappo_stats['min']:<20.2f}\n"
    if independent_cost > 0:
        table_text += f"{'总成本 (元)':<20} {independent_cost:<20.2f} {rule_based_cost:<20.2f} {mappo_cost:<20.2f}\n"
    if independent_return != 0:
        table_text += f"{'总回报':<20} {independent_return:<20.2f} {rule_based_return:<20.2f} {mappo_return:<20.2f}\n"
    table_text += "=" * 100 + "\n"
    
    print(table_text)
    
    # 保存到文件
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(table_text)
        print(f"汇总表已保存到: {save_path}")
    
    return table_text


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='绘制社区峰值负荷对比图')
    parser.add_argument('--data_file', type=str, default=None,
                       help='JSON数据文件路径（如果提供，从文件加载数据）')
    parser.add_argument('--baseline_peak', type=float, default=None,
                       help='基准峰值（如果为None，使用independent的峰值）')
    parser.add_argument('--output_dir', type=str, default='multi_agent/visualization/output',
                       help='输出目录 (default: multi_agent/visualization/output)')
    parser.add_argument('--show_plot', action='store_true',
                       help='显示图表 (默认: False, 只保存不显示)')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 必须提供数据文件
    if args.data_file is None:
        print("=" * 80)
        print("错误：必须提供数据文件")
        print("=" * 80)
        print("使用方法:")
        print("  python plot_peak_comparison.py --data_file path/to/comparison_data.json")
        print("=" * 80)
        print("数据文件可以通过运行 collect_visualization_data.py 生成")
        print("=" * 80)
        return
    
    data = load_data_from_json(args.data_file)
    
    # 绘制对比图
    output_path = os.path.join(args.output_dir, 'peak_comparison.png')
    plot_peak_comparison(data, baseline_peak=args.baseline_peak, 
                        save_path=output_path, show_plot=args.show_plot)
    
    # 生成汇总表
    table_path = os.path.join(args.output_dir, 'peak_comparison_summary.txt')
    generate_summary_table(data, baseline_peak=args.baseline_peak, save_path=table_path)
    
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
