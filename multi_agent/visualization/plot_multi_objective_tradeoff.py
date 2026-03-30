"""
多目标权衡雷达图
Plot multi-objective tradeoff radar chart
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


def normalize_value(value, min_val, max_val, reverse=False):
    """
    归一化值到0-1范围
    
    Args:
        value: 原始值
        min_val: 最小值
        max_val: 最大值
        reverse: 是否反向（值越大越好时reverse=False，值越小越好时reverse=True）
    
    Returns:
        归一化后的值（0-1）
    """
    if max_val == min_val:
        return 0.5
    
    normalized = (value - min_val) / (max_val - min_val)
    
    if reverse:
        normalized = 1.0 - normalized
    
    return np.clip(normalized, 0.0, 1.0)


def plot_multi_objective_tradeoff(data, baseline_peak=None, save_path=None, show_plot=True):
    """
    绘制多目标权衡雷达图
    
    Args:
        data: 数据字典（从comparison_data.json读取）
        baseline_peak: 基准峰值（如果为None，从data中获取）
        save_path: 保存路径
        show_plot: 是否显示图表
    """
    # 从data中获取基准峰值
    if baseline_peak is None:
        baseline_peak = data.get('baseline_peak', None)
    
    # 提取数据
    independent_data = data.get('independent', {})
    rule_based_data = data.get('rule_based', {})
    mappo_data = data.get('mappo', {})
    
    # 计算各个指标
    methods_data = {
        '独立学习基线': independent_data,
        '固定规则基线': rule_based_data,
        'MAPPO算法': mappo_data
    }
    
    # 收集所有方法的指标值（用于归一化）
    all_peaks = []
    all_avg_loads = []
    all_std_loads = []
    all_costs = []
    all_returns = []
    
    for method_name, method_data in methods_data.items():
        if method_data:
            all_peaks.append(method_data.get('peak_load', 0))
            all_avg_loads.append(method_data.get('avg_load', 0))
            all_std_loads.append(method_data.get('std_load', 0))
            all_costs.append(method_data.get('total_cost', 0))
            all_returns.append(method_data.get('total_return', 0))
    
    if not all_peaks:
        raise ValueError("没有可用的数据")
    
    # 计算归一化范围
    min_peak, max_peak = min(all_peaks), max(all_peaks)
    min_avg, max_avg = min(all_avg_loads), max(all_avg_loads)
    min_std, max_std = min(all_std_loads), max(all_std_loads)
    min_cost, max_cost = min(all_costs), max(all_costs)
    min_return, max_return = min(all_returns), max(all_returns)
    
    # 计算峰值降低率（相对于基准峰值）
    if baseline_peak is None:
        baseline_peak = max_peak
    
    # 准备雷达图数据
    categories = ['峰值降低率', '成本降低率', '负荷平滑度', '回报提升率', '储能利用率']
    num_vars = len(categories)
    
    # 计算每个方法的归一化值
    method_values = {}
    for method_name, method_data in methods_data.items():
        if not method_data:
            continue
        
        peak = method_data.get('peak_load', 0)
        avg_load = method_data.get('avg_load', 0)
        std_load = method_data.get('std_load', 0)
        cost = method_data.get('total_cost', 0)
        return_val = method_data.get('total_return', 0)
        
        # 计算归一化指标（0-1范围，值越大越好）
        peak_reduction_rate = (baseline_peak - peak) / baseline_peak if baseline_peak > 0 else 0
        peak_reduction_rate = np.clip(peak_reduction_rate, 0, 1)  # 峰值降低率（越高越好）
        
        cost_reduction_rate = normalize_value(cost, min_cost, max_cost, reverse=True)  # 成本越低越好
        
        load_smoothness = normalize_value(std_load, min_std, max_std, reverse=True)  # 标准差越小越好
        
        return_improvement = normalize_value(return_val, min_return, max_return, reverse=False)  # 回报越高越好
        
        # 储能利用率（如果有数据，否则使用默认值）
        # 这里简化处理，使用SOC变化范围作为利用率指标
        ess_soc = method_data.get('community_ess_soc', [])
        if len(ess_soc) > 0:
            soc_range = np.max(ess_soc) - np.min(ess_soc)
            ess_utilization = soc_range  # SOC变化范围（0-1）
        else:
            ess_utilization = 0.5  # 默认值
        
        method_values[method_name] = [
            peak_reduction_rate,
            cost_reduction_rate,
            load_smoothness,
            return_improvement,
            ess_utilization
        ]
    
    # 创建雷达图
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
    
    # 计算角度
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    # 绘制每个方法
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    linestyles = ['-', '--', '-.']
    
    for idx, (method_name, values) in enumerate(method_values.items()):
        values += values[:1]  # 闭合
        ax.plot(angles, values, 'o-', linewidth=2.5, label=method_name,
               color=colors[idx % len(colors)], linestyle=linestyles[idx % len(linestyles)],
               markersize=8, alpha=0.8)
        ax.fill(angles, values, alpha=0.15, color=colors[idx % len(colors)])
    
    # 设置类别标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    
    # 设置Y轴（0-1范围）
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 设置标题
    ax.set_title('多目标权衡雷达图\n（值越大越好，归一化到0-1范围）', 
                fontsize=14, pad=30, weight='bold')
    
    # 添加图例
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"多目标权衡雷达图已保存到: {save_path}")
    
    # 显示图表
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='绘制多目标权衡雷达图')
    parser.add_argument('--data_file', type=str, required=True,
                       help='JSON数据文件路径（comparison_data.json）')
    parser.add_argument('--baseline_peak', type=float, default=None,
                       help='基准峰值（如果为None，从data中获取）')
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
    output_path = os.path.join(args.output_dir, 'multi_objective_tradeoff.png')
    plot_multi_objective_tradeoff(data, baseline_peak=args.baseline_peak,
                                  save_path=output_path, show_plot=args.show_plot)
    
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
