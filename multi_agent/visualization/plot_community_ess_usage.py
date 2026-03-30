"""
社区储能使用可视化
Plot community ESS usage (SOC and charge/discharge power)
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


def plot_community_ess_usage(data, method='mappo', save_path=None, show_plot=True):
    """
    绘制社区储能使用可视化图
    
    Args:
        data: 数据字典（从comparison_data.json或单个方法的数据文件读取）
        method: 要绘制的方法（'independent', 'rule_based', 'mappo'）
        save_path: 保存路径
        show_plot: 是否显示图表
    """
    # 提取数据
    method_data = data.get(method, {})
    if not method_data:
        raise ValueError(f"数据中不包含方法 '{method}' 的数据")
    
    community_ess_soc = np.array(method_data.get('community_ess_soc', []))
    community_ess_charge_power = np.array(method_data.get('community_ess_charge_power', []))
    community_ess_discharge_power = np.array(method_data.get('community_ess_discharge_power', []))
    
    if len(community_ess_soc) == 0:
        raise ValueError(f"方法 '{method}' 的数据中没有社区储能SOC数据")
    
    # 时间步（转换为小时）
    time_steps = np.arange(len(community_ess_soc))
    hours = time_steps * 0.5
    
    # 创建图形（双Y轴）
    fig, ax1 = plt.subplots(figsize=(16, 8))
    
    # 左Y轴：SOC（0-1）
    color_soc = '#4169E1'
    ax1.set_xlabel('时间 (小时)', fontsize=12)
    ax1.set_ylabel('社区储能SOC', fontsize=12, color=color_soc)
    line1 = ax1.plot(hours, community_ess_soc, color=color_soc, linewidth=2.5, 
                     label='社区储能SOC', marker='o', markersize=3, alpha=0.8)
    ax1.tick_params(axis='y', labelcolor=color_soc)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 右Y轴：充放电功率
    ax2 = ax1.twinx()
    color_charge = '#32CD32'  # 绿色（充电）
    color_discharge = '#FF6347'  # 红色（放电）
    
    # 绘制充电功率（正值向上）
    ax2.fill_between(hours, 0, community_ess_charge_power, 
                     color=color_charge, alpha=0.5, label='充电功率')
    ax2.plot(hours, community_ess_charge_power, color=color_charge, 
             linewidth=1.5, linestyle='--', alpha=0.7)
    
    # 绘制放电功率（负值向下）
    ax2.fill_between(hours, 0, -community_ess_discharge_power, 
                     color=color_discharge, alpha=0.5, label='放电功率')
    ax2.plot(hours, -community_ess_discharge_power, color=color_discharge, 
             linewidth=1.5, linestyle='--', alpha=0.7)
    
    ax2.set_ylabel('功率 (kW)', fontsize=12, color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    
    # 标注关键点
    if len(community_ess_soc) > 0:
        min_soc_idx = np.argmin(community_ess_soc)
        max_soc_idx = np.argmax(community_ess_soc)
        
        ax1.annotate(f'最低SOC: {community_ess_soc[min_soc_idx]:.2%}\n时间: {hours[min_soc_idx]:.1f}h',
                    xy=(hours[min_soc_idx], community_ess_soc[min_soc_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax1.annotate(f'最高SOC: {community_ess_soc[max_soc_idx]:.2%}\n时间: {hours[max_soc_idx]:.1f}h',
                    xy=(hours[max_soc_idx], community_ess_soc[max_soc_idx]),
                    xytext=(10, -30), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # 设置标题
    method_names = {
        'independent': '独立学习基线',
        'rule_based': '固定规则基线',
        'mappo': 'MAPPO算法'
    }
    ax1.set_title(f'{method_names.get(method, method)} - 社区储能使用情况（24小时）', 
                  fontsize=14, pad=20)
    
    # 设置x轴
    ax1.set_xlim(0, 24)
    ax1.set_xticks(range(0, 25, 2))
    
    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10, framealpha=0.9)
    
    # 添加0轴参考线
    ax2.axhline(y=0, color='black', linewidth=0.8, linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"社区储能使用图已保存到: {save_path}")
    
    # 显示图表
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='绘制社区储能使用可视化图')
    parser.add_argument('--data_file', type=str, required=True,
                       help='JSON数据文件路径（comparison_data.json或单个方法的数据文件）')
    parser.add_argument('--method', type=str, default='mappo', 
                       choices=['independent', 'rule_based', 'mappo'],
                       help='要绘制的方法 (default: mappo)')
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
    output_path = os.path.join(args.output_dir, f'community_ess_usage_{args.method}.png')
    plot_community_ess_usage(data, method=args.method, 
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
