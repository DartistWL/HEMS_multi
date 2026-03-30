"""
积分余额变化曲线
Plot credit balance changes over time
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


def plot_credit_balance(data, method='mappo', save_path=None, show_plot=True):
    """
    绘制积分余额变化曲线
    
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
    
    agent_credit_balances = method_data.get('agent_credit_balances', [[], [], []])
    
    if len(agent_credit_balances) < 3 or len(agent_credit_balances[0]) == 0:
        raise ValueError(f"方法 '{method}' 的数据中没有积分余额数据")
    
    # 时间步（转换为小时）
    time_steps = np.arange(len(agent_credit_balances[0]))
    hours = time_steps * 0.5
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # 绘制三个家庭的积分余额曲线
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    labels = ['家庭 1', '家庭 2', '家庭 3']
    
    for i in range(3):
        if i < len(agent_credit_balances) and len(agent_credit_balances[i]) > 0:
            balances = np.array(agent_credit_balances[i])
            ax.plot(hours, balances, color=colors[i], linewidth=2.5, 
                   label=labels[i], marker='o', markersize=3, alpha=0.8)
    
    # 绘制初始积分线（100）
    initial_credit = 100.0
    ax.axhline(y=initial_credit, color='gray', linestyle='--', linewidth=1.5, 
              alpha=0.7, label=f'初始积分 ({initial_credit:.0f})')
    
    # 标注最终积分
    for i in range(3):
        if i < len(agent_credit_balances) and len(agent_credit_balances[i]) > 0:
            final_balance = agent_credit_balances[i][-1]
            change = final_balance - initial_credit
            ax.annotate(f'家庭 {i+1}\n最终: {final_balance:.2f}\n变化: {change:+.2f}',
                       xy=(hours[-1], final_balance),
                       xytext=(10, 10 if i == 0 else (-10 if i == 1 else 0)), 
                       textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[i], alpha=0.3),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # 设置标签和标题
    ax.set_xlabel('时间 (小时)', fontsize=12)
    ax.set_ylabel('积分余额', fontsize=12)
    
    method_names = {
        'independent': '独立学习基线',
        'rule_based': '固定规则基线',
        'mappo': 'MAPPO算法'
    }
    ax.set_title(f'{method_names.get(method, method)} - 积分余额变化（24小时）', 
                 fontsize=14, pad=20)
    
    # 设置x轴
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 添加图例
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"积分余额变化图已保存到: {save_path}")
    
    # 显示图表
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='绘制积分余额变化曲线')
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
    output_path = os.path.join(args.output_dir, f'credit_balance_{args.method}.png')
    plot_credit_balance(data, method=args.method, 
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
