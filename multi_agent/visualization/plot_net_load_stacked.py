"""
净负荷堆叠图可视化
绘制每个家庭的能量堆叠图，显示不同设备的负荷
Plot net load stacked charts for each household
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import json
import argparse

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目根目录到路径
# 脚本位于 HEMS/multi_agent/visualization/plot_net_load_stacked.py
# 向上三级得到项目根目录 HEMS
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)


def load_data_from_json(data_file):
    """从JSON文件加载数据"""
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"数据文件不存在: {data_file}")

    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def _get_episode_data_for_plot(data, date_type=None):
    """
    根据 date_type 从数据中取用于绘图的单条 episode 数据。
    date_type: None | 'weekday' | 'weekend'
    """
    summary = data.get('summary', {})
    episodes = data.get('episodes', [])
    if date_type == 'weekday':
        summary_typed = data.get('summary_weekday', {})
        if summary_typed and summary_typed.get('agent_net_loads'):
            return summary_typed
        typed_eps = [e for e in episodes if e.get('date_type') == '工作日']
        return typed_eps[0] if typed_eps else (summary_typed or summary)
    if date_type == 'weekend':
        summary_typed = data.get('summary_weekend', {})
        if summary_typed and summary_typed.get('agent_net_loads'):
            return summary_typed
        typed_eps = [e for e in episodes if e.get('date_type') == '双休日']
        return typed_eps[0] if typed_eps else (summary_typed or summary)
    # date_type is None: 保持原逻辑
    if not episodes:
        return summary
    return episodes[0]


def plot_net_load_stacked(independent_data_file=None, rule_based_data_file=None,
                          mappo_data_file=None, save_path=None, show_plot=True, date_type=None):
    """
    绘制三个模型的净负荷堆叠对比图。
    若数据中含 summary_weekday/summary_weekend 或 episodes 带 date_type，可通过 date_type 分别绘制工作日/双休日。

    Args:
        independent_data_file: 独立训练模型数据文件路径
        rule_based_data_file: 规则基线模型数据文件路径
        mappo_data_file: MAPPO模型数据文件路径
        save_path: 保存路径
        show_plot: 是否显示图表
        date_type: None | 'weekday' | 'weekend'，为 None 时使用整体汇总或第一个 episode；为 'weekday'/'weekend' 时使用对应类型数据
    """
    # 加载数据
    data_dict = {}
    if independent_data_file and os.path.exists(independent_data_file):
        data_dict['independent'] = load_data_from_json(independent_data_file)
    if rule_based_data_file and os.path.exists(rule_based_data_file):
        data_dict['rule_based'] = load_data_from_json(rule_based_data_file)
    if mappo_data_file and os.path.exists(mappo_data_file):
        data_dict['mappo'] = load_data_from_json(mappo_data_file)

    if not data_dict:
        raise ValueError("至少需要提供一个数据文件")

    # 创建图形：3行（每个模型一行），4列（3个家庭 + 1个社区总负荷）
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # 模型名称映射
    model_names = {
        'independent': '独立训练基线',
        'rule_based': '固定规则基线',
        'mappo': 'MAPPO算法'
    }

    # 颜色定义
    colors = {
        'home_load': '#FF6B6B',           # 家庭基础负荷 - 红色
        'ac': '#FFA500',                  # 空调负荷 - 橙色
        'ewh': '#FFD700',                 # 热水器 - 金色
        'wash_machine': '#FF69B4',        # 洗衣机 - 粉色
        'ess_charge': '#32CD32',          # ESS充电（从电网） - 绿色
        'ev_charge': '#00CED1',           # EV充电（从电网） - 深青色
        'community_charge': '#9370DB',    # 社区储能充电 - 紫色
        'pv_direct': '#90EE90',           # PV直接使用 - 浅绿色
        'ess_discharge': '#00FF00',       # ESS放电 - 亮绿色（负值）
        'ev_discharge': '#00FFFF',        # EV放电 - 青色（负值）
        'community_discharge': '#BA55D3', # 社区储能放电 - 中紫色（负值）
        'pv_sell': '#98FB98'              # PV售给电网 - 淡绿色（负值）
    }

    # 统一图例：所有子图使用相同的图例项（不论该子图是否绘制了对应曲线）
    legend_stack_labels = ['家庭负荷', '空调', '热水器', '洗衣机', 'ESS充电', 'EV充电', '社区储能充电',
                           'PV售电', 'ESS放电', 'EV放电', '社区储能放电']
    legend_stack_colors = [colors['home_load'], colors['ac'], colors['ewh'], colors['wash_machine'],
                           colors['ess_charge'], colors['ev_charge'], colors['community_charge'],
                           colors['pv_sell'], colors['ess_discharge'], colors['ev_discharge'], colors['community_discharge']]
    legend_handles_family = [mpatches.Patch(facecolor=c, alpha=0.7, label=l) for l, c in zip(legend_stack_labels, legend_stack_colors)]
    legend_handles_family.append(Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='净负荷'))
    legend_handles_community = [mpatches.Patch(facecolor=c, alpha=0.7, label=l) for l, c in zip(legend_stack_labels, legend_stack_colors)]
    legend_handles_community.append(Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='社区净负荷'))

    # 处理每个模型
    row = 0
    for method in ['independent', 'rule_based', 'mappo']:
        if method not in data_dict:
            # 如果某个模型没有数据，跳过
            continue

        data = data_dict[method]
        # 按 date_type 选择用于绘图的单条数据
        episode_data = _get_episode_data_for_plot(data, date_type)

        # 时间步（转换为小时）
        num_steps = len(episode_data.get('agent_net_loads', [[], [], []])[0]) if episode_data.get('agent_net_loads') else 48
        time_steps = np.arange(num_steps)
        hours = time_steps * 0.5

        # 获取每个家庭的净负荷
        agent_net_loads = episode_data.get('agent_net_loads', [[], [], []])
        community_net_loads = episode_data.get('community_net_loads', [])

        # 获取社区储能充放电功率
        community_ess_charge_power = episode_data.get('community_ess_charge_power', [0.0] * num_steps)
        community_ess_discharge_power = episode_data.get('community_ess_discharge_power', [0.0] * num_steps)

        # 绘制三个家庭的堆叠图
        for agent_idx in range(3):
            ax = fig.add_subplot(gs[row, agent_idx])

            if agent_idx < len(agent_net_loads) and len(agent_net_loads[agent_idx]) > 0:
                # 获取该家庭的详细设备功率数据
                home_loads = np.array(episode_data.get('agent_home_loads', [[], [], []])[agent_idx] if episode_data.get('agent_home_loads') else [0.0] * num_steps)
                ac_loads = np.array(episode_data.get('agent_ac_loads', [[], [], []])[agent_idx] if episode_data.get('agent_ac_loads') else [0.0] * num_steps)
                ewh_loads = np.array(episode_data.get('agent_ewh_loads', [[], [], []])[agent_idx] if episode_data.get('agent_ewh_loads') else [0.0] * num_steps)
                wash_loads = np.array(episode_data.get('agent_wash_loads', [[], [], []])[agent_idx] if episode_data.get('agent_wash_loads') else [0.0] * num_steps)
                pv_gen = np.array(episode_data.get('agent_pv_generation', [[], [], []])[agent_idx] if episode_data.get('agent_pv_generation') else [0.0] * num_steps)
                ess_charge = np.array(episode_data.get('agent_ess_charge', [[], [], []])[agent_idx] if episode_data.get('agent_ess_charge') else [0.0] * num_steps)
                ess_discharge = np.array(episode_data.get('agent_ess_discharge', [[], [], []])[agent_idx] if episode_data.get('agent_ess_discharge') else [0.0] * num_steps)
                ev_charge = np.array(episode_data.get('agent_ev_charge', [[], [], []])[agent_idx] if episode_data.get('agent_ev_charge') else [0.0] * num_steps)
                ev_discharge = np.array(episode_data.get('agent_ev_discharge', [[], [], []])[agent_idx] if episode_data.get('agent_ev_discharge') else [0.0] * num_steps)
                community_charge = np.array(episode_data.get('agent_community_charge', [[], [], []])[agent_idx] if episode_data.get('agent_community_charge') else [0.0] * num_steps)
                community_discharge = np.array(episode_data.get('agent_community_discharge', [[], [], []])[agent_idx] if episode_data.get('agent_community_discharge') else [0.0] * num_steps)

                # 获取PV流向信息（用于计算PV售给电网部分）
                pv_flows = episode_data.get('agent_pv_flows', [[], [], []])[agent_idx] if episode_data.get('agent_pv_flows') else []
                pv_sell = np.array([pv_flow.get('grid_sell', 0.0) if isinstance(pv_flow, dict) else 0.0 for pv_flow in pv_flows] if len(pv_flows) == num_steps else [0.0] * num_steps)

                # 计算从电网充电的部分（减去PV用于充电的部分）
                # 从PV流向中获取用于充电的PV
                pv_for_ess = np.array([pv_flow.get('ess_charge', 0.0) if isinstance(pv_flow, dict) else 0.0 for pv_flow in pv_flows] if len(pv_flows) == num_steps else [0.0] * num_steps)
                pv_for_ev = np.array([pv_flow.get('ev_charge', 0.0) if isinstance(pv_flow, dict) else 0.0 for pv_flow in pv_flows] if len(pv_flows) == num_steps else [0.0] * num_steps)
                pv_for_community = np.array([pv_flow.get('community_charge', 0.0) if isinstance(pv_flow, dict) else 0.0 for pv_flow in pv_flows] if len(pv_flows) == num_steps else [0.0] * num_steps)

                # 从电网充电的部分 = 总充电需求 - PV用于充电的部分
                grid_ess_charge = np.maximum(0, ess_charge - pv_for_ess)
                grid_ev_charge = np.maximum(0, ev_charge - pv_for_ev)
                grid_community_charge = np.maximum(0, community_charge - pv_for_community)

                # 构建堆叠数据（正方向：从电网购电）
                positive_stack = [
                    home_loads,  # 家庭基础负荷
                    ac_loads,    # 空调负荷
                    ewh_loads,   # 热水器负荷
                    wash_loads,  # 洗衣机负荷
                    grid_ess_charge,  # ESS充电（从电网）
                    grid_ev_charge,   # EV充电（从电网）
                    grid_community_charge  # 社区储能充电（从电网）
                ]
                positive_labels = ['家庭负荷', '空调', '热水器', '洗衣机', 'ESS充电', 'EV充电', '社区储能充电']
                positive_colors = [colors['home_load'], colors['ac'], colors['ewh'],
                                  colors['wash_machine'], colors['ess_charge'],
                                  colors['ev_charge'], colors['community_charge']]

                # 构建堆叠数据（负方向：向电网售电）
                negative_stack = [
                    -pv_sell,           # PV售给电网（负值）
                    -ess_discharge,     # ESS放电给电网（负值）
                    -ev_discharge,      # EV放电给电网（负值）
                    -community_discharge  # 社区储能放电给电网（负值）
                ]
                negative_labels = ['PV售电', 'ESS放电', 'EV放电', '社区储能放电']
                negative_colors = [colors['pv_sell'], colors['ess_discharge'],
                                  colors['ev_discharge'], colors['community_discharge']]

                # 绘制堆叠图（正方向，不传 labels，使用统一图例）
                if any(np.any(stack > 0) for stack in positive_stack):
                    ax.stackplot(hours, *positive_stack, colors=positive_colors, alpha=0.7)

                # 绘制堆叠图（负方向）
                if any(np.any(stack < 0) for stack in negative_stack):
                    ax.stackplot(hours, *negative_stack, colors=negative_colors, alpha=0.7)

                # 绘制净负荷曲线（用于验证）
                net_loads = np.array(agent_net_loads[agent_idx])
                ax.plot(hours, net_loads, color='black', linewidth=2, linestyle='--', alpha=0.8)

                # 绘制零线
                ax.axhline(y=0, color='black', linewidth=1, alpha=0.3)

                ax.set_title(f'{model_names[method]} - 家庭 {agent_idx+1}', fontsize=11, pad=10)
                ax.set_xlabel('时间 (小时)', fontsize=12)
                ax.set_ylabel('功率 (kW)', fontsize=12)
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.legend(handles=legend_handles_family, loc='lower left', fontsize=7, ncol=2)
                ax.set_xlim(0, 24)

                # 设置y轴范围
                y_max = max(np.abs(net_loads)) * 1.2 if len(net_loads) > 0 else 5
                ax.set_ylim(-y_max, y_max)
            else:
                # 无数据时仍显示统一图例与坐标
                ax.set_title(f'{model_names[method]} - 家庭 {agent_idx+1}', fontsize=11, pad=10)
                ax.set_xlabel('时间 (小时)', fontsize=12)
                ax.set_ylabel('功率 (kW)', fontsize=12)
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.legend(handles=legend_handles_family, loc='lower left', fontsize=7, ncol=2)
                ax.set_xlim(0, 24)
                ax.set_ylim(-5, 5)

        # 绘制社区总负荷（第四列）- 社区储能视为内部，只显示与电网的交互
        ax_community = fig.add_subplot(gs[row, 3])

        if len(community_net_loads) > 0:
            community_loads = np.array(community_net_loads)

            # 汇总三个家庭的设备功率（用于社区堆叠图）
            # 社区视角：只考虑与电网的交互，社区储能是内部的
            total_home_loads = np.zeros(num_steps)
            total_ac_loads = np.zeros(num_steps)
            total_ewh_loads = np.zeros(num_steps)
            total_wash_loads = np.zeros(num_steps)
            total_pv_sell = np.zeros(num_steps)
            total_ess_discharge = np.zeros(num_steps)
            total_ev_discharge = np.zeros(num_steps)

            for i in range(3):
                if i < len(episode_data.get('agent_home_loads', [[], [], []])):
                    total_home_loads += np.array(episode_data.get('agent_home_loads', [[], [], []])[i] if episode_data.get('agent_home_loads') else [0.0] * num_steps)
                    total_ac_loads += np.array(episode_data.get('agent_ac_loads', [[], [], []])[i] if episode_data.get('agent_ac_loads') else [0.0] * num_steps)
                    total_ewh_loads += np.array(episode_data.get('agent_ewh_loads', [[], [], []])[i] if episode_data.get('agent_ewh_loads') else [0.0] * num_steps)
                    total_wash_loads += np.array(episode_data.get('agent_wash_loads', [[], [], []])[i] if episode_data.get('agent_wash_loads') else [0.0] * num_steps)

                    # 获取PV售给电网的部分
                    pv_flows = episode_data.get('agent_pv_flows', [[], [], []])[i] if episode_data.get('agent_pv_flows') else []
                    pv_sell = np.array([pv_flow.get('grid_sell', 0.0) if isinstance(pv_flow, dict) else 0.0 for pv_flow in pv_flows] if len(pv_flows) == num_steps else [0.0] * num_steps)
                    total_pv_sell += pv_sell

                    # ESS和EV放电给电网
                    ess_discharge = np.array(episode_data.get('agent_ess_discharge', [[], [], []])[i] if episode_data.get('agent_ess_discharge') else [0.0] * num_steps)
                    ev_discharge = np.array(episode_data.get('agent_ev_discharge', [[], [], []])[i] if episode_data.get('agent_ev_discharge') else [0.0] * num_steps)
                    total_ess_discharge += ess_discharge
                    total_ev_discharge += ev_discharge

            # 正向堆叠：从电网购电（社区总需求）
            positive_stack = [
                total_home_loads,  # 家庭基础负荷
                total_ac_loads,    # 空调负荷
                total_ewh_loads,   # 热水器负荷
                total_wash_loads   # 洗衣机负荷
            ]
            positive_labels = ['家庭负荷', '空调', '热水器', '洗衣机']
            positive_colors = [colors['home_load'], colors['ac'], colors['ewh'], colors['wash_machine']]

            # 负向堆叠：向电网售电
            negative_stack = [
                -total_pv_sell,         # PV售给电网
                -total_ess_discharge,   # ESS放电给电网
                -total_ev_discharge     # EV放电给电网
            ]
            negative_labels = ['PV售电', 'ESS放电', 'EV放电']
            negative_colors = [colors['pv_sell'], colors['ess_discharge'], colors['ev_discharge']]

            # 绘制堆叠图（正方向，不传 labels，使用统一图例）
            if any(np.any(stack > 0) for stack in positive_stack):
                ax_community.stackplot(hours, *positive_stack, colors=positive_colors, alpha=0.7)

            # 绘制堆叠图（负方向）
            if any(np.any(stack < 0) for stack in negative_stack):
                ax_community.stackplot(hours, *negative_stack, colors=negative_colors, alpha=0.7)

            # 绘制社区净负荷曲线（验证）
            ax_community.plot(hours, community_loads, color='black', linewidth=2, linestyle='--', alpha=0.8)

            # 绘制零线
            ax_community.axhline(y=0, color='black', linewidth=1, alpha=0.3)

            ax_community.set_title(f'{model_names[method]} - 社区总负荷', fontsize=11, pad=10)
            ax_community.set_xlabel('时间 (小时)', fontsize=12)
            ax_community.set_ylabel('功率 (kW)', fontsize=12)
            ax_community.grid(True, alpha=0.3, linestyle='--')
            ax_community.legend(handles=legend_handles_community, loc='lower left', fontsize=7, ncol=2)
            ax_community.set_xlim(0, 24)

            y_max = max(np.abs(community_loads)) * 1.2 if len(community_loads) > 0 else 10
            ax_community.set_ylim(-y_max, y_max)

            # 标注峰值
            peak_idx = np.argmax(community_loads)
            peak_x = hours[peak_idx]
            peak_value = community_loads[peak_idx]
            box_w, box_h = 2.5, 1.0
            gap = 1.2
            if method == 'rule_based':
                text_x = max(0.2, min(16.0, peak_x - 6.0 - gap))
                text_y = max(-y_max + box_h + 0.2, min(y_max - 0.2, peak_value + 0.5))
                ha, va = 'left', 'top'
            else:
                text_x = max(0.2, min(24 - box_w - 0.5, peak_x + gap))
                text_y = max(-y_max + box_h + 0.2, min(y_max - 0.2, peak_value + 0.5))
                ha, va = 'left', 'bottom'
            ax_community.annotate(f'峰值: {peak_value:.2f} kW\n时间: {peak_x:.1f}h',
                                    xy=(peak_x, peak_value),
                                    xytext=(text_x, text_y),
                                    textcoords='data',
                                    ha=ha, va=va,
                                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7, zorder=2),
                                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', zorder=10),
                                    fontsize=8)
        else:
            # 无社区数据时仍显示统一图例与坐标
            ax_community.set_title(f'{model_names[method]} - 社区总负荷', fontsize=11, pad=10)
            ax_community.set_xlabel('时间 (小时)', fontsize=12)
            ax_community.set_ylabel('功率 (kW)', fontsize=12)
            ax_community.grid(True, alpha=0.3, linestyle='--')
            ax_community.legend(handles=legend_handles_community, loc='lower left', fontsize=7, ncol=2)
            ax_community.set_xlim(0, 24)
            ax_community.set_ylim(-10, 10)

        row += 1

    plt.tight_layout()

    # 保存图片（高分辨率）
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"堆叠图已保存到: {save_path}")

    # 显示图表
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='绘制净负荷堆叠对比图')
    parser.add_argument('--independent_data', type=str, default=None,
                       help='独立训练模型数据文件路径')
    parser.add_argument('--rule_based_data', type=str, default=None,
                       help='规则基线模型数据文件路径')
    parser.add_argument('--mappo_data', type=str, default=None,
                       help='MAPPO模型数据文件路径')
    parser.add_argument('--comparison_data', type=str, default=None,
                       help='对比数据文件路径（包含所有模型的数据）')
    parser.add_argument('--output_dir', type=str, default='multi_agent/visualization/output',
                       help='输出目录 (default: multi_agent/visualization/output)')
    parser.add_argument('--show_plot', action='store_true',
                       help='显示图表 (默认: False, 只保存不显示)')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 确定数据文件路径：优先使用命令行参数，否则尝试默认路径
    # 默认路径：项目根目录下的 multi_agent/multi_agent/visualization_data
    default_data_dir = os.path.join(project_root, 'multi_agent', 'multi_agent', 'visualization_data')
    independent_file = args.independent_data
    rule_based_file = args.rule_based_data
    mappo_file = args.mappo_data

    if not independent_file and os.path.exists(os.path.join(default_data_dir, 'independent_data.json')):
        independent_file = os.path.join(default_data_dir, 'independent_data.json')
    if not rule_based_file and os.path.exists(os.path.join(default_data_dir, 'rule_based_data.json')):
        rule_based_file = os.path.join(default_data_dir, 'rule_based_data.json')
    if not mappo_file and os.path.exists(os.path.join(default_data_dir, 'mappo_data.json')):
        mappo_file = os.path.join(default_data_dir, 'mappo_data.json')

    # 如果还是没有，尝试旧路径（兼容）
    if not any([independent_file, rule_based_file, mappo_file]):
        old_dir = os.path.join(project_root, 'multi_agent', 'visualization_data')
        if not independent_file and os.path.exists(os.path.join(old_dir, 'independent_data.json')):
            independent_file = os.path.join(old_dir, 'independent_data.json')
        if not rule_based_file and os.path.exists(os.path.join(old_dir, 'rule_based_data.json')):
            rule_based_file = os.path.join(old_dir, 'rule_based_data.json')
        if not mappo_file and os.path.exists(os.path.join(old_dir, 'mappo_data.json')):
            mappo_file = os.path.join(old_dir, 'mappo_data.json')

    # 检查是否有工作日/双休日分开的数据（任一数据文件含 summary_weekday/summary_weekend 或 episodes 带 date_type）
    has_weekday_weekend = False
    for _path in [independent_file, rule_based_file, mappo_file]:
        if _path and os.path.exists(_path):
            try:
                d = load_data_from_json(_path)
                if d.get('summary_weekday') or d.get('summary_weekend'):
                    has_weekday_weekend = True
                    break
                eps = d.get('episodes', [])
                if any(e.get('date_type') in ('工作日', '双休日') for e in eps):
                    has_weekday_weekend = True
                    break
            except Exception:
                pass

    if has_weekday_weekend:
        # 分别绘制工作日与双休日两张图
        out_weekday = os.path.join(args.output_dir, 'net_load_stacked_weekday.png')
        plot_net_load_stacked(
            independent_data_file=independent_file,
            rule_based_data_file=rule_based_file,
            mappo_data_file=mappo_file,
            save_path=out_weekday,
            show_plot=args.show_plot,
            date_type='weekday'
        )
        out_weekend = os.path.join(args.output_dir, 'net_load_stacked_weekend.png')
        plot_net_load_stacked(
            independent_data_file=independent_file,
            rule_based_data_file=rule_based_file,
            mappo_data_file=mappo_file,
            save_path=out_weekend,
            show_plot=args.show_plot,
            date_type='weekend'
        )
        print(f"已生成: {out_weekday}, {out_weekend}")
    else:
        output_path = os.path.join(args.output_dir, 'net_load_stacked_comparison.png')
        plot_net_load_stacked(
            independent_data_file=independent_file,
            rule_based_data_file=rule_based_file,
            mappo_data_file=mappo_file,
            save_path=output_path,
            show_plot=args.show_plot
        )
        print(f"已生成: {output_path}")

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