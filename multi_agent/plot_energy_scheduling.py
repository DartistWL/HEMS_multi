"""
绘制电力设备能源调度堆叠图
绘制一天48个时间步中，每个时刻的能量流向堆叠图
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import argparse

# 设置中文字体（Windows系统常用字体，按优先级排序）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题（将负号显示为正常字符而非Unicode字符）

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from multi_agent.algorithms.mappo import MAPPO
from multi_agent.environment.multi_agent_env import MultiAgentHEMEnv


def evaluate_with_detailed_records(mappo, baseline_peak=31.01, num_episodes=1, mode='eval', 
                                   agent_id=0, date_index=0):
    """
    评估MAPPO模型并记录详细数据
    
    Args:
        mappo: MAPPO算法实例
        baseline_peak: 基准峰值
        num_episodes: 评估轮数（通常为1，只评估一天）
        mode: 评估模式
        agent_id: 要记录的家庭ID（0, 1, 2）
        date_index: 日期索引
    
    Returns:
        dict: 详细记录的数据
    """
    # 创建环境
    env = MultiAgentHEMEnv(
        n_agents=3,
        community_ess_capacity=36.0,
        baseline_peak=baseline_peak,
        community_weight=0.2,
        pv_coefficients=[2.0, 2.0, 2.0]
    )
    
    # 设置评估日期
    eval_dates = ['2011-07-10', '2011-07-11', '2011-07-12']
    pv_coefficients_list = [
        [3.0, 3.0, 3.0],  # 晴天
        [1.0, 1.0, 1.0],  # 阴天
        [2.0, 2.0, 2.0],  # 正常
    ]
    env.set_evaluation_dates(eval_dates, pv_coefficients_list)
    
    # 记录数据（48个时间步）
    detailed_data = {
        'pv_generation': [],      # PV发电（kW）
        'grid_purchase': [],      # 从电网购电（kW，正值）
        'grid_sell': [],          # 向电网售电（kW，正值，但值为负）
        'community_ess_supply': [], # 从社区储能获取（kW，正值）
        'community_ess_discharge': [], # 向社区储能贡献（kW，正值，但值为负）
        'net_load': [],           # 净负荷（kW）
        'time_steps': [],         # 时间步（0-47）
        # PV流向相关数据
        'pv_direct_use': [],      # PV用于家庭直接用电（kW）
        'pv_ess_charge': [],      # PV用于ESS充电（kW）
        'pv_ev_charge': [],       # PV用于EV充电（kW）
        'pv_community_contribute': [],  # PV用于向社区储能贡献（kW）
        'pv_grid_sell': []        # PV用于向电网售电（kW）
    }
    
    # 重置环境
    states = env.reset(mode=mode, date_index=date_index)
    
    step_count = 0
    done = False
    
    # 运行一个episode（48个时间步）
    while not done and step_count < 48:
        # 选择动作（确定性策略）
        actions, _ = mappo.select_actions(states, deterministic=True)
        
        # 执行动作
        next_states, rewards, dones, info = env.step(actions)
        
        # 获取指定家庭的数据
        agent_state = states[agent_id]
        agent_action = actions[agent_id]
        
        # 记录PV发电
        pv_gen = agent_state.get('pv_generation', 0.0)
        detailed_data['pv_generation'].append(pv_gen)
        
        # 记录净负荷（从info中获取agent_net_loads列表）
        agent_net_loads = info.get('agent_net_loads', [0.0] * 3)
        net_load = agent_net_loads[agent_id] if agent_id < len(agent_net_loads) else 0.0
        detailed_data['net_load'].append(net_load)
        
        # 电网购电/售电（净负荷的正值部分和负值部分）
        if net_load > 0:
            detailed_data['grid_purchase'].append(net_load)
            detailed_data['grid_sell'].append(0.0)
        else:
            detailed_data['grid_purchase'].append(0.0)
            detailed_data['grid_sell'].append(abs(net_load))  # 转换为正值用于绘图
        
        # 记录社区储能交互
        community_power = agent_action.get('community_ess_power', 0.0)
        if community_power > 0:
            # 从社区储能获取（正值）
            detailed_data['community_ess_supply'].append(community_power)
            detailed_data['community_ess_discharge'].append(0.0)
        elif community_power < 0:
            # 向社区储能贡献（负值，转换为正值用于绘图）
            detailed_data['community_ess_supply'].append(0.0)
            detailed_data['community_ess_discharge'].append(abs(community_power))
        else:
            detailed_data['community_ess_supply'].append(0.0)
            detailed_data['community_ess_discharge'].append(0.0)
        
        # 计算PV流向（根据能量平衡推断）
        # 获取家庭负荷（基础负荷 + AC + 其他设备）
        home_load = agent_state.get('home_load', 0.0)
        ac_power = agent_state.get('Air_conditioner_power', 0.0)
        ac_power2 = agent_state.get('Air_conditioner_power2', 0.0)
        wash_machine_state = agent_state.get('wash_machine_state', 0)
        # 需要获取wash_machine_power，从环境获取
        try:
            wash_machine_power = env.agents[agent_id].env.wash_machine_power if hasattr(env.agents[agent_id].env, 'wash_machine_power') else 1.5
        except:
            wash_machine_power = 1.5
        wash_power = wash_machine_state * wash_machine_power
        ewh_power = agent_state.get('ewh_power', 0.0)
        household_load = home_load + ac_power + ac_power2 + wash_power + ewh_power
        
        # 获取充放电动作
        battery_power = agent_action.get('battery_power', 0.0)
        ev_power = agent_action.get('ev_power', 0.0)
        battery_charge = max(battery_power, 0.0)  # ESS充电功率
        ev_charge = max(ev_power, 0.0)  # EV充电功率
        community_contribute = abs(community_power) if community_power < 0 else 0.0  # 向社区储能贡献
        
        # 计算PV的各个去向（按优先级分配）
        # 注意：这是推断性的分配，基于能量平衡模型
        # 总需求 = household_load + battery_charge + ev_charge + community_contribute
        # PV可用于满足这些需求，剩余部分向电网售电
        pv_remaining = pv_gen
        
        # 1. 家庭直接用电（最高优先级）
        pv_direct_use = min(pv_remaining, household_load)
        pv_remaining = max(0, pv_remaining - pv_direct_use)
        
        # 2. ESS充电
        pv_ess_charge = min(pv_remaining, battery_charge) if pv_remaining > 0 and battery_charge > 0 else 0.0
        pv_remaining = max(0, pv_remaining - pv_ess_charge)
        
        # 3. EV充电
        pv_ev_charge = min(pv_remaining, ev_charge) if pv_remaining > 0 and ev_charge > 0 else 0.0
        pv_remaining = max(0, pv_remaining - pv_ev_charge)
        
        # 4. 向社区储能贡献
        pv_community_contribute = min(pv_remaining, community_contribute) if pv_remaining > 0 and community_contribute > 0 else 0.0
        pv_remaining = max(0, pv_remaining - pv_community_contribute)
        
        # 5. 向电网售电（剩余部分，如果净负荷为负）
        grid_sell_value = abs(net_load) if net_load < 0 else 0.0
        pv_grid_sell = min(pv_remaining, grid_sell_value) if pv_remaining > 0 else 0.0
        
        # 记录PV流向数据
        detailed_data['pv_direct_use'].append(pv_direct_use)
        detailed_data['pv_ess_charge'].append(pv_ess_charge)
        detailed_data['pv_ev_charge'].append(pv_ev_charge)
        detailed_data['pv_community_contribute'].append(pv_community_contribute)
        detailed_data['pv_grid_sell'].append(pv_grid_sell)
        
        # 记录时间步
        detailed_data['time_steps'].append(step_count)
        
        states = next_states
        done = all(dones)
        step_count += 1
    
    return detailed_data


def plot_energy_scheduling_stacked(detailed_data, agent_id=0, save_path=None, show_plot=True):
    """
    绘制能源调度堆叠图
    
    Args:
        detailed_data: 详细数据字典
        agent_id: 家庭ID
        save_path: 保存路径
        show_plot: 是否显示图表
    """
    time_steps = detailed_data['time_steps']
    num_steps = len(time_steps)
    
    # 提取数据
    pv_gen = np.array(detailed_data['pv_generation'])
    grid_purchase = np.array(detailed_data['grid_purchase'])
    grid_sell = np.array(detailed_data['grid_sell'])
    community_supply = np.array(detailed_data['community_ess_supply'])
    community_discharge = np.array(detailed_data['community_ess_discharge'])
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # 计算堆叠位置
    # 0轴上方：PV发电 + 电网购电 + 从社区储能获取
    bottom_positive = np.zeros(num_steps)
    
    # 0轴下方：向社区储能贡献 + 向电网售电
    bottom_negative = np.zeros(num_steps)
    
    # 绘制0轴上方的堆叠（能量来源）
    # 1. PV发电（最底层）
    ax.bar(time_steps, pv_gen, bottom=bottom_positive, label='PV发电', 
           color='#FFD700', alpha=0.8, width=0.6)
    bottom_positive += pv_gen
    
    # 2. 从社区储能获取（堆叠在PV上方）
    ax.bar(time_steps, community_supply, bottom=bottom_positive, 
           label='从社区储能获取', color='#32CD32', alpha=0.8, width=0.6)
    bottom_positive += community_supply
    
    # 3. 从电网购电（堆叠在最上方）
    ax.bar(time_steps, grid_purchase, bottom=bottom_positive, 
           label='从电网购电', color='#4169E1', alpha=0.8, width=0.6)
    
    # 绘制0轴下方的堆叠（能量去向）
    # 1. 向社区储能贡献
    ax.bar(time_steps, -community_discharge, bottom=-bottom_negative, 
           label='向社区储能贡献', color='#FF6347', alpha=0.8, width=0.6)
    bottom_negative += community_discharge
    
    # 2. 向电网售电（堆叠在向社区储能贡献下方）
    ax.bar(time_steps, -grid_sell, bottom=-bottom_negative, 
           label='向电网售电', color='#9370DB', alpha=0.8, width=0.6)
    
    # 添加0轴参考线
    ax.axhline(y=0, color='black', linewidth=1, linestyle='-', alpha=0.5)
    
    # 设置标签和标题
    ax.set_xlabel('时间步 (每步30分钟)', fontsize=12)
    ax.set_ylabel('功率 (kW)', fontsize=12)
    ax.set_title(f'家庭 {agent_id + 1} 能源调度堆叠图（24小时，48个时间步）', fontsize=14, pad=20)
    
    # 设置x轴刻度（每4步显示一个，即每2小时）
    step_interval = 4
    ax.set_xticks(time_steps[::step_interval])
    ax.set_xticklabels([f'{i*0.5:.1f}h' for i in time_steps[::step_interval]], rotation=45)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_axisbelow(True)
    
    # 添加图例
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"能源调度堆叠图已保存到: {save_path}")
    
    # 显示图表
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_pv_flow_direction(detailed_data, agent_id=0, save_path=None, show_plot=True):
    """
    绘制光伏流向堆叠面积图
    
    Args:
        detailed_data: 详细数据字典
        agent_id: 家庭ID
        save_path: 保存路径
        show_plot: 是否显示图表
    """
    time_steps = detailed_data['time_steps']
    hours = [t * 0.5 for t in time_steps]  # 转换为小时
    
    # 提取PV流向数据
    pv_direct_use = np.array(detailed_data['pv_direct_use'])
    pv_ess_charge = np.array(detailed_data['pv_ess_charge'])
    pv_ev_charge = np.array(detailed_data['pv_ev_charge'])
    pv_community_contribute = np.array(detailed_data['pv_community_contribute'])
    pv_grid_sell = np.array(detailed_data['pv_grid_sell'])
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 绘制堆叠面积图（从下往上堆叠）
    # 1. 家庭直接用电（最底层，黄色/橙色）
    ax.fill_between(hours, 0, pv_direct_use, 
                     label='家庭直接用电', color='#FFA500', alpha=0.7, linewidth=0.5)
    bottom = pv_direct_use
    
    # 2. 储能充电（蓝色）
    ax.fill_between(hours, bottom, bottom + pv_ess_charge, 
                     label='储能充电', color='#4169E1', alpha=0.7, linewidth=0.5)
    bottom += pv_ess_charge
    
    # 3. EV充电（紫色）
    ax.fill_between(hours, bottom, bottom + pv_ev_charge, 
                     label='EV充电', color='#9370DB', alpha=0.7, linewidth=0.5)
    bottom += pv_ev_charge
    
    # 4. 向社区储能贡献（红色/橙色）
    ax.fill_between(hours, bottom, bottom + pv_community_contribute, 
                     label='向社区储能贡献', color='#FF6347', alpha=0.7, linewidth=0.5)
    bottom += pv_community_contribute
    
    # 5. 售电给电网（绿色，最顶层）
    ax.fill_between(hours, bottom, bottom + pv_grid_sell, 
                     label='售电给电网', color='#32CD32', alpha=0.7, linewidth=0.5)
    
    # 绘制PV总发电量轮廓线（黑色）
    pv_total = np.array(detailed_data['pv_generation'])
    ax.plot(hours, pv_total, 'k-', linewidth=1.5, label='PV总发电量', alpha=0.8)
    
    # 设置标签和标题
    ax.set_xlabel('时间 (小时)', fontsize=12)
    ax.set_ylabel('功率 (kW)', fontsize=12)
    ax.set_title(f'家庭 {agent_id + 1} 光伏流向图（24小时）', fontsize=14, pad=20)
    
    # 设置x轴范围
    ax.set_xlim(0, 24)
    
    # 设置x轴刻度（每2小时显示一个）
    ax.set_xticks(range(0, 25, 2))
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--', axis='both')
    ax.set_axisbelow(True)
    
    # 添加图例
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9, ncol=1)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"光伏流向图已保存到: {save_path}")
    
    # 显示图表
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='绘制能源调度堆叠图（所有家庭，晴天和阴天）')
    parser.add_argument('--model_dir', type=str, default='multi_agent/algorithms/models',
                       help='模型目录 (default: multi_agent/algorithms/models)')
    parser.add_argument('--baseline_peak', type=float, default=31.01,
                       help='基准峰值 (default: 31.01)')
    parser.add_argument('--output_dir', type=str, default='multi_agent/algorithms/models',
                       help='输出目录 (default: multi_agent/algorithms/models)')
    parser.add_argument('--show_plot', action='store_true',
                       help='显示图表 (默认: False, 只保存不显示)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("能源调度堆叠图与光伏流向图绘制工具")
    print("=" * 80)
    print(f"模型目录: {args.model_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"将生成: 3个家庭 × 2种天气（晴天、阴天）× 2种图表 = 12张图")
    print("  - 能源调度堆叠图 (energy_scheduling_agent*_*.png)")
    print("  - 光伏流向图 (pv_flow_agent*_*.png)")
    print("=" * 80)
    
    # 创建环境（用于初始化MAPPO）
    env = MultiAgentHEMEnv(
        n_agents=3,
        community_ess_capacity=36.0,
        baseline_peak=args.baseline_peak,
        community_weight=0.2,
        pv_coefficients=[2.0, 2.0, 2.0]
    )
    
    # 创建MAPPO算法
    mappo = MAPPO(
        env=env,
        n_agents=3,
        hidden_dim=128,
        gamma=0.96,
        lmbda=0.95,
        eps=0.2,
        epochs=3,
        lr_actor=1e-4,
        lr_critic=1e-4,
        batch_size=64,
        ent_coef=0.01,
        max_grad_norm=1.0,
        use_state_normalization=True,
        reward_scale=10.0
    )
    
    # 加载模型
    print(f"\n加载模型从 {args.model_dir}...")
    if not os.path.exists(args.model_dir):
        print(f"错误: 模型目录 {args.model_dir} 不存在!")
        return
    
    mappo.load(args.model_dir)
    print("模型加载成功!")
    
    # 为所有家庭和天气条件生成图表
    date_configs = [
        (0, '晴天'),   # date_index=0, 晴天
        (1, '阴天'),   # date_index=1, 阴天
    ]
    
    generated_files = []
    
    for agent_id in range(3):  # 3个家庭
        for date_index, date_name in date_configs:  # 晴天和阴天
            print(f"\n{'='*80}")
            print(f"生成图表: 家庭 {agent_id + 1} - {date_name}")
            print(f"{'='*80}")
            
            # 评估并记录详细数据
            detailed_data = evaluate_with_detailed_records(
                mappo=mappo,
                baseline_peak=args.baseline_peak,
                num_episodes=1,
                mode='eval',
                agent_id=agent_id,
                date_index=date_index
            )
            
            # 绘制能源调度堆叠图
            output_filename_energy = f'energy_scheduling_agent{agent_id + 1}_{date_name}.png'
            output_path_energy = os.path.join(args.output_dir, output_filename_energy)
            
            plot_energy_scheduling_stacked(
                detailed_data=detailed_data,
                agent_id=agent_id,
                save_path=output_path_energy,
                show_plot=args.show_plot  # 只在用户指定时显示
            )
            
            generated_files.append(output_path_energy)
            print(f"已保存: {output_path_energy}")
            
            # 绘制PV流向图
            output_filename_pv = f'pv_flow_agent{agent_id + 1}_{date_name}.png'
            output_path_pv = os.path.join(args.output_dir, output_filename_pv)
            
            plot_pv_flow_direction(
                detailed_data=detailed_data,
                agent_id=agent_id,
                save_path=output_path_pv,
                show_plot=args.show_plot  # 只在用户指定时显示
            )
            
            generated_files.append(output_path_pv)
            print(f"已保存: {output_path_pv}")
    
    print("\n" + "=" * 80)
    print("所有图表生成完成!")
    print("=" * 80)
    print(f"共生成 {len(generated_files)} 张图表:")
    for file_path in generated_files:
        print(f"  - {file_path}")
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
