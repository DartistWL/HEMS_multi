"""
数据收集脚本
收集评估数据用于可视化
Collect evaluation data for visualization
"""
import sys
import os
import numpy as np
import json
import argparse
from datetime import datetime

# 添加项目根目录到路径
# 文件位于 multi_agent/collect_visualization_data.py，需要向上一级到项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from multi_agent.baselines.rule_based_baseline import RuleBasedBaseline
from multi_agent.baselines.independent_baseline import IndependentBaseline
from multi_agent.algorithms.mappo import MAPPO
from multi_agent.environment.multi_agent_env import MultiAgentHEMEnv

# 评估日期使用 2020 年的数据（与 StoreNet 2020 一致，且不与训练日期 01-01~01-07 重叠）
EVAL_DATES_WEEKDAY = ['2020-01-08', '2020-01-09']   # 周三、周四（工作日）
EVAL_DATES_WEEKEND = ['2020-01-11', '2020-01-12']   # 周六、周日（双休日）


# 注意：此函数已废弃，积分不再转换为金钱
# 积分余额会单独记录和展示，不再转换为金钱成本
# 保留此函数定义以避免导入错误，但不再使用
def calculate_credit_cost_in_money(credit_transaction, grid_price):
    """
    【已废弃】将积分交易转换为金钱成本

    注意：根据新的需求，积分不再转换为金钱。
    积分余额会单独记录和展示（在agent_credit_balances中），不再转换为金钱成本。

    此函数保留以避免导入错误，但不再被调用。
    """
    # 返回0.0，表示不再计算积分成本
    return 0.0


def collect_rule_based_data(num_episodes=3, output_file=None):
    """
    收集固定规则基线的数据

    Args:
        num_episodes: 评估轮数
        output_file: 输出文件路径（可选）

    Returns:
        dict: 收集的数据
    """
    print("=" * 80)
    print("收集固定规则基线数据")
    print("=" * 80)

    # 保存原始工作目录，并切换到 multi_agent 目录（使 ../data 相对路径正确）
    original_cwd = os.getcwd()
    os.chdir(os.path.join(project_root, 'multi_agent'))

    try:
        # 创建环境
        env = MultiAgentHEMEnv(
            n_agents=3,
            community_ess_capacity=36.0,
            baseline_peak=31.01,  # 临时值，实际应该使用独立基线的峰值
            community_weight=0.2,
            community_credit_cost_weight=0.1,
            community_credit_benefit_weight=0.1,
            pv_coefficients=[2.0, 2.0, 2.0]
        )

        # 工作日与双休日分开评估
        pv_list_weekday = [[3.0, 3.0, 3.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
        pv_list_weekend = [[3.0, 3.0, 3.0], [2.0, 2.0, 2.0]]

        # 创建基线
        baseline = RuleBasedBaseline(peak_threshold_factor=1.2)

        # 数据收集
        collected_data = {
            'method': 'rule_based',
            'episodes': [],
            'summary': {},
            'summary_weekday': {},
            'summary_weekend': {},
            'eval_dates_weekday': EVAL_DATES_WEEKDAY,
            'eval_dates_weekend': EVAL_DATES_WEEKEND,
        }

        all_community_loads = []
        all_peak_loads = []
        all_costs = []
        all_returns = []

        for date_type_label, eval_dates, pv_list in [
            ('工作日', EVAL_DATES_WEEKDAY, pv_list_weekday),
            ('双休日', EVAL_DATES_WEEKEND, pv_list_weekend),
        ]:
            env.set_evaluation_dates(eval_dates, pv_list)
            for episode in range(num_episodes):
                date_index = episode % len(eval_dates)
                date = eval_dates[date_index]
                print(f"\nEpisode [{date_type_label}] {episode + 1}/{num_episodes} - Date: {date}")

                # 重置环境
                states = env.reset(mode='eval', date_index=date_index)

                # 记录时间序列数据（48个时间步）
                episode_data = {
                    'date': date,
                    'date_index': date_index,
                    'date_type': date_type_label,
                    'community_net_loads': [],
                    'community_ess_soc': [],
                    'community_ess_charge_power': [],
                    'community_ess_discharge_power': [],
                    'agent_credit_balances': [[], [], []],  # 3个家庭的积分余额
                    'agent_net_loads': [[], [], []],  # 3个家庭的净负荷
                    # 详细的设备功率分解
                    'agent_home_loads': [[], [], []],  # 家庭基础负荷
                    'agent_ac_loads': [[], [], []],  # 空调负荷（AC1 + AC2）
                    'agent_ewh_loads': [[], [], []],  # 热水器负荷
                    'agent_wash_loads': [[], [], []],  # 洗衣机负荷
                    'agent_pv_generation': [[], [], []],  # PV发电
                    'agent_ess_charge': [[], [], []],  # ESS充电功率（从电网）
                    'agent_ess_discharge': [[], [], []],  # ESS放电功率（给电网）
                    'agent_ev_charge': [[], [], []],  # EV充电功率（从电网）
                    'agent_ev_discharge': [[], [], []],  # EV放电功率（给电网）
                    'agent_community_charge': [[], [], []],  # 向社区储能充电（负值）
                    'agent_community_discharge': [[], [], []],  # 从社区储能放电（正值）
                    'agent_pv_flows': [[], [], []],  # PV流向信息
                }

                step_count = 0
                done = False
                episode_cost = 0.0
                episode_return = 0.0
                episode_agent_costs = [0.0, 0.0, 0.0]  # 每个家庭的成本

                while not done and step_count < 48:
                    # 选择动作
                    actions = []
                    for i, state in enumerate(states):
                        action = baseline.select_action(
                            state, i, env.get_community_state(),
                            env.agents[i].net_load_history
                        )
                        actions.append(action)

                    # 执行动作
                    next_states, rewards, dones, info = env.step(actions)

                    # 记录数据
                    episode_data['community_net_loads'].append(info.get('community_net_load', 0.0))

                    # 社区储能状态
                    community_ess_state = env.community_ess.get_state()
                    episode_data['community_ess_soc'].append(community_ess_state.get('soc', 0.5))

                    # 社区储能充放电功率（从社区储能状态获取）
                    charge_power = 0.0
                    discharge_power = 0.0
                    if len(env.community_ess.charge_history) > step_count:
                        charge_power = env.community_ess.charge_history[step_count] / 0.5
                    if len(env.community_ess.discharge_history) > step_count:
                        discharge_power = env.community_ess.discharge_history[step_count] / 0.5
                    episode_data['community_ess_charge_power'].append(charge_power)
                    episode_data['community_ess_discharge_power'].append(discharge_power)

                    # 积分余额
                    credit_balances = env.credit_system.get_all_balances()
                    for i in range(3):
                        episode_data['agent_credit_balances'][i].append(credit_balances.get(i, 100.0))

                    # 每个家庭的净负荷
                    agent_net_loads = info.get('agent_net_loads', [0.0] * 3)
                    for i in range(3):
                        episode_data['agent_net_loads'][i].append(agent_net_loads[i] if i < len(agent_net_loads) else 0.0)

                    # 获取每个智能体的详细设备功率信息
                    agent_pv_flows = info.get('agent_pv_flows', [{}] * 3)
                    for i in range(3):
                        # 获取当前状态和动作
                        state = env.agents[i].get_state()
                        action = actions[i] if i < len(actions) else {}
                        home_load = state.get('home_load', 0.0)
                        episode_data['agent_home_loads'][i].append(home_load)
                        ac_load = state.get('Air_conditioner_power', 0.0) + state.get('Air_conditioner_power2', 0.0)
                        episode_data['agent_ac_loads'][i].append(ac_load)
                        ewh_load = state.get('ewh_power', 0.0)
                        episode_data['agent_ewh_loads'][i].append(ewh_load)
                        wash_load = state.get('wash_machine_state', 0) * env.agents[i].env.wash_machine_power
                        episode_data['agent_wash_loads'][i].append(wash_load)
                        pv_gen = state.get('pv_generation', 0.0)
                        episode_data['agent_pv_generation'][i].append(pv_gen)
                        ess_power = action.get('battery_power', 0.0)
                        ess_charge = max(ess_power, 0.0)
                        ess_discharge = max(-ess_power, 0.0)
                        episode_data['agent_ess_charge'][i].append(ess_charge)
                        episode_data['agent_ess_discharge'][i].append(ess_discharge)
                        ev_power = action.get('ev_power', 0.0)
                        ev_charge = max(ev_power, 0.0)
                        ev_discharge = max(-ev_power, 0.0)
                        episode_data['agent_ev_charge'][i].append(ev_charge)
                        episode_data['agent_ev_discharge'][i].append(ev_discharge)
                        community_power = action.get('community_ess_power', 0.0)
                        if community_power > 0:
                            episode_data['agent_community_discharge'][i].append(community_power)
                            episode_data['agent_community_charge'][i].append(0.0)
                        elif community_power < 0:
                            episode_data['agent_community_charge'][i].append(abs(community_power))
                            episode_data['agent_community_discharge'][i].append(0.0)
                        else:
                            episode_data['agent_community_charge'][i].append(0.0)
                            episode_data['agent_community_discharge'][i].append(0.0)
                        pv_flow = agent_pv_flows[i] if i < len(agent_pv_flows) else {}
                        episode_data['agent_pv_flows'][i].append(pv_flow)

                    # 成本（只记录电网成本）
                    for i in range(3):
                        grid_cost = env.agents[i].env.current_step_cost
                        episode_agent_costs[i] += grid_cost
                    step_cost = sum([env.agents[i].env.current_step_cost for i in range(3)])
                    episode_cost += step_cost
                    episode_return += sum(rewards)
                    states = next_states
                    done = all(dones)
                    step_count += 1

                # 计算峰值并记录
                peak_load = max(episode_data['community_net_loads']) if episode_data['community_net_loads'] else 0.0
                episode_data['peak_load'] = peak_load
                episode_data['total_cost'] = episode_cost
                episode_data['total_return'] = episode_return
                episode_data['agent_costs'] = episode_agent_costs
                collected_data['episodes'].append(episode_data)
                all_community_loads.extend(episode_data['community_net_loads'])
                all_peak_loads.append(peak_load)
                all_costs.append(episode_cost)
                all_returns.append(episode_return)

        # 汇总统计：summary 取第一个 episode；summary_weekday / summary_weekend 取各自第一个
        first_episode = collected_data['episodes'][0] if collected_data['episodes'] else {}
        weekday_eps = [e for e in collected_data['episodes'] if e.get('date_type') == '工作日']
        weekend_eps = [e for e in collected_data['episodes'] if e.get('date_type') == '双休日']
        first_weekday = weekday_eps[0] if weekday_eps else {}
        first_weekend = weekend_eps[0] if weekend_eps else {}

        def _make_summary(ep, mean_peak=None):
            if not ep:
                return {}
            peak = mean_peak if mean_peak is not None else ep.get('peak_load', 0.0)
            return {
                'community_net_loads': ep.get('community_net_loads', []),
                'peak_load': peak,
                'avg_load': np.mean(all_community_loads) if all_community_loads else 0.0,
                'std_load': np.std(all_community_loads) if all_community_loads else 0.0,
                'total_cost': np.mean(all_costs) if all_costs else 0.0,
                'total_return': np.mean(all_returns) if all_returns else 0.0,
                'community_ess_soc': ep.get('community_ess_soc', []),
                'community_ess_charge_power': ep.get('community_ess_charge_power', []),
                'community_ess_discharge_power': ep.get('community_ess_discharge_power', []),
                'agent_credit_balances': ep.get('agent_credit_balances', [[], [], []]),
                'agent_net_loads': ep.get('agent_net_loads', [[], [], []]),
                'agent_home_loads': ep.get('agent_home_loads', [[], [], []]),
                'agent_ac_loads': ep.get('agent_ac_loads', [[], [], []]),
                'agent_ewh_loads': ep.get('agent_ewh_loads', [[], [], []]),
                'agent_wash_loads': ep.get('agent_wash_loads', [[], [], []]),
                'agent_pv_generation': ep.get('agent_pv_generation', [[], [], []]),
                'agent_ess_charge': ep.get('agent_ess_charge', [[], [], []]),
                'agent_ess_discharge': ep.get('agent_ess_discharge', [[], [], []]),
                'agent_ev_charge': ep.get('agent_ev_charge', [[], [], []]),
                'agent_ev_discharge': ep.get('agent_ev_discharge', [[], [], []]),
                'agent_community_charge': ep.get('agent_community_charge', [[], [], []]),
                'agent_community_discharge': ep.get('agent_community_discharge', [[], [], []]),
                'agent_pv_flows': ep.get('agent_pv_flows', [[], [], []]),
            }

        collected_data['summary'] = _make_summary(first_episode, mean_peak=np.mean(all_peak_loads) if all_peak_loads else 0.0)
        collected_data['summary_weekday'] = _make_summary(first_weekday, mean_peak=np.mean([e['peak_load'] for e in weekday_eps]) if weekday_eps else 0.0)
        collected_data['summary_weekend'] = _make_summary(first_weekend, mean_peak=np.mean([e['peak_load'] for e in weekend_eps]) if weekend_eps else 0.0)

        # 保存数据
        if output_file:
            os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(collected_data, f, indent=2, ensure_ascii=False)
            print(f"\n数据已保存到: {output_file}")

        print("=" * 80)
        print(f"数据收集完成！共 {len(collected_data['episodes'])} 个episodes（工作日+双休日各 {num_episodes}）")
        print(f"平均峰值负荷: {collected_data['summary']['peak_load']:.2f} kW")
        print("=" * 80)

        return collected_data

    finally:
        # 恢复工作目录
        os.chdir(original_cwd)


def collect_independent_data(num_episodes=3, model_dir=None, output_file=None):
    """
    收集独立学习基线的数据（所有家庭使用同一个模型）

    Args:
        num_episodes: 评估轮数
        model_dir: 模型目录（如果为None，需要先训练）
        output_file: 输出文件路径（可选）

    Returns:
        dict: 收集的数据
    """
    print("=" * 80)
    print("收集独立学习基线数据")
    print("每个家庭使用各自的模型（不同的随机种子）")
    print("=" * 80)

    # 创建基线（use_community_env 从 config.json 的 independent_baseline.use_community_env 读取，默认 False）
    baseline = IndependentBaseline(
        n_agents=3,
        pv_coefficients=[2.0, 2.0, 2.0],
        random_seeds=None,
        use_community_env=None  # None = 从 config 读取
    )

    # 加载模型（如果提供）
    if model_dir:
        baseline.load_models(model_dir)
    else:
        print("错误：未提供模型目录，需要先训练模型")
        return None

    # 关键修复：强制评估环境使用训练住户（H1~H15），避免读取不存在的 H16 等文件
    if baseline.use_community_env and hasattr(baseline, 'eval_ma_env'):
        # 修改 eval_ma_env 中的测试住户列表为训练住户
        baseline.eval_ma_env.test_house_ids = [f"H{i}" for i in range(1, 16)]
        # 同时重置内部可能缓存的其他属性
        if hasattr(baseline.eval_ma_env, '_current_test_house_ids'):
            baseline.eval_ma_env._current_test_house_ids = baseline.eval_ma_env.test_house_ids
        print("已将评估环境的测试住户强制改为训练住户 (H1~H15)，避免 H16 缺失错误")

    # 数据收集
    collected_data = {
        'method': 'independent',
        'episodes': [],
        'summary': {},
        'summary_weekday': {},
        'summary_weekend': {},
        'eval_dates_weekday': EVAL_DATES_WEEKDAY,
        'eval_dates_weekend': EVAL_DATES_WEEKEND,
    }

    all_community_loads = []
    all_peak_loads = []
    all_costs = []
    all_returns = []

    # 方案1：使用多智能体环境（有社区储能）时的数据收集，工作日与双休日分开
    if baseline.use_community_env:
        pv_weekday = [[3.0, 3.0, 3.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
        pv_weekend = [[3.0, 3.0, 3.0], [2.0, 2.0, 2.0]]
        for date_type_label, eval_dates, pv_list in [
            ('工作日', EVAL_DATES_WEEKDAY, pv_weekday),
            ('双休日', EVAL_DATES_WEEKEND, pv_weekend),
        ]:
            baseline.eval_ma_env.set_evaluation_dates(eval_dates, pv_list)
            for episode in range(num_episodes):
                date_index = episode % len(eval_dates)
                date = eval_dates[date_index]
                for i in range(3):
                    baseline.eval_ma_env.agents[i].env.data_interface.set_pv_coefficient(pv_list[date_index][i])
                # 注意：这里 reset 时 house_index=0 会使用 test_house_ids[0] = H1（已修改）
                states = baseline.eval_ma_env.reset(mode='eval', date_index=date_index, house_index=0)
                episode_data = {
                    'date': date, 'date_index': date_index, 'date_type': date_type_label,
                    'community_net_loads': [], 'agent_net_loads': [[], [], []],
                    'agent_home_loads': [[], [], []], 'agent_ac_loads': [[], [], []],
                    'agent_ewh_loads': [[], [], []], 'agent_wash_loads': [[], [], []],
                    'agent_pv_generation': [[], [], []], 'agent_ess_charge': [[], [], []],
                    'agent_ess_discharge': [[], [], []], 'agent_ev_charge': [[], [], []],
                    'agent_ev_discharge': [[], [], []], 'agent_community_charge': [[], [], []],
                    'agent_community_discharge': [[], [], []], 'agent_pv_flows': [[], [], []],
                    'community_ess_soc': [], 'community_ess_charge_power': [], 'community_ess_discharge_power': [],
                    'agent_credit_balances': [[], [], []]
                }
                step_count = 0
                done = False
                episode_cost = 0.0
                episode_return = 0.0
                episode_agent_costs = [0.0, 0.0, 0.0]
                env = baseline.eval_ma_env
                while not done and step_count < 48:
                    actions = [baseline.agents[i].take_action(states[i]) for i in range(3)]
                    next_states, rewards, dones, info = env.step(actions)
                    community_net_load = info.get('community_net_load', 0)
                    episode_data['community_net_loads'].append(community_net_load)
                    agent_net_loads = info.get('agent_net_loads', [0.0, 0.0, 0.0])
                    for i in range(3):
                        episode_data['agent_net_loads'][i].append(agent_net_loads[i] if i < len(agent_net_loads) else 0.0)
                        sw = env.agents[i]
                        st = next_states[i]
                        episode_data['agent_home_loads'][i].append(st.get('home_load', 0.0))
                        episode_data['agent_ac_loads'][i].append(st.get('Air_conditioner_power', 0) + st.get('Air_conditioner_power2', 0))
                        episode_data['agent_ewh_loads'][i].append(st.get('ewh_power', 0))
                        episode_data['agent_wash_loads'][i].append(st.get('wash_machine_state', 0) * getattr(sw.env, 'wash_machine_power', 0))
                        episode_data['agent_pv_generation'][i].append(st.get('pv_generation', 0))
                        episode_data['agent_ess_charge'][i].append(max(actions[i].get('battery_power', 0), 0))
                        episode_data['agent_ess_discharge'][i].append(max(-actions[i].get('battery_power', 0), 0))
                        episode_data['agent_ev_charge'][i].append(max(actions[i].get('ev_power', 0), 0))
                        episode_data['agent_ev_discharge'][i].append(max(-actions[i].get('ev_power', 0), 0))
                        cp = actions[i].get('community_ess_power', 0)
                        episode_data['agent_community_charge'][i].append(abs(cp) if cp < 0 else 0)
                        episode_data['agent_community_discharge'][i].append(cp if cp > 0 else 0)
                        episode_data['agent_pv_flows'][i].append({})
                        episode_agent_costs[i] += env.agents[i].env.current_step_cost
                        episode_data['agent_credit_balances'][i].append(env.credit_system.get_balance(i))
                    episode_data['community_ess_soc'].append(info.get('community_ess_soc', env.community_ess.soc))
                    episode_data['community_ess_charge_power'].append(0.0)
                    episode_data['community_ess_discharge_power'].append(0.0)
                    episode_cost += sum(env.agents[i].env.current_step_cost for i in range(3))
                    episode_return += sum(rewards)
                    states = next_states
                    done = all(dones)
                    step_count += 1
                peak_load = max(episode_data['community_net_loads']) if episode_data['community_net_loads'] else 0
                episode_data['peak_load'] = peak_load
                episode_data['total_cost'] = episode_cost
                episode_data['total_return'] = episode_return
                episode_data['agent_costs'] = episode_agent_costs
                collected_data['episodes'].append(episode_data)
                all_community_loads.extend(episode_data['community_net_loads'])
                all_peak_loads.append(peak_load)
                all_costs.append(episode_cost)
                all_returns.append(episode_return)
                print(f"Episode [{date_type_label}] {episode + 1}/{num_episodes} - Date: {date}, Peak: {peak_load:.2f} kW, Cost: {episode_cost:.2f}")
        first_ep = collected_data['episodes'][0] if collected_data['episodes'] else {}
        weekday_eps = [e for e in collected_data['episodes'] if e.get('date_type') == '工作日']
        weekend_eps = [e for e in collected_data['episodes'] if e.get('date_type') == '双休日']
        first_wd = weekday_eps[0] if weekday_eps else {}
        first_we = weekend_eps[0] if weekend_eps else {}
        def _ind_summary(ep, mean_peak=None):
            p = mean_peak if mean_peak is not None else (ep.get('peak_load', 0.0) if ep else 0.0)
            return {
                'community_net_loads': ep.get('community_net_loads', []), 'peak_load': p,
                'avg_load': np.mean(all_community_loads) if all_community_loads else 0.0,
                'std_load': np.std(all_community_loads) if all_community_loads else 0.0,
                'total_cost': np.mean(all_costs) if all_costs else 0.0,
                'total_return': np.mean(all_returns) if all_returns else 0.0,
                'community_ess_soc': ep.get('community_ess_soc', []),
                'community_ess_charge_power': ep.get('community_ess_charge_power', []),
                'community_ess_discharge_power': ep.get('community_ess_discharge_power', []),
                'agent_credit_balances': ep.get('agent_credit_balances', [[], [], []]),
                'agent_net_loads': ep.get('agent_net_loads', [[], [], []]),
                'agent_home_loads': ep.get('agent_home_loads', [[], [], []]),
                'agent_ac_loads': ep.get('agent_ac_loads', [[], [], []]),
                'agent_ewh_loads': ep.get('agent_ewh_loads', [[], [], []]),
                'agent_wash_loads': ep.get('agent_wash_loads', [[], [], []]),
                'agent_pv_generation': ep.get('agent_pv_generation', [[], [], []]),
                'agent_ess_charge': ep.get('agent_ess_charge', [[], [], []]),
                'agent_ess_discharge': ep.get('agent_ess_discharge', [[], [], []]),
                'agent_ev_charge': ep.get('agent_ev_charge', [[], [], []]),
                'agent_ev_discharge': ep.get('agent_ev_discharge', [[], [], []]),
                'agent_community_charge': ep.get('agent_community_charge', [[], [], []]),
                'agent_community_discharge': ep.get('agent_community_discharge', [[], [], []]),
                'agent_pv_flows': ep.get('agent_pv_flows', [[], [], []])
            } if ep else {}
        collected_data['summary'] = _ind_summary(first_ep, np.mean(all_peak_loads) if all_peak_loads else 0.0)
        collected_data['summary_weekday'] = _ind_summary(first_wd, np.mean([e['peak_load'] for e in weekday_eps]) if weekday_eps else 0.0)
        collected_data['summary_weekend'] = _ind_summary(first_we, np.mean([e['peak_load'] for e in weekend_eps]) if weekend_eps else 0.0)
        if output_file:
            os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(collected_data, f, indent=2, ensure_ascii=False)
            print(f"\n数据已保存到: {output_file}")
        print("=" * 80)
        print(f"数据收集完成（方案1 社区储能）！共 {num_episodes} 个episodes")
        print(f"平均峰值负荷: {collected_data['summary']['peak_load']:.2f} kW")
        print("=" * 80)
        return collected_data

    # ---------- 原逻辑：单智能体环境（无社区储能）----------
    eval_dates = EVAL_DATES_WEEKDAY + EVAL_DATES_WEEKEND  # 合并用于单智能体路径
    for episode in range(num_episodes):
        date_index = episode % len(eval_dates)
        date = eval_dates[date_index]

        print(f"\nEpisode {episode + 1}/{num_episodes} - Date: {date}")

        # 重置所有环境（使用eval_envs，每个家庭使用各自的模型）
        states = []
        for env in baseline.eval_envs:
            state = env.reset()
            env.current_time = date
            env.current_time_index = 0
            states.append(state)

        # 记录时间序列数据
        episode_data = {
            'date': date,
            'date_index': date_index,
            'community_net_loads': [],
            'agent_net_loads': [[], [], []],
            # 详细的设备功率分解
            'agent_home_loads': [[], [], []],
            'agent_ac_loads': [[], [], []],
            'agent_ewh_loads': [[], [], []],
            'agent_wash_loads': [[], [], []],
            'agent_pv_generation': [[], [], []],
            'agent_ess_charge': [[], [], []],
            'agent_ess_discharge': [[], [], []],
            'agent_ev_charge': [[], [], []],
            'agent_ev_discharge': [[], [], []],
            'agent_community_charge': [[], [], []],  # 独立训练没有社区储能，保持为空
            'agent_community_discharge': [[], [], []],  # 独立训练没有社区储能，保持为空
            'agent_pv_flows': [[], [], []]  # 独立训练可以计算PV流向
        }

        step_count = 0
        done = False
        episode_cost = 0.0
        episode_return = 0.0
        episode_agent_costs = [0.0, 0.0, 0.0]  # 每个家庭的成本

        while not done and step_count < 48:
            # 选择动作（每个智能体使用各自的模型）
            actions = []
            for i, (state, agent) in enumerate(zip(states, baseline.agents)):
                action = agent.take_action(state)
                actions.append(action)

            # 执行动作（每个智能体独立执行，但使用相同的策略）
            next_states = []
            all_dones = []
            for i, (env, action, state) in enumerate(zip(baseline.eval_envs, actions, states)):
                next_state, reward, done = env.step(env.state, action)
                next_states.append(next_state)
                all_dones.append(done)

                # 计算净负荷（从环境的total_load_compute获取）
                total_load = env.total_load_compute()
                episode_data['agent_net_loads'][i].append(total_load)

                # 获取详细的设备功率信息
                # 注意：这里使用next_state，因为step已经执行了
                # 家庭基础负荷
                home_load = next_state.get('home_load', 0.0)
                episode_data['agent_home_loads'][i].append(home_load)

                # 空调负荷
                ac_load = next_state.get('Air_conditioner_power', 0.0) + next_state.get('Air_conditioner_power2', 0.0)
                episode_data['agent_ac_loads'][i].append(ac_load)

                # 热水器负荷
                ewh_load = next_state.get('ewh_power', 0.0)
                episode_data['agent_ewh_loads'][i].append(ewh_load)

                # 洗衣机负荷
                wash_load = next_state.get('wash_machine_state', 0) * env.wash_machine_power
                episode_data['agent_wash_loads'][i].append(wash_load)

                # PV发电
                pv_gen = next_state.get('pv_generation', 0.0)
                episode_data['agent_pv_generation'][i].append(pv_gen)

                # ESS充放电功率
                ess_power = action.get('battery_power', 0.0)
                ess_charge = max(ess_power, 0.0)
                ess_discharge = max(-ess_power, 0.0)
                episode_data['agent_ess_charge'][i].append(ess_charge)
                episode_data['agent_ess_discharge'][i].append(ess_discharge)

                # EV充放电功率
                ev_power = action.get('ev_power', 0.0)
                ev_charge = max(ev_power, 0.0)
                ev_discharge = max(-ev_power, 0.0)
                episode_data['agent_ev_charge'][i].append(ev_charge)
                episode_data['agent_ev_discharge'][i].append(ev_discharge)

                # 独立训练没有社区储能
                episode_data['agent_community_charge'][i].append(0.0)
                episode_data['agent_community_discharge'][i].append(0.0)

                # 计算PV流向（简化版本，独立训练没有社区储能）
                # PV优先用于直接负荷，然后充电，最后售给电网
                household_load = home_load + ac_load + ewh_load + wash_load
                pv_direct_use = min(pv_gen, household_load)
                pv_remaining = max(0, pv_gen - pv_direct_use)

                # PV用于ESS充电
                pv_for_ess = min(pv_remaining, ess_charge)
                pv_remaining = max(0, pv_remaining - pv_for_ess)

                # PV用于EV充电
                pv_for_ev = min(pv_remaining, ev_charge)
                pv_remaining = max(0, pv_remaining - pv_for_ev)

                # 剩余PV售给电网
                pv_sell = pv_remaining

                pv_flow = {
                    'direct_use': pv_direct_use,
                    'ess_charge': pv_for_ess,
                    'ev_charge': pv_for_ev,
                    'community_charge': 0.0,  # 独立训练没有社区储能
                    'grid_sell': pv_sell
                }
                episode_data['agent_pv_flows'][i].append(pv_flow)

                agent_step_cost = env.current_step_cost
                episode_agent_costs[i] += agent_step_cost
                episode_cost += env.current_step_cost
                episode_return += reward

            # 计算社区净负荷
            community_load = sum([loads[-1] if len(loads) > 0 else 0.0
                                 for loads in episode_data['agent_net_loads']])
            episode_data['community_net_loads'].append(community_load)

            states = next_states
            done = all(all_dones)
            step_count += 1

        # 计算峰值
        peak_load = max(episode_data['community_net_loads']) if episode_data['community_net_loads'] else 0.0

        episode_data['peak_load'] = peak_load
        episode_data['total_cost'] = episode_cost
        episode_data['total_return'] = episode_return
        episode_data['agent_costs'] = episode_agent_costs  # 每个家庭的成本

        # 独立基线没有社区储能和积分系统，所以这些数据为空
        episode_data['community_ess_soc'] = []
        episode_data['community_ess_charge_power'] = []
        episode_data['community_ess_discharge_power'] = []
        episode_data['agent_credit_balances'] = [[], [], []]

        collected_data['episodes'].append(episode_data)
        all_community_loads.extend(episode_data['community_net_loads'])
        all_peak_loads.append(peak_load)
        all_costs.append(episode_cost)
        all_returns.append(episode_return)

    # 汇总统计
    first_episode = collected_data['episodes'][0] if collected_data['episodes'] else {}
    collected_data['summary'] = {
        'community_net_loads': first_episode.get('community_net_loads', []),
        'peak_load': np.mean(all_peak_loads) if all_peak_loads else 0.0,
        'avg_load': np.mean(all_community_loads) if all_community_loads else 0.0,
        'std_load': np.std(all_community_loads) if all_community_loads else 0.0,
        'total_cost': np.mean(all_costs) if all_costs else 0.0,
        'total_return': np.mean(all_returns) if all_returns else 0.0,
        'community_ess_soc': [],
        'community_ess_charge_power': [],
        'community_ess_discharge_power': [],
        'agent_credit_balances': [[], [], []],
        'agent_net_loads': first_episode.get('agent_net_loads', [[], [], []]),
        # 详细的设备功率分解
        'agent_home_loads': first_episode.get('agent_home_loads', [[], [], []]),
        'agent_ac_loads': first_episode.get('agent_ac_loads', [[], [], []]),
        'agent_ewh_loads': first_episode.get('agent_ewh_loads', [[], [], []]),
        'agent_wash_loads': first_episode.get('agent_wash_loads', [[], [], []]),
        'agent_pv_generation': first_episode.get('agent_pv_generation', [[], [], []]),
        'agent_ess_charge': first_episode.get('agent_ess_charge', [[], [], []]),
        'agent_ess_discharge': first_episode.get('agent_ess_discharge', [[], [], []]),
        'agent_ev_charge': first_episode.get('agent_ev_charge', [[], [], []]),
        'agent_ev_discharge': first_episode.get('agent_ev_discharge', [[], [], []]),
        'agent_community_charge': first_episode.get('agent_community_charge', [[], [], []]),
        'agent_community_discharge': first_episode.get('agent_community_discharge', [[], [], []]),
        'agent_pv_flows': first_episode.get('agent_pv_flows', [[], [], []])
    }

    if output_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(collected_data, f, indent=2, ensure_ascii=False)
        print(f"\n数据已保存到: {output_file}")

    print("=" * 80)
    print(f"数据收集完成！共 {num_episodes} 个episodes")
    print(f"平均峰值负荷: {collected_data['summary']['peak_load']:.2f} kW")
    print("=" * 80)

    return collected_data


def collect_mappo_data(num_episodes=3, model_dir=None, baseline_peak=31.01,
                       config_path='multi_agent/config.json', output_file=None):
    """
    收集MAPPO算法的数据

    Args:
        num_episodes: 评估轮数
        model_dir: 模型目录
        baseline_peak: 基准峰值
        config_path: 配置文件路径（用于读取训练时的参数）
        output_file: 输出文件路径（可选）

    Returns:
        dict: 收集的数据
    """
    print("=" * 80)
    print("收集MAPPO算法数据")
    print("=" * 80)

    default_mappo_dir = 'multi_agent/algorithms/models'
    if not model_dir:
        model_dir = default_mappo_dir
    # 根据 config 的 credit_pricing.scheme 自动选择模型目录（contribution_based 时用 _contribution_based 后缀，不覆盖原模型）
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            _cfg = json.load(f)
        if (_cfg.get('credit_pricing') or {}).get('scheme') == 'contribution_based':
            _resolved = model_dir.rstrip(os.sep) + '_contribution_based'
            if _resolved != model_dir and os.path.exists(_resolved):
                model_dir = _resolved
                print(f"根据 config credit_pricing.scheme=contribution_based 使用模型目录: {model_dir}")

    if not os.path.exists(model_dir):
        print("错误：模型目录不存在，需要先训练MAPPO模型")
        print(f"  当前目录: {model_dir}")
        return None

    # 从配置文件读取参数（与训练时保持一致）
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        training_config = config.get('training', {})
        community_weight = training_config.get('community_weight', 0.2)
        community_credit_cost_weight = training_config.get('community_credit_cost_weight', 0.1)
        community_credit_benefit_weight = training_config.get('community_credit_benefit_weight', 0.1)
        initial_credit = training_config.get('initial_credit', 100.0)  # 默认初始积分为100.0
        peak_penalty_exponent = training_config.get('peak_penalty_exponent', 2.0)
        peak_discharge_bonus = training_config.get('peak_discharge_bonus', 0.0)
        peak_credit_cost_reduction = training_config.get('peak_credit_cost_reduction', 1.0)
        print(f"从配置文件读取参数: community_weight={community_weight}")
    else:
        print(f"警告：配置文件不存在 {config_path}，使用默认值")
        community_weight = 0.2
        community_credit_cost_weight = 0.1
        community_credit_benefit_weight = 0.1
        initial_credit = 100.0  # 默认初始积分为100.0
        peak_penalty_exponent = 2.0
        peak_discharge_bonus = 0.0
        peak_credit_cost_reduction = 1.0

    # 保存原始工作目录，并切换到 multi_agent 目录（使 ../data 相对路径正确）
    original_cwd = os.getcwd()
    os.chdir(os.path.join(project_root, 'multi_agent'))

    try:
        # 创建环境（使用与训练时相同的参数）
        env = MultiAgentHEMEnv(
            n_agents=3,
            community_ess_capacity=36.0,
            baseline_peak=baseline_peak,
            community_weight=community_weight,
            community_credit_cost_weight=community_credit_cost_weight,
            community_credit_benefit_weight=community_credit_benefit_weight,
            initial_credit=initial_credit,
            peak_penalty_exponent=peak_penalty_exponent,
            peak_discharge_bonus=peak_discharge_bonus,
            peak_credit_cost_reduction=peak_credit_cost_reduction,
            pv_coefficients=[2.0, 2.0, 2.0]
        )

        # 工作日与双休日分开评估（与 rule_based / independent 一致）
        pv_weekday = [[3.0, 3.0, 3.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
        pv_weekend = [[3.0, 3.0, 3.0], [2.0, 2.0, 2.0]]

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
        mappo.load(model_dir)

        # 数据收集（与rule_based类似）
        collected_data = {
            'method': 'mappo',
            'episodes': [],
            'summary': {},
            'summary_weekday': {},
            'summary_weekend': {},
            'eval_dates_weekday': EVAL_DATES_WEEKDAY,
            'eval_dates_weekend': EVAL_DATES_WEEKEND,
        }

        all_community_loads = []
        all_peak_loads = []
        all_costs = []
        all_returns = []

        for date_type_label, eval_dates, pv_list in [
            ('工作日', EVAL_DATES_WEEKDAY, pv_weekday),
            ('双休日', EVAL_DATES_WEEKEND, pv_weekend),
        ]:
            env.set_evaluation_dates(eval_dates, pv_list)
            for episode in range(num_episodes):
                date_index = episode % len(eval_dates)
                date = eval_dates[date_index]
                print(f"\nEpisode [{date_type_label}] {episode + 1}/{num_episodes} - Date: {date}")
                states = env.reset(mode='eval', date_index=date_index)
                episode_data = {
                    'date': date,
                    'date_index': date_index,
                    'date_type': date_type_label,
                    'community_net_loads': [],
                    'community_ess_soc': [],
                    'community_ess_charge_power': [],
                    'community_ess_discharge_power': [],
                    'agent_credit_balances': [[], [], []],
                    'agent_net_loads': [[], [], []],
                    # 详细的设备功率分解
                    'agent_home_loads': [[], [], []],
                    'agent_ac_loads': [[], [], []],
                    'agent_ewh_loads': [[], [], []],
                    'agent_wash_loads': [[], [], []],
                    'agent_pv_generation': [[], [], []],
                    'agent_ess_charge': [[], [], []],
                    'agent_ess_discharge': [[], [], []],
                    'agent_ev_charge': [[], [], []],
                    'agent_ev_discharge': [[], [], []],
                    'agent_community_charge': [[], [], []],
                    'agent_community_discharge': [[], [], []],
                    'agent_pv_flows': [[], [], []],
                }
                step_count = 0
                done = False
                episode_cost = 0.0
                episode_return = 0.0
                episode_agent_costs = [0.0, 0.0, 0.0]
                while not done and step_count < 48:
                    actions, _ = mappo.select_actions(states, deterministic=True)
                    next_states, rewards, dones, info = env.step(actions)
                    episode_data['community_net_loads'].append(info.get('community_net_load', 0.0))
                    community_ess_state = env.community_ess.get_state()
                    episode_data['community_ess_soc'].append(community_ess_state.get('soc', 0.5))
                    charge_power = 0.0
                    discharge_power = 0.0
                    if len(env.community_ess.charge_history) > step_count:
                        charge_power = env.community_ess.charge_history[step_count] / 0.5
                    if len(env.community_ess.discharge_history) > step_count:
                        discharge_power = env.community_ess.discharge_history[step_count] / 0.5
                    episode_data['community_ess_charge_power'].append(charge_power)
                    episode_data['community_ess_discharge_power'].append(discharge_power)
                    credit_balances = env.credit_system.get_all_balances()
                    for i in range(3):
                        episode_data['agent_credit_balances'][i].append(credit_balances.get(i, 100.0))
                    agent_net_loads = info.get('agent_net_loads', [0.0] * 3)
                    for i in range(3):
                        episode_data['agent_net_loads'][i].append(agent_net_loads[i] if i < len(agent_net_loads) else 0.0)
                    agent_pv_flows = info.get('agent_pv_flows', [{}] * 3)
                    for i in range(3):
                        state = env.agents[i].get_state()
                        action = actions[i] if i < len(actions) else {}
                        home_load = state.get('home_load', 0.0)
                        episode_data['agent_home_loads'][i].append(home_load)
                        ac_load = state.get('Air_conditioner_power', 0.0) + state.get('Air_conditioner_power2', 0.0)
                        episode_data['agent_ac_loads'][i].append(ac_load)
                        ewh_load = state.get('ewh_power', 0.0)
                        episode_data['agent_ewh_loads'][i].append(ewh_load)
                        wash_load = state.get('wash_machine_state', 0) * env.agents[i].env.wash_machine_power
                        episode_data['agent_wash_loads'][i].append(wash_load)
                        pv_gen = state.get('pv_generation', 0.0)
                        episode_data['agent_pv_generation'][i].append(pv_gen)
                        ess_power = action.get('battery_power', 0.0)
                        ess_charge = max(ess_power, 0.0)
                        ess_discharge = max(-ess_power, 0.0)
                        episode_data['agent_ess_charge'][i].append(ess_charge)
                        episode_data['agent_ess_discharge'][i].append(ess_discharge)
                        ev_power = action.get('ev_power', 0.0)
                        ev_charge = max(ev_power, 0.0)
                        ev_discharge = max(-ev_power, 0.0)
                        episode_data['agent_ev_charge'][i].append(ev_charge)
                        episode_data['agent_ev_discharge'][i].append(ev_discharge)
                        community_power = action.get('community_ess_power', 0.0)
                        if community_power > 0:
                            episode_data['agent_community_discharge'][i].append(community_power)
                            episode_data['agent_community_charge'][i].append(0.0)
                        elif community_power < 0:
                            episode_data['agent_community_charge'][i].append(abs(community_power))
                            episode_data['agent_community_discharge'][i].append(0.0)
                        else:
                            episode_data['agent_community_charge'][i].append(0.0)
                            episode_data['agent_community_discharge'][i].append(0.0)
                        pv_flow = agent_pv_flows[i] if i < len(agent_pv_flows) else {}
                        episode_data['agent_pv_flows'][i].append(pv_flow)
                    for i in range(3):
                        grid_cost = env.agents[i].env.current_step_cost
                        episode_agent_costs[i] += grid_cost
                    step_cost = sum([env.agents[i].env.current_step_cost for i in range(3)])
                    episode_cost += step_cost
                    episode_return += sum(rewards)
                    states = next_states
                    done = all(dones)
                    step_count += 1
                peak_load = max(episode_data['community_net_loads']) if episode_data['community_net_loads'] else 0.0
                episode_data['peak_load'] = peak_load
                episode_data['total_cost'] = episode_cost
                episode_data['total_return'] = episode_return
                episode_data['agent_costs'] = episode_agent_costs
                collected_data['episodes'].append(episode_data)
                all_community_loads.extend(episode_data['community_net_loads'])
                all_peak_loads.append(peak_load)
                all_costs.append(episode_cost)
                all_returns.append(episode_return)

        # 汇总统计：summary / summary_weekday / summary_weekend
        first_episode = collected_data['episodes'][0] if collected_data['episodes'] else {}
        weekday_eps = [e for e in collected_data['episodes'] if e.get('date_type') == '工作日']
        weekend_eps = [e for e in collected_data['episodes'] if e.get('date_type') == '双休日']
        first_wd = weekday_eps[0] if weekday_eps else {}
        first_we = weekend_eps[0] if weekend_eps else {}
        def _mappo_summary(ep, mean_peak=None):
            p = mean_peak if mean_peak is not None else (ep.get('peak_load', 0.0) if ep else 0.0)
            return {
                'community_net_loads': ep.get('community_net_loads', []), 'peak_load': p,
                'avg_load': np.mean(all_community_loads) if all_community_loads else 0.0,
                'std_load': np.std(all_community_loads) if all_community_loads else 0.0,
                'total_cost': np.mean(all_costs) if all_costs else 0.0,
                'total_return': np.mean(all_returns) if all_returns else 0.0,
                'community_ess_soc': ep.get('community_ess_soc', []),
                'community_ess_charge_power': ep.get('community_ess_charge_power', []),
                'community_ess_discharge_power': ep.get('community_ess_discharge_power', []),
                'agent_credit_balances': ep.get('agent_credit_balances', [[], [], []]),
                'agent_net_loads': ep.get('agent_net_loads', [[], [], []]),
                'agent_home_loads': ep.get('agent_home_loads', [[], [], []]),
                'agent_ac_loads': ep.get('agent_ac_loads', [[], [], []]),
                'agent_ewh_loads': ep.get('agent_ewh_loads', [[], [], []]),
                'agent_wash_loads': ep.get('agent_wash_loads', [[], [], []]),
                'agent_pv_generation': ep.get('agent_pv_generation', [[], [], []]),
                'agent_ess_charge': ep.get('agent_ess_charge', [[], [], []]),
                'agent_ess_discharge': ep.get('agent_ess_discharge', [[], [], []]),
                'agent_ev_charge': ep.get('agent_ev_charge', [[], [], []]),
                'agent_ev_discharge': ep.get('agent_ev_discharge', [[], [], []]),
                'agent_community_charge': ep.get('agent_community_charge', [[], [], []]),
                'agent_community_discharge': ep.get('agent_community_discharge', [[], [], []]),
                'agent_pv_flows': ep.get('agent_pv_flows', [[], [], []]),
            } if ep else {}
        collected_data['summary'] = _mappo_summary(first_episode, np.mean(all_peak_loads) if all_peak_loads else 0.0)
        collected_data['summary_weekday'] = _mappo_summary(first_wd, np.mean([e['peak_load'] for e in weekday_eps]) if weekday_eps else 0.0)
        collected_data['summary_weekend'] = _mappo_summary(first_we, np.mean([e['peak_load'] for e in weekend_eps]) if weekend_eps else 0.0)
        if output_file:
            os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(collected_data, f, indent=2, ensure_ascii=False)
            print(f"\n数据已保存到: {output_file}")

        print("=" * 80)
        print(f"数据收集完成！共 {len(collected_data['episodes'])} 个episodes（工作日+双休日各 {num_episodes}）")
        print(f"平均峰值负荷: {collected_data['summary']['peak_load']:.2f} kW")
        print("=" * 80)

        return collected_data

    finally:
        os.chdir(original_cwd)


def collect_all_data(output_dir='multi_agent/visualization_data',
                    independent_model_dir=None,
                    mappo_model_dir=None,
                    baseline_peak=31.01,
                    num_episodes=3):
    """
    收集所有方法的数据

    Args:
        output_dir: 输出目录
        independent_model_dir: 独立基线模型目录
        mappo_model_dir: MAPPO模型目录
        baseline_peak: 基准峰值
        num_episodes: 每个方法的评估轮数
    """
    os.makedirs(output_dir, exist_ok=True)

    # 收集固定规则基线数据
    rule_based_file = os.path.join(output_dir, 'rule_based_data.json')
    rule_based_data = collect_rule_based_data(num_episodes=num_episodes, output_file=rule_based_file)

    # 收集独立学习基线数据（如果提供模型目录）
    independent_data = None
    if independent_model_dir:
        independent_file = os.path.join(output_dir, 'independent_data.json')
        independent_data = collect_independent_data(
            num_episodes=num_episodes,
            model_dir=independent_model_dir,
            output_file=independent_file
        )

    # 收集MAPPO数据（如果提供模型目录）
    mappo_data = None
    if mappo_model_dir:
        mappo_file = os.path.join(output_dir, 'mappo_data.json')
        mappo_data = collect_mappo_data(
            num_episodes=num_episodes,
            model_dir=mappo_model_dir,
            baseline_peak=baseline_peak,
            output_file=mappo_file
        )

    # 创建对比数据文件
    comparison_data = {
        'baseline_peak': baseline_peak,
        'independent': independent_data['summary'] if independent_data else None,
        'rule_based': rule_based_data['summary'] if rule_based_data else None,
        'mappo': mappo_data['summary'] if mappo_data else None
    }

    comparison_file = os.path.join(output_dir, 'comparison_data.json')
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("所有数据收集完成！")
    print(f"数据保存在: {output_dir}")
    print("=" * 80)

    return comparison_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='收集可视化数据')
    parser.add_argument('--method', type=str, choices=['rule_based', 'independent', 'mappo', 'all'],
                       default='all', help='要收集数据的方法')
    parser.add_argument('--output_dir', type=str, default='multi_agent/visualization_data',
                       help='输出目录')
    parser.add_argument('--independent_model_dir', type=str, default='multi_agent/baselines/models',
                       help='独立基线模型目录 (default: multi_agent/baselines/models)')
    parser.add_argument('--mappo_model_dir', type=str, default='multi_agent/algorithms/models',
                       help='MAPPO模型目录 (default: multi_agent/algorithms/models)')
    parser.add_argument('--baseline_peak', type=float, default=31.01,
                       help='基准峰值')
    parser.add_argument('--num_episodes', type=int, default=3,
                       help='评估轮数')

    args = parser.parse_args()

    try:
        if args.method == 'rule_based':
            collect_rule_based_data(
                num_episodes=args.num_episodes,
                output_file=os.path.join(args.output_dir, 'rule_based_data.json')
            )
        elif args.method == 'independent':
            if not args.independent_model_dir or not os.path.exists(args.independent_model_dir):
                print(f"错误：独立基线模型目录不存在: {args.independent_model_dir}")
                print("请使用 --independent_model_dir 指定正确的模型目录")
            else:
                collect_independent_data(
                    num_episodes=args.num_episodes,
                    model_dir=args.independent_model_dir,
                    output_file=os.path.join(args.output_dir, 'independent_data.json')
                )
        elif args.method == 'mappo':
            if not args.mappo_model_dir or not os.path.exists(args.mappo_model_dir):
                print(f"错误：MAPPO模型目录不存在: {args.mappo_model_dir}")
                print("请使用 --mappo_model_dir 指定正确的模型目录")
            else:
                collect_mappo_data(
                    num_episodes=args.num_episodes,
                    model_dir=args.mappo_model_dir,
                    baseline_peak=args.baseline_peak,
                    output_file=os.path.join(args.output_dir, 'mappo_data.json')
                )
        elif args.method == 'all':
            collect_all_data(
                output_dir=args.output_dir,
                independent_model_dir=args.independent_model_dir,
                mappo_model_dir=args.mappo_model_dir,
                baseline_peak=args.baseline_peak,
                num_episodes=args.num_episodes
            )
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()