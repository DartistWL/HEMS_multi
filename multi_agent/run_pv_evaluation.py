"""
光伏场景评估：在不同光伏（天气）条件下评估 MAPPO 表现
- 全社区同一天气：sunny / cloudy / normal，记录社区峰值、总成本。
- 异质光伏：户0=高光伏、户1=中光伏、户2=低光伏，供论文对比高光伏户与低光伏户的户均成本与公平性。

用法:
  python multi_agent/run_pv_evaluation.py --output_dir multi_agent/pv_eval_results
  python multi_agent/run_pv_evaluation.py --num_episodes 5
  python multi_agent/run_pv_evaluation.py --debug   # 打印每场景光伏系数与中午PV发电量
"""
import sys
import os
import json
import argparse
import numpy as np
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from multi_agent.environment.multi_agent_env import MultiAgentHEMEnv
from multi_agent.algorithms.mappo import MAPPO
from multi_agent.utils.data_interface import MultiAgentDataInterface
from multi_agent.baselines.independent_baseline import IndependentBaseline


EVAL_DATE = '2011-07-10'  # 兼容旧逻辑单日评估
# 评估日期与训练日（07-03～07-09）不重叠，避免在训练日上评估泛化性
# 工作日：周一、周二（EV 白天外出）；双休日：周日（EV 全天在家）
EVAL_DATES_WEEKDAY = ['2011-07-11', '2011-07-12']   # 周一、周二
EVAL_DATES_WEEKEND = ['2011-07-10', '2011-07-17']   # 周日（07-10 与 07-17）
DEFAULT_NUM_EPISODES = 10  # 多做几次实验，使行为与积分更稳定、更易体现高光伏向社区储能充电
DEFAULT_NUM_RUNS = 1      # 评估轮数，1=不重复、不取平均；>1 时多轮取均值与标准差

# 每步时长（小时）：与 env 一致，48 步/天 => 0.5h/步。若 env 改为 24 步/天则应为 1.0。
STEP_HOURS_PER_DAY = 24.0
EXPECTED_STEPS_PER_DAY = 48


def load_env_config(config_path='multi_agent/config.json'):
    """从 config 读取环境参数。"""
    defaults = {
        'community_weight': 0.5,
        'community_credit_cost_weight': 0.05,
        'community_credit_benefit_weight': 0.05,
        'initial_credit': 100.0,
        'peak_penalty_exponent': 3.0,
        'peak_discharge_bonus': 0.5,
        'peak_credit_cost_reduction': 0.3,
        'baseline_peak': 25.0,
    }
    if not os.path.exists(config_path):
        return defaults
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        training = cfg.get('training', {})
        for k in defaults:
            if k in training:
                defaults[k] = training[k]
        return defaults
    except Exception:
        return defaults


def run_one_episode_mappo(env, mappo, date_index, collect_behavior=False):
    """在当前 env 的评估日期与光伏设置下跑一个 episode，返回指标；可选收集行为（社区储能充放电、PV 流向、积分净变化）。"""
    states = env.reset(mode='eval', date_index=date_index)
    episode_cost = 0.0
    episode_agent_costs = [0.0, 0.0, 0.0]
    episode_agent_credit_change = [0.0, 0.0, 0.0]  # 每户本 episode 积分净变化（正=收益，负=支出）
    community_net_loads = []
    agent_community_charge = [[], [], []]
    agent_community_discharge = [[], [], []]
    agent_pv_flows = [[], [], []]
    step_count = 0
    done = False
    while not done and step_count < 48:
        actions, _ = mappo.select_actions(states, deterministic=True)
        next_states, rewards, dones, info = env.step(actions)
        community_net_loads.append(info.get('community_net_load', 0.0))
        for i in range(3):
            episode_agent_costs[i] += env.agents[i].env.current_step_cost
        episode_cost += sum(env.agents[i].env.current_step_cost for i in range(3))
        for tr in info.get('credit_transactions', []):
            if tr.get('success') and 'transaction' in tr:
                aid = tr['transaction'].get('agent_id', -1)
                if 0 <= aid < 3:
                    episode_agent_credit_change[aid] += tr.get('credit_change', 0.0)
        if collect_behavior:
            for i in range(3):
                agent_community_charge[i].append(info.get('agent_community_charge', [0, 0, 0])[i] if i < len(info.get('agent_community_charge', [])) else 0.0)
                agent_community_discharge[i].append(info.get('agent_community_discharge', [0, 0, 0])[i] if i < len(info.get('agent_community_discharge', [])) else 0.0)
                agent_pv_flows[i].append((info.get('agent_pv_flows', [{}] * 3)[i] if i < len(info.get('agent_pv_flows', [])) else {}))
        states = next_states
        done = all(dones)
        step_count += 1
    peak_load = max(community_net_loads) if community_net_loads else 0.0
    cost_std = float(np.std(episode_agent_costs)) if episode_agent_costs else 0.0
    out = {
        'peak_load': peak_load,
        'total_cost': episode_cost,
        'agent_costs': list(episode_agent_costs),
        'cost_std': cost_std,
        'agent_credit_change': list(episode_agent_credit_change),
    }
    if collect_behavior:
        # 每户向社区储能充电/放电：info 中为每步功率(kW)，总能量 = sum(功率)×步长(h)。
        # 步长用实际收集的步数反推：一日 24h，n_steps 步 => dt = 24/n_steps（避免历史 24 步/天时误用 0.5 导致结果偏小 2 倍）。
        n_steps = len(agent_community_charge[0]) if agent_community_charge else 0
        step_dt = (STEP_HOURS_PER_DAY / n_steps) if n_steps > 0 else (STEP_HOURS_PER_DAY / EXPECTED_STEPS_PER_DAY)
        out['agent_community_charge_kwh'] = [sum(agent_community_charge[i]) * step_dt for i in range(3)]
        out['agent_community_discharge_kwh'] = [sum(agent_community_discharge[i]) * step_dt for i in range(3)]
        out['agent_pv_flows'] = agent_pv_flows  # 3 agents × 48 steps，每步为 dict(direct_use, ess_charge, ev_charge, community_charge, grid_sell)
        out['_n_steps'] = n_steps
        out['_step_dt'] = step_dt
    return out


def run_episodes_for_dates(env, mappo, dates_list, pv_list, num_episodes, collect_behavior=False):
    """在给定日期列表上跑 num_episodes（按日期轮转 date_index），汇总峰值、成本等；返回单组汇总与可选的一 episode 行为。"""
    env.set_evaluation_dates(dates_list, pv_list)
    episodes_data = []
    for ep in range(num_episodes):
        date_index = ep % len(dates_list)
        ep_metrics = run_one_episode_mappo(env, mappo, date_index=date_index, collect_behavior=collect_behavior)
        episodes_data.append(ep_metrics)
    peak_loads = [e['peak_load'] for e in episodes_data]
    total_costs = [e['total_cost'] for e in episodes_data]
    agent_costs_list = [e['agent_costs'] for e in episodes_data]
    cost_stds = [e['cost_std'] for e in episodes_data]
    mean_agent_costs = np.mean(agent_costs_list, axis=0).tolist()
    charge_list = [e.get('agent_community_charge_kwh', [0, 0, 0]) for e in episodes_data]
    discharge_list = [e.get('agent_community_discharge_kwh', [0, 0, 0]) for e in episodes_data]
    credit_list = [e.get('agent_credit_change', [0, 0, 0]) for e in episodes_data]
    # 对 num_episodes 个 episode 取平均，得到户均日充电/放电量 (kWh/天)
    mean_agent_community_charge = np.mean(charge_list, axis=0).tolist() if charge_list else [0, 0, 0]
    mean_agent_community_discharge = np.mean(discharge_list, axis=0).tolist() if discharge_list else [0, 0, 0]
    mean_agent_credit_change = np.mean(credit_list, axis=0).tolist() if credit_list else [0, 0, 0]
    out = {
        'mean_peak_load': float(np.mean(peak_loads)),
        'std_peak_load': float(np.std(peak_loads)) if len(peak_loads) > 1 else 0.0,
        'mean_total_cost': float(np.mean(total_costs)),
        'std_total_cost': float(np.std(total_costs)) if len(total_costs) > 1 else 0.0,
        'mean_agent_costs': mean_agent_costs,
        'mean_cost_std': float(np.mean(cost_stds)),
        'mean_agent_community_charge': mean_agent_community_charge,
        'mean_agent_community_discharge': mean_agent_community_discharge,
        'mean_agent_credit_change': mean_agent_credit_change,
    }
    if collect_behavior and episodes_data and 'agent_pv_flows' in episodes_data[0]:
        out['one_episode_agent_pv_flows'] = episodes_data[0]['agent_pv_flows']
        out['_n_steps'] = episodes_data[0].get('_n_steps')
        out['_step_dt'] = episodes_data[0].get('_step_dt')
    return out


def _aggregate_pv_flows_across_runs(list_of_pv_flows):
    """将多轮的 one_episode_agent_pv_flows（每轮 3×48 的 dict 列表）按步、按键取平均，返回一组 3×48 的均值。"""
    if not list_of_pv_flows:
        return None
    n_agents = len(list_of_pv_flows[0])
    n_steps = len(list_of_pv_flows[0][0]) if n_agents else 0
    keys = ['direct_use', 'ess_charge', 'ev_charge', 'community_charge', 'grid_sell']
    aggregated = []
    for i in range(n_agents):
        agent_flows = []
        for t in range(n_steps):
            vals = [run[i][t].get(k, 0) for run in list_of_pv_flows for k in keys]
            # 按键对齐：同一 (run, step) 的 dict 取各 key 的均值
            mean_dict = {}
            for k in keys:
                mean_dict[k] = float(np.mean([run[i][t].get(k, 0) for run in list_of_pv_flows]))
            agent_flows.append(mean_dict)
        aggregated.append(agent_flows)
    return aggregated


def run_episodes_for_dates_independent(baseline, dates_list, pv_list, num_episodes, collect_behavior=False):
    """独立基线：在给定日期列表上跑 num_episodes，汇总后返回单组汇总。"""
    env = baseline.eval_ma_env
    env.set_evaluation_dates(dates_list, pv_list)
    episodes_data = []
    for ep in range(num_episodes):
        date_index = ep % len(dates_list)
        ep_metrics = run_one_episode_independent(baseline, date_index=date_index, collect_behavior=collect_behavior)
        episodes_data.append(ep_metrics)
    peak_loads = [e['peak_load'] for e in episodes_data]
    total_costs = [e['total_cost'] for e in episodes_data]
    agent_costs_list = [e['agent_costs'] for e in episodes_data]
    cost_stds = [e['cost_std'] for e in episodes_data]
    mean_agent_costs = np.mean(agent_costs_list, axis=0).tolist()
    charge_list = [e.get('agent_community_charge_kwh', [0, 0, 0]) for e in episodes_data]
    discharge_list = [e.get('agent_community_discharge_kwh', [0, 0, 0]) for e in episodes_data]
    credit_list = [e.get('agent_credit_change', [0, 0, 0]) for e in episodes_data]
    # 对 num_episodes 个 episode 取平均，得到户均日充电/放电量 (kWh/天)
    mean_agent_community_charge = np.mean(charge_list, axis=0).tolist() if charge_list else [0, 0, 0]
    mean_agent_community_discharge = np.mean(discharge_list, axis=0).tolist() if discharge_list else [0, 0, 0]
    mean_agent_credit_change = np.mean(credit_list, axis=0).tolist() if credit_list else [0, 0, 0]
    return {
        'mean_peak_load': float(np.mean(peak_loads)),
        'std_peak_load': float(np.std(peak_loads)) if len(peak_loads) > 1 else 0.0,
        'mean_total_cost': float(np.mean(total_costs)),
        'std_total_cost': float(np.std(total_costs)) if len(total_costs) > 1 else 0.0,
        'mean_agent_costs': mean_agent_costs,
        'mean_cost_std': float(np.mean(cost_stds)),
        'mean_agent_community_charge': mean_agent_community_charge,
        'mean_agent_community_discharge': mean_agent_community_discharge,
        'mean_agent_credit_change': mean_agent_credit_change,
    }


def run_one_episode_independent(baseline, date_index, collect_behavior=False):
    """在 baseline 的 eval_ma_env 上跑一个 episode（已通过 set_evaluation_dates 设好光伏），返回与 MAPPO 相同的指标；可选收集行为。"""
    env = baseline.eval_ma_env
    states = env.reset(mode='eval', date_index=date_index)
    episode_cost = 0.0
    episode_agent_costs = [0.0, 0.0, 0.0]
    episode_agent_credit_change = [0.0, 0.0, 0.0]
    community_net_loads = []
    agent_community_charge = [[], [], []]
    agent_community_discharge = [[], [], []]
    agent_pv_flows = [[], [], []]
    for _ in range(48):
        actions = [baseline.agents[i].take_action(states[i]) for i in range(3)]
        next_states, rewards, dones, info = env.step(actions)
        community_net_loads.append(info.get('community_net_load', 0.0))
        for i in range(3):
            episode_agent_costs[i] += env.agents[i].env.current_step_cost
        for tr in info.get('credit_transactions', []):
            if tr.get('success') and 'transaction' in tr:
                aid = tr['transaction'].get('agent_id', -1)
                if 0 <= aid < 3:
                    episode_agent_credit_change[aid] += tr.get('credit_change', 0.0)
        if collect_behavior:
            for i in range(3):
                agent_community_charge[i].append(info.get('agent_community_charge', [0, 0, 0])[i] if i < len(info.get('agent_community_charge', [])) else 0.0)
                agent_community_discharge[i].append(info.get('agent_community_discharge', [0, 0, 0])[i] if i < len(info.get('agent_community_discharge', [])) else 0.0)
                agent_pv_flows[i].append((info.get('agent_pv_flows', [{}] * 3)[i] if i < len(info.get('agent_pv_flows', [])) else {}))
        states = next_states
    peak_load = max(community_net_loads) if community_net_loads else 0.0
    cost_std = float(np.std(episode_agent_costs))
    out = {
        'peak_load': peak_load,
        'total_cost': sum(episode_agent_costs),
        'agent_costs': list(episode_agent_costs),
        'cost_std': cost_std,
        'agent_credit_change': list(episode_agent_credit_change),
    }
    if collect_behavior:
        n_steps = len(agent_community_charge[0]) if agent_community_charge else 0
        step_dt = (STEP_HOURS_PER_DAY / n_steps) if n_steps > 0 else (STEP_HOURS_PER_DAY / EXPECTED_STEPS_PER_DAY)
        out['agent_community_charge_kwh'] = [sum(agent_community_charge[i]) * step_dt for i in range(3)]
        out['agent_community_discharge_kwh'] = [sum(agent_community_discharge[i]) * step_dt for i in range(3)]
        out['agent_pv_flows'] = agent_pv_flows
    return out


def main():
    parser = argparse.ArgumentParser(description='光伏场景评估：不同天气下 MAPPO 表现')
    parser.add_argument('--num_episodes', type=int, default=DEFAULT_NUM_EPISODES,
                        help='每个光伏场景跑的 episode 数')
    parser.add_argument('--num_runs', type=int, default=DEFAULT_NUM_RUNS,
                        help='整轮评估重复次数，多轮取平均后绘图更具说服力')
    parser.add_argument('--baseline_peak', type=float, default=25.0)
    parser.add_argument('--mappo_model_dir', type=str, default='multi_agent/algorithms/models')
    parser.add_argument('--config_path', type=str, default='multi_agent/config.json')
    parser.add_argument('--output_dir', type=str, default='multi_agent/pv_eval_results')
    parser.add_argument('--output_prefix', type=str, default='pv_eval')
    parser.add_argument('--debug', action='store_true', help='打印每场景下光伏系数与中午PV发电量，用于验证系数是否生效')
    parser.add_argument('--no_compare_independent', dest='compare_independent', action='store_false',
                        help='不跑独立基线对比（默认会跑）')
    parser.set_defaults(compare_independent=True)
    args = parser.parse_args()
    
    cfg = load_env_config(args.config_path)
    sunny = MultiAgentDataInterface.get_weather_coefficient('sunny')
    cloudy = MultiAgentDataInterface.get_weather_coefficient('cloudy')
    normal = MultiAgentDataInterface.get_weather_coefficient('normal')
    
    # 场景：全社区同一天气（晴天/阴天/正常）+ 异质光伏（高/中/低光伏户）
    # 异质光伏：agent0=高光伏(sunny), agent1=中光伏(normal), agent2=低光伏(cloudy)，用于论文对比高光伏户与低光伏户
    scenarios = [
        {'name': 'sunny', 'pv_coef': sunny, 'pv_list': [[sunny, sunny, sunny]]},
        {'name': 'cloudy', 'pv_coef': cloudy, 'pv_list': [[cloudy, cloudy, cloudy]]},
        {'name': 'normal', 'pv_coef': normal, 'pv_list': [[normal, normal, normal]]},
        {'name': 'heterogeneous', 'pv_coef': normal, 'pv_list': [[sunny, normal, cloudy]]},  # 高/中/低光伏户
    ]
    
    env = MultiAgentHEMEnv(
        n_agents=3,
        community_ess_capacity=36.0,
        baseline_peak=args.baseline_peak,
        community_weight=cfg['community_weight'],
        community_credit_cost_weight=cfg['community_credit_cost_weight'],
        community_credit_benefit_weight=cfg['community_credit_benefit_weight'],
        initial_credit=cfg['initial_credit'],
        peak_penalty_exponent=cfg['peak_penalty_exponent'],
        peak_discharge_bonus=cfg['peak_discharge_bonus'],
        peak_credit_cost_reduction=cfg['peak_credit_cost_reduction'],
        pv_coefficients=[2.0, 2.0, 2.0],
    )
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
        reward_scale=10.0,
    )
    model_dir = args.mappo_model_dir
    if os.path.exists(args.config_path):
        try:
            with open(args.config_path, 'r', encoding='utf-8') as f:
                cfg_json = json.load(f)
            if (cfg_json.get('credit_pricing') or {}).get('scheme') == 'contribution_based':
                resolved = model_dir.rstrip(os.sep) + '_contribution_based'
                if os.path.exists(resolved):
                    model_dir = resolved
                    print(f"根据 config 使用模型目录: {model_dir}")
        except Exception:
            pass
    mappo.load(model_dir)
    
    runs_results = []   # 每轮 (results, results_independent)
    for run_id in range(args.num_runs):
        if args.num_runs > 1:
            print(f"\n===== 第 {run_id + 1}/{args.num_runs} 轮评估 =====")
        results = []
        for scenario in scenarios:
            name = scenario['name']
            pv_list = scenario['pv_list']
            if name == 'heterogeneous':
                print(f"场景: {name} (光伏系数 户0/户1/户2={pv_list[0][0]:.1f}/{pv_list[0][1]:.1f}/{pv_list[0][2]:.1f} 即 高/中/低)")
            else:
                print(f"场景: {name} (光伏系数={pv_list[0][0]:.1f})")
            if args.debug:
                env.set_evaluation_dates([EVAL_DATE], pv_list)
                env.reset(mode='eval', date_index=0)
                for i in range(3):
                    di = env.agents[i].env.data_interface
                    pv_noon = di.get_pv_generation(EVAL_DATE, 24)
                    print(f"  agent{i}: pv_coef={di.pv_coefficient}, PV(12时)={pv_noon:.4f} kW")
            # 工作日与双休日分开评估
            print(f"  评估工作日 ({EVAL_DATES_WEEKDAY}) ...")
            weekday_out = run_episodes_for_dates(env, mappo, EVAL_DATES_WEEKDAY, pv_list, args.num_episodes, collect_behavior=True)
            if args.debug and name == scenarios[0]['name']:
                print(f"  [充放电诊断] n_steps={weekday_out.get('_n_steps')}, step_dt={weekday_out.get('_step_dt')} h, 户均日充电(kWh)={weekday_out.get('mean_agent_community_charge')}, 户均日放电(kWh)={weekday_out.get('mean_agent_community_discharge')}")
            print(f"  评估双休日 ({EVAL_DATES_WEEKEND}) ...")
            weekend_out = run_episodes_for_dates(env, mappo, EVAL_DATES_WEEKEND, pv_list, args.num_episodes, collect_behavior=True)
            rec = {
                'scenario': name,
                'pv_coef': scenario['pv_coef'],
                'weekday': weekday_out,
                'weekend': weekend_out,
            }
            if name == 'heterogeneous' and weekend_out.get('one_episode_agent_pv_flows'):
                rec['one_episode_agent_pv_flows'] = weekend_out['one_episode_agent_pv_flows']  # 绘图用：异质光伏下高/中/低户 PV 流向（取双休日一 episode）
            results.append(rec)

        results_independent = []
        if args.compare_independent:
            print("\n" + "=" * 60)
            print("运行独立基线（相同光伏场景）用于对比")
            print("=" * 60)
            try:
                baseline = IndependentBaseline(use_community_env=True)
                baseline.load_models()
                if not baseline.use_community_env:
                    print("警告: 独立基线未使用社区环境，跳过对比。请在 config 中设置 independent_baseline.use_community_env=true 并重训。")
                else:
                    for scenario in scenarios:
                        name = scenario['name']
                        pv_list = scenario['pv_list']
                        print(f"独立基线 场景: {name} (工作日+双休日)")
                        weekday_ind = run_episodes_for_dates_independent(baseline, EVAL_DATES_WEEKDAY, pv_list, args.num_episodes, collect_behavior=True)
                        weekend_ind = run_episodes_for_dates_independent(baseline, EVAL_DATES_WEEKEND, pv_list, args.num_episodes, collect_behavior=True)
                        results_independent.append({
                            'scenario': name,
                            'pv_coef': scenario['pv_coef'],
                            'weekday': weekday_ind,
                            'weekend': weekend_ind,
                        })
            except Exception as e:
                print(f"独立基线对比失败: {e}")
                import traceback
                traceback.print_exc()

        runs_results.append((results, results_independent))

    # 多轮汇总：对每场景、每 date_type 的指标取均值和标准差（跨轮）
    def agg(w_list):
        """w_list: 多轮的同结构 dict 列表，返回 mean 与 std（标量或列表逐项）。"""
        if not w_list:
            return {}, {}
        mean_out = {}
        std_out = {}
        for k in w_list[0].keys():
            if k == 'one_episode_agent_pv_flows' or k in ('_n_steps', '_step_dt'):
                continue
            vals = [w.get(k) for w in w_list]
            if vals[0] is None:
                mean_out[k] = None
                std_out[k] = 0.0
            elif isinstance(vals[0], (list, np.ndarray)):
                arr = np.array(vals)
                mean_out[k] = np.mean(arr, axis=0).tolist()
                std_out[k] = np.std(arr, axis=0).tolist() if len(vals) > 1 else [0.0] * len(vals[0])
            else:
                try:
                    a = np.array(vals, dtype=float)
                    mean_out[k] = float(np.mean(a))
                    std_out[k] = float(np.std(a)) if len(a) > 1 else 0.0
                except (TypeError, ValueError):
                    mean_out[k] = vals[0]
                    std_out[k] = 0.0
        return mean_out, std_out

    results = []
    results_independent = []
    scenario_names = [s['name'] for s in scenarios]
    for idx, name in enumerate(scenario_names):
        # MAPPO 汇总
        runs_mappo = [run_res[0][idx] for run_res in runs_results]
        wd_list = [r.get('weekday', {}) for r in runs_mappo]
        we_list = [r.get('weekend', {}) for r in runs_mappo]
        wd_mean, wd_std = agg([{k: v for k, v in w.items() if k != 'one_episode_agent_pv_flows'} for w in wd_list])
        we_mean, we_std = agg([{k: v for k, v in w.items() if k != 'one_episode_agent_pv_flows'} for w in we_list])
        rec = {
            'scenario': name,
            'pv_coef': scenarios[idx]['pv_coef'],
            'weekday': {
                **wd_mean,
                'std_peak_load': wd_std.get('mean_peak_load', 0),
                'std_total_cost': wd_std.get('mean_total_cost', 0),
            },
            'weekend': {
                **we_mean,
                'std_peak_load': we_std.get('mean_peak_load', 0),
                'std_total_cost': we_std.get('mean_total_cost', 0),
            },
        }
        if name == 'heterogeneous':
            pv_flows_list = [r.get('one_episode_agent_pv_flows') for r in runs_mappo if r.get('one_episode_agent_pv_flows')]
            if pv_flows_list:
                rec['one_episode_agent_pv_flows'] = _aggregate_pv_flows_across_runs(pv_flows_list)
        results.append(rec)

    if args.compare_independent and runs_results and runs_results[0][1]:
        for idx, name in enumerate(scenario_names):
            runs_ind = [run_res[1][idx] for run_res in runs_results if len(run_res[1]) > idx]
            if not runs_ind:
                continue
            wd_list = [r.get('weekday', {}) for r in runs_ind]
            we_list = [r.get('weekend', {}) for r in runs_ind]
            wd_mean, wd_std = agg(wd_list)
            we_mean, we_std = agg(we_list)
            results_independent.append({
                'scenario': name,
                'pv_coef': scenarios[idx]['pv_coef'],
                'weekday': {
                    **wd_mean,
                    'std_peak_load': wd_std.get('mean_peak_load', 0),
                    'std_total_cost': wd_std.get('mean_total_cost', 0),
                },
                'weekend': {
                    **we_mean,
                    'std_peak_load': we_std.get('mean_peak_load', 0),
                    'std_total_cost': we_std.get('mean_total_cost', 0),
                },
            })

    os.makedirs(args.output_dir, exist_ok=True)
    prefix = args.output_prefix or 'pv_eval'
    # 写入 JSON：每场景含 weekday/weekend 两组汇总；保留 one_episode_agent_pv_flows 供绘图
    results_for_json = []
    for r in results:
        rj = dict(r)
        # 从 weekday/weekend 中移除 one_episode_agent_pv_flows 避免重复（已在 rec 顶层为 heterogeneous）
        if 'weekday' in rj and isinstance(rj['weekday'], dict):
            rj['weekday'] = {k: v for k, v in rj['weekday'].items() if k != 'one_episode_agent_pv_flows'}
        if 'weekend' in rj and isinstance(rj['weekend'], dict):
            rj['weekend'] = {k: v for k, v in rj['weekend'].items() if k != 'one_episode_agent_pv_flows'}
        results_for_json.append(rj)
    out = {
        'timestamp': datetime.now().isoformat(),
        'eval_dates_weekday': EVAL_DATES_WEEKDAY,
        'eval_dates_weekend': EVAL_DATES_WEEKEND,
        'num_episodes_per_scenario': args.num_episodes,
        'num_runs': args.num_runs,
        'scenarios': [s['name'] for s in scenarios],
        'results': results_for_json,
        'results_independent': results_independent if results_independent else None,
    }
    json_path = os.path.join(args.output_dir, f'{prefix}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"已保存 JSON: {json_path}")

    csv_path = os.path.join(args.output_dir, f'{prefix}.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("scenario,date_type,method,pv_coef,mean_peak_load,std_peak_load,mean_total_cost,std_total_cost,"
                "mean_agent1_cost,mean_agent2_cost,mean_agent3_cost,mean_cost_std\n")
        for r in results:
            for date_type, label in [('weekday', '工作日'), ('weekend', '双休日')]:
                w = r.get(date_type, {})
                mc = w.get('mean_agent_costs', [0, 0, 0])
                f.write(f"{r['scenario']},{label},mappo,{r['pv_coef']:.2f},"
                        f"{w.get('mean_peak_load', 0):.4f},{w.get('std_peak_load', 0):.4f},"
                        f"{w.get('mean_total_cost', 0):.4f},{w.get('std_total_cost', 0):.4f},"
                        f"{mc[0]:.4f},{mc[1]:.4f},{mc[2]:.4f},{w.get('mean_cost_std', 0):.4f}\n")
        for r in results_independent:
            for date_type, label in [('weekday', '工作日'), ('weekend', '双休日')]:
                w = r.get(date_type, {})
                mc = w.get('mean_agent_costs', [0, 0, 0])
                f.write(f"{r['scenario']},{label},independent,{r['pv_coef']:.2f},"
                        f"{w.get('mean_peak_load', 0):.4f},{w.get('std_peak_load', 0):.4f},"
                        f"{w.get('mean_total_cost', 0):.4f},{w.get('std_total_cost', 0):.4f},"
                        f"{mc[0]:.4f},{mc[1]:.4f},{mc[2]:.4f},{w.get('mean_cost_std', 0):.4f}\n")
    print(f"已保存 CSV: {csv_path}")
    print("光伏场景评估完成。")


if __name__ == '__main__':
    main()
