"""
参数扫描实验脚本（仅 MAPPO）
循环 (社区储能初始 SOC, 各家庭初始积分) 等参数跑 MAPPO 评估，记录净负荷峰值、成本及公平性等指标，输出 JSON/CSV 供绘图。

用法示例:
  python multi_agent/run_parameter_sweep.py --output_dir multi_agent/sweep_results
  python multi_agent/run_parameter_sweep.py --initial_soc 0 0.3 0.5 0.7 1.0 --initial_credit 50 100 200
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


# ---------- 默认参数网格 ----------
DEFAULT_INITIAL_SOC = [0.0, 0.3, 0.5, 0.7, 1.0]
DEFAULT_INITIAL_CREDIT = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
DEFAULT_NUM_EPISODES = 3

# 评估日期使用 2020 年 StoreNet 数据（不与训练日期 01-01~01-07 重叠）
EVAL_DATES = [
    '2020-01-08',  # 周三（工作日）
    '2020-01-09',  # 周四（工作日）
    '2020-01-11',  # 周六（双休日）
    '2020-01-12',  # 周日（双休日）
]
# 光伏系数列表长度必须与 EVAL_DATES 一致，每个日期对应一组 [agent1, agent2, agent3]
PV_COEFFICIENTS_LIST = [
    [3.0, 3.0, 3.0],  # 2020-01-08
    [1.0, 1.0, 1.0],  # 2020-01-09
    [3.0, 3.0, 3.0],  # 2020-01-11
    [2.0, 2.0, 2.0],  # 2020-01-12
]


def load_env_config(config_path='multi_agent/config.json'):
    """从 config 读取环境与训练相关参数。"""
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


def run_one_episode_mappo(env, mappo, initial_soc, initial_credit, date_index):
    """
    在给定 initial_soc、initial_credit 下跑一个 episode（MAPPO），返回该 episode 的指标。
    """
    env.credit_system.initial_credit = initial_credit
    # 关键修改：显式指定 house_index=0，使用训练住户 H1（确保数据存在）
    states = env.reset(mode='eval', date_index=date_index,
                       initial_community_soc=initial_soc, house_index=0)
    episode_cost = 0.0
    episode_agent_costs = [0.0, 0.0, 0.0]
    community_net_loads = []
    credit_balances_history = [[], [], []]
    step_count = 0
    done = False
    while not done and step_count < 48:
        actions, _ = mappo.select_actions(states, deterministic=True)
        next_states, rewards, dones, info = env.step(actions)
        community_net_loads.append(info.get('community_net_load', 0.0))
        for i in range(3):
            episode_agent_costs[i] += env.agents[i].env.current_step_cost
            credit_balances_history[i].append(env.credit_system.get_balance(i))
        episode_cost += sum(env.agents[i].env.current_step_cost for i in range(3))
        states = next_states
        done = all(dones)
        step_count += 1
    peak_load = max(community_net_loads) if community_net_loads else 0.0
    credit_final = [env.credit_system.get_balance(i) for i in range(3)]
    ess = env.community_ess
    ess_charge_total = sum(ess.charge_history) if getattr(ess, 'charge_history', None) else 0.0
    ess_discharge_total = sum(ess.discharge_history) if getattr(ess, 'discharge_history', None) else 0.0
    cost_std = float(np.std(episode_agent_costs)) if episode_agent_costs else 0.0
    mn, mx = min(episode_agent_costs), max(episode_agent_costs)
    cost_max_min_ratio = (mx / mn) if mn > 1e-9 else 999999.0
    return {
        'peak_load': peak_load,
        'total_cost': episode_cost,
        'agent_costs': list(episode_agent_costs),
        'cost_std': cost_std,
        'cost_max_min_ratio': cost_max_min_ratio,
        'credit_final': list(credit_final),
        'ess_charge_total': ess_charge_total,
        'ess_discharge_total': ess_discharge_total,
        'community_net_loads': community_net_loads,
    }


def run_sweep_mappo(initial_soc_list, initial_credit_list, num_episodes, baseline_peak,
                    mappo_model_dir, config_path, output_dir, output_prefix):
    """对 MAPPO 做参数扫描。"""
    cfg = load_env_config(config_path)

    # 显式指定住户列表，避免默认测试住户 H16 缺失
    train_house_ids = [f"H{i}" for i in range(1, 16)]   # H1~H15
    test_house_ids = [f"H{i}" for i in range(16, 21)]   # H16~H20

    env = MultiAgentHEMEnv(
        n_agents=3,
        community_ess_capacity=36.0,
        baseline_peak=baseline_peak,
        community_weight=cfg['community_weight'],
        community_credit_cost_weight=cfg['community_credit_cost_weight'],
        community_credit_benefit_weight=cfg['community_credit_benefit_weight'],
        initial_credit=cfg['initial_credit'],
        peak_penalty_exponent=cfg['peak_penalty_exponent'],
        peak_discharge_bonus=cfg['peak_discharge_bonus'],
        peak_credit_cost_reduction=cfg['peak_credit_cost_reduction'],
        pv_coefficients=[2.0, 2.0, 2.0],
        train_house_ids=train_house_ids,
        test_house_ids=test_house_ids,
    )
    env.set_evaluation_dates(EVAL_DATES, PV_COEFFICIENTS_LIST)

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
    mappo.load(mappo_model_dir)

    results = []
    total_combos = len(initial_soc_list) * len(initial_credit_list)
    idx = 0
    for initial_soc in initial_soc_list:
        for initial_credit in initial_credit_list:
            idx += 1
            print(f"[{idx}/{total_combos}] initial_soc={initial_soc}, initial_credit={initial_credit}")
            episodes_data = []
            for ep in range(num_episodes):
                date_index = ep % len(EVAL_DATES)
                ep_metrics = run_one_episode_mappo(env, mappo, initial_soc, initial_credit, date_index)
                episodes_data.append(ep_metrics)
            peak_loads = [e['peak_load'] for e in episodes_data]
            total_costs = [e['total_cost'] for e in episodes_data]
            agent_costs_list = [e['agent_costs'] for e in episodes_data]
            cost_stds = [e['cost_std'] for e in episodes_data]
            mean_agent_costs = np.mean(agent_costs_list, axis=0).tolist()
            results.append({
                'initial_soc': initial_soc,
                'initial_credit': initial_credit,
                'mean_peak_load': float(np.mean(peak_loads)),
                'std_peak_load': float(np.std(peak_loads)) if len(peak_loads) > 1 else 0.0,
                'mean_total_cost': float(np.mean(total_costs)),
                'std_total_cost': float(np.std(total_costs)) if len(total_costs) > 1 else 0.0,
                'mean_agent_costs': mean_agent_costs,
                'mean_cost_std': float(np.mean(cost_stds)),
                'episodes': episodes_data,
            })
    return results


def save_results(method, results, initial_soc_list, initial_credit_list, num_episodes,
                 output_dir, output_prefix):
    """保存 JSON（完整）和 CSV（扁平表，便于绘图）。"""
    os.makedirs(output_dir, exist_ok=True)
    prefix = output_prefix or 'sweep'
    out = {
        'method': method,
        'timestamp': datetime.now().isoformat(),
        'param_grid': {
            'initial_soc': initial_soc_list,
            'initial_credit': initial_credit_list,
            'num_episodes_per_combo': num_episodes,
        },
        'eval_dates': EVAL_DATES,
        'results': results,
    }
    json_path = os.path.join(output_dir, f'{prefix}_{method}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"已保存 JSON: {json_path}")
    # CSV: 每行 = 一个参数组合的汇总
    csv_path = os.path.join(output_dir, f'{prefix}_{method}.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("initial_soc,initial_credit,mean_peak_load,std_peak_load,mean_total_cost,std_total_cost,"
                "mean_agent1_cost,mean_agent2_cost,mean_agent3_cost,mean_cost_std\n")
        for r in results:
            mc = r.get('mean_agent_costs', [0, 0, 0])
            f.write(f"{r['initial_soc']},{r['initial_credit']},"
                    f"{r['mean_peak_load']:.4f},{r['std_peak_load']:.4f},"
                    f"{r['mean_total_cost']:.4f},{r['std_total_cost']:.4f},"
                    f"{mc[0]:.4f},{mc[1]:.4f},{mc[2]:.4f},{r.get('mean_cost_std', 0):.4f}\n")
    print(f"已保存 CSV: {csv_path}")
    # 每 episode 一行 CSV，便于画散点或箱线
    csv_ep_path = os.path.join(output_dir, f'{prefix}_{method}_episodes.csv')
    with open(csv_ep_path, 'w', encoding='utf-8') as f:
        f.write("initial_soc,initial_credit,episode_id,peak_load,total_cost,agent1_cost,agent2_cost,agent3_cost,cost_std\n")
        for r in results:
            for ep_id, ep in enumerate(r.get('episodes', [])):
                ac = ep.get('agent_costs', [0, 0, 0])
                f.write(f"{r['initial_soc']},{r['initial_credit']},{ep_id},"
                        f"{ep['peak_load']:.4f},{ep['total_cost']:.4f},"
                        f"{ac[0]:.4f},{ac[1]:.4f},{ac[2]:.4f},{ep.get('cost_std', 0):.4f}\n")
    print(f"已保存 episodes CSV: {csv_ep_path}")


def main():
    parser = argparse.ArgumentParser(description='参数扫描（仅 MAPPO）：循环 initial_soc / initial_credit 跑评估并记录指标')
    parser.add_argument('--initial_soc', type=float, nargs='+', default=DEFAULT_INITIAL_SOC,
                        help='社区储能初始 SOC 列表，如 0 0.3 0.5 0.7 1.0')
    parser.add_argument('--initial_credit', type=float, nargs='+', default=DEFAULT_INITIAL_CREDIT,
                        help='各家庭初始积分列表，如 50 100 200')
    parser.add_argument('--num_episodes', type=int, default=DEFAULT_NUM_EPISODES,
                        help='每个参数组合跑的 episode 数（不同日期）')
    parser.add_argument('--baseline_peak', type=float, default=25.0, help='基准峰值（与训练一致）')
    parser.add_argument('--mappo_model_dir', type=str, default='multi_agent/algorithms/models',
                        help='MAPPO 模型目录')
    parser.add_argument('--config_path', type=str, default='multi_agent/config.json')
    parser.add_argument('--output_dir', type=str, default='multi_agent/sweep_results',
                        help='结果输出目录')
    parser.add_argument('--output_prefix', type=str, default='sweep',
                        help='输出文件名前缀，如 sweep -> sweep_mappo.json / sweep_mappo.csv')
    args = parser.parse_args()
    initial_soc_list = sorted(set(args.initial_soc))
    initial_credit_list = sorted(set(args.initial_credit))
    print("参数网格: initial_soc =", initial_soc_list, ", initial_credit =", initial_credit_list)
    print("每个组合 episode 数:", args.num_episodes)
    results = run_sweep_mappo(
        initial_soc_list, initial_credit_list, args.num_episodes, args.baseline_peak,
        args.mappo_model_dir, args.config_path, args.output_dir, args.output_prefix,
    )
    save_results(
        'mappo', results, initial_soc_list, initial_credit_list, args.num_episodes,
        args.output_dir, args.output_prefix,
    )
    print("参数扫描完成。")


if __name__ == '__main__':
    main()