"""
MAPPO评估脚本（支持多户测试）
Evaluate trained MAPPO model on test households
"""
import sys
import os
import json
import numpy as np
import argparse

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from multi_agent.algorithms.mappo import MAPPO
from multi_agent.environment.multi_agent_env import MultiAgentHEMEnv
from multi_agent.utils.data_interface import MultiAgentDataInterface
from multi_agent.algorithms.action_utils import state_dict_to_vector


def _resolve_mappo_model_dir(model_dir, config_path='multi_agent/config.json'):
    """根据 config 的 credit_pricing.scheme 解析 MAPPO 模型目录"""
    default_dir = 'multi_agent/algorithms/models'
    if model_dir != default_dir:
        return model_dir
    if not os.path.exists(config_path):
        return model_dir
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        if (cfg.get('credit_pricing') or {}).get('scheme') == 'contribution_based':
            resolved = model_dir.rstrip(os.sep) + '_contribution_based'
            if os.path.exists(resolved):
                return resolved
    except Exception:
        pass
    return model_dir


def evaluate_mappo(model_dir='multi_agent/algorithms/models', baseline_peak=31.01,
                   num_episodes=10, mode='eval', config_path='multi_agent/config.json'):
    """
    评估MAPPO模型（使用测试住户和2020年日期）
    """
    model_dir = _resolve_mappo_model_dir(model_dir, config_path)
    print("=" * 80)
    print("Evaluating MAPPO Model on Test Households")
    print("=" * 80)
    print(f"Model directory: {model_dir}")
    print(f"Baseline peak: {baseline_peak} kW")
    print(f"Number of episodes: {num_episodes}")
    print(f"Mode: {mode}")
    print("=" * 80)

    # 定义训练和测试住户（与训练一致）
    train_house_ids = [f"H{i}" for i in range(1, 16)]  # 仅用于环境创建
    test_house_ids = [f"H{i}" for i in range(16, 21)]  # H16~H20

    # 创建环境（传入住户列表）
    env = MultiAgentHEMEnv(
        n_agents=3,
        community_ess_capacity=36.0,
        baseline_peak=baseline_peak,
        community_weight=0.2,
        community_credit_cost_weight=0.1,
        community_credit_benefit_weight=0.1,
        initial_credit=100.0,
        peak_penalty_exponent=2.0,
        peak_discharge_bonus=0.0,
        peak_credit_cost_reduction=1.0,
        pv_coefficients=[2.0, 2.0, 2.0],
        train_house_ids=train_house_ids,
        test_house_ids=test_house_ids
    )

    # 设置评估日期（2020年1月8-10日，共3天）
    eval_dates = ['2020-01-08', '2020-01-09', '2020-01-10']
    # 获取天气系数（与训练一致）
    sunny_coef = MultiAgentDataInterface.get_weather_coefficient('sunny')
    cloudy_coef = MultiAgentDataInterface.get_weather_coefficient('cloudy')
    normal_coef = MultiAgentDataInterface.get_weather_coefficient('normal')
    pv_coefficients_list = [
        [sunny_coef, sunny_coef, sunny_coef],  # 晴天
        [cloudy_coef, cloudy_coef, cloudy_coef],  # 阴天
        [normal_coef, normal_coef, normal_coef],  # 正常
    ]
    env.set_evaluation_dates(eval_dates, pv_coefficients_list)

    # 手动获取一次状态以推断维度（避免在 MAPPO 初始化中 reset 时出错）
    sample_states = env.reset(mode='eval', date_index=0, house_index=0)
    local_state_dim = len(state_dict_to_vector(sample_states[0]))
    global_state_dim = sum(len(state_dict_to_vector(s)) for s in sample_states)
    print(f"实际局部状态维度: {local_state_dim}")
    print(f"实际全局状态维度: {global_state_dim}")

    # 创建MAPPO算法实例（与训练时超参一致）
    mappo = MAPPO(
        env=env,
        n_agents=3,
        local_state_dim=local_state_dim,
        global_state_dim=global_state_dim,
        hidden_dim=128,
        gamma=0.96,
        lmbda=0.95,
        eps=0.2,
        epochs=3,  # 评估时不需要更新，仅用于加载模型
        lr_actor=1e-4,
        lr_critic=1e-4,
        batch_size=32,
        ent_coef=0.05,
        max_grad_norm=1.0,
        use_state_normalization=False,
        use_popart=False,
        reward_scale=1.0,
        use_reward_normalization=False,
        critic_loss_type='mse'
    )

    # 加载模型
    print(f"\nLoading model from {model_dir}...")
    if not os.path.exists(model_dir):
        print(f"Error: Model directory {model_dir} does not exist!")
        return None

    mappo.load(model_dir)
    print("Model loaded successfully!")

    # 评估结果
    results = {
        'returns': [],
        'costs': [],
        'peak_loads': [],
        'peak_penalties': [],
        'agent_returns': [[] for _ in range(3)],
        'agent_costs': [[] for _ in range(3)]
    }

    num_eval_dates = len(eval_dates)
    num_test_houses = len(test_house_ids)

    print("\n" + "=" * 80)
    print("Starting Evaluation...")
    print("=" * 80)

    for episode in range(num_episodes):
        date_index = episode % num_eval_dates
        house_index = episode % num_test_houses  # 循环使用测试住户
        date = eval_dates[date_index]
        pv_coeff = pv_coefficients_list[date_index]

        # 重置环境（传入 house_index）
        states = env.reset(mode=mode, date_index=date_index, house_index=house_index)

        episode_return = 0.0
        episode_cost = 0.0
        episode_peak_load = 0.0
        episode_peak_penalty = 0.0
        agent_returns = [0.0] * 3
        agent_costs = [0.0] * 3
        step_count = 0
        done = False

        while not done:
            # 确定性策略
            actions, _ = mappo.select_actions(states, deterministic=True)

            next_states, rewards, dones, info = env.step(actions)

            episode_return += sum(rewards)
            step_cost = sum([env.agents[i].env.current_step_cost for i in range(3)])
            episode_cost += step_cost
            episode_peak_load = max(episode_peak_load, info['community_net_load'])
            episode_peak_penalty += info['peak_penalty']

            for i in range(3):
                agent_returns[i] += rewards[i]
                agent_costs[i] += env.agents[i].env.current_step_cost

            states = next_states
            done = all(dones)
            step_count += 1
            if step_count >= 48:
                done = True

        results['returns'].append(episode_return)
        results['costs'].append(episode_cost)
        results['peak_loads'].append(episode_peak_load)
        results['peak_penalties'].append(episode_peak_penalty)
        for i in range(3):
            results['agent_returns'][i].append(agent_returns[i])
            results['agent_costs'][i].append(agent_costs[i])

        print(f"Episode {episode + 1}/{num_episodes} | "
              f"Date: {date} (PV={pv_coeff[0]:.1f}) | House: {test_house_ids[house_index]} | "
              f"Return: {episode_return:.2f} | Cost: {episode_cost:.2f} | Peak: {episode_peak_load:.2f} kW")

    # 打印汇总结果
    print("\n" + "=" * 80)
    print("Evaluation Results Summary")
    print("=" * 80)
    print(f"Average Total Return: {np.mean(results['returns']):.2f} ± {np.std(results['returns']):.2f}")
    print(f"Average Total Cost: {np.mean(results['costs']):.2f} ± {np.std(results['costs']):.2f}")
    print(f"Average Peak Load: {np.mean(results['peak_loads']):.2f} ± {np.std(results['peak_loads']):.2f} kW")
    print(f"Max Peak Load: {np.max(results['peak_loads']):.2f} kW")
    print(f"Min Peak Load: {np.min(results['peak_loads']):.2f} kW")
    print(f"Average Peak Penalty: {np.mean(results['peak_penalties']):.4f} ± {np.std(results['peak_penalties']):.4f}")

    print("\nPer-Agent Results:")
    for i in range(3):
        avg_return = np.mean(results['agent_returns'][i])
        avg_cost = np.mean(results['agent_costs'][i])
        print(f"  Agent {i + 1}: Return={avg_return:.2f}, Cost={avg_cost:.2f}")

    print("=" * 80)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained MAPPO model on test households')
    parser.add_argument('--model_dir', type=str, default='multi_agent/algorithms/models',
                        help='Directory containing the model (default: multi_agent/algorithms/models)')
    parser.add_argument('--baseline_peak', type=float, default=31.01,
                        help='Baseline peak load (default: 31.01)')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of evaluation episodes (default: 10)')
    parser.add_argument('--mode', type=str, default='eval', choices=['eval', 'train'],
                        help='Evaluation mode (default: eval)')

    args = parser.parse_args()

    try:
        results = evaluate_mappo(
            model_dir=args.model_dir,
            baseline_peak=args.baseline_peak,
            num_episodes=args.episodes,
            mode=args.mode
        )
        if results is not None:
            print("\n" + "=" * 80)
            print("Evaluation completed successfully!")
            print("=" * 80)
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
    except Exception as e:
        print(f"\n\nError during evaluation: {e}")
        import traceback

        traceback.print_exc()
