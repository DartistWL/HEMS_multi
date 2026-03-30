"""
QMIX 训练脚本：与 MAPPO 相同的环境与 episode 数，训练结束后保存模型与 training_stats.json、CSV。
不周期性保存模型，仅结束时保存一次。
"""
import sys
import os
import json
import argparse
import numpy as np
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from multi_agent.algorithms.qmix import QMIX
from multi_agent.algorithms.action_utils import get_global_state_vector
from multi_agent.environment.multi_agent_env import MultiAgentHEMEnv
from multi_agent.utils.data_interface import MultiAgentDataInterface


def load_config(config_path='multi_agent/config.json'):
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        return cfg.get('training', {}) or {}
    return {}


def train_qmix(num_episodes=1000, baseline_peak=None, save_dir='multi_agent/algorithms/models_qmix',
               load_model=None, config_path='multi_agent/config.json'):
    config = load_config(config_path)
    if baseline_peak is None:
        baseline_peak = config.get('baseline_peak', 31.01)
    community_weight = config.get('community_weight', 0.2)
    community_credit_cost_weight = config.get('community_credit_cost_weight', 0.1)
    community_credit_benefit_weight = config.get('community_credit_benefit_weight', 0.1)
    initial_credit = config.get('initial_credit', 100.0)
    peak_penalty_exponent = config.get('peak_penalty_exponent', 2.0)
    peak_discharge_bonus = config.get('peak_discharge_bonus', 0.0)
    peak_credit_cost_reduction = config.get('peak_credit_cost_reduction', 1.0)

    print("=" * 60)
    print("Training QMIX Algorithm")
    print("=" * 60)
    print(f"Episodes: {num_episodes}, Baseline peak: {baseline_peak} kW, Save dir: {save_dir}")

    # 定义住户列表（新增）
    train_house_ids = [f"H{i}" for i in range(1, 16)]  # H1~H15
    test_house_ids = [f"H{i}" for i in range(16, 21)]  # H16~H20

    # 排查代码
    # print("1. 开始创建环境...")
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
        pv_coefficients=[2.0, 2.0, 2.0],
        train_house_ids=train_house_ids,   # 新增
        test_house_ids=test_house_ids,     # 新增
    )
    # print("2. 环境创建成功")
    # training_dates = [
    #     '2011-07-03', '2011-07-04', '2011-07-05', '2011-07-06',
    #     '2011-07-07', '2011-07-08', '2011-07-09',
    # ]
    training_dates = [
        '2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04',
        '2020-01-05', '2020-01-06', '2020-01-07',
    ]
    sunny_coef = MultiAgentDataInterface.get_weather_coefficient('sunny')
    cloudy_coef = MultiAgentDataInterface.get_weather_coefficient('cloudy')
    pv_list = [
        [sunny_coef, sunny_coef, sunny_coef],
    ] * 4 + [[cloudy_coef, cloudy_coef, cloudy_coef]] * 3
    # print("3. 开始设置训练日期...")
    env.set_training_dates(training_dates, pv_list)
    # print("4. 训练日期设置成功")

    # ---------- QMIX 超参 ----------
    # 学习率：前段保持高位快速学策略，再线性衰减到末期
    lr_start = 3e-4
    lr_end = 5e-5
    lr_hold_episodes = 200
    # 奖励归一化：QMIX 无 baseline，TD target 尺度直接受奖励影响，建议 True 以稳定收敛（False 易震荡/更差）
    use_reward_normalization = True
    reward_scale = 10.0
    td_target_clip = 200.0
    target_tau = 0.003
    epsilon_decay_episodes = 400
    # 状态归一化
    use_actor_state_normalization = False
    use_critic_state_normalization = True
    # 前期快速学策略：前 N 个 episode 每步都 update，之后每 4 步 update，降低方差的同时前期多学
    update_every_steps_early = 1
    update_every_steps_late = 4
    update_early_episodes = 250  # 前 250 episode 每步更新，之后每 4 步更新
    # VDN：Q_tot = sum_i Q_i，无 mixing 网络
    use_vdn = False
    vdn_aux_coef = 0.1
    # 优先经验回放（PER）：按 |TD error| 多采样“意外”样本，常能加快收敛；需配合重要性采样权重
    # use_prioritized_replay = True
    use_prioritized_replay = False
    per_alpha = 0.6      # 优先级指数：priority = |delta|^alpha
    per_beta_start = 0.4 # 重要性采样 beta 起始，逐渐增至 per_beta_end 以减轻偏差
    per_beta_end = 1.0
    per_beta_episodes = 500
    print(f"QMIX 超参: use_vdn={use_vdn}, vdn_aux_coef={vdn_aux_coef}, PER={use_prioritized_replay} (alpha={per_alpha}, beta={per_beta_start}->{per_beta_end}), lr {lr_start}(hold {lr_hold_episodes}ep)->{lr_end}, reward_norm={use_reward_normalization}, reward_scale={reward_scale}, td_clip={td_target_clip}, target_tau={target_tau}, epsilon_decay={epsilon_decay_episodes}, update_early: every {update_every_steps_early} for first {update_early_episodes}ep then every {update_every_steps_late}")

    qmix = QMIX(
        env=env,
        n_agents=3,
        # 强制使用cpu
        device='cpu',
        hidden_dim=128,
        gamma=0.95,
        lr=lr_start,
        use_actor_state_normalization=use_actor_state_normalization,
        use_critic_state_normalization=use_critic_state_normalization,
        reward_scale=reward_scale,
        use_reward_normalization=use_reward_normalization,
        use_vdn=use_vdn,
        vdn_aux_coef=vdn_aux_coef,
        use_prioritized_replay=use_prioritized_replay,
        per_alpha=per_alpha,
        per_beta_start=per_beta_start,
        per_beta_end=per_beta_end,
        per_beta_episodes=per_beta_episodes,
        # replay_capacity=50000,
        # batch_size=128,
        batch_size=32,  # 原 128 → 32
        replay_capacity=10000,  # 原 50000 → 10000
        target_tau=target_tau,
        td_target_clip=td_target_clip,
        max_grad_norm=5.0,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_episodes=epsilon_decay_episodes,
    )
    if load_model and os.path.exists(load_model):
        qmix.load(load_model)
        print(f"Loaded model from {load_model}")

    os.makedirs(save_dir, exist_ok=True)
    training_stats = {
        'episode_returns': [],
        'episode_costs': [],
        'peak_loads': [],
        'td_loss': [],
        'aux_loss': [],
    }
    num_dates = len(training_dates)

    # 在 train_qmix 函数中，创建 env 之后定义住户列表长度
    train_house_ids = [f"H{i}" for i in range(1, 16)]  # 与创建 env 时保持一致
    num_houses = len(train_house_ids)

    for episode in range(num_episodes):

        # 学习率：前 lr_hold_episodes 保持 lr_start，再线性衰减到 lr_end，前期快速学、后期稳
        if episode < lr_hold_episodes:
            lr = lr_start
        else:
            progress = (episode - lr_hold_episodes) / max(1, num_episodes - lr_hold_episodes)
            lr = lr_start + (lr_end - lr_start) * progress
        qmix.set_learning_rate(lr)

        date_index = episode % len(training_dates)
        house_index = episode % num_houses
        # print("5. 开始第一次 reset...")
        states = env.reset(mode='train', date_index=date_index, house_index=house_index)
        # print("6. reset 成功")
        global_state = get_global_state_vector(env)

        episode_return = 0.0
        episode_cost = 0.0
        episode_peak_load = 0.0
        episode_td_losses = []
        episode_aux_losses = []
        step_count = 0
        done = False

        while not done:
            actions_dict, action_indices = qmix.select_actions(states, deterministic=False)
            next_states, rewards, dones, info = env.step(actions_dict)
            next_global_state = get_global_state_vector(env)

            qmix.store_transition(
                states, global_state, action_indices, rewards,
                next_states, next_global_state, dones,
            )
            episode_return += sum(rewards)
            episode_cost += sum(env.agents[i].env.current_step_cost for i in range(3))
            episode_peak_load = max(episode_peak_load, info.get('community_net_load', 0.0))

            # 前期每步更新以快速学策略，后期每 4 步更新以降低方差、减轻震荡
            update_every = update_every_steps_early if episode < update_early_episodes else update_every_steps_late
            u = None
            if step_count % update_every == 0:
                u = qmix.update()
            if u:
                if u.get('skip_nan'):
                    continue  # 本次 update 因 nan 已跳过，不参与 episode 统计
                episode_td_losses.append(u.get('td_loss', 0))
                if 'aux_loss' in u:
                    episode_aux_losses.append(u['aux_loss'])

            states = next_states
            global_state = next_global_state
            done = all(dones)
            step_count += 1
            if step_count >= 48:
                done = True

        qmix.on_episode_end()
        training_stats['episode_returns'].append(episode_return)
        training_stats['episode_costs'].append(episode_cost)
        training_stats['peak_loads'].append(episode_peak_load)
        training_stats['td_loss'].append(float(np.mean(episode_td_losses)) if episode_td_losses else 0.0)
        training_stats['aux_loss'].append(float(np.mean(episode_aux_losses)) if episode_aux_losses else 0.0)

        if (episode + 1) % 10 == 0:
            w = min(10, len(training_stats['episode_returns']))
            avg_r = np.mean(training_stats['episode_returns'][-w:])
            avg_c = np.mean(training_stats['episode_costs'][-w:])
            avg_p = np.mean(training_stats['peak_loads'][-w:])
            print(f"Episode {episode + 1}/{num_episodes} | Return: {episode_return:.2f} (avg: {avg_r:.2f}) | "
                  f"Cost: {episode_cost:.2f} | Peak: {episode_peak_load:.2f} kW")

    qmix.save(save_dir)
    print(f"Model saved to {save_dir}")

    stats_file = os.path.join(save_dir, 'training_stats.json')
    stats_dict = {
        'episode_returns': training_stats['episode_returns'],
        'episode_costs': training_stats['episode_costs'],
        'peak_loads': training_stats['peak_loads'],
        'td_loss': training_stats.get('td_loss', []),
        'aux_loss': training_stats.get('aux_loss', []),
    }
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats_dict, f, indent=2, ensure_ascii=False)
    print(f"training_stats.json saved to {stats_file}")

    csv_path = os.path.join(save_dir, 'returns_data.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("episode_id,return,cost,peak_load\n")
        for i in range(len(training_stats['episode_returns'])):
            f.write(f"{i},{training_stats['episode_returns'][i]:.6f},"
                    f"{training_stats['episode_costs'][i]:.6f},{training_stats['peak_loads'][i]:.6f}\n")
    print(f"returns_data.csv saved to {csv_path}")

    print("=" * 60)
    print("QMIX training completed.")
    if len(training_stats['episode_returns']) >= 100:
        print(f"Avg return (last 100): {np.mean(training_stats['episode_returns'][-100:]):.2f}")
    return qmix, training_stats


def main():
    parser = argparse.ArgumentParser(description='Train QMIX for multi-agent HEMS')
    parser.add_argument('--num_episodes', type=int, default=1500)
    parser.add_argument('--baseline_peak', type=float, default=None)
    parser.add_argument('--save_dir', type=str, default='multi_agent/algorithms/models_qmix')
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--config_path', type=str, default='multi_agent/config.json')
    args = parser.parse_args()
    train_qmix(
        num_episodes=args.num_episodes,
        baseline_peak=args.baseline_peak,
        save_dir=args.save_dir,
        load_model=args.load_model,
        config_path=args.config_path,
    )


if __name__ == '__main__':
    main()
