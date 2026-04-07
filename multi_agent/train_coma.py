"""
COMA 训练脚本：与 MAPPO 相同的环境与 episode 数，训练结束后保存模型与 training_stats.json、CSV。
不周期性保存模型，仅结束时保存一次。
"""

import sys
import os
import json
import argparse
import numpy as np
from datetime import datetime
import torch
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from multi_agent.algorithms.coma import COMA
from multi_agent.algorithms.action_utils import get_global_state_vector
from multi_agent.environment.multi_agent_env import MultiAgentHEMEnv
from multi_agent.utils.data_interface import MultiAgentDataInterface


def load_config(config_path='multi_agent/config.json'):
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        return cfg.get('training', {}) or {}
    return {}


def train_coma(num_episodes=1000, baseline_peak=None, save_dir='multi_agent/algorithms/models_coma',
               load_model=None, config_path='multi_agent/config.json', n_episodes_per_update=2):
    # 强制每更新一次只用一个 episode（更稳定）
    n_episodes_per_update = 1

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
    print("Training COMA Algorithm")
    print("=" * 60)
    print(f"Episodes: {num_episodes}, Baseline peak: {baseline_peak} kW, Save dir: {save_dir}")

    # 超参数（已调稳）
    lr_actor = 1e-5
    lr_critic = 5e-6
    use_popart = False
    use_advantage_normalization = True
    max_grad_norm = 1.0
    td_target_clip = 20.0
    ent_coef = 0.005
    use_actor_state_normalization = False
    use_critic_state_normalization = True
    reward_scale = 1.0
    n_actor_epochs = 1
    print(f"n_episodes_per_update: {n_episodes_per_update}, use_popart: {use_popart}, "
          f"actor_state_norm: {use_actor_state_normalization}, critic_state_norm: {use_critic_state_normalization}, "
          f"use_adv_norm: {use_advantage_normalization}, reward_scale: {reward_scale}, ent_coef: {ent_coef}, "
          f"lr_actor: {lr_actor}, lr_critic: {lr_critic}, n_actor_epochs: {n_actor_epochs}, "
          f"td_target_clip: {td_target_clip}")

    # 定义训练和测试住户（15 户训练，5 户测试）
    train_house_ids = [f"H{i}" for i in range(1, 16)]
    test_house_ids = [f"H{i}" for i in range(16, 21)]

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
        train_house_ids=train_house_ids,
        test_house_ids=test_house_ids,
    )

    training_dates = [
        '2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04',
        '2020-01-05', '2020-01-06', '2020-01-07',
    ]
    sunny_coef = MultiAgentDataInterface.get_weather_coefficient('sunny')
    cloudy_coef = MultiAgentDataInterface.get_weather_coefficient('cloudy')
    pv_list = [[sunny_coef, sunny_coef, sunny_coef]] * 4 + [[cloudy_coef, cloudy_coef, cloudy_coef]] * 3
    env.set_training_dates(training_dates, pv_list)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coma = COMA(
        env=env,
        n_agents=3,
        device=device,
        hidden_dim=64,          # 降低隐藏层维度，减少计算量
        gamma=0.96,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        use_actor_state_normalization=use_actor_state_normalization,
        use_critic_state_normalization=use_critic_state_normalization,
        reward_scale=reward_scale,
        use_reward_normalization=False,
        max_grad_norm=max_grad_norm,
        td_target_clip=td_target_clip,
        ent_coef=ent_coef,
        n_actor_epochs=n_actor_epochs,
        use_popart=use_popart,
        use_advantage_normalization=use_advantage_normalization,
    )

    if load_model and os.path.exists(load_model):
        coma.load(load_model)
        print(f"Loaded model from {load_model}")

    os.makedirs(save_dir, exist_ok=True)
    training_stats = {
        'episode_returns': [],
        'episode_costs': [],
        'peak_loads': [],
        'actor_loss': [],
        'critic_loss': [],
    }
    last_update_stats = None

    # 创建 checkpoint 目录
    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ===== 新增：自动加载最新的 checkpoint =====
    start_episode = 0
    if load_model is None:
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_') and f.endswith('.pt')]
        if checkpoints:
            # 按 episode 数字排序
            checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
            latest = checkpoints[-1]
            checkpoint_path = os.path.join(checkpoint_dir, latest)
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            coma.load_state_dict(checkpoint['model_state_dict'])
            start_episode = checkpoint['episode'] + 1
            training_stats = checkpoint['training_stats']
            last_update_stats = checkpoint.get('last_update_stats')
            print(f"Resumed from episode {start_episode}")
    else:
        if os.path.exists(load_model):
            coma.load(load_model)
            print(f"Loaded model from {load_model}")
    # ===== 新增结束 =====

    for episode in range(start_episode, num_episodes):
    # # 创建 checkpoint 目录
    # checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    # os.makedirs(checkpoint_dir, exist_ok=True)
    #
    # for episode in range(num_episodes):
        if episode % n_episodes_per_update == 0:
            coma.reset_buffer()

        date_index = episode % len(training_dates)
        house_index = episode % len(train_house_ids)
        states = env.reset(mode='train', date_index=date_index, house_index=house_index)
        global_state = get_global_state_vector(env)

        episode_return = 0.0
        episode_cost = 0.0
        episode_peak_load = 0.0
        step_count = 0
        done = False

        while not done:
            actions_dict, action_indices = coma.select_actions(states, deterministic=False)
            next_states, rewards, dones, info = env.step(actions_dict)
            next_global_state = get_global_state_vector(env)

            coma.store_transition(
                states, global_state, action_indices, rewards,
                next_states, next_global_state, dones,
            )
            episode_return += sum(rewards)
            episode_cost += sum(env.agents[i].env.current_step_cost for i in range(3))
            episode_peak_load = max(episode_peak_load, info.get('community_net_load', 0.0))

            states = next_states
            global_state = next_global_state
            done = all(dones)
            step_count += 1
            if step_count >= 48:
                done = True

        training_stats['episode_returns'].append(episode_return)
        training_stats['episode_costs'].append(episode_cost)
        training_stats['peak_loads'].append(episode_peak_load)

        do_update = (episode + 1) % n_episodes_per_update == 0 or \
                    (episode == num_episodes - 1 and len(coma.buffer['rewards']) > 0)
        if do_update:
            update_stats = coma.update()
            last_update_stats = update_stats
            if update_stats:
                training_stats['actor_loss'].append(update_stats.get('actor_loss', 0))
                training_stats['critic_loss'].append(update_stats.get('critic_loss', 0))

        # 每 10 集保存一次 checkpoint 并打印进度
        if (episode + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{episode}.pt')
            torch.save({
                'episode': episode,
                'model_state_dict': coma.state_dict(),
                'training_stats': training_stats,
                'last_update_stats': last_update_stats,
            }, checkpoint_path)
            # 打印训练进度
            w = min(10, len(training_stats['episode_returns']))
            avg_r = np.mean(training_stats['episode_returns'][-w:])
            avg_c = np.mean(training_stats['episode_costs'][-w:])
            avg_p = np.mean(training_stats['peak_loads'][-w:])
            update_stats = last_update_stats
            al = update_stats.get('actor_loss') if update_stats else None
            cl = update_stats.get('critic_loss') if update_stats else None
            al_str = f"{al:.6f}" if al is not None and al == al else (str(al) if al is not None else "N/A")
            cl_str = f"{cl:.6f}" if cl is not None and cl == cl else (str(cl) if cl is not None else "N/A")
            print(f"Episode {episode + 1}/{num_episodes} | Return: {episode_return:.2f} (avg: {avg_r:.2f}) | "
                  f"Cost: {episode_cost:.2f} | Peak: {episode_peak_load:.2f} kW | "
                  f"actor_loss: {al_str} | critic_loss: {cl_str}")

    # 最终保存模型和统计
    coma.save(save_dir)
    print(f"Model saved to {save_dir}")

    stats_file = os.path.join(save_dir, 'training_stats.json')
    stats_dict = {
        'episode_returns': training_stats['episode_returns'],
        'episode_costs': training_stats['episode_costs'],
        'peak_loads': training_stats['peak_loads'],
        'actor_loss': training_stats.get('actor_loss', []),
        'critic_loss': training_stats.get('critic_loss', []),
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
    print("COMA training completed.")
    if len(training_stats['episode_returns']) >= 100:
        print(f"Avg return (last 100): {np.mean(training_stats['episode_returns'][-100:]):.2f}")
    return coma, training_stats


def main():
    parser = argparse.ArgumentParser(description='Train COMA for multi-agent HEMS')
    parser.add_argument('--num_episodes', type=int, default=1500)
    parser.add_argument('--baseline_peak', type=float, default=None)
    parser.add_argument('--save_dir', type=str, default='multi_agent/algorithms/models_coma')
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--config_path', type=str, default='multi_agent/config.json')
    parser.add_argument('--n_episodes_per_update', type=int, default=2, help='积累多少个 episode 再更新（推荐 2，更稳）')
    args = parser.parse_args()
    train_coma(
        num_episodes=args.num_episodes,
        baseline_peak=args.baseline_peak,
        save_dir=args.save_dir,
        load_model=args.load_model,
        config_path=args.config_path,
        n_episodes_per_update=args.n_episodes_per_update,
    )


if __name__ == '__main__':
    main()