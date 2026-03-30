"""
Independent Learning Baseline
独立学习基线

3户家庭完全独立训练:
- 每个家庭只能看到自家信息（使用社区环境时可见社区储能/积分等状态，但奖励仅个人目标）
- 奖励函数只有个人目标：使用社区环境时，环境以 reward_for_individual_only=True 仅返回「个人奖励 - 积分成本」，
  不含峰值惩罚、社区储能激励、排队惩罚等，从而不对峰值做优化，以体现 MAPPO 的削峰优势
- 使用社区环境时仍可与社区储能交互（充放电），但策略不因峰值/社区激励而优化
"""
import numpy as np
import sys
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# 修复导入路径：使用importlib直接加载，避免路径冲突
import importlib.util
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
env_file_path = os.path.join(project_root, "environment.py")
spec = importlib.util.spec_from_file_location("home_energy_env", env_file_path)
home_energy_env = importlib.util.module_from_spec(spec)
spec.loader.exec_module(home_energy_env)
HomeEnergyManagementEnv = home_energy_env.HomeEnergyManagementEnv

from multi_agent.utils.data_interface import MultiAgentDataInterface
from multi_agent.baselines.ppo import HomeEnergyPPO


def _load_use_community_env_from_config(default=False):
    """从 config.json 读取 independent_baseline.use_community_env，默认 default。"""
    try:
        config_path = os.path.join(project_root, 'multi_agent', 'config.json')
        if os.path.isfile(config_path):
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            ib = cfg.get('independent_baseline', {})
            return bool(ib.get('use_community_env', default))
    except Exception:
        pass
    return default


def _load_credit_pricing_scheme_from_config(default='uniform'):
    """从 config.json 读取 credit_pricing.scheme，用于模型保存/加载后缀。默认 uniform。"""
    try:
        config_path = os.path.join(project_root, 'multi_agent', 'config.json')
        if os.path.isfile(config_path):
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            cp = cfg.get('credit_pricing', {})
            return str(cp.get('scheme', default)).strip() or default
    except Exception:
        pass
    return default


def _load_training_env_params_from_config():
    """从 config.json 读取 training 段的环境参数，与 MAPPO 训练时一致（公平对比、避免独立基线 env 只用 0.2 等硬编码）。"""
    defaults = {
        'baseline_peak': 31.01,
        'community_weight': 0.2,
        'community_credit_cost_weight': 0.05,
        'community_credit_benefit_weight': 0.05,
        'initial_credit': 100.0,
        'peak_penalty_exponent': 2.0,
        'peak_discharge_bonus': 0.0,
        'peak_credit_cost_reduction': 1.0,
    }
    try:
        config_path = os.path.join(project_root, 'multi_agent', 'config.json')
        if os.path.isfile(config_path):
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            tr = cfg.get('training', {})
            for k in defaults:
                if k in tr:
                    v = tr[k]
                    defaults[k] = float(v) if isinstance(v, (int, float)) else v
    except Exception:
        pass
    return defaults


class IndependentBaseline:
    """
    独立学习基线
    为每个家庭训练独立的模型，使用不同的随机种子
    """
    
    def __init__(self, n_agents=3, pv_coefficients=None, 
                 ev_capacity=24, ess_capacity=24, random_seeds=None,
                 use_community_env=None):
        """
        初始化独立学习基线
        
        Args:
            n_agents: 家庭数量
            pv_coefficients: 每个家庭的光伏系数列表
            ev_capacity: EV容量
            ess_capacity: 家庭储能容量
            random_seeds: 每个家庭的随机种子列表，如果为None则使用[0, 1, 2]
            use_community_env: 是否使用带社区储能的多智能体环境（方案1）。
                True=多智能体环境+社区储能，False=原单智能体环境。
                None=从 config.json 的 independent_baseline.use_community_env 读取，默认 False。
        """
        self.n_agents = n_agents
        if use_community_env is None:
            use_community_env = _load_use_community_env_from_config(default=False)
        self.use_community_env = use_community_env
        
        # 初始化随机种子（每个家庭使用不同的随机种子）
        if random_seeds is None:
            random_seeds = list(range(n_agents))  # [0, 1, 2]
        elif len(random_seeds) != n_agents:
            raise ValueError(f"random_seeds长度必须等于n_agents ({n_agents})")
        self.random_seeds = random_seeds
        
        # 初始化光伏系数
        if pv_coefficients is None:
            pv_coefficients = [1.0] * n_agents
        self.pv_coefficients = pv_coefficients
        
        self.train_envs = []
        self.eval_envs = []
        self.agents = []
        
        if not self.use_community_env:
            # ---------- 原逻辑：单智能体环境，无社区储能 ----------
            for i in range(n_agents):
                train_data_interface = MultiAgentDataInterface(
                    'data/daily_pivot_cons_2011-2012.csv',
                    'data/daily_pivot_prod_2011-2012.csv',
                    pv_coefficient=pv_coefficients[i]
                )
                train_env = HomeEnergyManagementEnv(
                    ev_capacity=ev_capacity,
                    ess_capacity=ess_capacity
                )
                train_env.data_interface = train_data_interface
                train_env.data_interface.np_random = np.random.RandomState(random_seeds[i])
                self.train_envs.append(train_env)
                
                state_dim = len(train_env.state_space)
                action_space_config = train_env.action_space
                agent = HomeEnergyPPO(
                    env=train_env,
                    state_dim=state_dim,
                    hidden_dim=128,
                    action_space_config=action_space_config,
                    gamma=0.96,
                    lmbda=0.95,
                    eps=0.2,
                    epochs=10,
                    lr=1e-4,
                    constraint_mode='lagrangian',
                    use_state_normalization=True,
                    reward_scale=1.0,
                    use_reward_normalization=True,
                    use_popart=False,
                    critic_loss_type='mse'
                )
                self.agents.append(agent)
            
            for i in range(n_agents):
                data_interface = MultiAgentDataInterface(
                    'data/daily_pivot_cons_2011-2012.csv',
                    'data/daily_pivot_prod_2011-2012.csv',
                    pv_coefficient=pv_coefficients[i]
                )
                env = HomeEnergyManagementEnv(
                    ev_capacity=ev_capacity,
                    ess_capacity=ess_capacity
                )
                env.data_interface = data_interface
                env.data_interface.np_random = np.random.RandomState(random_seeds[i])
                self.eval_envs.append(env)
        else:
            # ---------- 方案1：多智能体环境，有社区储能 ----------
            from multi_agent.environment.multi_agent_env import MultiAgentHEMEnv
            # 扩展状态键（与 multi_agent_env._add_community_info_to_states 一致，含 is_weekend）
            _COMMUNITY_STATE_KEYS = [
                'community_ess_soc', 'community_ess_capacity', 'community_credit_balance',
                'community_net_load', 'community_avg_load', 'community_peak_threshold',
                'is_weekend'
            ]
            # 从 config.json training 读取环境参数，与 MAPPO 训练时一致（公平对比、避免独立基线只用 0.2 等硬编码）
            env_params = _load_training_env_params_from_config()
            
            train_ma_env = MultiAgentHEMEnv(
                n_agents=n_agents,
                community_ess_capacity=36.0,
                baseline_peak=env_params['baseline_peak'],
                community_weight=env_params['community_weight'],
                community_credit_cost_weight=env_params['community_credit_cost_weight'],
                community_credit_benefit_weight=env_params['community_credit_benefit_weight'],
                initial_credit=env_params['initial_credit'],
                peak_penalty_exponent=env_params['peak_penalty_exponent'],
                peak_discharge_bonus=env_params['peak_discharge_bonus'],
                peak_credit_cost_reduction=env_params['peak_credit_cost_reduction'],
                pv_coefficients=pv_coefficients,
                reward_for_individual_only=True  # 仅优化个人目标，不对峰值/社区储能激励优化，以体现 MAPPO 削峰优势
            )
            for i in range(n_agents):
                train_ma_env.agents[i].env.data_interface.np_random = np.random.RandomState(random_seeds[i])
            
            base_state_keys = sorted(train_ma_env.agents[0].state_space.keys())
            extended_keys = sorted(set(base_state_keys) | set(_COMMUNITY_STATE_KEYS))
            state_dim = len(extended_keys)
            action_space_config = train_ma_env.agents[0].action_space
            
            self.train_ma_env = train_ma_env
            self.eval_ma_env = MultiAgentHEMEnv(
                n_agents=n_agents,
                community_ess_capacity=36.0,
                baseline_peak=env_params['baseline_peak'],
                community_weight=env_params['community_weight'],
                community_credit_cost_weight=env_params['community_credit_cost_weight'],
                community_credit_benefit_weight=env_params['community_credit_benefit_weight'],
                initial_credit=env_params['initial_credit'],
                peak_penalty_exponent=env_params['peak_penalty_exponent'],
                peak_discharge_bonus=env_params['peak_discharge_bonus'],
                peak_credit_cost_reduction=env_params['peak_credit_cost_reduction'],
                pv_coefficients=pv_coefficients,
                reward_for_individual_only=True
            )
            for i in range(n_agents):
                self.eval_ma_env.agents[i].env.data_interface.np_random = np.random.RandomState(random_seeds[i])
            
            for _ in range(n_agents):
                agent = HomeEnergyPPO(
                    env=None,
                    state_dim=state_dim,
                    hidden_dim=128,
                    action_space_config=action_space_config,
                    gamma=0.96,
                    lmbda=0.95,
                    eps=0.2,
                    epochs=10,
                    lr=1e-4,
                    constraint_mode='lagrangian',
                    use_state_normalization=True,
                    reward_scale=1.0,
                    use_reward_normalization=True,
                    use_popart=False,
                    critic_loss_type='mse'
                )
                self.agents.append(agent)
    
    def train(self, num_episodes=500, save_dir='multi_agent/baselines/models', 
              training_dates=None, pv_coefficients_list=None):
        """
        训练独立学习基线（为每个家庭训练独立的模型）
        
        Args:
            num_episodes: 训练轮数
            save_dir: 模型保存目录
            training_dates: 训练日期列表（7天，和MAPPO一样），如果为None则使用默认日期
            pv_coefficients_list: 每天的光伏系数列表，如果为None则使用默认值
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        credit_scheme = _load_credit_pricing_scheme_from_config(default='uniform')
        save_suffix = ('_contribution_based' if credit_scheme == 'contribution_based' else '')
        if save_suffix:
            print(f"credit_pricing.scheme={credit_scheme} -> 模型保存带后缀 {save_suffix}，不覆盖原模型")
        
        if training_dates is None:
            training_dates = [
                '2011-07-03', '2011-07-04', '2011-07-05', '2011-07-06',
                '2011-07-07', '2011-07-08', '2011-07-09',
            ]
        if pv_coefficients_list is None:
            sunny_coef = MultiAgentDataInterface.get_weather_coefficient('sunny')
            cloudy_coef = MultiAgentDataInterface.get_weather_coefficient('cloudy')
            pv_coefficients_list = [
                [sunny_coef, sunny_coef, sunny_coef],
                [sunny_coef, sunny_coef, sunny_coef],
                [sunny_coef, sunny_coef, sunny_coef],
                [sunny_coef, sunny_coef, sunny_coef],
                [cloudy_coef, cloudy_coef, cloudy_coef],
                [cloudy_coef, cloudy_coef, cloudy_coef],
                [cloudy_coef, cloudy_coef, cloudy_coef],
            ]
        num_training_dates = len(training_dates)
        
        print("=" * 60)
        print("Training Independent Learning Baseline")
        print(f"use_community_env={self.use_community_env}")
        print(f"Training {self.n_agents} independent models (one for each agent)")
        print(f"Random seeds: {self.random_seeds}")
        print(f"Training dates: {training_dates}")
        print(f"Number of episodes: {num_episodes}")
        print("=" * 60)
        
        if self.use_community_env:
            self._train_with_community_env(
                num_episodes, save_dir, training_dates, pv_coefficients_list, num_training_dates,
                credit_scheme=credit_scheme
            )
            return
        
        # ---------- 原逻辑：单智能体环境 ----------
        all_training_histories = []
        
        for agent_id in range(self.n_agents):
            print(f"\n{'='*60}")
            print(f"Training Agent {agent_id + 1}/{self.n_agents}")
            print(f"Random seed: {self.random_seeds[agent_id]}")
            print(f"{'='*60}")
            
            env = self.train_envs[agent_id]
            agent = self.agents[agent_id]
            
            return_list = []
            loss_list = {'actor_loss': [], 'critic_loss': [], 'entropy': []}
            
            for episode in range(num_episodes):
                # 选择训练日期（循环使用）
                date_index = episode % num_training_dates
                training_date = training_dates[date_index]
                
                # 更新光伏系数（根据日期）
                pv_coefs = pv_coefficients_list[date_index]
                env.data_interface.set_pv_coefficient(pv_coefs[agent_id])
                
                # 每个episode开始时打印（每10个）
                if (episode + 1) % 10 == 0:
                    print(f"Agent {agent_id + 1} - Episode {episode + 1}/{num_episodes} - Date: {training_date}...", flush=True)
                
                # 重置环境并设置日期
                state = env.reset()
                env.current_time = training_date
                env.current_time_index = 0
                
                episode_return = 0
                transition_dict = {
                    'states': [],
                    'actions': [],
                    'next_states': [],
                    'rewards': [],
                    'dones': []
                }
                done = False
                step_count = 0
                
                while not done:
                    # 选择动作
                    action = agent.take_action(state)
                    
                    # 执行动作
                    next_state, reward, done = env.step(state, action)
                    
                    # 存储经验
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    
                    state = next_state
                    episode_return += reward
                    step_count += 1
                    
                    # 防止无限循环
                    if step_count >= 48:  # 一天最多48个时间步
                        done = True
                
                # 每10个episode打印完成信息
                if (episode + 1) % 10 == 0:
                    print(f"Agent {agent_id + 1} - Episode {episode + 1} completed: steps={step_count}, return={episode_return:.2f}", flush=True)
                
                # 更新策略并获取统计信息
                update_stats = agent.update(transition_dict)
                return_list.append(episode_return)
                
                # 记录loss和熵
                if update_stats:
                    loss_list['actor_loss'].append(update_stats.get('actor_loss', 0))
                    loss_list['critic_loss'].append(update_stats.get('critic_loss', 0))
                    loss_list['entropy'].append(update_stats.get('entropy', 0))
                
                # 打印进度（每10轮打印一次详细信息）
                if (episode + 1) % 10 == 0:
                    window_size = min(10, len(return_list))
                    avg_return = np.mean(return_list[-window_size:])
                    recent_returns = return_list[-window_size:] if len(return_list) >= window_size else return_list
                    std_return = np.std(recent_returns)
                    
                    print(f"Agent {agent_id + 1} - Episode {episode + 1}/{num_episodes}")
                    print(f"  Return: avg={avg_return:.2f}, std={std_return:.2f}, latest={episode_return:.2f}")
                    
                    if update_stats and len(loss_list['actor_loss']) > 0:
                        window_size_loss = min(10, len(loss_list['actor_loss']))
                        avg_actor_loss = np.mean(loss_list['actor_loss'][-window_size_loss:])
                        avg_critic_loss = np.mean(loss_list['critic_loss'][-window_size_loss:])
                        avg_entropy = np.mean(loss_list['entropy'][-window_size_loss:])
                        print(f"  Loss: Actor={avg_actor_loss:.4f}, Critic={avg_critic_loss:.4f}, Entropy={avg_entropy:.4f}")
                    print()
            
            # 保存训练历史
            training_history = {
                'returns': return_list,
                'actor_loss': loss_list['actor_loss'],
                'critic_loss': loss_list['critic_loss'],
                'entropy': loss_list['entropy']
            }
            all_training_histories.append(training_history)
            
            # 保存模型（每个家庭一个）；scheme=contribution_based 时带 _contribution_based 后缀，不覆盖原模型
            model_path = os.path.join(save_dir, f'independent_agent_{agent_id}{save_suffix}.pth')
            checkpoint = {
                'episode': num_episodes,
                'shared_backbone': agent.shared_backbone.state_dict(),
                'action_branches': {name: branch.state_dict() for name, branch in agent.action_branches.items()},
                'value_net': agent.value_net.state_dict(),
                'running_stats': {
                    'mean': agent.running_stats.mean.cpu(),
                    'std': agent.running_stats.std.cpu(),
                    'count': agent.running_stats.count
                } if hasattr(agent, 'running_stats') else None,
                'state_keys': list(env.state_space.keys()),
                'random_seed': self.random_seeds[agent_id]
            }
            torch.save(checkpoint, model_path)
            print(f"Agent {agent_id + 1} model saved to {model_path}")
        
        print("\n" + "=" * 60)
        print("Independent Learning Baseline Training Completed!")
        print(f"All {self.n_agents} models saved to {save_dir}" + (f" (suffix {save_suffix})" if save_suffix else ""))
        print("=" * 60)
        
        # 绘制训练曲线（所有模型）
        self._plot_training_curves(all_training_histories, save_dir)
    
    def _train_with_community_env(self, num_episodes, save_dir, training_dates, pv_coefficients_list, num_training_dates, credit_scheme='uniform'):
        """方案1：在共享多智能体环境（含社区储能）中独立训练每个智能体。credit_scheme=contribution_based 时保存带 _contribution_based 后缀，不覆盖原模型。"""
        ma_env = self.train_ma_env
        ma_env.set_training_dates(training_dates, pv_coefficients_list)
        
        all_training_histories = [
            {'returns': [], 'actor_loss': [], 'critic_loss': [], 'entropy': []}
            for _ in range(self.n_agents)
        ]
        
        for episode in range(num_episodes):
            date_index = episode % num_training_dates
            for i in range(self.n_agents):
                ma_env.agents[i].env.data_interface.set_pv_coefficient(pv_coefficients_list[date_index][i])
            
            states = ma_env.reset(mode='train', date_index=date_index)
            transition_dicts = [
                {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                for _ in range(self.n_agents)
            ]
            done = False
            step_count = 0
            
            while not done:
                actions = [self.agents[i].take_action(states[i]) for i in range(self.n_agents)]
                next_states, rewards, dones, info = ma_env.step(actions)
                for i in range(self.n_agents):
                    transition_dicts[i]['states'].append(states[i])
                    transition_dicts[i]['actions'].append(actions[i])
                    transition_dicts[i]['next_states'].append(next_states[i])
                    transition_dicts[i]['rewards'].append(rewards[i])
                    transition_dicts[i]['dones'].append(dones[i])
                states = next_states
                done = all(dones)
                step_count += 1
                if step_count >= 48:
                    done = True
            
            for i in range(self.n_agents):
                update_stats = self.agents[i].update(transition_dicts[i])
                episode_return_i = sum(transition_dicts[i]['rewards'])
                all_training_histories[i]['returns'].append(episode_return_i)
                if update_stats:
                    all_training_histories[i]['actor_loss'].append(update_stats.get('actor_loss', 0))
                    all_training_histories[i]['critic_loss'].append(update_stats.get('critic_loss', 0))
                    all_training_histories[i]['entropy'].append(update_stats.get('entropy', 0))
            
            if (episode + 1) % 10 == 0:
                total_return = sum(all_training_histories[i]['returns'][-1] for i in range(self.n_agents))
                print(f"Episode {episode + 1}/{num_episodes} (date_index={date_index}) - Total return={total_return:.2f}, steps={step_count}")
        
        community_suffix = '_community' + ('_contribution_based' if credit_scheme == 'contribution_based' else '')
        for agent_id in range(self.n_agents):
            agent = self.agents[agent_id]
            model_path = os.path.join(save_dir, f'independent_agent_{agent_id}{community_suffix}.pth')
            checkpoint = {
                'episode': num_episodes,
                'shared_backbone': agent.shared_backbone.state_dict(),
                'action_branches': {name: branch.state_dict() for name, branch in agent.action_branches.items()},
                'value_net': agent.value_net.state_dict(),
                'running_stats': {
                    'mean': agent.running_stats.mean.cpu(),
                    'std': agent.running_stats.std.cpu(),
                    'count': agent.running_stats.count
                } if hasattr(agent, 'running_stats') and agent.running_stats is not None else None,
                'state_keys': None,
                'random_seed': self.random_seeds[agent_id],
                'use_community_env': True
            }
            torch.save(checkpoint, model_path)
            print(f"Agent {agent_id + 1} model (community) saved to {model_path}")
        
        print("\n" + "=" * 60)
        print("Independent Learning Baseline (with community env) Training Completed!")
        print(f"All {self.n_agents} models saved to {save_dir} (filename suffix {community_suffix})")
        print("=" * 60)
        self._plot_training_curves(all_training_histories, save_dir)
    
    def _plot_training_curves(self, all_training_histories, save_dir):
        """
        绘制训练曲线（所有模型的）
        
        Args:
            all_training_histories: 所有模型的训练历史列表
            save_dir: 保存目录
        """
        if not all_training_histories:
            print("No training history to plot.")
            return
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Independent Learning Baseline Training Curves ({self.n_agents} Models)', fontsize=16)
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        labels = [f'Agent {i+1} (seed={self.random_seeds[i]})' for i in range(self.n_agents)]
        
        # 1. Returns曲线
        ax1 = axes[0, 0]
        for i, training_history in enumerate(all_training_histories):
            if training_history and training_history.get('returns'):
                returns = training_history['returns']
                episodes = range(1, len(returns) + 1)
                ax1.plot(episodes, returns, alpha=0.7, color=colors[i], label=labels[i])
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Return')
        ax1.set_title('Episode Returns')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Actor Loss曲线
        ax2 = axes[0, 1]
        for i, training_history in enumerate(all_training_histories):
            actor_loss = training_history.get('actor_loss', [])
            if len(actor_loss) > 0:
                updates = range(1, len(actor_loss) + 1)
                ax2.plot(updates, actor_loss, alpha=0.7, color=colors[i], label=labels[i])
        ax2.set_xlabel('Update Step')
        ax2.set_ylabel('Actor Loss')
        ax2.set_title('Actor Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Critic Loss曲线
        ax3 = axes[1, 0]
        for i, training_history in enumerate(all_training_histories):
            critic_loss = training_history.get('critic_loss', [])
            if len(critic_loss) > 0:
                updates = range(1, len(critic_loss) + 1)
                ax3.plot(updates, critic_loss, alpha=0.7, color=colors[i], label=labels[i])
        ax3.set_xlabel('Update Step')
        ax3.set_ylabel('Critic Loss')
        ax3.set_title('Critic Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Entropy曲线
        ax4 = axes[1, 1]
        for i, training_history in enumerate(all_training_histories):
            entropy = training_history.get('entropy', [])
            if len(entropy) > 0:
                updates = range(1, len(entropy) + 1)
                ax4.plot(updates, entropy, alpha=0.7, color=colors[i], label=labels[i])
        ax4.set_xlabel('Update Step')
        ax4.set_ylabel('Entropy')
        ax4.set_title('Policy Entropy')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        plot_path = os.path.join(save_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {plot_path}")
        
        # 显示图片（如果在交互式环境中）
        try:
            plt.show()
        except:
            pass
        
        plt.close()
    
    def load_models(self, model_dir='multi_agent/baselines/models'):
        """
        加载训练好的模型（每个家庭使用各自的模型）
        根据 config 的 use_community_env 与 credit_pricing.scheme 自动选择文件名：
        use_community_env=True 时带 _community，scheme=contribution_based 时带 _contribution_based，不覆盖原模型。
        
        Args:
            model_dir: 模型目录
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scheme = _load_credit_pricing_scheme_from_config(default='uniform')
        suffix = ('_community' if self.use_community_env else '') + ('_contribution_based' if scheme == 'contribution_based' else '')
        all_loaded = True
        for agent_id in range(self.n_agents):
            model_path = os.path.join(model_dir, f'independent_agent_{agent_id}{suffix}.pth')
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=device)
                
                # 加载网络参数（每个家庭使用各自的模型）
                agent = self.agents[agent_id]
                agent.shared_backbone.load_state_dict(checkpoint['shared_backbone'])
                for name, branch in agent.action_branches.items():
                    if name in checkpoint['action_branches']:
                        branch.load_state_dict(checkpoint['action_branches'][name])
                agent.value_net.load_state_dict(checkpoint['value_net'])
                
                # 加载运行统计（如果存在）
                if checkpoint.get('running_stats') is not None and hasattr(agent, 'running_stats'):
                    agent.running_stats.mean = checkpoint['running_stats']['mean'].to(device)
                    agent.running_stats.std = checkpoint['running_stats']['std'].to(device)
                    agent.running_stats.count = checkpoint['running_stats']['count']
                
                # 如果checkpoint中有随机种子，使用它
                if 'random_seed' in checkpoint:
                    self.random_seeds[agent_id] = checkpoint['random_seed']
                    if self.use_community_env:
                        self.eval_ma_env.agents[agent_id].env.data_interface.np_random = np.random.RandomState(self.random_seeds[agent_id])
                    else:
                        self.eval_envs[agent_id].data_interface.np_random = np.random.RandomState(self.random_seeds[agent_id])
                
                print(f"Loaded model for Agent {agent_id + 1} from {model_path}")
            else:
                print(f"Warning: Model not found for Agent {agent_id + 1}: {model_path}")
                all_loaded = False
        
        if all_loaded:
            print(f"All {self.n_agents} models loaded successfully")
        else:
            print("Please train the models first using baseline.train()")
    
    def evaluate(self, num_episodes=10, dates=None):
        """
        评估独立学习基线。
        use_community_env=True 时在共享多智能体环境中评估，否则在各自单智能体环境中评估。
        
        Args:
            num_episodes: 评估轮数
            dates: 评估日期列表，如果为None则使用默认日期
        
        Returns:
            dict: 评估结果
        """
        if dates is None:
            dates = ['2011-07-03'] * num_episodes
        
        results = {
            'episode_returns': [],
            'episode_costs': [],
            'agent_returns': [[] for _ in range(self.n_agents)],
            'agent_costs': [[] for _ in range(self.n_agents)],
            'community_peak_loads': []
        }
        
        print("=" * 60)
        print("Evaluating Independent Learning Baseline")
        print(f"use_community_env={self.use_community_env}")
        print(f"Each agent uses its own model")
        print(f"Random seeds: {self.random_seeds}")
        print("=" * 60)
        
        if self.use_community_env:
            eval_dates = ['2011-07-10', '2011-07-11', '2011-07-12']
            pv_list = [
                [3.0, 3.0, 3.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]
            ]
            self.eval_ma_env.set_evaluation_dates(eval_dates, pv_list)
            for episode in range(num_episodes):
                date_index = episode % len(eval_dates)
                for i in range(self.n_agents):
                    self.eval_ma_env.agents[i].env.data_interface.set_pv_coefficient(pv_list[date_index][i])
                states = self.eval_ma_env.reset(mode='eval', date_index=date_index)
                done = False
                step = 0
                episode_returns = [0.0] * self.n_agents
                episode_costs = [0.0] * self.n_agents
                community_loads = []
                while not done:
                    actions = [self.agents[i].take_action(states[i]) for i in range(self.n_agents)]
                    next_states, rewards, dones, info = self.eval_ma_env.step(actions)
                    for i in range(self.n_agents):
                        episode_returns[i] += rewards[i]
                        episode_costs[i] += self.eval_ma_env.agents[i].env.current_step_cost
                    community_loads.append(info.get('community_net_load', 0))
                    states = next_states
                    done = all(dones)
                    step += 1
                    if step >= 48:
                        done = True
                results['episode_returns'].append(sum(episode_returns))
                results['episode_costs'].append(sum(episode_costs))
                for i in range(self.n_agents):
                    results['agent_returns'][i].append(episode_returns[i])
                    results['agent_costs'][i].append(episode_costs[i])
                peak_load = max(community_loads) if community_loads else 0
                results['community_peak_loads'].append(peak_load)
                print(f"Episode {episode + 1}/{num_episodes}: Total Return = {sum(episode_returns):.2f}, Total Cost = {sum(episode_costs):.2f}, Peak Load = {peak_load:.2f} kW")
            avg_return = np.mean(results['episode_returns'])
            avg_cost = np.mean(results['episode_costs'])
            avg_peak_load = np.mean(results['community_peak_loads'])
            print("\n" + "=" * 60)
            print("Independent Learning Baseline (with community env) Results:")
            print(f"Average Total Return: {avg_return:.2f}")
            print(f"Average Total Cost: {avg_cost:.2f}")
            print(f"Average Peak Load: {avg_peak_load:.2f} kW")
            print("=" * 60)
            return results
        
        # ---------- 原逻辑：单智能体环境 ----------
        for episode in range(num_episodes):
            states = []
            for env in self.eval_envs:
                state = env.reset()
                if dates[episode] != env.current_time:
                    env.current_time = dates[episode]
                    env.current_time_index = 0
                states.append(state)
            
            done = False
            episode_returns = [0.0] * self.n_agents
            episode_costs = [0.0] * self.n_agents
            community_loads = []
            step = 0
            while not done:
                actions = []
                for i, (state, agent) in enumerate(zip(states, self.agents)):
                    action = agent.take_action(state)
                    actions.append(action)
                next_states = []
                all_dones = []
                for i, (env, state, action) in enumerate(zip(self.eval_envs, states, actions)):
                    next_state, reward, done = env.step(state, action)
                    next_states.append(next_state)
                    all_dones.append(done)
                    episode_returns[i] += reward
                    episode_costs[i] += env.current_step_cost
                agent_net_loads = []
                for env in self.eval_envs:
                    state = env.state
                    total_consumption = (
                        state['home_load']
                        + max(0, env.current_ev_power)
                        + max(0, env.current_battery_power)
                        + state['Air_conditioner_power']
                        + state['Air_conditioner_power2']
                        + state.get('wash_machine_state', 0) * env.wash_machine_power
                        + state.get('ewh_power', 0)
                    )
                    total_generation = (
                        state['pv_generation']
                        + max(0, -env.current_ev_power)
                        + max(0, -env.current_battery_power)
                    )
                    agent_net_loads.append(total_consumption - total_generation)
                community_loads.append(sum(agent_net_loads))
                states = next_states
                done = all(all_dones)
                step += 1
                if step >= 48:
                    done = True
            
            results['episode_returns'].append(sum(episode_returns))
            results['episode_costs'].append(sum(episode_costs))
            for i in range(self.n_agents):
                results['agent_returns'][i].append(episode_returns[i])
                results['agent_costs'][i].append(episode_costs[i])
            peak_load = max(community_loads) if community_loads else 0
            results['community_peak_loads'].append(peak_load)
            print(f"Episode {episode + 1}/{num_episodes}: Total Return = {sum(episode_returns):.2f}, Total Cost = {sum(episode_costs):.2f}, Peak Load = {peak_load:.2f} kW")
        
        avg_return = np.mean(results['episode_returns'])
        avg_cost = np.mean(results['episode_costs'])
        avg_peak_load = np.mean(results['community_peak_loads'])
        print("\n" + "=" * 60)
        print("Independent Learning Baseline Results:")
        print(f"Average Total Return: {avg_return:.2f}")
        print(f"Average Total Cost: {avg_cost:.2f}")
        print(f"Average Peak Load: {avg_peak_load:.2f} kW")
        print("=" * 60)
        return results
    
    def calculate_baseline_peak(self, num_episodes=10):
        """
        计算基准峰值（用于归一化峰值惩罚）
        
        Args:
            num_episodes: 评估轮数
        
        Returns:
            float: 基准峰值
        """
        print("Calculating baseline peak load...")
        results = self.evaluate(num_episodes=num_episodes)
        baseline_peak = np.mean(results['community_peak_loads'])
        print(f"Baseline Peak Load: {baseline_peak:.2f} kW")
        return baseline_peak


if __name__ == "__main__":
 
    import argparse
    
    parser = argparse.ArgumentParser(description='Train and evaluate independent learning baseline')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes per agent (default: 1000)')
    parser.add_argument('--eval_only', action='store_true',
                       help='Only evaluate (requires trained models)')
    parser.add_argument('--calc_peak_only', action='store_true',
                       help='Only calculate baseline peak (requires trained models)')
    
    args = parser.parse_args()
    
    # 创建独立学习基线
    baseline = IndependentBaseline(
        n_agents=3,
        pv_coefficients=[4.0,4.0,4.0]  # 正常天气
    )
    
    if args.calc_peak_only:
        # 仅计算基准峰值（需要先训练）
        baseline.load_models()
        baseline_peak = baseline.calculate_baseline_peak(num_episodes=10)
        print(f"\n{'='*60}")
        print(f"计算得到的基准峰值: {baseline_peak:.2f} kW")
        print(f"请在创建MultiAgentHEMEnv时使用: baseline_peak={baseline_peak:.2f}")
        print(f"{'='*60}")
    elif args.eval_only:
        # 仅评估（需要先训练）
        baseline.load_models()
        results = baseline.evaluate(num_episodes=10)
        print(f"\n{'='*60}")
        print("评估完成")
        print(f"{'='*60}")
    else:
        # 训练 + 计算基准峰值
        print(f"\n{'='*60}")
        print("开始训练独立学习基线")
        print(f"每个智能体训练 {args.episodes} 轮")
        print(f"{'='*60}")
        
        # 训练
        baseline.train(num_episodes=args.episodes, save_dir='multi_agent/baselines/models')
        
        # 计算基准峰值
        print(f"\n{'='*60}")
        print("计算基准峰值")
        print(f"{'='*60}")
        baseline_peak = baseline.calculate_baseline_peak(num_episodes=10)
        
        print(f"\n{'='*60}")
        print(f"训练完成！基准峰值: {baseline_peak:.2f} kW")
        print(f"\n请在使用MultiAgentHEMEnv时使用以下参数:")
        print(f"  baseline_peak={baseline_peak:.2f}")
        print(f"{'='*60}")
