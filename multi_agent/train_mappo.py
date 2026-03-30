"""
MAPPO训练脚本
Train MAPPO algorithm for multi-agent home energy management
"""
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from multi_agent.algorithms.mappo import MAPPO
from multi_agent.environment.multi_agent_env import MultiAgentHEMEnv
from multi_agent.utils.data_interface import MultiAgentDataInterface


def load_config(config_path='multi_agent/config.json', training_only=False):
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        training_only: 若 True 仅返回 training 段（兼容旧用法）；若 False 返回完整 config
    
    Returns:
        dict: 完整配置或 training 段
    """
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config.get('training', {}) if training_only else config
    else:
        print(f"Warning: Config file not found at {config_path}, using default values.")
        return {} if training_only else {}


def train_mappo(num_episodes=1000, baseline_peak=None, save_dir='multi_agent/algorithms/models', 
                load_model=None, eval_freq=100, save_freq=500, config_path='multi_agent/config.json'):
    """
    训练MAPPO算法
    
    Args:
        num_episodes: 训练轮数
        baseline_peak: 基准峰值（用于归一化峰值惩罚）
        save_dir: 模型保存目录
        load_model: 要加载的模型目录（如果为None则不加载）
        eval_freq: 评估频率（每多少轮评估一次）
        save_freq: 保存频率（每多少轮保存一次）
        config_path: 配置文件路径
    """
    # 加载配置（完整 config，用于 training 与 credit_pricing）
    full_config = load_config(config_path)
    config = full_config.get('training', {}) if isinstance(full_config, dict) and full_config else {}
    credit_scheme = (full_config.get('credit_pricing') or {}).get('scheme', 'uniform') if isinstance(full_config, dict) else 'uniform'
    if credit_scheme == 'contribution_based':
        save_dir = save_dir.rstrip(os.sep) + '_contribution_based'
        print(f"credit_pricing.scheme=contribution_based -> 模型保存到 {save_dir}，不覆盖原模型")
    use_popart = config.get('use_popart_normalization', False)
    use_state_norm = config.get('use_state_normalization', True)
    use_reward_norm = config.get('use_reward_normalization', False)
    ent_coef = config.get('ent_coef', 0.05)
    community_weight = config.get('community_weight', 0.2)
    community_credit_cost_weight = config.get('community_credit_cost_weight', 0.1)
    community_credit_benefit_weight = config.get('community_credit_benefit_weight', 0.1)
    initial_credit = config.get('initial_credit', 100.0)  # 默认初始积分为100.0
    peak_penalty_exponent = config.get('peak_penalty_exponent', 2.0)
    peak_discharge_bonus = config.get('peak_discharge_bonus', 0.0)
    peak_credit_cost_reduction = config.get('peak_credit_cost_reduction', 1.0)
    critic_loss_type = config.get('critic_loss_type', 'mse')
    huber_delta = config.get('huber_delta', 1.0)
    # baseline_peak优先级：命令行参数 > 配置文件 > 默认值(31.01)
    # 如果函数参数baseline_peak为None（未从命令行提供），则从配置文件读取
    if baseline_peak is None:
        baseline_peak = config.get('baseline_peak', 31.01)  # 配置文件没有时使用默认值31.01
    
    print("=" * 80)
    print("Training MAPPO Algorithm")
    print("=" * 80)
    print(f"Number of episodes: {num_episodes}")
    print(f"Baseline peak: {baseline_peak} kW")
    print(f"Save directory: {save_dir}")
    print(f"Use Pop-Art normalization: {use_popart}")
    print(f"Use state normalization: {use_state_norm}")
    print(f"Use reward normalization: {use_reward_norm}")
    print(f"Entropy coefficient: {ent_coef}")
    print(f"Community weight: {community_weight}")
    print(f"Credit cost weight: {community_credit_cost_weight}")
    print(f"Credit benefit weight: {community_credit_benefit_weight}")
    print(f"Initial credit: {initial_credit}")
    print(f"Peak penalty exponent: {peak_penalty_exponent}")
    print(f"Peak discharge bonus: {peak_discharge_bonus}")
    print(f"Peak credit cost reduction: {peak_credit_cost_reduction}")
    print(f"Critic loss type: {critic_loss_type}")
    if critic_loss_type == 'huber':
        print(f"Huber delta: {huber_delta}")
    print("=" * 80)
    
    # 创建环境
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
        pv_coefficients=[2.0, 2.0, 2.0]  # 初始PV系数（会被训练数据覆盖）
    )
    
    # 设置训练日期（7天：4天晴天+3天阴天）
    training_dates = [
        '2011-07-03',  # 晴天
        '2011-07-04',  # 晴天
        '2011-07-05',  # 晴天
        '2011-07-06',  # 晴天
        '2011-07-07',  # 阴天
        '2011-07-08',  # 阴天
        '2011-07-09',  # 阴天
    ]
    
    # 使用统一的天气系数函数获取光伏系数
    sunny_coef = MultiAgentDataInterface.get_weather_coefficient('sunny')
    cloudy_coef = MultiAgentDataInterface.get_weather_coefficient('cloudy')
    
    # 设置光伏系数（从data_interface中获取）
    pv_coefficients_list = [
        [sunny_coef, sunny_coef, sunny_coef],  # 晴天
        [sunny_coef, sunny_coef, sunny_coef],  # 晴天
        [sunny_coef, sunny_coef, sunny_coef],  # 晴天
        [sunny_coef, sunny_coef, sunny_coef],  # 晴天
        [cloudy_coef, cloudy_coef, cloudy_coef],  # 阴天
        [cloudy_coef, cloudy_coef, cloudy_coef],  # 阴天
        [cloudy_coef, cloudy_coef, cloudy_coef],  # 阴天
    ]
    
    env.set_training_dates(training_dates, pv_coefficients_list)
    
    # 创建MAPPO算法
    mappo = MAPPO(
        env=env,
        n_agents=3,
        hidden_dim=128,
        gamma=0.96,
        lmbda=0.95,
        eps=0.2,
        epochs=10,
        lr_actor=1e-4,
        lr_critic=1e-4,
        batch_size=64,
        ent_coef=ent_coef,
        max_grad_norm=1.0,
        use_state_normalization=use_state_norm,
        use_popart=use_popart,
        reward_scale=1.0,
        use_reward_normalization=use_reward_norm,
        critic_loss_type=critic_loss_type,
        huber_delta=huber_delta
    )
    
    # 加载模型（如果指定）
    if load_model is not None and os.path.exists(load_model):
        print(f"\nLoading model from {load_model}...")
        mappo.load(load_model)
        print("Model loaded successfully!")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 训练统计
    training_stats = {
        'episode_returns': [],
        'episode_costs': [],
        'peak_loads': [],
        'actor_loss': [],
        'critic_loss': [],
        'entropy': [],
        'credit_balances': [[], [], []]  # 三户家庭的积分余额
    }
    
    # 最后一个episode的详细数据（用于绘制时间步级别的积分变化）
    last_episode_details = {
        'time_steps': [],
        'credit_balances': [[], [], []],  # 每个时间步的积分余额（三个智能体）
        'credit_changes': [[], [], []],   # 每个时间步的积分变化（三个智能体）
        'grid_prices': [],                # 每个时间步的电价
        'agent_loads': [[], [], []],      # 每个时间步的负荷（三个智能体）
        'community_load': [],             # 每个时间步的社区总负荷
        'community_ess_soc': []           # 每个时间步的社区储能SOC
    }
    
    print("\n" + "=" * 80)
    print("Starting Training...")
    print("=" * 80)
    
    # 最后一个episode的详细数据（用于绘制时间步级别的积分变化）
    last_episode_details = {
        'time_steps': [],
        'credit_balances': [[], [], []],  # 每个时间步的积分余额（三个智能体）
        'credit_changes': [[], [], []],   # 每个时间步的积分变化（三个智能体）
        'grid_prices': [],                # 每个时间步的电价
        'agent_loads': [[], [], []],      # 每个时间步的负荷（三个智能体）
        'community_load': []              # 每个时间步的社区总负荷
    }
    
    # 训练循环
    num_training_dates = len(training_dates)
    
    for episode in range(num_episodes):
        # 选择训练日期（循环使用）
        date_index = episode % num_training_dates
        
        # 重置环境
        states = env.reset(mode='train', date_index=date_index)
        global_state = env.get_community_state()  # 用于兼容性（实际不使用）
        
        # 重置经验缓冲区
        mappo.reset_buffer()
        
        # Episode统计
        episode_return = 0.0
        episode_cost = 0.0
        episode_peak_load = 0.0
        step_count = 0
        done = False
        
        # 如果是最后一个episode，记录详细数据
        is_last_episode = (episode == num_episodes - 1)
        if is_last_episode:
            # 清空之前的数据
            last_episode_details = {
                'time_steps': [],
                'credit_balances': [[], [], []],
                'credit_changes': [[], [], []],
                'grid_prices': [],
                'agent_loads': [[], [], []],
                'community_load': [],
                'community_ess_soc': []
            }
            # 记录初始积分余额
            initial_credits = env.credit_system.get_all_balances()
            for i in range(3):
                last_episode_details['credit_balances'][i].append(initial_credits.get(i, 100.0))  # 初始积分为100.0
                last_episode_details['credit_changes'][i].append(0.0)
        
        # 运行一个episode
        while not done:
            # 选择动作
            actions, action_log_probs = mappo.select_actions(states, deterministic=False)
            
            # 如果是最后一个episode，记录当前时间步的电价和负荷
            if is_last_episode:
                current_time_idx = env.agents[0].env.current_time_index
                grid_price = env.agents[0].env.data_interface.get_electricity_price(
                    env.current_date, current_time_idx
                )
                last_episode_details['grid_prices'].append(grid_price)
                last_episode_details['time_steps'].append(current_time_idx)
                
                # 记录每个智能体的负荷
                for i in range(3):
                    agent_net_load = env.agents[i].get_net_load()
                    last_episode_details['agent_loads'][i].append(agent_net_load)
            
            # 执行动作
            next_states, rewards, dones, info = env.step(actions)
            next_global_state = env.get_community_state()  # 用于兼容性（实际不使用）
            
            # 如果是最后一个episode，记录积分变化和社区负荷
            if is_last_episode:
                # 记录社区总负荷
                last_episode_details['community_load'].append(info['community_net_load'])
                
                # 记录社区储能SOC（从info中获取，确保数据一致性）
                last_episode_details['community_ess_soc'].append(info['community_ess_soc'])
                
                # 记录积分余额和变化（在step之后，积分已经更新）
                current_credits = env.credit_system.get_all_balances()
                for i in range(3):
                    current_balance = current_credits.get(i, 1000.0)
                    # 计算积分变化（相对于上一个时间步）
                    if len(last_episode_details['credit_balances'][i]) > 0:
                        previous_balance = last_episode_details['credit_balances'][i][-1]
                        credit_change = current_balance - previous_balance
                    else:
                        credit_change = 0.0  # 初始时间步，无变化
                    
                    last_episode_details['credit_balances'][i].append(current_balance)
                    last_episode_details['credit_changes'][i].append(credit_change)
            
            # 存储经验
            mappo.store_transition(
                local_states=states,
                global_state=global_state,
                actions=actions,
                action_log_probs=action_log_probs,
                rewards=rewards,
                next_local_states=next_states,
                next_global_state=next_global_state,
                dones=dones
            )
            
            # 更新统计
            episode_return += sum(rewards)
            episode_cost += sum([env.agents[i].env.current_step_cost for i in range(3)])
            episode_peak_load = max(episode_peak_load, info['community_net_load'])
            
            states = next_states
            global_state = next_global_state
            done = all(dones)
            step_count += 1
            
            # 防止无限循环
            if step_count >= 48:
                done = True
        
        # 更新策略（每个episode结束后）
        update_stats = mappo.update()
        
        # 获取三户家庭的积分余额
        credit_balances = env.credit_system.get_all_balances()
        for i in range(3):
            training_stats['credit_balances'][i].append(credit_balances.get(i, 100.0))  # 初始积分为100.0
        
        # 记录统计信息
        training_stats['episode_returns'].append(episode_return)
        training_stats['episode_costs'].append(episode_cost)
        training_stats['peak_loads'].append(episode_peak_load)
        
        if update_stats:
            training_stats['actor_loss'].append(update_stats.get('actor_loss', 0))
            training_stats['critic_loss'].append(update_stats.get('critic_loss', 0))
            training_stats['entropy'].append(update_stats.get('entropy', 0))
        
        # 打印进度
        if (episode + 1) % 10 == 0:
            window_size = min(10, len(training_stats['episode_returns']))
            avg_return = np.mean(training_stats['episode_returns'][-window_size:])
            avg_cost = np.mean(training_stats['episode_costs'][-window_size:])
            avg_peak = np.mean(training_stats['peak_loads'][-window_size:])
            
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Return: {episode_return:.2f} (avg: {avg_return:.2f}) | "
                  f"Cost: {episode_cost:.2f} (avg: {avg_cost:.2f}) | "
                  f"Peak: {episode_peak_load:.2f} kW (avg: {avg_peak:.2f} kW)")
            
            # 打印奖励分解（诊断信息）
            if hasattr(env, '_reward_breakdown') and len(env._reward_breakdown['total_reward']) > 0:
                breakdown = env._reward_breakdown
                # 计算这一episode的平均值
                n_steps = len(breakdown['total_reward'])
                avg_ind = np.mean(breakdown['individual_reward'][-n_steps:]) if breakdown['individual_reward'] else 0
                avg_peak_pen = np.mean(breakdown['peak_penalty_value'][-n_steps:]) if breakdown['peak_penalty_value'] else 0
                avg_queue_pen = np.mean(breakdown['queue_penalty'][-n_steps:]) if breakdown['queue_penalty'] else 0
                avg_ess_reward = np.mean(breakdown['community_ess_reward'][-n_steps:]) if breakdown['community_ess_reward'] else 0
                avg_credit_pen = np.mean(breakdown['credit_penalty'][-n_steps:]) if breakdown['credit_penalty'] else 0
                
                # 打印充电功率统计（诊断queue_penalty为什么是0）
                charge_stats = env._charge_power_stats if hasattr(env, '_charge_power_stats') else None
                if charge_stats and len(charge_stats['total_charge_power_all']) > 0:
                    max_charge = charge_stats['max_total_charge']
                    trigger_count = charge_stats['trigger_count']
                    avg_charge = np.mean(charge_stats['total_charge_power_all'][-n_steps:]) if len(charge_stats['total_charge_power_all']) >= n_steps else np.mean(charge_stats['total_charge_power_all'])
                    print(f"  Reward Breakdown (per step avg): "
                          f"Individual={avg_ind:.2f}, "
                          f"PeakPenalty={avg_peak_pen:.2f}, "
                          f"QueuePenalty={avg_queue_pen:.2f}, "
                          f"ESSReward={avg_ess_reward:.2f}, "
                          f"CreditPenalty={avg_credit_pen:.2f}")
                    print(f"  Charge Power Stats: "
                          f"Avg={avg_charge:.2f}kW, "
                          f"Max={max_charge:.2f}kW, "
                          f"TriggerCount={trigger_count}/{n_steps} "
                          f"(target={env.queue_target_power:.1f}kW)")
                else:
                    print(f"  Reward Breakdown (per step avg): "
                          f"Individual={avg_ind:.2f}, "
                          f"PeakPenalty={avg_peak_pen:.2f}, "
                          f"QueuePenalty={avg_queue_pen:.2f}, "
                          f"ESSReward={avg_ess_reward:.2f}, "
                          f"CreditPenalty={avg_credit_pen:.2f}")
            
            if update_stats:
                print(f"  Loss: Actor={update_stats.get('actor_loss', 0):.4f}, "
                      f"Critic={update_stats.get('critic_loss', 0):.4f}, "
                      f"Entropy={update_stats.get('entropy', 0):.4f}")
        
        # 定期评估
        if (episode + 1) % eval_freq == 0:
            print(f"\nEvaluating at episode {episode + 1}...")
            eval_results = evaluate_mappo(mappo, baseline_peak=baseline_peak, num_episodes=3)
            print(f"Eval - Avg Return: {np.mean(eval_results['returns']):.2f}, "
                  f"Avg Cost: {np.mean(eval_results['costs']):.2f}, "
                  f"Avg Peak: {np.mean(eval_results['peak_loads']):.2f} kW")
        
        # 定期保存模型
        if (episode + 1) % save_freq == 0:
            print(f"\nSaving model at episode {episode + 1}...")
            mappo.save(save_dir)
            print("Model saved!")
    
    # 训练完成，保存最终模型
    print("\n" + "=" * 80)
    print("Training Completed! Saving final model...")
    print("=" * 80)
    mappo.save(save_dir)
    
    # 保存训练统计数据到JSON文件（便于后续分析）
    stats_file = os.path.join(save_dir, 'training_stats.json')
    # 转换为可序列化的格式
    stats_dict = {
        'episode_returns': training_stats['episode_returns'],
        'episode_costs': training_stats['episode_costs'],
        'peak_loads': training_stats['peak_loads'],
        'credit_balances': {
            f'agent_{i}': training_stats['credit_balances'][i] 
            for i in range(3)
        },
        'actor_loss': training_stats.get('actor_loss', []),
        'critic_loss': training_stats.get('critic_loss', []),
        'entropy': training_stats.get('entropy', [])
    }
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats_dict, f, indent=2, ensure_ascii=False)
    print(f"训练统计数据已保存到: {stats_file}")
    
    # 绘制训练曲线
    plot_training_curves(training_stats, save_dir, initial_credit=initial_credit)
    
    # 绘制最后一个episode的详细积分变化（如果存在）
    if len(last_episode_details['time_steps']) > 0:
        plot_last_episode_credit_details(last_episode_details, save_dir, initial_credit=initial_credit)
    
    print("\n" + "=" * 80)
    print("Training Summary")
    print("=" * 80)
    print(f"Average Return (last 100): {np.mean(training_stats['episode_returns'][-100:]):.2f}")
    print(f"Average Cost (last 100): {np.mean(training_stats['episode_costs'][-100:]):.2f}")
    print(f"Average Peak Load (last 100): {np.mean(training_stats['peak_loads'][-100:]):.2f} kW")
    print(f"Final model saved to: {save_dir}")
    print("=" * 80)
    
    return mappo, training_stats


def evaluate_mappo(mappo, baseline_peak=31.01, num_episodes=10, mode='eval'):
    """
    评估MAPPO算法
    
    Args:
        mappo: MAPPO算法实例
        baseline_peak: 基准峰值
        num_episodes: 评估轮数
        mode: 评估模式（'eval'或'train'）
    
    Returns:
        dict: 评估结果
    """
    # 创建评估环境
    env = MultiAgentHEMEnv(
        n_agents=3,
        community_ess_capacity=36.0,
        baseline_peak=baseline_peak,
        community_weight=0.2,
        pv_coefficients=[2.0, 2.0, 2.0]
    )
    
    # 设置评估日期（3天：1天晴天+1天阴天+1天正常）
    eval_dates = ['2011-07-10', '2011-07-11', '2011-07-12']
    
    # 使用统一的天气系数函数获取光伏系数
    sunny_coef = MultiAgentDataInterface.get_weather_coefficient('sunny')
    cloudy_coef = MultiAgentDataInterface.get_weather_coefficient('cloudy')
    normal_coef = MultiAgentDataInterface.get_weather_coefficient('normal')
    
    pv_coefficients_list = [
        [sunny_coef, sunny_coef, sunny_coef],  # 晴天
        [cloudy_coef, cloudy_coef, cloudy_coef],  # 阴天
        [normal_coef, normal_coef, normal_coef],  # 正常
    ]
    env.set_evaluation_dates(eval_dates, pv_coefficients_list)
    
    results = {
        'returns': [],
        'costs': [],
        'peak_loads': [],
        'peak_penalties': []
    }
    
    num_eval_dates = len(eval_dates)
    
    for episode in range(num_episodes):
        # 选择评估日期（循环使用）
        date_index = episode % num_eval_dates
        
        # 重置环境
        states = env.reset(mode=mode, date_index=date_index)
        
        episode_return = 0.0
        episode_cost = 0.0
        episode_peak_load = 0.0
        episode_peak_penalty = 0.0
        step_count = 0
        done = False
        
        # 运行一个episode
        while not done:
            # 选择动作（确定性策略）
            actions, _ = mappo.select_actions(states, deterministic=True)
            
            # 执行动作
            next_states, rewards, dones, info = env.step(actions)
            
            # 更新统计
            episode_return += sum(rewards)
            # 计算成本（从每个智能体的环境中获取）
            step_cost = sum([env.agents[i].env.current_step_cost for i in range(3)])
            episode_cost += step_cost
            episode_peak_load = max(episode_peak_load, info['community_net_load'])
            episode_peak_penalty += info['peak_penalty']
            
            states = next_states
            done = all(dones)
            step_count += 1
            
            if step_count >= 48:
                done = True
        
        results['returns'].append(episode_return)
        results['costs'].append(episode_cost)
        results['peak_loads'].append(episode_peak_load)
        results['peak_penalties'].append(episode_peak_penalty)
    
    return results


def plot_last_episode_credit_details(last_episode_details, save_dir, initial_credit=100.0):
    """
    绘制最后一个episode中每个时间步的积分变化、电价和负荷关系图
    
    Args:
        last_episode_details: 最后一个episode的详细数据字典
        save_dir: 保存目录
        initial_credit: 初始积分值（用于绘制参考线）
    """
    time_steps = last_episode_details['time_steps']
    if len(time_steps) == 0:
        print("No detailed episode data to plot.")
        return
    
    # 创建图形：3行2列
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('最后一个Episode的积分、电价和负荷变化关系', fontsize=16, fontweight='bold')
    
    colors_agent = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    labels_agent = ['家庭 1', '家庭 2', '家庭 3']
    
    # 1. 积分余额变化（左上方）
    ax1 = axes[0, 0]
    for i in range(3):
        credit_balances = last_episode_details['credit_balances'][i]
        if len(credit_balances) > 0:
            # credit_balances包含初始值和所有时间步的值，需要与time_steps对齐
            # 如果credit_balances比time_steps多1，去掉第一个初始值
            if len(credit_balances) == len(time_steps) + 1:
                balance_data = credit_balances[1:]
            elif len(credit_balances) == len(time_steps):
                balance_data = credit_balances
            elif len(credit_balances) > len(time_steps):
                balance_data = credit_balances[-len(time_steps):]
            else:
                # 数据不足，补初始值
                initial_balance = credit_balances[0] if len(credit_balances) > 0 else 1000.0
                balance_data = [initial_balance] * (len(time_steps) - len(credit_balances)) + credit_balances
            
            ax1.plot(time_steps, balance_data, 
                    color=colors_agent[i], label=labels_agent[i], 
                    linewidth=2, alpha=0.8, marker='o', markersize=4)
    
    # 计算y轴范围（根据实际数据自动调整）
    all_balance_values = []
    for i in range(3):
        credit_balances = last_episode_details['credit_balances'][i]
        if len(credit_balances) > 0:
            if len(credit_balances) == len(time_steps) + 1:
                balance_data = credit_balances[1:]
            elif len(credit_balances) == len(time_steps):
                balance_data = credit_balances
            elif len(credit_balances) > len(time_steps):
                balance_data = credit_balances[-len(time_steps):]
            else:
                initial_balance = credit_balances[0] if len(credit_balances) > 0 else initial_credit
                balance_data = [initial_balance] * (len(time_steps) - len(credit_balances)) + credit_balances
            all_balance_values.extend(balance_data)
    
    # 获取初始积分（从配置中读取，或在数据中查找）
    if len(all_balance_values) > 0:
        # 通常初始积分是第一个值或最大值附近的某个值
        # 尝试从数据中推断初始积分
        possible_initial = max(all_balance_values) if all_balance_values else initial_credit
        # 或者使用数据中的第一个值
        first_balance = all_balance_values[0] if all_balance_values else initial_credit
        initial_credit_estimate = max(possible_initial, first_balance) if all_balance_values else initial_credit
    else:
        initial_credit_estimate = initial_credit
    
    # 设置y轴范围：基于实际数据范围，添加50%的上边界
    if all_balance_values:
        min_balance = min(all_balance_values)
        max_balance = max(all_balance_values)
        balance_range = max_balance - min_balance
        y_min = max(0, min_balance - balance_range * 0.1)  # 下边界留10%余量，但不低于0
        y_max = max_balance + balance_range * 0.5  # 上边界留50%余量
        # 确保y_max至少是初始积分的1.5倍（如果初始积分在数据范围内）
        if initial_credit_estimate > max_balance * 0.8:  # 如果初始积分接近最大值
            y_max = max(y_max, initial_credit_estimate * 1.5)
        ax1.set_ylim([y_min, y_max])
    
    # 绘制初始积分线（使用估计值或配置值）
    ax1.axhline(y=initial_credit_estimate, color='gray', linestyle='--', 
               linewidth=1.5, alpha=0.7, label=f'初始积分 ({initial_credit_estimate:.0f})')
    ax1.set_xlabel('时间步 (Time Step)')
    ax1.set_ylabel('积分余额')
    ax1.set_title('积分余额随时间步变化')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. 积分变化量（右上方）
    ax2 = axes[0, 1]
    for i in range(3):
        credit_changes = last_episode_details['credit_changes'][i]
        if len(credit_changes) > 0:
            # 确保长度一致
            if len(credit_changes) > len(time_steps):
                credit_changes_plot = credit_changes[:len(time_steps)]
            elif len(credit_changes) < len(time_steps):
                credit_changes_plot = credit_changes + [0.0] * (len(time_steps) - len(credit_changes))
            else:
                credit_changes_plot = credit_changes
            
            # 使用柱状图显示正负变化
            colors_bars = [colors_agent[i] if x >= 0 else '#CC0000' for x in credit_changes_plot]
            ax2.bar([t + i*0.25 for t in time_steps], credit_changes_plot, 
                   width=0.25, color=colors_agent[i], label=labels_agent[i],
                   alpha=0.7)
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax2.set_xlabel('时间步 (Time Step)')
    ax2.set_ylabel('积分变化量')
    ax2.set_title('积分变化量随时间步变化（负值=消耗，正值=获得）')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. 电价变化（左中）
    ax3 = axes[1, 0]
    grid_prices = last_episode_details['grid_prices']
    if len(grid_prices) > 0:
        if len(grid_prices) > len(time_steps):
            grid_prices_plot = grid_prices[:len(time_steps)]
        elif len(grid_prices) < len(time_steps):
            grid_prices_plot = grid_prices + [0.0] * (len(time_steps) - len(grid_prices))
        else:
            grid_prices_plot = grid_prices
            
        ax3.plot(time_steps, grid_prices_plot, color='#FF9800', 
                linewidth=2, marker='s', markersize=5, label='电网电价')
        ax3.fill_between(time_steps, grid_prices_plot, alpha=0.3, color='#FF9800')
    ax3.set_xlabel('时间步 (Time Step)')
    ax3.set_ylabel('电价 (元/kWh)')
    ax3.set_title('电网电价随时间步变化')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    # 4. 每个智能体的负荷（右中）
    ax4 = axes[1, 1]
    for i in range(3):
        agent_loads = last_episode_details['agent_loads'][i]
        if len(agent_loads) > 0:
            if len(agent_loads) > len(time_steps):
                agent_loads_plot = agent_loads[:len(time_steps)]
            elif len(agent_loads) < len(time_steps):
                agent_loads_plot = agent_loads + [0.0] * (len(time_steps) - len(agent_loads))
            else:
                agent_loads_plot = agent_loads
            
            ax4.plot(time_steps, agent_loads_plot, 
                    color=colors_agent[i], label=labels_agent[i], 
                    linewidth=2, alpha=0.8, marker='o', markersize=4)
    
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax4.set_xlabel('时间步 (Time Step)')
    ax4.set_ylabel('净负荷 (kW)')
    ax4.set_title('各家庭净负荷随时间步变化')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # 5. 社区总负荷（左下方）
    ax5 = axes[2, 0]
    community_load = last_episode_details['community_load']
    if len(community_load) > 0:
        if len(community_load) > len(time_steps):
            community_load_plot = community_load[:len(time_steps)]
        elif len(community_load) < len(time_steps):
            community_load_plot = community_load + [0.0] * (len(time_steps) - len(community_load))
        else:
            community_load_plot = community_load
        
        ax5.plot(time_steps, community_load_plot, color='#9C27B0', 
                linewidth=2.5, marker='s', markersize=5, label='社区总负荷')
        ax5.fill_between(time_steps, community_load_plot, alpha=0.3, color='#9C27B0')
    ax5.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax5.set_xlabel('时间步 (Time Step)')
    ax5.set_ylabel('负荷 (kW)')
    ax5.set_title('社区总净负荷随时间步变化')
    ax5.legend(loc='best')
    ax5.grid(True, alpha=0.3)
    
    # 6. 社区储能SOC变化（右下方）
    ax6 = axes[2, 1]
    community_ess_soc = last_episode_details['community_ess_soc']
    if len(community_ess_soc) > 0:
        if len(community_ess_soc) > len(time_steps):
            community_ess_soc_plot = community_ess_soc[:len(time_steps)]
        elif len(community_ess_soc) < len(time_steps):
            community_ess_soc_plot = community_ess_soc + [0.0] * (len(time_steps) - len(community_ess_soc))
        else:
            community_ess_soc_plot = community_ess_soc
        
        ax6.plot(time_steps, community_ess_soc_plot, color='#FF5722', 
                linewidth=2.5, marker='o', markersize=5, label='社区储能SOC')
        ax6.fill_between(time_steps, community_ess_soc_plot, alpha=0.3, color='#FF5722')
        
        # 绘制SOC限制线
        ax6.axhline(y=0.9, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='SOC上限 (0.9)')
        ax6.axhline(y=0.1, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='SOC下限 (0.1)')
        
    ax6.set_xlabel('时间步 (Time Step)')
    ax6.set_ylabel('SOC (0-1)')
    ax6.set_title('社区储能SOC随时间步变化')
    ax6.set_ylim([0, 1])  # SOC范围是0-1
    ax6.legend(loc='best', fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    plot_path = os.path.join(save_dir, 'last_episode_credit_details.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Last episode credit details plot saved to {plot_path}")
    
    try:
        plt.show()
    except:
        pass
    
    plt.close()


def plot_training_curves(training_stats, save_dir, initial_credit=100.0):
    """
    绘制训练曲线
    
    Args:
        training_stats: 训练统计数据字典
        save_dir: 保存目录
        initial_credit: 初始积分值（用于绘制参考线）
    """
    # 扩展为3x2布局，添加积分变化图
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle('MAPPO Training Curves', fontsize=16)
    
    # 1. Returns
    ax1 = axes[0, 0]
    ax1.plot(training_stats['episode_returns'], alpha=0.7)
    if len(training_stats['episode_returns']) > 10:
        window = min(50, len(training_stats['episode_returns']) // 10)
        smoothed = np.convolve(training_stats['episode_returns'], 
                              np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(training_stats['episode_returns'])), smoothed, 
                label=f'Smoothed (window={window})', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Return')
    ax1.set_title('Episode Returns')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Costs
    ax2 = axes[0, 1]
    ax2.plot(training_stats['episode_costs'], alpha=0.7)
    if len(training_stats['episode_costs']) > 10:
        window = min(50, len(training_stats['episode_costs']) // 10)
        smoothed = np.convolve(training_stats['episode_costs'], 
                              np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(training_stats['episode_costs'])), smoothed, 
                label=f'Smoothed (window={window})', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Cost')
    ax2.set_title('Episode Costs')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Peak Loads
    ax3 = axes[1, 0]
    ax3.plot(training_stats['peak_loads'], alpha=0.7)
    if len(training_stats['peak_loads']) > 10:
        window = min(50, len(training_stats['peak_loads']) // 10)
        smoothed = np.convolve(training_stats['peak_loads'], 
                              np.ones(window)/window, mode='valid')
        ax3.plot(range(window-1, len(training_stats['peak_loads'])), smoothed, 
                label=f'Smoothed (window={window})', linewidth=2)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Peak Load (kW)')
    ax3.set_title('Community Peak Loads')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Credit Balances (三户家庭积分变化)
    ax4 = axes[1, 1]
    colors_credit = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    labels_credit = ['家庭 1', '家庭 2', '家庭 3']
    for i in range(3):
        if len(training_stats['credit_balances'][i]) > 0:
            episodes = range(len(training_stats['credit_balances'][i]))
            ax4.plot(episodes, training_stats['credit_balances'][i], 
                    color=colors_credit[i], label=labels_credit[i], 
                    linewidth=2, alpha=0.8)
    # 绘制初始积分线
    ax4.axhline(y=initial_credit, color='gray', linestyle='--', 
               linewidth=1.5, alpha=0.7, label=f'初始积分 ({initial_credit:.0f})')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('积分余额')
    ax4.set_title('社区积分余额变化')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # 5. Loss and Entropy
    ax5 = axes[2, 0]
    if training_stats['actor_loss']:
        ax5_twin = ax5.twinx()
        ax5.plot(training_stats['actor_loss'], label='Actor Loss', alpha=0.7, color='blue')
        ax5.plot(training_stats['critic_loss'], label='Critic Loss', alpha=0.7, color='orange')
        ax5_twin.plot(training_stats['entropy'], label='Entropy', alpha=0.7, color='green')
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('Loss', color='black')
        ax5_twin.set_ylabel('Entropy', color='green')
        ax5.set_title('Training Loss and Entropy')
        ax5.legend(loc='upper left')
        ax5_twin.legend(loc='upper right')
        ax5.grid(True, alpha=0.3)
    
    # 6. 空白子图（用于布局平衡）
    ax6 = axes[2, 1]
    ax6.axis('off')
    
    plt.tight_layout()
    
    # 保存图片
    plot_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {plot_path}")
    
    try:
        plt.show()
    except:
        pass
    
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MAPPO algorithm')
    parser.add_argument('--episodes', type=int, default=1500,
                       help='Number of training episodes (default: 1000)')
    parser.add_argument('--baseline_peak', type=float, default=None,
                       help='Baseline peak load for normalization (default: from config.json, or 25.0)')
    parser.add_argument('--save_dir', type=str, default='multi_agent/algorithms/models',
                       help='Directory to save models (default: multi_agent/algorithms/models)')
    parser.add_argument('--load_model', type=str, default=None,
                       help='Path to load model from (default: None)')
    parser.add_argument('--eval_freq', type=int, default=1000,
                       help='Evaluation frequency (default: 100)')
    parser.add_argument('--save_freq', type=int, default=1000,
                       help='Save frequency (default: 500)')
    
    args = parser.parse_args()
    
    try:
        train_mappo(
            num_episodes=args.episodes,
            baseline_peak=args.baseline_peak,
            save_dir=args.save_dir,
            load_model=args.load_model,
            eval_freq=args.eval_freq,
            save_freq=args.save_freq
        )
        
        print("\n" + "=" * 80)
        print("Training completed successfully!")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
