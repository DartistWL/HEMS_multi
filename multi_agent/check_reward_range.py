"""
检测多智能体环境中奖励函数的奖励值范围
Check reward value range in multi-agent environment
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from multi_agent.environment.multi_agent_env import MultiAgentHEMEnv
from multi_agent.baselines.rule_based_baseline import RuleBasedBaseline


def check_reward_range(num_episodes=10, baseline_peak=31.01):
    """
    检测奖励值范围
    
    Args:
        num_episodes: 测试轮数
        baseline_peak: 基准峰值
    """
    print("=" * 80)
    print("检测奖励函数奖励值范围")
    print("=" * 80)
    
    # 创建环境
    env = MultiAgentHEMEnv(
        n_agents=3,
        community_ess_capacity=36.0,
        baseline_peak=baseline_peak,
        community_weight=0.2,
        pv_coefficients=[2.0, 2.0, 2.0]
    )
    
    # 设置评估日期（3天：1天晴天+1天阴天+1天正常）
    eval_dates = ['2011-07-10', '2011-07-11', '2011-07-12']
    pv_coefficients_list = [
        [3.0, 3.0, 3.0],  # 晴天
        [1.0, 1.0, 1.0],  # 阴天
        [2.0, 2.0, 2.0],  # 正常
    ]
    env.set_evaluation_dates(eval_dates, pv_coefficients_list)
    
    # 使用规则基线策略（简单策略，用于测试）
    baseline = RuleBasedBaseline(peak_threshold_factor=1.2)
    
    # 收集奖励数据
    all_step_rewards = []  # 每一步的奖励（所有智能体的总和）
    all_agent_rewards = [[] for _ in range(3)]  # 每个智能体的奖励
    all_episode_returns = []  # 每个episode的总回报
    all_episode_agent_returns = [[] for _ in range(3)]  # 每个智能体每个episode的回报
    
    print(f"\n运行 {num_episodes} 个episode收集奖励数据...")
    
    for episode in range(num_episodes):
        date_index = episode % len(eval_dates)
        states = env.reset(mode='eval', date_index=date_index)
        
        episode_return = 0.0
        episode_agent_returns = [0.0] * 3
        step_count = 0
        done = False
        
        while not done:
            # 使用规则基线策略选择动作
            community_state = env.get_community_state()
            agent_net_load_histories = [agent.net_load_history for agent in env.agents]
            current_community_load = sum([agent.get_net_load() for agent in env.agents])
            community_state['community_net_load'] = current_community_load
            
            actions = []
            for i, state in enumerate(states):
                action = baseline.select_action(
                    state, i, community_state, agent_net_load_histories[i]
                )
                actions.append(action)
            
            # 执行动作
            next_states, rewards, dones, info = env.step(actions)
            
            # 记录奖励
            step_total_reward = sum(rewards)
            all_step_rewards.append(step_total_reward)
            for i in range(3):
                all_agent_rewards[i].append(rewards[i])
                episode_agent_returns[i] += rewards[i]
            
            episode_return += step_total_reward
            states = next_states
            done = all(dones)
            step_count += 1
            
            if step_count >= 48:
                done = True
        
        # 记录episode回报
        all_episode_returns.append(episode_return)
        for i in range(3):
            all_episode_agent_returns[i].append(episode_agent_returns[i])
        
        if (episode + 1) % 5 == 0:
            print(f"  完成 {episode + 1}/{num_episodes} episodes")
    
    # 转换为numpy数组
    all_step_rewards = np.array(all_step_rewards)
    all_agent_rewards = [np.array(rewards) for rewards in all_agent_rewards]
    all_episode_returns = np.array(all_episode_returns)
    all_episode_agent_returns = [np.array(returns) for returns in all_episode_agent_returns]
    
    # 打印统计信息
    print("\n" + "=" * 80)
    print("奖励值统计信息")
    print("=" * 80)
    
    print("\n【每步总奖励（所有智能体总和）】")
    print(f"  最小值: {np.min(all_step_rewards):.4f}")
    print(f"  最大值: {np.max(all_step_rewards):.4f}")
    print(f"  平均值: {np.mean(all_step_rewards):.4f}")
    print(f"  中位数: {np.median(all_step_rewards):.4f}")
    print(f"  标准差: {np.std(all_step_rewards):.4f}")
    print(f"  25%分位数: {np.percentile(all_step_rewards, 25):.4f}")
    print(f"  75%分位数: {np.percentile(all_step_rewards, 75):.4f}")
    
    print("\n【每个智能体每步奖励】")
    for i in range(3):
        rewards = all_agent_rewards[i]
        print(f"  智能体 {i+1}:")
        print(f"    最小值: {np.min(rewards):.4f}")
        print(f"    最大值: {np.max(rewards):.4f}")
        print(f"    平均值: {np.mean(rewards):.4f}")
        print(f"    标准差: {np.std(rewards):.4f}")
    
    print("\n【每个Episode总回报（所有智能体总和）】")
    print(f"  最小值: {np.min(all_episode_returns):.4f}")
    print(f"  最大值: {np.max(all_episode_returns):.4f}")
    print(f"  平均值: {np.mean(all_episode_returns):.4f}")
    print(f"  标准差: {np.std(all_episode_returns):.4f}")
    
    print("\n【每个智能体每个Episode回报】")
    for i in range(3):
        returns = all_episode_agent_returns[i]
        print(f"  智能体 {i+1}:")
        print(f"    最小值: {np.min(returns):.4f}")
        print(f"    最大值: {np.max(returns):.4f}")
        print(f"    平均值: {np.mean(returns):.4f}")
        print(f"    标准差: {np.std(returns):.4f}")
    
    # 奖励缩放建议
    print("\n" + "=" * 80)
    print("奖励缩放建议")
    print("=" * 80)
    
    # 计算合理的缩放因子（使奖励范围在-10到10之间）
    max_abs_value = max(np.abs(np.min(all_step_rewards)), np.abs(np.max(all_step_rewards)))
    suggested_scale = max_abs_value / 10.0
    
    print(f"\n当前每步奖励范围: [{np.min(all_step_rewards):.2f}, {np.max(all_step_rewards):.2f}]")
    print(f"当前每步奖励绝对值最大值: {max_abs_value:.2f}")
    print(f"\n建议的奖励缩放因子: {suggested_scale:.1f}")
    print(f"  (使用此缩放因子后，奖励范围约为: [{np.min(all_step_rewards)/suggested_scale:.2f}, {np.max(all_step_rewards)/suggested_scale:.2f}])")
    
    # 考虑Episode回报的缩放
    max_abs_episode_return = max(np.abs(np.min(all_episode_returns)), np.abs(np.max(all_episode_returns)))
    suggested_scale_episode = max_abs_episode_return / 100.0
    
    print(f"\n当前Episode回报范围: [{np.min(all_episode_returns):.2f}, {np.max(all_episode_returns):.2f}]")
    print(f"当前Episode回报绝对值最大值: {max_abs_episode_return:.2f}")
    print(f"基于Episode回报的建议缩放因子: {suggested_scale_episode:.1f}")
    print(f"  (使用此缩放因子后，Episode回报范围约为: [{np.min(all_episode_returns)/suggested_scale_episode:.2f}, {np.max(all_episode_returns)/suggested_scale_episode:.2f}])")
    
    # 绘制奖励分布图
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Reward Value Distribution', fontsize=16)
        
        # 1. 每步总奖励分布
        ax1 = axes[0, 0]
        ax1.hist(all_step_rewards, bins=50, alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(all_step_rewards), color='r', linestyle='--', label=f'Mean: {np.mean(all_step_rewards):.2f}')
        ax1.set_xlabel('Step Total Reward')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Step Total Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 每个智能体每步奖励分布
        ax2 = axes[0, 1]
        for i in range(3):
            ax2.hist(all_agent_rewards[i], bins=30, alpha=0.5, label=f'Agent {i+1}')
        ax2.set_xlabel('Step Reward')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Step Rewards per Agent')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Episode总回报分布
        ax3 = axes[1, 0]
        ax3.hist(all_episode_returns, bins=20, alpha=0.7, edgecolor='black')
        ax3.axvline(np.mean(all_episode_returns), color='r', linestyle='--', label=f'Mean: {np.mean(all_episode_returns):.2f}')
        ax3.set_xlabel('Episode Total Return')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Episode Total Returns')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 每个智能体Episode回报分布
        ax4 = axes[1, 1]
        for i in range(3):
            ax4.hist(all_episode_agent_returns[i], bins=20, alpha=0.5, label=f'Agent {i+1}')
        ax4.set_xlabel('Episode Return')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Episode Returns per Agent')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        plot_path = 'multi_agent/reward_distribution.png'
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n奖励分布图已保存到: {plot_path}")
        
        try:
            plt.show()
        except:
            pass
        
        plt.close()
    except Exception as e:
        print(f"\n绘制分布图时出错: {e}")
    
    print("\n" + "=" * 80)
    
    return {
        'step_rewards': all_step_rewards,
        'agent_rewards': all_agent_rewards,
        'episode_returns': all_episode_returns,
        'episode_agent_returns': all_episode_agent_returns,
        'suggested_scale': suggested_scale,
        'suggested_scale_episode': suggested_scale_episode
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Check reward value range')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to test (default: 10)')
    parser.add_argument('--baseline_peak', type=float, default=31.01,
                       help='Baseline peak load (default: 31.01)')
    
    args = parser.parse_args()
    
    try:
        results = check_reward_range(
            num_episodes=args.episodes,
            baseline_peak=args.baseline_peak
        )
        
        print("\n检测完成！")
        
    except KeyboardInterrupt:
        print("\n\n检测被用户中断。")
    except Exception as e:
        print(f"\n\n检测过程中出错: {e}")
        import traceback
        traceback.print_exc()
