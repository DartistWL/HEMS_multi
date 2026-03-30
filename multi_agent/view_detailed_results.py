"""
查看固定规则基线的详细结果
View detailed results of rule-based baseline
"""
import sys
import os
import numpy as np

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from multi_agent.baselines.rule_based_baseline import RuleBasedBaseline
from multi_agent.environment.multi_agent_env import MultiAgentHEMEnv


def view_detailed_results():
    """运行并显示固定规则基线的详细结果"""
    print("=" * 80)
    print("固定规则基线详细结果分析")
    print("=" * 80)
    
    # 创建环境
    env = MultiAgentHEMEnv(
        n_agents=3,
        community_ess_capacity=36.0,
        baseline_peak=50.0,
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
    
    # 创建固定规则基线
    baseline = RuleBasedBaseline(peak_threshold_factor=1.2)
    
    # 评估
    print("\n正在运行评估...")
    results = baseline.evaluate(env, num_episodes=3, mode='eval')
    
    # 显示详细结果
    print("\n" + "=" * 80)
    print("详细结果分析")
    print("=" * 80)
    
    num_episodes = len(results['peak_loads'])
    num_agents = len(results['episode_returns'][0]) if results['episode_returns'] else 3
    
    # 逐轮显示结果
    for episode in range(num_episodes):
        print(f"\n{'='*80}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"日期: {eval_dates[episode]}")
        print(f"天气类型: {['晴天', '阴天', '正常'][episode]}")
        print(f"光伏系数: {pv_coefficients_list[episode]}")
        print(f"{'='*80}")
        
        # 峰值信息
        peak_load = results['peak_loads'][episode]
        peak_penalty = results['peak_penalties'][episode]
        print(f"\n峰值信息:")
        print(f"  峰值负荷: {peak_load:.2f} kW")
        print(f"  峰值惩罚: {peak_penalty:.4f}")
        
        # 每个智能体的回报和成本
        episode_returns = results['episode_returns'][episode]
        episode_costs = results['episode_costs'][episode]
        
        print(f"\n各智能体表现:")
        for agent_id in range(num_agents):
            print(f"  智能体 {agent_id + 1}:")
            print(f"    回报: {episode_returns[agent_id]:.2f}")
            print(f"    成本: {episode_costs[agent_id]:.2f}")
        
        # 总回报和总成本
        total_return = sum(episode_returns)
        total_cost = sum(episode_costs)
        print(f"\n总体表现:")
        print(f"  总回报: {total_return:.2f}")
        print(f"  总成本: {total_cost:.2f}")
        
        # 积分余额
        if episode < len(results['credit_balances']):
            credit_balances = results['credit_balances'][episode]
            print(f"\n积分余额:")
            for agent_id in range(num_agents):
                balance = credit_balances.get(agent_id, 0.0)
                print(f"  智能体 {agent_id + 1}: {balance:.2f}")
    
    # 统计摘要
    print(f"\n{'='*80}")
    print("统计摘要")
    print(f"{'='*80}")
    
    # 峰值负荷统计
    peak_loads = results['peak_loads']
    print(f"\n峰值负荷统计:")
    print(f"  平均: {np.mean(peak_loads):.2f} kW")
    print(f"  最大: {np.max(peak_loads):.2f} kW")
    print(f"  最小: {np.min(peak_loads):.2f} kW")
    print(f"  标准差: {np.std(peak_loads):.2f} kW")
    
    # 峰值惩罚统计
    peak_penalties = results['peak_penalties']
    print(f"\n峰值惩罚统计:")
    print(f"  平均: {np.mean(peak_penalties):.4f}")
    print(f"  最大: {np.max(peak_penalties):.4f}")
    print(f"  最小: {np.min(peak_penalties):.4f}")
    
    # 回报统计
    total_returns = [sum(r) for r in results['episode_returns']]
    print(f"\n总回报统计:")
    print(f"  平均: {np.mean(total_returns):.2f}")
    print(f"  最大: {np.max(total_returns):.2f}")
    print(f"  最小: {np.min(total_returns):.2f}")
    print(f"  标准差: {np.std(total_returns):.2f}")
    
    # 成本统计
    total_costs = [sum(c) for c in results['episode_costs']]
    print(f"\n总成本统计:")
    print(f"  平均: {np.mean(total_costs):.2f}")
    print(f"  最大: {np.max(total_costs):.2f}")
    print(f"  最小: {np.min(total_costs):.2f}")
    print(f"  标准差: {np.std(total_costs):.2f}")
    
    # 各智能体平均表现
    print(f"\n各智能体平均表现:")
    for agent_id in range(num_agents):
        agent_returns = [r[agent_id] for r in results['episode_returns']]
        agent_costs = [c[agent_id] for c in results['episode_costs']]
        print(f"  智能体 {agent_id + 1}:")
        print(f"    平均回报: {np.mean(agent_returns):.2f} (范围: {np.min(agent_returns):.2f} - {np.max(agent_returns):.2f})")
        print(f"    平均成本: {np.mean(agent_costs):.2f} (范围: {np.min(agent_costs):.2f} - {np.max(agent_costs):.2f})")
    
    print(f"\n{'='*80}")
    print("详细结果分析完成")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    view_detailed_results()
