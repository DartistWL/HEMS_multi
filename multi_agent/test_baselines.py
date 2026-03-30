"""
测试基线实现
Test baseline implementations
"""
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from multi_agent.baselines.rule_based_baseline import RuleBasedBaseline
from multi_agent.baselines.independent_baseline import IndependentBaseline
from multi_agent.environment.multi_agent_env import MultiAgentHEMEnv


def test_rule_based_baseline():
    """测试固定规则基线"""
    print("=" * 60)
    print("Testing Rule-Based Baseline")
    print("=" * 60)
    
    # 创建环境
    env = MultiAgentHEMEnv(
        n_agents=3,
        community_ess_capacity=36.0,
        baseline_peak=50.0,
        community_weight=0.2,
        pv_coefficients=[2.0, 2.0, 2.0]  # 正常天气
    )
    
    # 设置评估日期
    eval_dates = ['2011-07-10', '2011-07-11', '2011-07-12']
    env.set_evaluation_dates(eval_dates, [[2.0, 2.0, 2.0]] * len(eval_dates))
    
    # 创建固定规则基线
    baseline = RuleBasedBaseline(peak_threshold_factor=1.2)
    
    # 评估
    results = baseline.evaluate(env, num_episodes=3, mode='eval')
    
    print("\nRule-Based Baseline Test Completed!")
    return results


def test_independent_baseline():
    """测试独立学习基线（仅评估，不训练）"""
    print("=" * 60)
    print("Testing Independent Learning Baseline (Evaluation Only)")
    print("=" * 60)
    
    # 创建独立学习基线
    baseline = IndependentBaseline(
        n_agents=3,
        pv_coefficients=[2.0, 2.0, 2.0]
    )
    
    # 尝试加载模型（如果存在）
    baseline.load_models()
    
    # 评估（使用简单规则策略，因为模型可能不存在）
    print("\nNote: If models are not found, this will use random actions.")
    print("To train models, use: baseline.train(num_episodes=5000)")
    
    # 计算基准峰值
    baseline_peak = baseline.calculate_baseline_peak(num_episodes=3)
    
    print(f"\nBaseline Peak Load: {baseline_peak:.2f} kW")
    print("\nIndependent Learning Baseline Test Completed!")
    
    return baseline_peak


def test_multi_agent_env():
    """测试多智能体环境"""
    print("=" * 60)
    print("Testing Multi-Agent Environment")
    print("=" * 60)
    
    # 创建环境
    env = MultiAgentHEMEnv(
        n_agents=3,
        community_ess_capacity=36.0,
        baseline_peak=50.0,
        community_weight=0.2
    )
    
    # 重置环境
    states = env.reset(mode='train')
    print(f"Environment reset successful. Number of agents: {len(states)}")
    
    # 执行几步
    for step in range(5):
        # 随机动作
        actions = []
        for agent in env.agents:
            action = {}
            for key, values in agent.action_space.items():
                if isinstance(values, tuple):
                    import random
                    action[key] = random.choice(values)
                else:
                    action[key] = 0
            actions.append(action)
        
        next_states, rewards, dones, info = env.step(actions)
        
        print(f"Step {step + 1}: Community Net Load = {info['community_net_load']:.2f} kW, "
              f"Peak Penalty = {info['peak_penalty']:.4f}, "
              f"Community ESS SOC = {info['community_ess_soc']:.3f}")
        
        states = next_states
        if all(dones):
            break
    
    print("\nMulti-Agent Environment Test Completed!")
    return True


if __name__ == "__main__":
    print("Starting Baseline Tests...\n")
    
    try:
        # 测试多智能体环境
        test_multi_agent_env()
        print()
        
        # 测试固定规则基线
        test_rule_based_baseline()
        print()
        
        # 测试独立学习基线
        test_independent_baseline()
        print()
        
        print("=" * 60)
        print("All Tests Completed Successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
