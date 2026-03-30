"""
训练基线方案
Train baseline methods
"""
import sys
import os
import numpy as np

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from multi_agent.baselines.rule_based_baseline import RuleBasedBaseline
from multi_agent.baselines.independent_baseline import IndependentBaseline
from multi_agent.environment.multi_agent_env import MultiAgentHEMEnv


def train_rule_based_baseline():
    """训练（评估）固定规则基线"""
    print("=" * 80)
    print("Training/Evaluating Rule-Based Baseline")
    print("=" * 80)
    
    # 创建环境
    env = MultiAgentHEMEnv(
        n_agents=3,
        community_ess_capacity=36.0,
        baseline_peak=50.0,  # 临时值，后续可以用独立基线计算
        community_weight=0.2,
        pv_coefficients=[2.0, 2.0, 2.0]  # 正常天气
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
    print("\nStarting evaluation...")
    results = baseline.evaluate(env, num_episodes=3, mode='eval')
    
    # 打印结果
    print("\n" + "=" * 80)
    print("Rule-Based Baseline Results Summary")
    print("=" * 80)
    print(f"Average Peak Load: {np.mean(results['peak_loads']):.2f} kW")
    print(f"Max Peak Load: {np.max(results['peak_loads']):.2f} kW")
    print(f"Min Peak Load: {np.min(results['peak_loads']):.2f} kW")
    print(f"Average Peak Penalty: {np.mean(results['peak_penalties']):.4f}")
    print(f"Average Total Return: {np.mean([sum(r) for r in results['episode_returns']]):.2f}")
    print(f"Average Total Cost: {np.mean([sum(c) for c in results['episode_costs']]):.2f}")
    
    return results


def train_independent_baseline(num_episodes=1000):
    """训练独立学习基线"""
    print("=" * 80)
    print("Training Independent Learning Baseline")
    print("=" * 80)
    print(f"Training episodes per agent: {num_episodes}")
    print("Note: This will take some time. Each agent trains independently.")
    
    # 创建独立学习基线（use_community_env 从 config.json 的 independent_baseline.use_community_env 读取，默认 False）
    baseline = IndependentBaseline(
        n_agents=3,
        pv_coefficients=[2.0, 2.0, 2.0],  # 正常天气
        use_community_env=None  # None = 从 config 读取；True = 方案1 多智能体+社区储能
    )
    
    # 训练
    print("\nStarting training...")
    baseline.train(num_episodes=num_episodes, save_dir='multi_agent/baselines/models')
    
    # 计算基准峰值
    print("\n" + "=" * 80)
    print("Calculating Baseline Peak Load")
    print("=" * 80)
    baseline_peak = baseline.calculate_baseline_peak(num_episodes=3)
    
    print(f"\nBaseline Peak Load: {baseline_peak:.2f} kW")
    print("This value should be used as 'baseline_peak' in MultiAgentHEMEnv")
    
    return baseline, baseline_peak


def evaluate_independent_baseline():
    """评估独立学习基线（需要先训练）"""
    print("=" * 80)
    print("Evaluating Independent Learning Baseline")
    print("=" * 80)
    
    # 创建独立学习基线（use_community_env 从 config 读取）
    baseline = IndependentBaseline(
        n_agents=3,
        pv_coefficients=[2.0, 2.0, 2.0],
        use_community_env=None
    )
    
    # 加载模型
    print("\nLoading models...")
    baseline.load_models()
    
    # 评估
    print("\nStarting evaluation...")
    results = baseline.evaluate(num_episodes=3)
    
    # 打印结果
    print("\n" + "=" * 80)
    print("Independent Learning Baseline Results Summary")
    print("=" * 80)
    print(f"Average Peak Load: {np.mean(results['community_peak_loads']):.2f} kW")
    print(f"Max Peak Load: {np.max(results['community_peak_loads']):.2f} kW")
    print(f"Min Peak Load: {np.min(results['community_peak_loads']):.2f} kW")
    print(f"Average Total Return: {np.mean(results['episode_returns']):.2f}")
    print(f"Average Total Cost: {np.mean(results['episode_costs']):.2f}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train baseline methods')
    parser.add_argument('--baseline', type=str, default='rule', 
                       choices=['rule', 'independent', 'both', 'eval_independent'],
                       help='Which baseline to train/evaluate')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes for independent baseline')
    
    args = parser.parse_args()
    
    try:
        if args.baseline == 'rule' or args.baseline == 'both':
            print("\n" + "=" * 80)
            print("PART 1: Rule-Based Baseline")
            print("=" * 80)
            train_rule_based_baseline()
        
        if args.baseline == 'independent' or args.baseline == 'both':
            print("\n" + "=" * 80)
            print("PART 2: Independent Learning Baseline")
            print("=" * 80)
            train_independent_baseline(num_episodes=args.episodes)
        
        if args.baseline == 'eval_independent':
            evaluate_independent_baseline()
        
        print("\n" + "=" * 80)
        print("All Baseline Training/Evaluation Completed!")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
