"""
训练脚本启动器 - 修复导入路径问题
Training script launcher - fixes import path issues
"""
import sys
import os

# 确保项目根目录在sys.path的最前面
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 移除multi_agent/environment从路径（如果存在），避免导入冲突
multi_agent_env_path = os.path.join(project_root, 'multi_agent', 'environment')
if multi_agent_env_path in sys.path:
    sys.path.remove(multi_agent_env_path)

# 现在导入并运行训练脚本
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train baseline methods')
    parser.add_argument('--baseline', type=str, default='rule', 
                       choices=['rule', 'independent', 'both', 'eval_independent'],
                       help='Which baseline to train/evaluate')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes for independent baseline')
    
    args = parser.parse_args()
    
    # 导入训练函数
    from multi_agent.train_baselines import (
        train_rule_based_baseline,
        train_independent_baseline,
        evaluate_independent_baseline
    )
    
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
