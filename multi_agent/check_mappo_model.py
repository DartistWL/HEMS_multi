"""
检查MAPPO模型的表现和问题
Diagnose MAPPO model performance
"""
import sys
import os
import numpy as np
import json
import torch
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from multi_agent.algorithms.mappo import MAPPO
from multi_agent.environment.multi_agent_env import MultiAgentHEMEnv


def load_training_stats(stats_path='multi_agent/algorithms/models/mappo_stats.pth'):
    """加载训练统计信息"""
    if os.path.exists(stats_path):
        stats = torch.load(stats_path, map_location='cpu')
        return stats
    return None


def check_model_performance(model_dir='multi_agent/algorithms/models', 
                           baseline_peak=31.01, num_episodes=3):
    """检查模型表现"""
    print("=" * 80)
    print("检查MAPPO模型表现")
    print("=" * 80)
    
    # 加载配置文件
    config_path = 'multi_agent/config.json'
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        training_config = config.get('training', {})
        community_weight = training_config.get('community_weight', 50)
        print(f"从配置文件读取: community_weight = {community_weight}")
    else:
        community_weight = 50
    
    # 创建环境
    env = MultiAgentHEMEnv(
        n_agents=3,
        community_ess_capacity=36.0,
        baseline_peak=baseline_peak,
        community_weight=community_weight,
        community_credit_cost_weight=training_config.get('community_credit_cost_weight', 0.3),
        community_credit_benefit_weight=training_config.get('community_credit_benefit_weight', 0.2),
        initial_credit=training_config.get('initial_credit', 50.0),
        peak_penalty_exponent=training_config.get('peak_penalty_exponent', 1.5),
        peak_discharge_bonus=training_config.get('peak_discharge_bonus', 0.1),
        peak_credit_cost_reduction=training_config.get('peak_credit_cost_reduction', 0.5),
        pv_coefficients=[2.0, 2.0, 2.0]
    )
    
    # 设置评估日期
    eval_dates = ['2011-07-10', '2011-07-11', '2011-07-12']
    pv_coefficients_list = [
        [3.0, 3.0, 3.0],  # 晴天
        [1.0, 1.0, 1.0],  # 阴天
        [2.0, 2.0, 2.0],  # 正常
    ]
    env.set_evaluation_dates(eval_dates, pv_coefficients_list)
    
    # 创建MAPPO算法
    mappo = MAPPO(
        env=env,
        n_agents=3,
        hidden_dim=128,
        gamma=0.96,
        lmbda=0.95,
        eps=0.2,
        epochs=3,
        lr_actor=1e-4,
        lr_critic=1e-4,
        batch_size=64,
        ent_coef=0.01,
        max_grad_norm=1.0,
        use_state_normalization=True,
        reward_scale=10.0
    )
    
    # 加载模型
    mappo.load(model_dir)
    
    # 评估统计
    all_peak_loads = []
    all_peak_penalties = []
    all_community_ess_soc = []
    all_credit_balances = []
    episode_data = {
        'peak_loads': [],
        'peak_penalties': [],
        'community_ess_utilization': [],
        'credit_balance_changes': []
    }
    
    print(f"\n评估 {num_episodes} 个episodes...")
    
    for episode in range(num_episodes):
        date_index = episode % len(eval_dates)
        date = eval_dates[date_index]
        
        print(f"\nEpisode {episode + 1}/{num_episodes} - Date: {date}")
        
        states = env.reset(mode='eval', date_index=date_index)
        done = False
        step_count = 0
        
        episode_peak_load = 0.0
        episode_peak_penalties = []
        episode_ess_soc = []
        episode_credit_balances = []
        
        while not done and step_count < 48:
            # 选择动作（确定性策略）
            actions, _ = mappo.select_actions(states, deterministic=True)
            
            # 执行动作
            next_states, rewards, dones, info = env.step(actions)
            
            # 记录数据
            community_load = info['community_net_load']
            peak_penalty = info['peak_penalty']
            ess_soc = info['community_ess_soc']
            credits = env.credit_system.get_all_balances()
            
            episode_peak_load = max(episode_peak_load, community_load)
            episode_peak_penalties.append(peak_penalty)
            episode_ess_soc.append(ess_soc)
            episode_credit_balances.append(list(credits.values()) if isinstance(credits, dict) else credits)
            
            states = next_states
            done = all(dones)
            step_count += 1
        
        all_peak_loads.append(episode_peak_load)
        all_peak_penalties.append(np.sum(episode_peak_penalties))
        all_community_ess_soc.append(episode_ess_soc)
        all_credit_balances.append(episode_credit_balances)
        
        print(f"  峰值负荷: {episode_peak_load:.2f} kW")
        print(f"  峰值惩罚总和: {np.sum(episode_peak_penalties):.4f}")
        print(f"  社区储能SOC范围: [{min(episode_ess_soc):.2f}, {max(episode_ess_soc):.2f}]")
        print(f"  社区储能SOC变化: {max(episode_ess_soc) - min(episode_ess_soc):.2f}")
    
    # 统计结果
    print("\n" + "=" * 80)
    print("诊断结果汇总")
    print("=" * 80)
    print(f"平均峰值负荷: {np.mean(all_peak_loads):.2f} kW")
    print(f"峰值负荷范围: [{np.min(all_peak_loads):.2f}, {np.max(all_peak_loads):.2f}] kW")
    print(f"基准峰值: {baseline_peak:.2f} kW")
    print(f"峰值降低率: {(baseline_peak - np.mean(all_peak_loads)) / baseline_peak * 100:.2f}%")
    
    print(f"\n平均峰值惩罚总和: {np.mean(all_peak_penalties):.4f}")
    print(f"峰值惩罚总和范围: [{np.min(all_peak_penalties):.4f}, {np.max(all_peak_penalties):.4f}]")
    
    # 检查社区储能使用
    all_soc_ranges = [max(soc_list) - min(soc_list) for soc_list in all_community_ess_soc]
    print(f"\n社区储能SOC变化范围（每episode）: {all_soc_ranges}")
    print(f"平均SOC变化: {np.mean(all_soc_ranges):.4f}")
    if np.mean(all_soc_ranges) < 0.1:
        print("⚠️  警告: 社区储能SOC变化很小，可能没有被充分利用！")
    
    # 检查峰值惩罚是否生效
    if np.mean(all_peak_penalties) < 0.01:
        print("⚠️  警告: 峰值惩罚总和很小，可能没有触发惩罚或惩罚权重太小！")
    
    # 判断是否需要重新训练
    print("\n" + "=" * 80)
    print("诊断结论")
    print("=" * 80)
    
    need_retrain = False
    issues = []
    
    if np.mean(all_peak_loads) > baseline_peak:
        need_retrain = True
        issues.append(f"❌ 峰值负荷 ({np.mean(all_peak_loads):.2f} kW) 高于基准峰值 ({baseline_peak:.2f} kW)")
    
    if np.mean(all_soc_ranges) < 0.1:
        need_retrain = True
        issues.append(f"❌ 社区储能利用率低 (SOC变化: {np.mean(all_soc_ranges):.4f})")
    
    if np.mean(all_peak_penalties) < 0.01:
        issues.append(f"⚠️  峰值惩罚很小 (总和: {np.mean(all_peak_penalties):.4f})，需要检查峰值追踪器逻辑")
    
    if need_retrain:
        print("需要重新训练模型，原因:")
        for issue in issues:
            print(f"  {issue}")
        print("\n建议:")
        print("  1. 我已经修复了净负荷计算错误（社区储能不应影响净负荷）")
        print("  2. 请重新运行训练脚本，使用修复后的代码")
        print("  3. 训练时关注峰值负荷是否在下降")
    else:
        print("✓ 模型表现正常，但峰值仍高于预期")
        print("建议:")
        print("  1. 检查训练曲线，确认训练是否收敛")
        print("  2. 可能需要增加训练轮数")
        print("  3. 检查峰值惩罚权重是否合适")
    
    return {
        'avg_peak_load': np.mean(all_peak_loads),
        'baseline_peak': baseline_peak,
        'peak_reduction': (baseline_peak - np.mean(all_peak_loads)) / baseline_peak * 100,
        'avg_peak_penalties': np.mean(all_peak_penalties),
        'avg_soc_range': np.mean(all_soc_ranges),
        'need_retrain': need_retrain,
        'issues': issues
    }


def check_training_curves(stats_path='multi_agent/algorithms/models/mappo_stats.pth',
                         curve_path='multi_agent/algorithms/models/training_curves.png'):
    """检查训练曲线"""
    print("\n" + "=" * 80)
    print("检查训练曲线")
    print("=" * 80)
    
    # 检查统计文件
    if os.path.exists(stats_path):
        stats = torch.load(stats_path, map_location='cpu')
        print(f"找到训练统计文件: {stats_path}")
        if isinstance(stats, dict):
            print("训练统计包含的键:", list(stats.keys()))
            if 'peak_loads' in stats:
                peak_loads = stats['peak_loads']
                print(f"峰值负荷数据点数: {len(peak_loads)}")
                if len(peak_loads) > 0:
                    print(f"初始峰值: {peak_loads[0]:.2f} kW")
                    print(f"最终峰值: {peak_loads[-1]:.2f} kW")
                    if len(peak_loads) > 10:
                        print(f"前10个episode平均峰值: {np.mean(peak_loads[:10]):.2f} kW")
                        print(f"后10个episode平均峰值: {np.mean(peak_loads[-10:]):.2f} kW")
                        if peak_loads[-1] < peak_loads[0]:
                            print("✓ 峰值在训练过程中下降")
                        else:
                            print("❌ 峰值在训练过程中上升或未明显下降")
    else:
        print(f"未找到训练统计文件: {stats_path}")
    
    # 检查训练曲线图
    if os.path.exists(curve_path):
        print(f"\n找到训练曲线图: {curve_path}")
        print("请手动查看该图片，检查:")
        print("  1. 峰值负荷是否在下降")
        print("  2. 回报是否在上升")
        print("  3. 成本是否在下降")
        print("  4. 训练是否收敛")
    else:
        print(f"未找到训练曲线图: {curve_path}")


if __name__ == "__main__":
    # 检查训练曲线
    check_training_curves()
    
    # 检查模型表现
    results = check_model_performance()
    
    print("\n" + "=" * 80)
    print("检查完成！")
    print("=" * 80)
