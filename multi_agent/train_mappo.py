"""
MAPPO 训练脚本 - 支持 100 天数据，20 个住户（16 训练，4 测试）
功能：训练多智能体深度强化学习模型，用于家庭能源管理（含社区储能、积分系统）
作者：基于原脚本修改
"""

import os
# 解决 OpenMP 库冲突（仅在绘图时可能触发）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import argparse
import numpy as np
import torch
import matplotlib
# 使用无交互后端，防止绘图时 OpenMP 崩溃
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random

# ---------------------------- 绘图中文设置 ----------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi',
                                   'FangSong', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------- 项目路径 ----------------------------
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 导入自定义模块
from multi_agent.algorithms.mappo import MAPPO          # MAPPO 算法主类
from multi_agent.environment.multi_agent_env import MultiAgentHEMEnv  # 多智能体环境
from multi_agent.utils.data_interface import MultiAgentDataInterface   # 数据接口（电价、光伏等）


def load_config(config_path='config.json', training_only=False):
    """
    加载配置文件（位于 multi_agent 目录下）
    :param config_path: 配置文件名
    :param training_only: 若为 True，只返回 training 部分的配置字典
    :return: 配置字典（全量或 training 子字典）
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_config_path = os.path.join(script_dir, config_path)
    if os.path.exists(abs_config_path):
        with open(abs_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config.get('training', {}) if training_only else config
    else:
        print(f"警告：配置文件 {abs_config_path} 不存在，使用默认值。")
        return {} if training_only else {}


def generate_training_dates_and_coeffs(num_days=100, start_date='2020-01-01', seed=42):
    """
    生成连续 num_days 天的日期列表及对应的光伏系数列表。
    每天为 3 个家庭随机分配天气（sunny/cloudy/normal），保证可重复性。
    :param num_days: 天数（默认 100）
    :param start_date: 起始日期字符串
    :param seed: 随机种子，使结果可复现
    :return: (dates_list, pv_coeffs_list)
             dates_list: 字符串日期列表，如 ['2020-01-01', ...]
             pv_coeffs_list: 每个元素是长度为 3 的列表，表示当天三个家庭的光伏系数
    """
    random.seed(seed)
    np.random.seed(seed)

    start = datetime.strptime(start_date, '%Y-%m-%d')
    dates = [(start + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(num_days)]

    weather_types = ['sunny', 'cloudy', 'normal']
    pv_coeffs_list = []
    for _ in range(num_days):
        # 为当天随机选择一种天气（三个家庭共用同一系数，与原逻辑一致）
        weather = random.choice(weather_types)
        coeff = MultiAgentDataInterface.get_weather_coefficient(weather)
        pv_coeffs_list.append([coeff, coeff, coeff])

    return dates, pv_coeffs_list


def train_mappo(num_episodes=1000, baseline_peak=None, save_dir='multi_agent/algorithms/models',
                load_model=None, eval_freq=100, save_freq=500, config_path='multi_agent/config.json'):
    """
    训练 MAPPO 算法的主函数（100 天数据，20 户，16 训练 / 4 测试）

    :param num_episodes:       训练的总 episode 数
    :param baseline_peak:      基准峰值（kW），用于归一化峰值惩罚，若 None 则从 config 读取
    :param save_dir:           模型保存目录
    :param load_model:         若提供路径，则从该路径加载已有模型继续训练
    :param eval_freq:          每隔多少 episode 执行一次评估（本函数内评估部分已占位，可自行实现）
    :param save_freq:          每隔多少 episode 保存一次模型
    :param config_path:        配置文件路径
    :return: (训练好的 MAPPO 对象, 训练统计数据字典)
    """
    # ---------------------------- 1. 加载配置参数 ----------------------------
    full_config = load_config(config_path)                     # 全量配置
    config = full_config.get('training', {}) if isinstance(full_config, dict) else {}
    # 积分定价方案（uniform 或 contribution_based），影响模型保存路径
    credit_scheme = (full_config.get('credit_pricing') or {}).get('scheme', 'uniform') \
                    if isinstance(full_config, dict) else 'uniform'
    if credit_scheme == 'contribution_based':
        save_dir = save_dir.rstrip(os.sep) + '_contribution_based'
        print(f"积分方案为 contribution_based，模型保存至 {save_dir}")

    # 读取训练相关超参数
    use_popart = config.get('use_popart_normalization', False)          # 是否使用 Pop-Art 归一化（价值网络）
    use_state_norm = config.get('use_state_normalization', True)        # 是否对状态进行归一化
    use_reward_norm = config.get('use_reward_normalization', False)     # 是否对奖励进行归一化
    ent_coef = config.get('ent_coef', 0.05)                             # 熵系数（鼓励探索）
    community_weight = config.get('community_weight', 0.2)              # 社区奖励权重
    community_credit_cost_weight = config.get('community_credit_cost_weight', 0.1)   # 积分成本项权重
    community_credit_benefit_weight = config.get('community_credit_benefit_weight', 0.1) # 积分收益项权重
    initial_credit = config.get('initial_credit', 100.0)                # 每个家庭的初始积分
    peak_penalty_exponent = config.get('peak_penalty_exponent', 2.0)    # 峰值惩罚指数（惩罚 = (峰值/基准)^指数）
    peak_discharge_bonus = config.get('peak_discharge_bonus', 0.0)      # 放电时的峰值奖励系数（预留）
    peak_credit_cost_reduction = config.get('peak_credit_cost_reduction', 1.0)  # 峰值惩罚对积分成本的影响系数
    critic_loss_type = config.get('critic_loss_type', 'mse')            # 价值网络损失类型：'mse' 或 'huber'
    huber_delta = config.get('huber_delta', 1.0)                        # Huber 损失的 delta 参数

    if baseline_peak is None:
        baseline_peak = config.get('baseline_peak', 31.01)              # 默认基准峰值 31.01 kW

    # ---------------------------- 2. 打印训练配置 ----------------------------
    print("=" * 80)
    print("训练 MAPPO 算法 - 100天数据，20户（16训练/4测试）")
    print("=" * 80)
    print(f"Episode 总数: {num_episodes}")
    print(f"基准峰值: {baseline_peak} kW")
    print(f"模型保存目录: {save_dir}")
    print(f"Pop-Art 归一化: {use_popart}")
    print(f"状态归一化: {use_state_norm}")
    print(f"奖励归一化: {use_reward_norm}")
    print(f"熵系数: {ent_coef}")
    print(f"社区权重: {community_weight}")
    print(f"积分成本权重: {community_credit_cost_weight}")
    print(f"积分收益权重: {community_credit_benefit_weight}")
    print(f"初始积分: {initial_credit}")
    print(f"峰值惩罚指数: {peak_penalty_exponent}")
    print(f"放电峰值奖励: {peak_discharge_bonus}")
    print(f"峰值积分成本降低系数: {peak_credit_cost_reduction}")
    print(f"Critic 损失类型: {critic_loss_type}")
    if critic_loss_type == 'huber':
        print(f"Huber delta: {huber_delta}")
    print("=" * 80)

    # ---------------------------- 3. 定义训练 / 测试住户 ----------------------------
    # 共 20 户，编号 H1 ~ H20。训练使用 H1~H16（16 户），测试使用 H17~H20（4 户）
    train_house_ids = [f"H{i}" for i in range(1, 17)]   # H1 ... H16
    test_house_ids = [f"H{i}" for i in range(17, 21)]   # H17 ... H20

    # ---------------------------- 4. 创建多智能体环境 ----------------------------
    # 注意：环境内部会根据 train_house_ids / test_house_ids 从 DataInterface 中抽取对应的住户数据
    env = MultiAgentHEMEnv(
        n_agents=3,                                 # 每个 episode 同时控制 3 个家庭（硬编码）
        community_ess_capacity=36.0,                # 社区储能容量（kWh）
        baseline_peak=baseline_peak,
        community_weight=community_weight,
        community_credit_cost_weight=community_credit_cost_weight,
        community_credit_benefit_weight=community_credit_benefit_weight,
        initial_credit=initial_credit,
        peak_penalty_exponent=peak_penalty_exponent,
        peak_discharge_bonus=peak_discharge_bonus,
        peak_credit_cost_reduction=peak_credit_cost_reduction,
        pv_coefficients=[2.0, 2.0, 2.0],            # 初始占位光伏系数，后续会被 set_training_dates 覆盖
        train_house_ids=train_house_ids,
        test_house_ids=test_house_ids
    )

    # ---------------------------- 5. 生成 100 天的训练数据（日期 + 光伏系数） ----------------------------
    training_dates, pv_coefficients_list = generate_training_dates_and_coeffs(num_days=100)
    env.set_training_dates(training_dates, pv_coefficients_list)   # 将日期和光伏系数注入环境

    # ---------------------------- 6. 计算状态维度（用于初始化神经网络） ----------------------------
    from multi_agent.algorithms.action_utils import state_dict_to_vector
    # 重置环境，获取一个样本状态（返回三个智能体的局部状态字典）
    sample_states = env.reset(mode='train', date_index=0, house_index=0)
    # 单个智能体的局部状态向量长度
    local_state_dim = len(state_dict_to_vector(sample_states[0]))
    # 全局状态 = 拼接所有智能体的局部状态（MAPPO 中 Critic 使用全局状态）
    global_state_dim = sum(len(state_dict_to_vector(s)) for s in sample_states)

    # ---------------------------- 7. 设备配置（GPU/CPU） ----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ---------------------------- 8. 创建 MAPPO 算法实例 ----------------------------
    mappo = MAPPO(
        env=env,                         # 环境（用于获取某些动态信息，如动作空间边界）
        n_agents=3,
        device=device,
        local_state_dim=local_state_dim,
        global_state_dim=global_state_dim,
        hidden_dim=128,                  # Actor 和 Critic 网络的隐藏层维度
        gamma=0.96,                      # 折扣因子
        lmbda=0.95,                      # GAE 参数
        eps=0.2,                         # PPO clip 范围
        epochs=10,                       # 每次更新时，使用同一批数据训练的次数
        lr_actor=1e-4,                   # Actor 学习率
        lr_critic=1e-4,                  # Critic 学习率
        batch_size=32,                   # 每次更新用的 batch 大小
        ent_coef=ent_coef,
        max_grad_norm=1.0,               # 梯度裁剪阈值
        use_state_normalization=use_state_norm,
        use_popart=use_popart,
        reward_scale=1.0,
        use_reward_normalization=use_reward_norm,
        critic_loss_type=critic_loss_type,
        huber_delta=huber_delta
    )

    # 如果指定了加载已有模型，则加载
    if load_model and os.path.exists(load_model):
        print(f"\n从 {load_model} 加载模型...")
        mappo.load(load_model)
        print("模型加载完成。")

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # ---------------------------- 9. 初始化训练统计数据结构 ----------------------------
    training_stats = {
        'episode_returns': [],          # 每个 episode 的总奖励（所有智能体奖励之和）
        'episode_costs': [],            # 每个 episode 的总电费（元）
        'peak_loads': [],               # 每个 episode 的社区峰值负荷（kW）
        'actor_loss': [],               # Actor 网络损失（每更新一次记录一次）
        'critic_loss': [],              # Critic 网络损失
        'entropy': [],                  # 策略熵（用于监控探索程度）
        'credit_balances': [[], [], []] # 三个家庭的积分余额变化（每个 episode 结束后的值）
    }

    # 最后一个 episode 的详细时间步数据（用于绘制积分变化细节图）
    last_episode_details = {
        'time_steps': [],               # 时间步索引（0~47）
        'credit_balances': [[], [], []],# 每个时间步后的积分余额
        'credit_changes': [[], [], []], # 每个时间步的积分变化量
        'grid_prices': [],              # 每个时间步的电网电价
        'agent_loads': [[], [], []],    # 每个家庭的净负荷（kW）
        'community_load': [],           # 社区总净负荷
        'community_ess_soc': []         # 社区储能 SOC
    }

    print("\n" + "=" * 80)
    print("开始训练...")
    print("=" * 80)

    num_training_dates = len(training_dates)   # 100 天
    num_train_houses = len(train_house_ids)    # 16 户

    # ---------------------------- 10. 主训练循环 ----------------------------
    for episode in range(num_episodes):
        # 循环使用训练日期和训练住户（保证每个 episode 数据不同）
        date_index = episode % num_training_dates
        house_index = episode % num_train_houses

        # 重置环境，得到初始局部状态（列表，每个智能体一个字典）
        states = env.reset(mode='train', date_index=date_index, house_index=house_index)
        global_state = env.get_community_state()   # 全局状态 = 拼接所有智能体局部状态

        mappo.reset_buffer()   # 清空经验池

        episode_return = 0.0
        episode_cost = 0.0
        episode_peak_load = 0.0
        step_count = 0
        done = False

        # 标记是否为最后一个 episode（用于收集详细数据绘图）
        is_last_episode = (episode == num_episodes - 1)
        if is_last_episode:
            # 清空上一轮残留数据
            last_episode_details = {k: ([] if isinstance(v, list) else v) for k, v in last_episode_details.items()}
            initial_credits = env.credit_system.get_all_balances()
            for i in range(3):
                last_episode_details['credit_balances'][i].append(initial_credits.get(i, 100.0))
                last_episode_details['credit_changes'][i].append(0.0)  # 初始变化为 0

        # ---------------------------- 一个 episode 内的步进循环（48 个时间步，半小时一步） ----------------------------
        while not done:
            # 根据当前状态选择动作（训练时使用随机策略 deterministic=False）
            actions, action_log_probs = mappo.select_actions(states, deterministic=False)

            # 如果是最后一个 episode，记录当前时间步的电价、各家庭负荷等信息
            if is_last_episode:
                current_time_idx = env.agents[0].env.current_time_index
                grid_price = env.agents[0].env.data_interface.get_electricity_price(
                    env.current_date, current_time_idx
                )
                last_episode_details['grid_prices'].append(grid_price)
                last_episode_details['time_steps'].append(current_time_idx)
                for i in range(3):
                    agent_net_load = env.agents[i].get_net_load()
                    last_episode_details['agent_loads'][i].append(agent_net_load)

            # 执行动作，环境返回下一状态、奖励、终止标志、额外信息
            next_states, rewards, dones, info = env.step(actions)
            next_global_state = env.get_community_state()

            # 最后一个 episode 中记录积分变化和社区信息
            if is_last_episode:
                last_episode_details['community_load'].append(info['community_net_load'])
                last_episode_details['community_ess_soc'].append(info['community_ess_soc'])
                current_credits = env.credit_system.get_all_balances()
                for i in range(3):
                    current_balance = current_credits.get(i, 100.0)
                    if len(last_episode_details['credit_balances'][i]) > 0:
                        prev = last_episode_details['credit_balances'][i][-1]
                        credit_change = current_balance - prev
                    else:
                        credit_change = 0.0
                    last_episode_details['credit_balances'][i].append(current_balance)
                    last_episode_details['credit_changes'][i].append(credit_change)

            # 存储 transition 到经验池
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

            # 累加 episode 统计量
            episode_return += sum(rewards)
            episode_cost += sum([env.agents[i].env.current_step_cost for i in range(3)])
            episode_peak_load = max(episode_peak_load, info['community_net_load'])

            # 状态更新
            states = next_states
            global_state = next_global_state
            done = all(dones)          # 三个智能体都终止时才算 episode 结束
            step_count += 1
            if step_count >= 48:       # 一天 48 个时间步（半小时步长）
                done = True

        # ---------------------------- 一个 episode 结束，进行策略更新 ----------------------------
        update_stats = mappo.update()   # 执行 PPO 更新，返回损失等信息

        # 记录 episode 结束后的积分余额
        credit_balances = env.credit_system.get_all_balances()
        for i in range(3):
            training_stats['credit_balances'][i].append(credit_balances.get(i, 100.0))

        # 保存 episode 级别的统计
        training_stats['episode_returns'].append(episode_return)
        training_stats['episode_costs'].append(episode_cost)
        training_stats['peak_loads'].append(episode_peak_load)

        if update_stats:
            training_stats['actor_loss'].append(update_stats.get('actor_loss', 0))
            training_stats['critic_loss'].append(update_stats.get('critic_loss', 0))
            training_stats['entropy'].append(update_stats.get('entropy', 0))

        # ---------------------------- 打印训练进度 ----------------------------
        if (episode + 1) % 10 == 0:
            window = min(10, len(training_stats['episode_returns']))
            avg_return = np.mean(training_stats['episode_returns'][-window:])
            avg_cost = np.mean(training_stats['episode_costs'][-window:])
            avg_peak = np.mean(training_stats['peak_loads'][-window:])
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Return: {episode_return:.2f} (avg: {avg_return:.2f}) | "
                  f"Cost: {episode_cost:.2f} (avg: {avg_cost:.2f}) | "
                  f"Peak: {episode_peak_load:.2f} kW (avg: {avg_peak:.2f} kW)")

            # 如果环境记录了奖励分解，打印出来便于分析
            if hasattr(env, '_reward_breakdown') and len(env._reward_breakdown['total_reward']) > 0:
                brk = env._reward_breakdown
                n_steps = len(brk['total_reward'])
                avg_ind = np.mean(brk['individual_reward'][-n_steps:]) if brk['individual_reward'] else 0
                avg_peak_pen = np.mean(brk['peak_penalty_value'][-n_steps:]) if brk['peak_penalty_value'] else 0
                avg_queue_pen = np.mean(brk['queue_penalty'][-n_steps:]) if brk['queue_penalty'] else 0
                avg_ess_reward = np.mean(brk['community_ess_reward'][-n_steps:]) if brk['community_ess_reward'] else 0
                avg_credit_pen = np.mean(brk['credit_penalty'][-n_steps:]) if brk['credit_penalty'] else 0
                print(f"  奖励分解（平均每步）：个体奖励={avg_ind:.2f}, 峰值惩罚={avg_peak_pen:.2f}, "
                      f"队列惩罚={avg_queue_pen:.2f}, ESS奖励={avg_ess_reward:.2f}, 积分惩罚={avg_credit_pen:.2f}")
            if update_stats:
                print(f"  损失：Actor={update_stats.get('actor_loss', 0):.4f}, "
                      f"Critic={update_stats.get('critic_loss', 0):.4f}, "
                      f"Entropy={update_stats.get('entropy', 0):.4f}")

        # 定期评估（此处仅占位，用户可自行实现 evaluate_mappo 函数）
        if (episode + 1) % eval_freq == 0:
            print(f"\n第 {episode + 1} episode 评估中...")
            # 调用评估函数（下面已定义 evaluate_mappo）
            # eval_results = evaluate_mappo(mappo, baseline_peak, num_episodes=5)
            pass

        # 定期保存模型
        if (episode + 1) % save_freq == 0:
            print(f"\n第 {episode + 1} episode 保存模型...")
            mappo.save(save_dir)

    # ---------------------------- 11. 训练结束，保存最终模型及统计数据 ----------------------------
    print("\n" + "=" * 80)
    print("训练完成！保存最终模型...")
    print("=" * 80)
    mappo.save(save_dir)

    # 保存训练统计为 JSON 文件
    stats_file = os.path.join(save_dir, 'training_stats.json')
    stats_dict = {
        'episode_returns': training_stats['episode_returns'],
        'episode_costs': training_stats['episode_costs'],
        'peak_loads': training_stats['peak_loads'],
        'credit_balances': {f'agent_{i}': training_stats['credit_balances'][i] for i in range(3)},
        'actor_loss': training_stats.get('actor_loss', []),
        'critic_loss': training_stats.get('critic_loss', []),
        'entropy': training_stats.get('entropy', [])
    }
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats_dict, f, indent=2, ensure_ascii=False)
    print(f"训练统计数据已保存至: {stats_file}")

    # 绘制训练曲线图
    try:
        plot_training_curves(training_stats, save_dir, initial_credit=initial_credit)
    except Exception as e:
        print(f"绘制训练曲线失败（忽略）: {e}")

    # 绘制最后一个 episode 的积分变化细节图
    try:
        if len(last_episode_details['time_steps']) > 0:
            plot_last_episode_credit_details(last_episode_details, save_dir, initial_credit=initial_credit)
    except Exception as e:
        print(f"绘制最后一个 episode 细节失败（忽略）: {e}")

    # 打印最终汇总
    print("\n" + "=" * 80)
    print("训练总结")
    print("=" * 80)
    print(f"最后 100 个 episode 的平均总奖励: {np.mean(training_stats['episode_returns'][-100:]):.2f}")
    print(f"最后 100 个 episode 的平均电费: {np.mean(training_stats['episode_costs'][-100:]):.2f}")
    print(f"最后 100 个 episode 的平均社区峰值负荷: {np.mean(training_stats['peak_loads'][-100:]):.2f} kW")
    print(f"最终模型保存至: {save_dir}")
    print("=" * 80)

    return mappo, training_stats


def evaluate_mappo(mappo, baseline_peak=31.01, num_episodes=10, mode='eval'):
    """
    评估训练好的 MAPPO 模型（使用 4 户测试住户，3 天评估日期）
    :param mappo:         训练好的 MAPPO 对象
    :param baseline_peak: 基准峰值（用于环境）
    :param num_episodes:  评估 episode 数
    :param mode:          环境模式，通常为 'eval'
    :return:              字典，包含 'returns', 'costs', 'peak_loads', 'peak_penalties'
    """
    # 训练和测试住户划分（必须与训练时一致）
    train_house_ids = [f"H{i}" for i in range(1, 17)]   # 16 户训练
    test_house_ids = [f"H{i}" for i in range(17, 21)]   # 4 户测试

    # 创建评估环境（使用相同的参数，但测试住户）
    env = MultiAgentHEMEnv(
        n_agents=3,
        community_ess_capacity=36.0,
        baseline_peak=baseline_peak,
        community_weight=0.2,
        community_credit_cost_weight=0.1,
        community_credit_benefit_weight=0.1,
        initial_credit=100.0,
        peak_penalty_exponent=2.0,
        peak_discharge_bonus=0.0,
        peak_credit_cost_reduction=1.0,
        pv_coefficients=[2.0, 2.0, 2.0],
        train_house_ids=train_house_ids,
        test_house_ids=test_house_ids
    )

    # 评估日期（3 天示例，可自行修改）
    eval_dates = ['2020-01-08', '2020-01-09', '2020-01-10']
    sunny_coef = MultiAgentDataInterface.get_weather_coefficient('sunny')
    cloudy_coef = MultiAgentDataInterface.get_weather_coefficient('cloudy')
    normal_coef = MultiAgentDataInterface.get_weather_coefficient('normal')
    pv_coefficients_list = [
        [sunny_coef, sunny_coef, sunny_coef],
        [cloudy_coef, cloudy_coef, cloudy_coef],
        [normal_coef, normal_coef, normal_coef],
    ]
    env.set_evaluation_dates(eval_dates, pv_coefficients_list)

    results = {'returns': [], 'costs': [], 'peak_loads': [], 'peak_penalties': []}
    num_eval_dates = len(eval_dates)          # 3
    num_test_houses = len(test_house_ids)     # 4

    for episode in range(num_episodes):
        date_index = episode % num_eval_dates
        house_index = episode % num_test_houses
        states = env.reset(mode=mode, date_index=date_index, house_index=house_index)

        episode_return = 0.0
        episode_cost = 0.0
        episode_peak_load = 0.0
        episode_peak_penalty = 0.0
        step_count = 0
        done = False

        while not done:
            # 评估时使用确定性策略 (deterministic=True)
            actions, _ = mappo.select_actions(states, deterministic=True)
            next_states, rewards, dones, info = env.step(actions)

            episode_return += sum(rewards)
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


def plot_training_curves(training_stats, save_dir, initial_credit=100.0):
    """
    绘制训练曲线：总奖励、电费、峰值负荷、积分余额、损失及熵
    :param training_stats: 训练统计数据字典
    :param save_dir:       图片保存目录
    :param initial_credit: 初始积分（用于绘图参考线）
    """
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle('MAPPO 训练曲线 (100天数据, 16/4户)', fontsize=16)

    # 1. Episode Returns
    ax1 = axes[0, 0]
    ax1.plot(training_stats['episode_returns'], alpha=0.7)
    if len(training_stats['episode_returns']) > 10:
        window = min(50, len(training_stats['episode_returns']) // 10)
        smoothed = np.convolve(training_stats['episode_returns'],
                               np.ones(window) / window, mode='valid')
        ax1.plot(range(window - 1, len(training_stats['episode_returns'])), smoothed,
                 label=f'平滑 (窗口={window})', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('总奖励')
    ax1.set_title('Episode 总奖励')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Episode Costs
    ax2 = axes[0, 1]
    ax2.plot(training_stats['episode_costs'], alpha=0.7)
    if len(training_stats['episode_costs']) > 10:
        window = min(50, len(training_stats['episode_costs']) // 10)
        smoothed = np.convolve(training_stats['episode_costs'],
                               np.ones(window) / window, mode='valid')
        ax2.plot(range(window - 1, len(training_stats['episode_costs'])), smoothed,
                 label=f'平滑 (窗口={window})', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('电费 (元)')
    ax2.set_title('Episode 电费')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Peak Loads
    ax3 = axes[1, 0]
    ax3.plot(training_stats['peak_loads'], alpha=0.7)
    if len(training_stats['peak_loads']) > 10:
        window = min(50, len(training_stats['peak_loads']) // 10)
        smoothed = np.convolve(training_stats['peak_loads'],
                               np.ones(window) / window, mode='valid')
        ax3.plot(range(window - 1, len(training_stats['peak_loads'])), smoothed,
                 label=f'平滑 (窗口={window})', linewidth=2)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('峰值负荷 (kW)')
    ax3.set_title('社区峰值负荷')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Credit Balances
    ax4 = axes[1, 1]
    colors_credit = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    labels_credit = ['家庭 1', '家庭 2', '家庭 3']
    for i in range(3):
        if len(training_stats['credit_balances'][i]) > 0:
            episodes = range(len(training_stats['credit_balances'][i]))
            ax4.plot(episodes, training_stats['credit_balances'][i],
                     color=colors_credit[i], label=labels_credit[i], linewidth=2, alpha=0.8)
    ax4.axhline(y=initial_credit, color='gray', linestyle='--', linewidth=1.5, alpha=0.7,
                label=f'初始积分 ({initial_credit:.0f})')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('积分余额')
    ax4.set_title('社区积分余额变化')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)

    # 5. Loss and Entropy
    ax5 = axes[2, 0]
    if training_stats.get('actor_loss') and len(training_stats['actor_loss']) > 0:
        ax5_twin = ax5.twinx()
        ax5.plot(training_stats['actor_loss'], label='Actor 损失', alpha=0.7, color='blue')
        ax5.plot(training_stats['critic_loss'], label='Critic 损失', alpha=0.7, color='orange')
        ax5_twin.plot(training_stats['entropy'], label='熵', alpha=0.7, color='green')
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('损失', color='black')
        ax5_twin.set_ylabel('熵', color='green')
        ax5.set_title('训练损失与熵')
        ax5.legend(loc='upper left')
        ax5_twin.legend(loc='upper right')
        ax5.grid(True, alpha=0.3)

    # 6. 空白子图（占位）
    ax6 = axes[2, 1]
    ax6.axis('off')

    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"训练曲线图已保存至 {plot_path}")
    plt.close()


def plot_last_episode_credit_details(last_episode_details, save_dir, initial_credit=100.0):
    """
    绘制最后一个 episode 中每个时间步的积分变化、电价、负荷关系图
    :param last_episode_details: 最后一个 episode 的详细数据字典
    :param save_dir:             图片保存目录
    :param initial_credit:      初始积分（用于参考线）
    """
    time_steps = last_episode_details['time_steps']
    if len(time_steps) == 0:
        print("没有详细的 episode 数据可供绘图。")
        return

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('最后一个 Episode 的积分、电价和负荷变化关系 (100天数据)', fontsize=16, fontweight='bold')

    colors_agent = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    labels_agent = ['家庭 1', '家庭 2', '家庭 3']

    # 1. 积分余额变化
    ax1 = axes[0, 0]
    for i in range(3):
        credit_balances = last_episode_details['credit_balances'][i]
        if len(credit_balances) > 0:
            # 对齐长度（积分余额比时间步多一个初始值，取后续）
            if len(credit_balances) == len(time_steps) + 1:
                balance_data = credit_balances[1:]
            elif len(credit_balances) == len(time_steps):
                balance_data = credit_balances
            else:
                balance_data = [credit_balances[0]] * (len(time_steps) - len(credit_balances)) + credit_balances
            ax1.plot(time_steps, balance_data,
                     color=colors_agent[i], label=labels_agent[i],
                     linewidth=2, alpha=0.8, marker='o', markersize=4)

    # 计算 y 轴范围
    all_balance_vals = []
    for i in range(3):
        cb = last_episode_details['credit_balances'][i]
        if cb:
            if len(cb) == len(time_steps) + 1:
                data = cb[1:]
            elif len(cb) == len(time_steps):
                data = cb
            else:
                data = [cb[0]] * (len(time_steps) - len(cb)) + cb
            all_balance_vals.extend(data)
    if all_balance_vals:
        min_b = min(all_balance_vals)
        max_b = max(all_balance_vals)
        y_min = max(0, min_b - (max_b - min_b) * 0.1)
        y_max = max_b + (max_b - min_b) * 0.5
        if initial_credit > max_b * 0.8:
            y_max = max(y_max, initial_credit * 1.5)
        ax1.set_ylim([y_min, y_max])

    ax1.axhline(y=initial_credit, color='gray', linestyle='--', linewidth=1.5, alpha=0.7,
                label=f'初始积分 ({initial_credit:.0f})')
    ax1.set_xlabel('时间步')
    ax1.set_ylabel('积分余额')
    ax1.set_title('积分余额随时间步变化')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. 积分变化量（柱状图）
    ax2 = axes[0, 1]
    for i in range(3):
        credit_changes = last_episode_details['credit_changes'][i]
        if len(credit_changes) > 0:
            if len(credit_changes) > len(time_steps):
                changes = credit_changes[:len(time_steps)]
            elif len(credit_changes) < len(time_steps):
                changes = credit_changes + [0.0] * (len(time_steps) - len(credit_changes))
            else:
                changes = credit_changes
            ax2.bar([t + i * 0.25 for t in time_steps], changes,
                    width=0.25, color=colors_agent[i], label=labels_agent[i], alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax2.set_xlabel('时间步')
    ax2.set_ylabel('积分变化量')
    ax2.set_title('积分变化量随时间步变化（负值=消耗，正值=获得）')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. 电价变化
    ax3 = axes[1, 0]
    grid_prices = last_episode_details['grid_prices']
    if grid_prices:
        if len(grid_prices) > len(time_steps):
            prices = grid_prices[:len(time_steps)]
        elif len(grid_prices) < len(time_steps):
            prices = grid_prices + [0.0] * (len(time_steps) - len(grid_prices))
        else:
            prices = grid_prices
        ax3.plot(time_steps, prices, color='#FF9800', linewidth=2, marker='s', markersize=5, label='电网电价')
        ax3.fill_between(time_steps, prices, alpha=0.3, color='#FF9800')
    ax3.set_xlabel('时间步')
    ax3.set_ylabel('电价 (元/kWh)')
    ax3.set_title('电网电价随时间步变化')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)

    # 4. 各家庭净负荷
    ax4 = axes[1, 1]
    for i in range(3):
        agent_loads = last_episode_details['agent_loads'][i]
        if agent_loads:
            if len(agent_loads) > len(time_steps):
                loads = agent_loads[:len(time_steps)]
            elif len(agent_loads) < len(time_steps):
                loads = agent_loads + [0.0] * (len(time_steps) - len(agent_loads))
            else:
                loads = agent_loads
            ax4.plot(time_steps, loads, color=colors_agent[i], label=labels_agent[i],
                     linewidth=2, alpha=0.8, marker='o', markersize=4)
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax4.set_xlabel('时间步')
    ax4.set_ylabel('净负荷 (kW)')
    ax4.set_title('各家庭净负荷随时间步变化')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)

    # 5. 社区总负荷
    ax5 = axes[2, 0]
    community_load = last_episode_details['community_load']
    if community_load:
        if len(community_load) > len(time_steps):
            load_data = community_load[:len(time_steps)]
        elif len(community_load) < len(time_steps):
            load_data = community_load + [0.0] * (len(time_steps) - len(community_load))
        else:
            load_data = community_load
        ax5.plot(time_steps, load_data, color='#9C27B0', linewidth=2.5, marker='s', markersize=5, label='社区总负荷')
        ax5.fill_between(time_steps, load_data, alpha=0.3, color='#9C27B0')
    ax5.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax5.set_xlabel('时间步')
    ax5.set_ylabel('负荷 (kW)')
    ax5.set_title('社区总净负荷随时间步变化')
    ax5.legend(loc='best')
    ax5.grid(True, alpha=0.3)

    # 6. 社区储能 SOC
    ax6 = axes[2, 1]
    community_ess_soc = last_episode_details['community_ess_soc']
    if community_ess_soc:
        if len(community_ess_soc) > len(time_steps):
            soc_data = community_ess_soc[:len(time_steps)]
        elif len(community_ess_soc) < len(time_steps):
            soc_data = community_ess_soc + [0.0] * (len(time_steps) - len(community_ess_soc))
        else:
            soc_data = community_ess_soc
        ax6.plot(time_steps, soc_data, color='#FF5722', linewidth=2.5, marker='o', markersize=5, label='社区储能 SOC')
        ax6.fill_between(time_steps, soc_data, alpha=0.3, color='#FF5722')
        ax6.axhline(y=0.9, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='SOC 上限 (0.9)')
        ax6.axhline(y=0.1, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='SOC 下限 (0.1)')
    ax6.set_xlabel('时间步')
    ax6.set_ylabel('SOC (0-1)')
    ax6.set_title('社区储能 SOC 随时间步变化')
    ax6.set_ylim([0, 1])
    ax6.legend(loc='best', fontsize=9)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'last_episode_credit_details.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"最后一个 episode 的积分细节图已保存至 {plot_path}")
    plt.close()


if __name__ == "__main__":
    """
    命令行入口：支持自定义 episode 数、基准峰值、保存/加载路径等。
    示例：
        python train_mappo.py --episodes 2000 --save_dir ./my_model
    """
    parser = argparse.ArgumentParser(description='训练 MAPPO 算法（100天数据，20户，16训练/4测试）')
    parser.add_argument('--episodes', type=int, default=1500, help='训练 episode 总数')
    parser.add_argument('--baseline_peak', type=float, default=None, help='基准峰值 (kW)')
    parser.add_argument('--save_dir', type=str, default='multi_agent/algorithms/models', help='模型保存目录')
    parser.add_argument('--load_model', type=str, default=None, help='加载已有模型路径（若提供）')
    parser.add_argument('--eval_freq', type=int, default=1000, help='评估频率（episode 数）')
    parser.add_argument('--save_freq', type=int, default=1000, help='模型保存频率（episode 数）')
    args = parser.parse_args()

    train_mappo(
        num_episodes=args.episodes,
        baseline_peak=args.baseline_peak,
        save_dir=args.save_dir,
        load_model=args.load_model,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq
    )