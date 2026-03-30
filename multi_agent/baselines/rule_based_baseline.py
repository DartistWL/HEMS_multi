"""
Rule-Based Baseline
固定规则基线策略

策略规则：
1. 光伏优先自家使用
2. 余电充入私有储能
3. 社区储能仅在社区净负荷超过阈值时放电
4. EV策略：低价充电，高价放电
"""
import numpy as np
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from multi_agent.environment.multi_agent_env import MultiAgentHEMEnv


class RuleBasedBaseline:
    """
    固定规则基线策略
    """
    
    def __init__(self, peak_threshold_factor=1.2):
        """
        初始化固定规则基线
        
        Args:
            peak_threshold_factor: 峰值阈值系数（社区净负荷超过平均值*此系数时，社区储能放电）
        """
        self.peak_threshold_factor = peak_threshold_factor
        
        # 动作空间限制
        self.max_battery_charge = 4.4  # 家庭储能最大充电功率
        self.max_battery_discharge = 4.4  # 家庭储能最大放电功率
        self.max_ev_charge = 6.6  # EV最大充电功率
        self.max_ev_discharge = 6.6  # EV最大放电功率
        self.max_community_charge = 5.0  # 社区储能最大充电功率
        self.max_community_discharge = 5.0  # 社区储能最大放电功率
        
    def select_action(self, agent_state, agent_id, community_state, agent_net_load_history):
        """
        选择动作（固定规则策略）
        
        Args:
            agent_state: 智能体状态
            agent_id: 智能体ID
            community_state: 社区状态
            agent_net_load_history: 智能体历史净负荷（用于计算平均值）
        
        Returns:
            dict: 动作字典
        """
        action = {}
        
        # 获取当前状态信息
        pv_generation = agent_state['pv_generation']
        home_load = agent_state['home_load']
        ess_soc = agent_state['ess_state'] / 24.0  # 归一化到0-1
        ev_soc = agent_state['ev_battery_state'] / 24.0  # 归一化到0-1
        electricity_price = agent_state['electricity_price']
        time_index = agent_state['time_index']
        
        # 计算光伏余量
        pv_surplus = pv_generation - home_load
        
        # 1. 光伏优先自家使用，余电充入私有储能
        if pv_surplus > 0:
            # 有余电，优先充私有储能
            battery_charge_power = min(pv_surplus, self.max_battery_charge)
            # 考虑储能容量限制
            max_charge_by_soc = (1.0 - ess_soc) * 24.0 / 0.5 * 0.95  # 考虑效率
            battery_charge_power = min(battery_charge_power, max_charge_by_soc)
            
            action['battery_power'] = battery_charge_power
            remaining_surplus = pv_surplus - battery_charge_power
            
            # 如果还有余电，考虑充社区储能或EV
            if remaining_surplus > 0:
                # 优先充EV（如果在家且SOC较低）
                ev_at_home = agent_state.get('ev_at_home', True)  # 假设可以获取EV在家状态
                if ev_at_home and ev_soc < 0.8:
                    ev_charge_power = min(remaining_surplus, self.max_ev_charge)
                    max_ev_charge_by_soc = (1.0 - ev_soc) * 24.0 / 0.5 * 0.95
                    ev_charge_power = min(ev_charge_power, max_ev_charge_by_soc)
                    action['ev_power'] = ev_charge_power
                    remaining_surplus -= ev_charge_power
                else:
                    action['ev_power'] = 0
                
                # 如果还有余电，充社区储能
                if remaining_surplus > 0:
                    community_charge_power = min(remaining_surplus, self.max_community_charge)
                    action['community_ess_power'] = -community_charge_power  # 负值表示充电
                else:
                    action['community_ess_power'] = 0
            else:
                action['ev_power'] = 0
                action['community_ess_power'] = 0
        else:
            # 需要从外部获取能量
            deficit = -pv_surplus
            
            # 2. 社区储能仅在社区净负荷超过阈值时放电
            community_net_load = community_state.get('community_net_load', 0)
            community_avg_load = community_state.get('community_avg_load', 0)
            threshold = community_avg_load * self.peak_threshold_factor
            
            community_discharge_power = 0.0
            if community_net_load > threshold:
                # 社区净负荷超过阈值，从社区储能放电
                community_discharge_power = min(deficit, self.max_community_discharge)
                # 考虑社区储能SOC限制
                community_ess_soc = community_state.get('community_ess_soc', 0.5)
                if community_ess_soc > 0.1:  # 确保有足够能量
                    max_community_discharge = (community_ess_soc - 0.1) * 36.0 / 0.5 * 0.95
                    community_discharge_power = min(community_discharge_power, max_community_discharge)
                else:
                    community_discharge_power = 0
                
                action['community_ess_power'] = community_discharge_power
                deficit -= community_discharge_power
            else:
                action['community_ess_power'] = 0
            
            # 从私有储能放电
            if deficit > 0:
                battery_discharge_power = min(deficit, self.max_battery_discharge)
                # 考虑储能SOC限制
                max_discharge_by_soc = (ess_soc - 0.1) * 24.0 / 0.5 * 0.95
                battery_discharge_power = min(battery_discharge_power, max_discharge_by_soc)
                action['battery_power'] = -battery_discharge_power
                deficit -= battery_discharge_power
            else:
                action['battery_power'] = 0
            
            # 3. EV策略：低价充电，高价放电
            if deficit > 0:
                # 如果还有缺额，考虑EV
                ev_at_home = agent_state.get('ev_at_home', True)
                if ev_at_home:
                    if electricity_price < 0.3:  # 低价时段
                        # 充电
                        ev_charge_power = min(deficit, self.max_ev_charge)
                        max_ev_charge_by_soc = (1.0 - ev_soc) * 24.0 / 0.5 * 0.95
                        ev_charge_power = min(ev_charge_power, max_ev_charge_by_soc)
                        action['ev_power'] = ev_charge_power
                    elif electricity_price > 0.7:  # 高价时段
                        # 放电
                        ev_discharge_power = min(deficit, self.max_ev_discharge)
                        max_ev_discharge_by_soc = (ev_soc - 0.1) * 24.0 / 0.5 * 0.95
                        ev_discharge_power = min(ev_discharge_power, max_ev_discharge_by_soc)
                        action['ev_power'] = -ev_discharge_power
                    else:
                        action['ev_power'] = 0
                else:
                    action['ev_power'] = 0
            else:
                # 如果已经满足需求，根据价格决定EV动作
                ev_at_home = agent_state.get('ev_at_home', True)
                if ev_at_home:
                    if electricity_price < 0.3 and ev_soc < 0.8:
                        # 低价且SOC较低，充电
                        ev_charge_power = min(self.max_ev_charge, (1.0 - ev_soc) * 24.0 / 0.5 * 0.95)
                        action['ev_power'] = ev_charge_power
                    elif electricity_price > 0.7 and ev_soc > 0.2:
                        # 高价且SOC较高，放电
                        ev_discharge_power = min(self.max_ev_discharge, (ev_soc - 0.1) * 24.0 / 0.5 * 0.95)
                        action['ev_power'] = -ev_discharge_power
                    else:
                        action['ev_power'] = 0
                else:
                    action['ev_power'] = 0
        
        # 4. 其他设备动作（使用简单规则）
        # 洗衣机：在低价时段运行
        if electricity_price < 0.4:
            action['wash_machine_schedule'] = 1  # 运行
        else:
            action['wash_machine_schedule'] = 0  # 不运行
        
        # 空调：根据温度设置
        indoor_temp = agent_state.get('temperature', 25)
        if indoor_temp > 26:
            action['Air_conditioner_set_temp'] = 24
        elif indoor_temp < 20:
            action['Air_conditioner_set_temp'] = 22
        else:
            action['Air_conditioner_set_temp'] = 22
        
        action['Air_conditioner_set_temp2'] = action['Air_conditioner_set_temp']
        
        # 热水器：保持适中温度
        ewh_temp = agent_state.get('ewh_temp', 50)
        if ewh_temp < 45:
            action['ewh_set_temp'] = 55
        elif ewh_temp > 65:
            action['ewh_set_temp'] = 50
        else:
            action['ewh_set_temp'] = 55
        
        return action
    
    def evaluate(self, env, num_episodes=10, mode='eval'):
        """
        评估固定规则策略
        
        Args:
            env: 多智能体环境
            num_episodes: 评估轮数
            mode: 'train' 或 'eval'
        
        Returns:
            dict: 评估结果
        """
        results = {
            'episode_returns': [],
            'episode_costs': [],
            'peak_loads': [],
            'peak_penalties': [],
            'community_ess_utilization': [],
            'credit_balances': []
        }
        
        for episode in range(num_episodes):
            states = env.reset(mode=mode, date_index=episode)
            done = False
            episode_return = [0.0] * env.n_agents
            episode_cost = [0.0] * env.n_agents
            peak_loads = []
            peak_penalties = []
            
            # 用于计算社区平均负荷
            community_net_loads = []
            step_count = 0
            
            while not done:
                # 获取社区状态
                community_state = env.get_community_state()
                
                # 计算每个智能体的平均净负荷
                agent_net_load_histories = []
                for agent in env.agents:
                    agent_net_load_histories.append(agent.net_load_history)
                
                # 更新社区平均负荷
                current_community_load = sum([agent.get_net_load() for agent in env.agents])
                community_net_loads.append(current_community_load)
                if len(community_net_loads) > 48:
                    community_net_loads.pop(0)
                community_state['community_avg_load'] = np.mean(community_net_loads) if len(community_net_loads) > 0 else current_community_load
                community_state['community_net_load'] = current_community_load
                
                # 选择动作
                actions = []
                for i, state in enumerate(states):
                    action = self.select_action(
                        state, i, community_state, 
                        agent_net_load_histories[i]
                    )
                    actions.append(action)
                
                # 执行动作
                next_states, rewards, dones, info = env.step(actions)
                
                # 记录
                for i in range(env.n_agents):
                    episode_return[i] += rewards[i]
                    # 从环境中获取成本
                    if hasattr(env.agents[i].env, 'current_step_cost'):
                        episode_cost[i] += env.agents[i].env.current_step_cost
                
                peak_loads.append(info['community_net_load'])
                peak_penalties.append(info['peak_penalty'])
                
                states = next_states
                done = all(dones)
                step_count += 1
                
                # 防止无限循环
                if step_count >= 48:
                    done = True
            
            # 记录结果
            results['episode_returns'].append(episode_return)
            results['episode_costs'].append(episode_cost)
            results['peak_loads'].append(max(peak_loads) if peak_loads else 0)
            results['peak_penalties'].append(sum(peak_penalties))
            
            # 获取最终积分余额
            final_credit_balances = env.credit_system.get_all_balances()
            results['credit_balances'].append(final_credit_balances)
            
            print(f"Episode {episode+1}/{num_episodes}: "
                  f"Peak Load = {max(peak_loads):.2f} kW, "
                  f"Total Peak Penalty = {sum(peak_penalties):.4f}")
        
        # 计算统计信息
        avg_peak_load = np.mean(results['peak_loads'])
        avg_peak_penalty = np.mean(results['peak_penalties'])
        avg_return = np.mean([sum(r) for r in results['episode_returns']])
        
        print(f"\n=== Rule-Based Baseline Results ===")
        print(f"Average Peak Load: {avg_peak_load:.2f} kW")
        print(f"Average Peak Penalty: {avg_peak_penalty:.4f}")
        print(f"Average Return: {avg_return:.2f}")
        
        return results
