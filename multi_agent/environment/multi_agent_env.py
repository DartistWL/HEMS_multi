"""
Multi-Agent Home Energy Management Environment
多智能体家庭能源管理环境
"""
import numpy as np
import sys
import os
from collections import deque

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from multi_agent.environment.single_agent_wrapper import SingleAgentWrapper
from multi_agent.environment.community_ess import CommunityESS
from multi_agent.environment.credit_system import CommunityCreditSystem
from multi_agent.environment.peak_tracker import PeakTracker


class MultiAgentHEMEnv:
    """
    多智能体家庭能源管理环境
    
    管理3户家庭，社区共享储能，积分系统，峰值追踪
    """

    def __init__(self, n_agents=3, community_ess_capacity=36.0,
                 baseline_peak=31.01, community_weight=0.2,
                 community_credit_cost_weight=0.1, community_credit_benefit_weight=0.1,
                 initial_credit=100.0, peak_penalty_exponent=2.0,  # 默认初始积分为100.0
                 peak_discharge_bonus=0.0, peak_credit_cost_reduction=1.0,
                 pv_coefficients=None, reward_for_individual_only=False,
                 train_house_ids=None, test_house_ids=None):
        """
        初始化多智能体环境
        
        Args:
            n_agents: 家庭数量
            community_ess_capacity: 社区共享储能容量（kWh）
            baseline_peak: 基准峰值（用于归一化峰值惩罚）
            community_weight: 社区峰值惩罚权重（固定值0.2）
            community_credit_cost_weight: 社区积分成本权重（从社区储能获取能量消耗积分的惩罚权重）
            community_credit_benefit_weight: 社区积分收益权重（向社区储能贡献能量获得积分的奖励权重）
            initial_credit: 每个家庭的初始积分余额（默认100.0）
            peak_penalty_exponent: 峰值惩罚的指数（1.0=线性，2.0=二次，1.5=介于两者之间）
            peak_discharge_bonus: 在峰值时段使用社区储能放电的额外奖励系数（每kW）
            peak_credit_cost_reduction: 在峰值时段使用社区储能时，积分成本降低的倍数（0.5=降低50%）
            pv_coefficients: 每个家庭的光伏系数列表，如果为None则都使用1.0
            reward_for_individual_only: 若为 True，则每个智能体仅收到「个人奖励 - 积分成本」，
                不含峰值惩罚、社区储能激励、排队惩罚等，用于独立基线以体现 MAPPO 的削峰优势。
        """
        # 设置训练/测试住户列表
        if train_house_ids is None:
            train_house_ids = [f"H{i}" for i in range(1, 16)]  # H1~H15
        if test_house_ids is None:
            test_house_ids = [f"H{i}" for i in range(16, 21)]  # H16~H20
        self.train_house_ids = train_house_ids
        self.test_house_ids = test_house_ids
        self.n_agents = n_agents
        self.reward_for_individual_only = bool(reward_for_individual_only)
        self.community_weight = community_weight
        self.community_credit_cost_weight = community_credit_cost_weight
        self.community_credit_benefit_weight = community_credit_benefit_weight
        self.baseline_peak = baseline_peak
        self.peak_penalty_exponent = peak_penalty_exponent
        self.peak_discharge_bonus = peak_discharge_bonus
        self.peak_credit_cost_reduction = peak_credit_cost_reduction
        # 低价时段排队充电惩罚：限制社区总充电功率，鼓励按顺序/分散充电（多端同时充放后需加强以控峰）
        self.queue_target_power = 15.0  # 允许的“安全充电功率阈值”（kW），超过则惩罚；可从 config.json training 覆盖
        self.queue_penalty_weight = 3.5  # 惩罚强度；可从 config 覆盖
        self.peak_grid_purchase_penalty_weight = 2.5  # 峰值时段从电网购电的惩罚权重；可从 config 覆盖
        try:
            config_path = os.path.join(project_root, 'multi_agent', 'config.json')
            if os.path.isfile(config_path):
                import json as _json
                with open(config_path, 'r', encoding='utf-8') as _f:
                    _cfg = _json.load(_f)
                _tr = _cfg.get('training', {})
                if 'queue_target_power' in _tr:
                    self.queue_target_power = float(_tr['queue_target_power'])
                if 'queue_penalty_weight' in _tr:
                    self.queue_penalty_weight = float(_tr['queue_penalty_weight'])
                if 'peak_grid_purchase_penalty_weight' in _tr:
                    self.peak_grid_purchase_penalty_weight = float(_tr['peak_grid_purchase_penalty_weight'])
        except Exception:
            pass

        # 社区储能使用激励：鼓励在低电价时充电、在高电价时放电；SOC 低时加强充电激励，避免放空后不再动
        self.low_price_charge_bonus = 0.5
        self.high_price_discharge_bonus = 0.3
        self.low_soc_charge_bonus_scale = 2.0  # SOC<0.3 时充电奖励乘 (1 + scale*(1-soc))
        self.low_soc_threshold = 0.3
        self.high_price_threshold = 0.6  # 电价≥此值为高电价，高电价不给充电奖励
        self.medium_price_charge_ratio = 0.6  # 中段电价时充电奖励相对低电价的比例
        try:
            _cfg_path = os.path.join(project_root, 'multi_agent', 'config.json')
            if os.path.isfile(_cfg_path):
                import json as _json
                with open(_cfg_path, 'r', encoding='utf-8') as _f:
                    _cfg = _json.load(_f)
                _tr = _cfg.get('training', {})
                if 'low_price_charge_bonus' in _tr:
                    self.low_price_charge_bonus = float(_tr['low_price_charge_bonus'])
                if 'high_price_discharge_bonus' in _tr:
                    self.high_price_discharge_bonus = float(_tr['high_price_discharge_bonus'])
                if 'low_soc_charge_bonus_scale' in _tr:
                    self.low_soc_charge_bonus_scale = float(_tr['low_soc_charge_bonus_scale'])
                if 'low_soc_threshold' in _tr:
                    self.low_soc_threshold = float(_tr['low_soc_threshold'])
                if 'high_price_threshold' in _tr:
                    self.high_price_threshold = float(_tr['high_price_threshold'])
                if 'medium_price_charge_ratio' in _tr:
                    self.medium_price_charge_ratio = float(_tr['medium_price_charge_ratio'])
        except Exception:
            pass

        # 初始化光伏系数
        if pv_coefficients is None:
            pv_coefficients = [1.0] * n_agents
        self.pv_coefficients = pv_coefficients

        # 创建单智能体环境
        self.agents = []
        for i in range(n_agents):
            agent = SingleAgentWrapper(
                agent_id=i,
                pv_coefficient=pv_coefficients[i],
                storenet_base_dir="../data/storenet_ireland_2020",  # 新增
                price_profile="lee2020",  # 新增
                steps_per_day=48,  # 新增
                use_carbon_in_reward=False  # 新增
            )
            self.agents.append(agent)

        # 创建社区共享储能
        self.community_ess = CommunityESS(capacity=community_ess_capacity)

        # 创建积分系统
        self.credit_system = CommunityCreditSystem(n_agents=n_agents, initial_credit=initial_credit)

        # 积分定价方案（从 config 读取，便于与现有方案对比）
        self.credit_pricing_scheme = 'uniform'
        self.credit_contribution_discount_factor = 0.0
        try:
            config_path = os.path.join(project_root, 'multi_agent', 'config.json')
            if os.path.isfile(config_path):
                import json
                with open(config_path, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                cp = cfg.get('credit_pricing', {})
                self.credit_pricing_scheme = cp.get('scheme', 'uniform')
                self.credit_contribution_discount_factor = float(cp.get('contribution_discount_factor', 0.0))
        except Exception:
            pass
        # 贡献定价按“向社区储能充电量”（kWh）；每户累计充电量，每 episode 在 reset 时清零
        self._agent_energy_charged_to_community = {i: 0.0 for i in range(n_agents)}

        # 创建峰值追踪器
        self.peak_tracker = PeakTracker(
            window_size=48,
            threshold_factor=1.2,
            baseline_peak=baseline_peak,
            penalty_exponent=peak_penalty_exponent
        )

        # 训练/评估数据配置
        self.training_dates = []  # 训练日期列表（7天）
        self.evaluation_dates = []  # 评估日期列表（3天）
        self.current_date_index = 0
        self.current_date = None

        # 记录
        self.episode_records = {
            'community_net_loads': [],
            'peak_penalties': [],
            'credit_transactions': [],
            'community_ess_soc': []
        }

    def reset(self, mode='train', date_index=0, house_index=None, initial_community_soc=None):
        """
        重置环境
        
        Args:
            mode: 'train' 或 'eval'
            date_index: 日期索引
            initial_community_soc: 社区储能初始 SOC（0~1），若为 None 则使用默认值（如 0.5），用于敏感性实验
        
        Returns:
            list: 所有智能体的初始状态
        """

        # 确定日期
        if mode == 'train':
            dates = self.training_dates
        else:
            dates = self.evaluation_dates

        if len(dates) == 0:
            # 如果没有设置日期，使用默认日期
            self.current_date = '2011-07-03'
        else:
            self.current_date = dates[date_index % len(dates)]
        self.current_date_index = date_index

        # 确定使用哪个住户
        if mode == 'train':
            house_pool = self.train_house_ids
        else:
            house_pool = self.test_house_ids

        if house_index is None:
            # 默认按 date_index 循环（为了兼容旧调用）
            house_index = date_index % len(house_pool)
        house_id = house_pool[house_index]

        # 为所有 agent 设置相同的住户
        for agent in self.agents:
            agent.set_house_id(house_id)

        # 评估/训练时按日期应用光伏系数（不同天气对应不同光伏发电）
        pv_list = getattr(self, 'evaluation_pv_coefficients_list', None) if mode != 'train' else getattr(self,
                                                                                                         'training_pv_coefficients_list',
                                                                                                         None)
        if pv_list and date_index < len(pv_list):
            for i, agent in enumerate(self.agents):
                coefs = pv_list[date_index]
                agent.env.data_interface.set_pv_coefficient(coefs[i] if i < len(coefs) else coefs[0])

        # 重置所有智能体；训练与评估均为单日 48 步/episode，与独立基线一致（7 天由 episode % 7 循环）
        states = []
        for agent in self.agents:
            agent.env.single_day_episode = True
            state = agent.reset(start_date=self.current_date, start_time_index=0)
            states.append(state)

        # 重置社区系统（可选指定初始 SOC，用于参数扫描实验）
        self.community_ess.reset(initial_soc=initial_community_soc)
        self.credit_system.reset()
        self.peak_tracker.reset()
        # 重置每户向社区储能的累计充电量（用于贡献定价 energy_charged）
        self._agent_energy_charged_to_community = {i: 0.0 for i in range(self.n_agents)}

        # 重置记录
        self.episode_records = {
            'community_net_loads': [],
            'peak_penalties': [],
            'credit_transactions': [],
            'community_ess_soc': []
        }

        # 重置奖励分解记录（用于诊断）
        self._reward_breakdown = {
            'individual_reward': [],
            'peak_penalty_value': [],
            'credit_penalty': [],
            'peak_discharge_reward': [],
            'peak_grid_purchase_penalty': [],
            'community_ess_reward': [],
            'queue_penalty': [],
            'total_reward': []
        }

        # 重置充电功率统计（用于诊断）
        self._charge_power_stats = {
            'total_charge_power_all': [],
            'max_total_charge': 0.0,
            'trigger_count': 0
        }

        # 扩展局部状态，添加社区信息
        states = self._add_community_info_to_states(states)

        return states

    def step(self, actions):
        """
        执行一步（所有智能体同时执行）
        
        Args:
            actions: 动作列表，每个元素是一个动作字典
        
        Returns:
            tuple: (next_states, rewards, dones, info)
        """
        # 1. 收集所有智能体的社区储能交互动作
        community_charge_powers = []  # 向社区储能充电的功率（正值）
        community_discharge_powers = []  # 从社区储能放电的功率（正值）

        for i, action in enumerate(actions):
            community_power = action.get('community_ess_power', 0.0)
            if community_power > 0:
                # 从社区储能充电
                community_discharge_powers.append(community_power)
                community_charge_powers.append(0.0)
            elif community_power < 0:
                # 向社区储能充电
                community_charge_powers.append(-community_power)
                community_discharge_powers.append(0.0)
            else:
                community_charge_powers.append(0.0)
                community_discharge_powers.append(0.0)

        # 2. 更新社区共享储能（多端口/逆变器：可同时向多户放电或接受多户充电，如电网般并行；SOC 按净充放更新）
        total_charge = sum(community_charge_powers)
        total_discharge = sum(community_discharge_powers)
        community_ess_state = self.community_ess.update(total_charge, total_discharge)
        actual_total_charge = community_ess_state['charge_power']
        actual_total_discharge = community_ess_state['discharge_power']

        # 3. 按请求比例分配：充电户分得实际充电功率，放电户分得实际放电功率（可同时进行）
        charge_scale = (actual_total_charge / total_charge) if total_charge > 0 else 0.0
        discharge_scale = (actual_total_discharge / total_discharge) if total_discharge > 0 else 0.0
        actual_community_powers = []
        for i in range(self.n_agents):
            if community_charge_powers[i] > 0:
                actual_community_powers.append(-charge_scale * community_charge_powers[i])  # 负=充电
            elif community_discharge_powers[i] > 0:
                actual_community_powers.append(discharge_scale * community_discharge_powers[i])  # 正=放电
            else:
                actual_community_powers.append(0.0)

        # 4. 处理积分交易（使用实际功率）
        grid_price = self.agents[0].env.data_interface.get_electricity_price(
            self.current_date, self.agents[0].env.current_time_index
        )
        community_demand = sum([agent.get_net_load() for agent in self.agents])

        # 判断电价档位（用于社区储能使用激励）：低电价、中段电价给充电奖励，高电价不给
        low_price_threshold = 0.35
        high_price_threshold = getattr(self, 'high_price_threshold', 0.6)
        is_low_price = grid_price < low_price_threshold
        is_high_price = grid_price >= high_price_threshold
        is_medium_price = (grid_price >= low_price_threshold) and (grid_price < high_price_threshold)

        # 贡献定价用：本步前的每户累计充电量及最大值（用于购电折扣）
        max_energy_charged = max(
            self._agent_energy_charged_to_community.values()) if self._agent_energy_charged_to_community else 0.0

        credit_transactions = []
        for i, action in enumerate(actions):
            actual_power = actual_community_powers[i]
            if abs(actual_power) > 1e-6:  # 避免浮点误差
                energy_amount = abs(actual_power) * 0.5  # 转换为kWh

                if actual_power > 0:
                    # 从社区储能购买（放电）
                    transaction_type = 'buy'
                    price = self.credit_system.calculate_dynamic_price(
                        grid_price, community_ess_state['soc'],
                        community_demand, transaction_type,
                        agent_id=i,
                        pricing_scheme=self.credit_pricing_scheme,
                        contribution_discount_factor=self.credit_contribution_discount_factor,
                        agent_energy_charged=self._agent_energy_charged_to_community.get(i, 0.0),
                        max_energy_charged=max_energy_charged
                    )
                else:
                    # 向社区储能出售（充电）
                    transaction_type = 'sell'
                    price = self.credit_system.calculate_dynamic_price(
                        grid_price, community_ess_state['soc'],
                        community_demand, transaction_type,
                        agent_id=i,
                        pricing_scheme=self.credit_pricing_scheme,
                        contribution_discount_factor=self.credit_contribution_discount_factor,
                        agent_energy_charged=self._agent_energy_charged_to_community.get(i, 0.0),
                        max_energy_charged=max_energy_charged
                    )

                result = self.credit_system.process_transaction(
                    i, energy_amount, price, transaction_type
                )
                credit_transactions.append(result)
                # 向社区储能出售后，累计该户充电量（用于下一步及之后的贡献定价）
                if transaction_type == 'sell':
                    self._agent_energy_charged_to_community[i] = self._agent_energy_charged_to_community.get(i,
                                                                                                             0.0) + energy_amount

        # 5. 执行所有智能体的步骤
        next_states = []
        individual_rewards = []
        dones = []
        agent_infos = []

        for i, (agent, action) in enumerate(zip(self.agents, actions)):
            # 使用按比例分配的实际功率
            actual_community_power = actual_community_powers[i]

            next_state, reward, done, info = agent.step(action, actual_community_power)
            next_states.append(next_state)
            individual_rewards.append(reward)
            dones.append(done)
            # info中已经包含了pv_flow信息（从single_agent_wrapper返回）
            agent_infos.append(info)

        # 5. 计算社区净负荷（与电网的总交换功率，用于峰值统计与论文指标）
        # 必须使用 sum(agent_net_loads)，即含「从电网购电向社区储能充电」部分，才能反映电网侧真实负荷；
        # 若用 grid_only（排除社区储能），会漏计向社区储能的充电功率，导致峰值偏低、积分系统削峰效果无法正确体现。
        agent_net_loads = [info['net_load'] for info in agent_infos]
        agent_grid_only_net_loads = [info.get('grid_only_net_load', info['net_load']) for info in agent_infos]
        community_net_load = sum(agent_net_loads)  # 社区与电网的总交换功率（含向社区储能充电的购电）= 电网侧真实负荷

        # 6. 更新峰值追踪器
        self.peak_tracker.update(community_net_load)
        # 取消归一化，让峰值惩罚信号更明确（use_normalization=False）
        peak_penalty, threshold, is_peak = self.peak_tracker.calculate_peak_penalty(community_net_load,
                                                                                    use_normalization=False)

        # 7. 计算每个智能体的峰值惩罚（按贡献分配）
        agent_peak_penalties = []
        agent_contributions = []

        for i, (net_load, avg_load) in enumerate(zip(agent_net_loads,
                                                     [agent.get_avg_net_load() for agent in self.agents])):
            contribution = self.peak_tracker.calculate_agent_contribution(net_load, avg_load)
            agent_contributions.append(contribution)

        total_contribution = sum(agent_contributions)

        for i, contribution in enumerate(agent_contributions):
            if total_contribution > 0:
                agent_penalty = peak_penalty * (contribution / total_contribution)
            else:
                agent_penalty = 0.0
            agent_peak_penalties.append(agent_penalty)

        # 8. 计算积分成本/收益（从credit_transactions中提取）
        agent_credit_costs = [0.0] * self.n_agents  # 每个智能体的积分成本（负值表示消耗）
        for transaction_result in credit_transactions:
            if transaction_result.get('success', False):
                agent_id = transaction_result['transaction']['agent_id']
                credit_change = transaction_result.get('credit_change', 0.0)
                agent_credit_costs[agent_id] = credit_change  # credit_change已经是负值（消耗）或正值（收益）

        # 8.1 计算社区排队充电惩罚：避免低电价时所有家庭同时大功率为EV和家庭ESS充电
        # 思路：统计当前时间步每个家庭的总充电功率（EV + 私有ESS + 社区ESS充电），
        # 如果社区总充电功率超过queue_target_power，则按超出部分平方惩罚，并按各家庭的充电占比进行分摊。
        agent_charge_powers = []
        for i, agent in enumerate(self.agents):
            env_i = agent.env
            # EV充电功率（正值表示充电）
            ev_charge = max(0.0, getattr(env_i, "current_ev_power", 0.0))
            # 私有ESS充电功率（正值表示充电）
            ess_charge = max(0.0, getattr(env_i, "current_battery_power", 0.0))
            # 社区ESS充电功率：actual_community_powers为负值表示向社区储能充电
            community_charge = max(0.0, -actual_community_powers[i])
            total_charge_power = ev_charge + ess_charge + community_charge
            agent_charge_powers.append(total_charge_power)

        total_charge_power_all = sum(agent_charge_powers)

        # 记录充电功率信息（用于诊断）
        if not hasattr(self, '_charge_power_stats'):
            self._charge_power_stats = {
                'total_charge_power_all': [],
                'max_total_charge': 0.0,
                'trigger_count': 0
            }
        self._charge_power_stats['total_charge_power_all'].append(total_charge_power_all)
        self._charge_power_stats['max_total_charge'] = max(self._charge_power_stats['max_total_charge'],
                                                           total_charge_power_all)

        agent_queue_penalties = [0.0] * self.n_agents
        if total_charge_power_all > self.queue_target_power:
            self._charge_power_stats['trigger_count'] += 1
            # 超出安全充电阈值的部分
            excess_charge = total_charge_power_all - self.queue_target_power
            # 取消归一化，直接使用超出部分的平方作为惩罚基础，让信号更明确
            # 使用平方惩罚，鼓励将充电在时间上摊开
            # 修改前：normalized_excess_charge = (excess_charge / queue_target_power) ** 2
            # 修改后：直接使用 excess_charge ** 2，然后除以一个缩放因子（queue_target_power）来平衡量级
            raw_excess_penalty = excess_charge ** 2
            # 除以queue_target_power作为缩放因子（而不是queue_target_power^2），使惩罚量级合理
            scaled_excess_penalty = raw_excess_penalty / self.queue_target_power
            for i, charge_power in enumerate(agent_charge_powers):
                if charge_power > 0:
                    share = charge_power / (total_charge_power_all + 1e-6)
                else:
                    share = 0.0
                agent_queue_penalties[i] = self.queue_penalty_weight * scaled_excess_penalty * share

        # 9. 计算总奖励（个人奖励 - 社区惩罚 - 积分成本 + 积分收益 + 峰值时段放电奖励 - 额外强峰值惩罚）
        total_rewards = []
        for i, (individual_reward, peak_penalty) in enumerate(zip(individual_rewards, agent_peak_penalties)):
            # 积分成本（credit_change是负值，所以这里加上就是减去成本）
            credit_cost = agent_credit_costs[i]  # 负值表示消耗积分（成本），正值表示获得积分（收益）

            # 计算积分惩罚（在峰值时段可能降低）
            if credit_cost < 0:
                # 消耗积分（成本）
                # 在峰值时段且使用了社区储能（放电）时，降低积分成本
                actual_power = actual_community_powers[i]
                if is_peak and actual_power > 0:  # 峰值时段且从社区储能放电
                    effective_cost_weight = self.community_credit_cost_weight * self.peak_credit_cost_reduction
                else:
                    effective_cost_weight = self.community_credit_cost_weight

                # 独立基线模式：提高积分成本权重，使积分成本 ≈ 积分价格（与电网价格相当），
                # 避免智能体无限制地从社区储能放电（因为节省的电网成本 > 积分成本）
                if self.reward_for_individual_only:
                    # 将积分成本权重提高到 1.0，使得积分成本 = 积分价格（真实反映使用社区储能的成本）
                    effective_cost_weight = 1.0

                credit_penalty = effective_cost_weight * abs(credit_cost)
            else:
                # 获得积分（收益）
                # 独立基线模式：同样提高收益权重，保持对称
                if self.reward_for_individual_only:
                    benefit_weight = 1.0
                else:
                    benefit_weight = self.community_credit_benefit_weight
                credit_penalty = -benefit_weight * credit_cost  # 负号因为这是奖励

            # 峰值时段使用社区储能放电的额外奖励（适度加强）
            # 这是关键机制：在峰值时段，鼓励从社区储能放电，而不是从电网购电
            # 注意：奖励不能太大，否则会导致回报不收敛
            peak_discharge_reward = 0.0
            actual_power = actual_community_powers[i]
            if is_peak and actual_power > 0:  # 峰值时段且从社区储能放电
                # 适度奖励，避免回报过大（从5.0倍降低到1.5倍）
                peak_discharge_reward = self.peak_discharge_bonus * actual_power * 1.5  # 奖励提升1.5倍

            # 峰值时段从电网购电的额外惩罚（新增）
            # 当社区净负荷高（峰值时段）时，如果家庭从电网购电（而不是从社区储能），会受到额外惩罚
            peak_grid_purchase_penalty = 0.0
            if is_peak:
                # 获取家庭当前从电网的净购电功率（正值表示购电）
                agent_grid_net_load = agent_grid_only_net_loads[i]  # 与电网的直接交互（不包括社区储能）
                if agent_grid_net_load > 0:  # 从电网购电
                    # 计算惩罚：购电功率越大，惩罚越大
                    # 但如果从社区储能放电（actual_power > 0），可以减少惩罚
                    if actual_power > 0:  # 同时从社区储能放电，减少惩罚
                        # 从社区储能放电的部分不计入惩罚
                        effective_grid_purchase = max(0.0, agent_grid_net_load - actual_power)
                    else:
                        effective_grid_purchase = agent_grid_net_load

                    # 惩罚与购电功率成正比
                    peak_grid_purchase_penalty = self.peak_grid_purchase_penalty_weight * effective_grid_purchase

            # 社区储能使用激励：低电价和中段电价给充电奖励，高电价不给；高电价时鼓励放电
            community_ess_reward = 0.0
            actual_power = actual_community_powers[i]
            soc = community_ess_state['soc']
            if actual_power < 0:  # 向社区储能充电（负值）
                if not is_high_price:  # 仅低电价和中段电价给充电奖励，高电价不给
                    base_bonus = self.low_price_charge_bonus * abs(actual_power)
                    if soc < getattr(self, 'low_soc_threshold', 0.3):
                        scale = 1.0 + getattr(self, 'low_soc_charge_bonus_scale', 2.0) * max(0.0, 1.0 - soc)
                    else:
                        scale = 1.0
                    if is_low_price:
                        community_ess_reward = base_bonus * scale
                    else:  # 中段电价，按比例给奖励
                        ratio = getattr(self, 'medium_price_charge_ratio', 0.6)
                        community_ess_reward = base_bonus * scale * ratio
            elif not is_low_price and actual_power > 0:  # 高/中电价时段且从社区储能放电（正值）
                community_ess_reward = self.high_price_discharge_bonus * actual_power

            # 排队充电惩罚：鼓励在低价时段分散充电，而不是所有家庭在同一时间步同时大功率充电
            # 这是主要的压峰机制，通过限制同时充电功率来避免低电价时段出现峰值
            queue_penalty = agent_queue_penalties[i]

            # 计算各项奖励/惩罚的数值（用于诊断）
            peak_penalty_value = self.community_weight * peak_penalty
            if self.reward_for_individual_only:
                # 独立基线模式：仅个人目标（个人奖励 - 积分成本），不对峰值/社区储能激励做优化，以体现 MAPPO 削峰优势
                total_reward = individual_reward - credit_penalty
            else:
                total_reward = (
                        individual_reward
                        - peak_penalty_value  # 原有的PeakTracker峰值惩罚（保留作为基础机制）
                        - credit_penalty
                        + peak_discharge_reward  # 峰值时段从社区储能放电的奖励（大幅加强）
                        - peak_grid_purchase_penalty  # 峰值时段从电网购电的惩罚（新增，关键机制）
                        + community_ess_reward  # 社区储能使用激励
                        - queue_penalty  # 排队充电惩罚（提高权重后）
                )
            total_rewards.append(total_reward)

            # 记录奖励分解（用于诊断，只在第一个智能体时记录一次）
            if i == 0:
                if not hasattr(self, '_reward_breakdown'):
                    self._reward_breakdown = {
                        'individual_reward': [],
                        'peak_penalty_value': [],
                        'credit_penalty': [],
                        'peak_discharge_reward': [],
                        'peak_grid_purchase_penalty': [],
                        'community_ess_reward': [],
                        'queue_penalty': [],
                        'total_reward': []
                    }
                self._reward_breakdown['individual_reward'].append(individual_reward)
                self._reward_breakdown['peak_penalty_value'].append(peak_penalty_value)
                self._reward_breakdown['credit_penalty'].append(credit_penalty)
                self._reward_breakdown['peak_discharge_reward'].append(peak_discharge_reward)
                self._reward_breakdown['peak_grid_purchase_penalty'].append(peak_grid_purchase_penalty)
                self._reward_breakdown['community_ess_reward'].append(community_ess_reward)
                self._reward_breakdown['queue_penalty'].append(queue_penalty)
                self._reward_breakdown['total_reward'].append(total_reward)

        # 10. 记录
        self.episode_records['community_net_loads'].append(community_net_load)
        self.episode_records['peak_penalties'].append(peak_penalty)
        self.episode_records['community_ess_soc'].append(community_ess_state['soc'])
        if credit_transactions:
            self.episode_records['credit_transactions'].extend(credit_transactions)

        # 11. 扩展局部状态，添加社区信息
        # 使用get_state()获取完整状态（包含capacity等字段）
        community_ess_full_state = self.community_ess.get_state()
        next_states = self._add_community_info_to_states(next_states,
                                                         community_ess_full_state,
                                                         community_net_load,
                                                         threshold)

        # 12. 构建info
        # 收集所有智能体的PV流向信息
        agent_pv_flows = [info.get('pv_flow', {}) for info in agent_infos]
        # 每户向社区储能充电/放电功率（kW），用于行为差异分析：高光伏户充电多、低光伏户放电多
        agent_community_charge = [max(0.0, -p) for p in actual_community_powers]
        agent_community_discharge = [max(0.0, p) for p in actual_community_powers]

        info = {
            'community_net_load': community_net_load,
            'peak_penalty': peak_penalty,
            'peak_threshold': threshold,
            'is_peak': is_peak,
            'community_ess_soc': community_ess_state['soc'],
            'agent_net_loads': agent_net_loads,
            'agent_peak_penalties': agent_peak_penalties,
            'credit_balances': self.credit_system.get_all_balances(),
            'agent_pv_flows': agent_pv_flows,  # 添加PV流向信息
            'agent_community_charge': agent_community_charge,  # 每户向社区储能充电功率（kW）
            'agent_community_discharge': agent_community_discharge,  # 每户从社区储能放电功率（kW）
            'credit_transactions': credit_transactions  # 添加积分交易信息，用于成本计算
        }

        return next_states, total_rewards, dones, info

    def set_training_dates(self, dates, pv_coefficients_list):
        """
        设置训练日期和对应的光伏系数
        
        Args:
            dates: 日期列表（7天：4天晴天+3天阴天）
            pv_coefficients_list: 每个日期对应的光伏系数列表（每个家庭一个系数）
        """
        self.training_dates = dates
        self.training_pv_coefficients_list = pv_coefficients_list if pv_coefficients_list else []

    def set_evaluation_dates(self, dates, pv_coefficients_list):
        """
        设置评估日期和对应的光伏系数
        
        Args:
            dates: 日期列表（3天：1天晴天+1天阴天+1天正常）
            pv_coefficients_list: 每个日期对应的光伏系数列表
        """
        self.evaluation_dates = dates
        self.evaluation_pv_coefficients_list = pv_coefficients_list if pv_coefficients_list else []

    def _add_community_info_to_states(self, states, community_ess_state=None,
                                      community_net_load=None, peak_threshold=None):
        """
        向局部状态添加社区信息
        
        Args:
            states: 状态列表（每个智能体的局部状态）
            community_ess_state: 社区储能状态（如果为None，则从self获取）
            community_net_load: 社区净负荷（如果为None，则计算）
            peak_threshold: 峰值阈值（如果为None，则计算）
        
        Returns:
            list: 扩展后的状态列表
        """
        # 获取社区信息
        if community_ess_state is None:
            community_ess_state = self.community_ess.get_state()
        if community_net_load is None:
            community_net_load = sum([agent.get_net_load() for agent in self.agents])
        if peak_threshold is None:
            peak_threshold = self.peak_tracker.avg_load * self.peak_tracker.threshold_factor

        community_avg_load = self.peak_tracker.avg_load
        credit_balances = self.credit_system.get_all_balances()

        # 当日是否为双休日（0=工作日，1=双休日），供策略区分 EV 在家/外出 等
        try:
            from datetime import datetime as _dt
            weekday = _dt.strptime(self.current_date, '%Y-%m-%d').weekday()
            is_weekend = 1.0 if weekday >= 5 else 0.0
        except Exception:
            is_weekend = 0.0

        # 为每个智能体的状态添加社区信息
        extended_states = []
        for agent_id, state in enumerate(states):
            # 创建状态的副本（避免修改原始状态）
            extended_state = state.copy()

            # 添加社区信息
            extended_state['community_ess_soc'] = community_ess_state['soc']
            extended_state['community_ess_capacity'] = community_ess_state['capacity']
            extended_state['community_credit_balance'] = credit_balances.get(agent_id, 0.0)
            extended_state['community_net_load'] = community_net_load
            extended_state['community_avg_load'] = community_avg_load
            extended_state['community_peak_threshold'] = peak_threshold
            extended_state['is_weekend'] = is_weekend

            extended_states.append(extended_state)

        return extended_states

    def get_community_state(self):
        """
        获取社区状态（用于共享Critic）
        
        Returns:
            dict: 社区状态信息
        """
        return {
            'community_ess_soc': self.community_ess.soc,
            'community_ess_capacity': self.community_ess.capacity,
            'community_net_load': sum([agent.get_net_load() for agent in self.agents]),
            'community_avg_load': self.peak_tracker.avg_load,
            'peak_threshold': self.peak_tracker.avg_load * self.peak_tracker.threshold_factor,
            'credit_balances': self.credit_system.get_all_balances()
        }
