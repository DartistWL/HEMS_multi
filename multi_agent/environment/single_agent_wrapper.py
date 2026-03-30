"""
Single Agent Environment Wrapper
单智能体环境包装，用于多智能体系统中的每个家庭
"""
import sys
import os
import numpy as np

# 获取当前脚本所在目录（multi_agent/）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 项目根目录（multi_agent/ 的上一级）
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
# 将项目根目录加入 sys.path（如果需要）
sys.path.insert(0, PROJECT_ROOT)

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 从项目根目录导入environment模块（避免与multi_agent/environment冲突）
import importlib.util

env_file_path = os.path.join(project_root, "environment.py")
spec = importlib.util.spec_from_file_location("home_energy_env", env_file_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load environment.py from {env_file_path}")

home_energy_env = importlib.util.module_from_spec(spec)
spec.loader.exec_module(home_energy_env)
HomeEnergyManagementEnv = home_energy_env.HomeEnergyManagementEnv

from multi_agent.utils.data_interface import MultiAgentDataInterface


class SingleAgentWrapper:
    """
    单智能体环境包装
    包装原有的HomeEnergyManagementEnv，使其可以在多智能体系统中使用
    """

    def __init__(self, agent_id, pv_coefficient=1.0,
                 ev_capacity=24, ess_capacity=24,
                 charge_efficiency=0.95, discharge_efficiency=0.95,
                 storenet_base_dir="../data/storenet_ireland_2020",
                 price_profile="lee2020",
                 steps_per_day=48):
        self.agent_id = agent_id
        self.pv_coefficient = pv_coefficient
        self.ev_capacity = ev_capacity
        self.ess_capacity = ess_capacity
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.storennet_base_dir = storenet_base_dir
        self.price_profile = price_profile
        self.steps_per_day = steps_per_day

        # 当前使用的住户 ID，默认为 "H1"
        self.current_house_id = "H1"

        # 调用内部初始化方法创建环境
        self._init_env()

        # 记录平均净负荷（用于峰值贡献计算）
        self.net_load_history = []
        self.avg_net_load = 0.0

        # PV流向追踪（用于分析和可视化）
        self.pv_flow_history = []  # 每个时间步的PV流向字典

    def _init_env(self):
        # 测试代码
        # print(f"[Agent {self.agent_id}] 初始化环境，住户: {self.current_house_id}")
        """根据当前 self.current_house_id 和 pv_coefficient 创建底层环境"""
        data_interface = MultiAgentDataInterface.from_storenet_ireland_2020(
            house_id=self.current_house_id,
            base_dir=self.storennet_base_dir,
            price_profile=self.price_profile,
            steps_per_day=self.steps_per_day,
            pv_coefficient=self.pv_coefficient
        )
        self.env = HomeEnergyManagementEnv(
            ev_capacity=self.ev_capacity,
            ess_capacity=self.ess_capacity,
            charge_efficiency=self.charge_efficiency,
            discharge_efficiency=self.discharge_efficiency,
            data_interface=data_interface
        )
        # print(f"[Agent {self.agent_id}] 切换到住户 {self.current_house_id}")
        self.env.data_interface = data_interface

    def set_house_id(self, house_id: str):
        """切换住户数据，例如 'H1', 'H2', ..."""
        if self.current_house_id == house_id:
            return
        self.current_house_id = house_id
        self._init_env()  # 重建环境

    def reset(self, start_date='2011-07-03', start_time_index=0):
        """
        重置环境

        Args:
            start_date: 开始日期
            start_time_index: 开始时间索引
        """
        state = self.env.reset()
        self.env.current_time = start_date
        self.env.current_time_index = start_time_index
        self.net_load_history = []
        self.avg_net_load = 0.0
        self.pv_flow_history = []  # 重置PV流向历史
        return state

    def step(self, action, community_ess_power=0.0):
        """
        执行一步

        Args:
            action: 动作字典（包含原有动作 + community_ess_power）
            community_ess_power: 与社区储能的交互功率
                            正值：从社区储能充电
                            负值：向社区储能放电

        Returns:
            tuple: (next_state, reward, done, info)
        """
        # 提取原有动作（不包括community_ess_power）
        original_action = {k: v for k, v in action.items() if k != 'community_ess_power'}

        # 执行环境步骤
        next_state, reward, done = self.env.step(self.env.state, original_action)

        # 计算净负荷和PV流向（与电网的交互）
        net_load, pv_flow, grid_only_net_load = self._calculate_net_load_with_pv_flow(
            next_state, original_action, community_ess_power
        )

        # 更新平均净负荷
        self.net_load_history.append(net_load)
        if len(self.net_load_history) > 48:  # 保持最近48个时间步
            self.net_load_history.pop(0)
        self.avg_net_load = np.mean(self.net_load_history) if len(self.net_load_history) > 0 else net_load

        # 记录PV流向
        self.pv_flow_history.append(pv_flow)
        if len(self.pv_flow_history) > 48:  # 保持最近48个时间步
            self.pv_flow_history.pop(0)

        info = {
            'net_load': net_load,  # 家庭净负荷（包含社区储能交互）
            'grid_only_net_load': grid_only_net_load,  # 家庭与电网的直接交互（不包括社区储能）
            'avg_net_load': self.avg_net_load,
            'agent_id': self.agent_id,
            'pv_flow': pv_flow  # 添加PV流向信息
        }

        return next_state, reward, done, info
    
    def step(self, action, community_ess_power=0.0):
        """
        执行一步
        
        Args:
            action: 动作字典（包含原有动作 + community_ess_power）
            community_ess_power: 与社区储能的交互功率
                            正值：从社区储能充电
                            负值：向社区储能放电
        
        Returns:
            tuple: (next_state, reward, done, info)
        """
        # 提取原有动作（不包括community_ess_power）
        original_action = {k: v for k, v in action.items() if k != 'community_ess_power'}
        
        # 执行环境步骤
        next_state, reward, done = self.env.step(self.env.state, original_action)
        
        # 计算净负荷和PV流向（与电网的交互）
        # 使用 env 在 step 后记录的实际执行功率（current_battery_power / current_ev_power），
        # 与 env 内部能量平衡一致，避免 action 与真实执行功率不一致导致 PV 流向仅剩自用与售电
        net_load, pv_flow, grid_only_net_load = self._calculate_net_load_with_pv_flow(
            next_state, original_action, community_ess_power
        )
        
        # 更新平均净负荷
        self.net_load_history.append(net_load)
        if len(self.net_load_history) > 48:  # 保持最近48个时间步
            self.net_load_history.pop(0)
        self.avg_net_load = np.mean(self.net_load_history) if len(self.net_load_history) > 0 else net_load
        
        # 记录PV流向
        self.pv_flow_history.append(pv_flow)
        if len(self.pv_flow_history) > 48:  # 保持最近48个时间步
            self.pv_flow_history.pop(0)
        
        info = {
            'net_load': net_load,  # 家庭净负荷（包含社区储能交互）
            'grid_only_net_load': grid_only_net_load,  # 家庭与电网的直接交互（不包括社区储能）
            'avg_net_load': self.avg_net_load,
            'agent_id': self.agent_id,
            'pv_flow': pv_flow  # 添加PV流向信息
        }
        
        return next_state, reward, done, info
    
    def _calculate_net_load_with_pv_flow(self, state, action, community_ess_power):
        """
        计算与电网的净负荷，并追踪PV流向
        
        Args:
            state: 当前状态
            action: 动作
            community_ess_power: 与社区储能的交互功率
        
        Returns:
            tuple: (net_load, pv_flow_dict)
            - net_load: 净负荷（kW），正值表示从电网购电，负值表示向电网售电
            - pv_flow_dict: PV流向字典，包含各个方向的PV使用量
        """
        # 获取PV发电量
        pv_gen = state['pv_generation']

        # 使用 env 在 step 后记录的实际执行功率（与 env 内部能量平衡一致），
        # 避免与 action 不同步导致 PV 流向中 ess_charge/ev_charge 恒为 0
        ev_charge = max(getattr(self.env, 'current_ev_power', 0), 0)
        ev_discharge = max(-getattr(self.env, 'current_ev_power', 0), 0)
        battery_charge = max(getattr(self.env, 'current_battery_power', 0), 0)
        battery_discharge = max(-getattr(self.env, 'current_battery_power', 0), 0)
        
        # 计算家庭直接负荷（不包括储能和EV）
        household_load = (
            state['home_load']
            + state['Air_conditioner_power']
            + state['Air_conditioner_power2']
            + state.get('wash_machine_state', 0) * self.env.wash_machine_power
            + state.get('ewh_power', 0)
        )
        
        # PV流向分配（按优先级）
        pv_flow = {
            'direct_use': 0.0,          # 直接用于家庭负荷
            'ess_charge': 0.0,          # 充入家庭储能
            'ev_charge': 0.0,           # 充入EV
            'community_charge': 0.0,    # 充入社区储能（负值community_ess_power表示充电）
            'grid_sell': 0.0            # 售给电网
        }
        
        pv_remaining = pv_gen
        
        # 1. 家庭直接用电（最高优先级）
        pv_flow['direct_use'] = min(pv_remaining, household_load)
        pv_remaining = max(0.0, pv_remaining - pv_flow['direct_use'])
        
        # 2. ESS充电（第二优先级）
        if pv_remaining > 0 and battery_charge > 0:
            pv_flow['ess_charge'] = min(pv_remaining, battery_charge)
            pv_remaining = max(0.0, pv_remaining - pv_flow['ess_charge'])
        
        # 3. EV充电（第三优先级）
        if pv_remaining > 0 and ev_charge > 0:
            pv_flow['ev_charge'] = min(pv_remaining, ev_charge)
            pv_remaining = max(0.0, pv_remaining - pv_flow['ev_charge'])
        
        # 4. 社区储能充电（第四优先级）
        community_charge_power = abs(community_ess_power) if community_ess_power < 0 else 0.0
        if pv_remaining > 0 and community_charge_power > 0:
            pv_flow['community_charge'] = min(pv_remaining, community_charge_power)
            pv_remaining = max(0.0, pv_remaining - pv_flow['community_charge'])
        
        # 5. 售给电网（剩余部分）
        # 注意：这里计算的是从PV剩余部分售给电网，实际售电量还要考虑储能和EV的放电
        pv_flow['grid_sell'] = pv_remaining
        
        # 计算实际从电网充电的部分（未使用PV的部分）
        grid_ess_charge = max(0.0, battery_charge - pv_flow['ess_charge'])
        grid_ev_charge = max(0.0, ev_charge - pv_flow['ev_charge'])
        
        # 计算向社区储能充电时，未使用PV的部分（这部分必须从电网购电）
        community_charge_power = abs(community_ess_power) if community_ess_power < 0 else 0.0
        grid_community_charge = max(0.0, community_charge_power - pv_flow['community_charge'])
        
        # 计算从社区储能放电的部分（这部分会减少从电网的购电需求）
        community_discharge_power = community_ess_power if community_ess_power > 0 else 0.0
        
        # 计算总需求（只包括与电网交互的部分）
        # 注意：向社区储能充电时，如果不是全部来自PV，剩余部分是从电网购电，应该计入净负荷
        total_consumption = (
            household_load
            - pv_flow['direct_use']  # 减去PV直接用电部分（已在总需求中）
            + grid_ess_charge        # 从电网充电ESS（个人储能）
            + grid_ev_charge         # 从电网充电EV（个人EV）
            + grid_community_charge  # 从电网购电用于向社区储能充电（未使用PV的部分）
        )
        
        # 计算总发电（只包括与电网交互的部分）
        # 注意：从社区储能放电会减少从电网的购电需求，所以应该从净负荷中扣除
        total_generation = (
            pv_flow['grid_sell']     # PV售给电网
            + ev_discharge           # EV放电给电网
            + battery_discharge      # ESS放电给电网
            + community_discharge_power  # 从社区储能放电（减少从电网的购电需求）
        )
        
        # 净负荷 = 总需求 - 总发电
        # 注意：
        # 1. 对于单个家庭：净负荷包含与社区储能的交互（社区储能被视为外部能量来源）
        #    - 从社区储能充电（获取能量）：增加家庭的净负荷
        #    - 向社区储能充电（提供能量）：如果是从电网购电，增加净负荷；如果是用PV，不增加
        # 2. 对于整个社区：社区总净负荷应该排除社区储能的交互，只计算与电网的直接交互
        #    这需要在multi_agent_env.py中单独计算
        net_load = total_consumption - total_generation
        
        # 计算家庭与电网的直接交互（不包括社区储能）
        # 用于计算社区总净负荷
        grid_only_consumption = (
            household_load
            - pv_flow['direct_use']
            + grid_ess_charge
            + grid_ev_charge
            # 不包含 grid_community_charge（社区储能充电不是与电网的直接交互）
        )
        grid_only_generation = (
            pv_flow['grid_sell']
            + ev_discharge
            + battery_discharge
            # 不包含 community_discharge_power（社区储能放电不是与电网的直接交互）
        )
        grid_only_net_load = grid_only_consumption - grid_only_generation
        
        return net_load, pv_flow, grid_only_net_load
    
    def _calculate_net_load(self, state, action, community_ess_power):
        """
        计算与电网的净负荷（保持向后兼容）
        
        Args:
            state: 当前状态
            action: 动作
            community_ess_power: 与社区储能的交互功率
        
        Returns:
            净负荷（kW），正值表示从电网购电，负值表示向电网售电
        """
        net_load, _, _ = self._calculate_net_load_with_pv_flow(state, action, community_ess_power)
        return net_load
    
    def get_state(self):
        """获取当前状态"""
        return self.env.state
    
    def get_net_load(self):
        """获取当前净负荷"""
        if len(self.net_load_history) > 0:
            return self.net_load_history[-1]
        return 0.0
    
    def get_avg_net_load(self):
        """获取平均净负荷"""
        return self.avg_net_load
    
    def get_pv_flow(self):
        """获取当前时间步的PV流向"""
        if len(self.pv_flow_history) > 0:
            return self.pv_flow_history[-1]
        return {
            'direct_use': 0.0,
            'ess_charge': 0.0,
            'ev_charge': 0.0,
            'community_charge': 0.0,
            'grid_sell': 0.0
        }
    
    def get_pv_flow_history(self):
        """获取PV流向历史"""
        return self.pv_flow_history
    
    @property
    def state_space(self):
        """状态空间"""
        return self.env.state_space
    
    @property
    def action_space(self):
        """动作空间（扩展，包含community_ess_power）"""
        action_space = self.env.action_space.copy()
        # 添加社区储能交互动作
        action_space['community_ess_power'] = (-5.0, -2.5, 0, 2.5, 5.0)
        return action_space
