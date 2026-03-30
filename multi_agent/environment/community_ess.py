"""
Community Shared Energy Storage System
社区共享储能系统
"""
import numpy as np


class CommunityESS:
    """
    社区共享储能系统
    管理社区共享储能池的充放电、SOC等状态
    """
    
    def __init__(self, capacity=36.0, charge_efficiency=0.95, discharge_efficiency=0.95,
                 soc_min=0.1, soc_max=0.9, initial_soc=0.5):
        """
        初始化社区共享储能
        
        Args:
            capacity: 储能容量 (kWh)
            charge_efficiency: 充电效率
            discharge_efficiency: 放电效率
            soc_min: 最小SOC（安全下限）
            soc_max: 最大SOC（安全上限）
            initial_soc: 初始SOC（0-1之间）
        """
        self.capacity = capacity
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.soc_min = soc_min
        self.soc_max = soc_max
        
        # 当前状态
        self.soc = initial_soc  # 当前SOC（0-1之间）
        self.energy = self.soc * self.capacity  # 当前能量 (kWh)
        
        # 记录
        self.soc_history = []
        self.charge_history = []  # 总充电量
        self.discharge_history = []  # 总放电量
        
    def update(self, total_charge_power, total_discharge_power, dt=0.5):
        """
        更新储能状态，考虑物理约束限制
        
        Args:
            total_charge_power: 总充电功率（所有家庭向社区储能充电的总和，kW）
                              正值表示充电
            total_discharge_power: 总放电功率（所有家庭从社区储能放电的总和，kW）
                                 正值表示放电
            dt: 时间步长（小时），默认0.5小时（半小时）
        
        Returns:
            dict: 更新后的状态信息，包含实际功率和请求功率
        """
        # 1. 限制充电功率（基于SOC上限约束）
        if total_charge_power > 0:
            # 计算最多可以充入的能量（考虑效率）
            max_charge_energy = (self.soc_max * self.capacity - self.energy) / self.charge_efficiency
            max_charge_power = max_charge_energy / dt if dt > 0 else 0.0
            actual_charge_power = min(total_charge_power, max(0.0, max_charge_power))
        else:
            actual_charge_power = 0.0
        
        # 2. 限制放电功率（基于SOC下限约束）
        if total_discharge_power > 0:
            # 计算最多可以放出的能量（考虑效率）
            max_discharge_energy = (self.energy - self.soc_min * self.capacity) * self.discharge_efficiency
            max_discharge_power = max_discharge_energy / dt if dt > 0 else 0.0
            actual_discharge_power = min(total_discharge_power, max(0.0, max_discharge_power))
        else:
            actual_discharge_power = 0.0
        
        # 3. 计算净充电能量（允许同一时间步内同时充、放，如多端口/逆变器；净效果更新 SOC）
        net_charge_energy = (
            actual_charge_power * self.charge_efficiency * dt
            - actual_discharge_power / self.discharge_efficiency * dt
        )
        
        # 4. 更新能量
        new_energy = self.energy + net_charge_energy
        
        # 5. 双重保险：再次应用约束（防止数值误差）
        new_energy = np.clip(new_energy, 
                            self.soc_min * self.capacity, 
                            self.soc_max * self.capacity)
        
        # 6. 更新SOC和能量
        self.energy = new_energy
        self.soc = self.energy / self.capacity
        
        # 7. 记录实际功率（而不是请求功率）
        self.soc_history.append(self.soc)
        self.charge_history.append(actual_charge_power * dt)
        self.discharge_history.append(actual_discharge_power * dt)
        
        return {
            'soc': self.soc,
            'energy': self.energy,
            'charge_power': actual_charge_power,  # 实际可用功率
            'discharge_power': actual_discharge_power,  # 实际可用功率
            'requested_charge_power': total_charge_power,  # 请求功率（用于调试）
            'requested_discharge_power': total_discharge_power  # 请求功率（用于调试）
        }
    
    def get_state(self):
        """
        获取当前状态
        
        Returns:
            dict: 当前状态信息
        """
        return {
            'soc': self.soc,
            'energy': self.energy,
            'capacity': self.capacity,
            'soc_min': self.soc_min,
            'soc_max': self.soc_max
        }
    
    def reset(self, initial_soc=None):
        """
        重置储能状态
        
        Args:
            initial_soc: 初始SOC，如果为None则使用默认值
        """
        if initial_soc is not None:
            self.soc = np.clip(initial_soc, self.soc_min, self.soc_max)
        else:
            self.soc = 0.5  # 默认50%
        
        self.energy = self.soc * self.capacity
        self.soc_history = []
        self.charge_history = []
        self.discharge_history = []
    
    def check_constraints(self, charge_power, discharge_power):
        """
        检查充放电功率是否满足约束
        
        Args:
            charge_power: 充电功率
            discharge_power: 放电功率
            
        Returns:
            tuple: (是否满足约束, 违反信息)
        """
        violations = []
        
        # 检查是否同时充放电
        if charge_power > 0 and discharge_power > 0:
            violations.append("Cannot charge and discharge simultaneously")
        
        # 检查SOC上限（充电时）
        if charge_power > 0:
            max_charge = (self.soc_max * self.capacity - self.energy) / (self.charge_efficiency * 0.5)
            if charge_power > max_charge:
                violations.append(f"Charge power exceeds SOC limit: {charge_power:.2f} > {max_charge:.2f}")
        
        # 检查SOC下限（放电时）
        if discharge_power > 0:
            max_discharge = (self.energy - self.soc_min * self.capacity) * self.discharge_efficiency / 0.5
            if discharge_power > max_discharge:
                violations.append(f"Discharge power exceeds SOC limit: {discharge_power:.2f} > {max_discharge:.2f}")
        
        return len(violations) == 0, violations
