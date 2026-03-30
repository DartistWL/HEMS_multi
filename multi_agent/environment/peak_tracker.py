"""
Peak Load Tracker
峰值负荷追踪器，用于计算峰值惩罚
"""
import numpy as np
from collections import deque


class PeakTracker:
    """
    峰值负荷追踪器
    追踪社区净负荷，计算峰值惩罚
    """
    
    def __init__(self, window_size=48, threshold_factor=1.2, baseline_peak=50.0, penalty_exponent=2.0):
        """
        初始化峰值追踪器
        
        Args:
            window_size: 滚动窗口大小（时间步数），默认48（24小时）
            threshold_factor: 阈值系数，阈值 = 平均值 * threshold_factor
            baseline_peak: 基准峰值（无协调时的历史峰值），用于归一化
            penalty_exponent: 峰值惩罚的指数（1.0=线性，2.0=二次，1.5=介于两者之间）
        """
        self.window_size = window_size
        self.threshold_factor = threshold_factor
        self.baseline_peak = baseline_peak
        self.penalty_exponent = penalty_exponent
        
        # 负荷历史
        self.load_history = deque(maxlen=window_size)
        self.avg_load = 0.0
        self.max_load = 0.0
        
        # 统计信息
        self.peak_count = 0  # 超过阈值的次数
        self.total_peak_penalty = 0.0
        
    def update(self, current_load):
        """
        更新负荷历史
        
        Args:
            current_load: 当前社区净负荷（kW）
        """
        self.load_history.append(current_load)
        
        # 更新统计信息
        if len(self.load_history) > 0:
            self.avg_load = np.mean(self.load_history)
            self.max_load = max(self.load_history)
        
        # 检查是否超过阈值
        threshold = self.avg_load * self.threshold_factor
        if current_load > threshold:
            self.peak_count += 1
    
    def calculate_peak_penalty(self, current_load, use_normalization=False):
        """
        计算峰值惩罚（可选择是否归一化）
        
        Args:
            current_load: 当前社区净负荷（kW）
            use_normalization: 是否使用归一化（默认False，取消归一化让信号更明确）
        
        Returns:
            tuple: (惩罚值, 阈值, 是否超过阈值)
        """
        if len(self.load_history) < 2:
            # 历史数据不足，返回0
            return 0.0, 0.0, False
        
        threshold = self.avg_load * self.threshold_factor
        
        if current_load > threshold:
            excess_load = current_load - threshold
            # 原始惩罚（根据指数计算）
            raw_penalty = excess_load ** self.penalty_exponent
            
            if use_normalization:
                # 归一化：除以基准峰值的对应次方，使其与个人奖励在同一数量级
                # 假设个人奖励在-10到10之间，基准峰值约为50kW
                penalty = raw_penalty / (self.baseline_peak ** self.penalty_exponent)
            else:
                # 不归一化：直接使用原始惩罚，让信号更明确
                # 使用一个缩放因子，使惩罚的量级合理（除以baseline_peak，而不是baseline_peak^exponent）
                # 这样惩罚会更大，但不会过大
                penalty = raw_penalty / self.baseline_peak
            
            self.total_peak_penalty += penalty
            
            return penalty, threshold, True
        else:
            return 0.0, threshold, False
    
    def calculate_agent_contribution(self, agent_net_load, agent_avg_load):
        """
        计算智能体对峰值的贡献
        
        Args:
            agent_net_load: 智能体当前净负荷
            agent_avg_load: 智能体平均净负荷
        
        Returns:
            贡献值（非负）
        """
        contribution = max(0, agent_net_load - agent_avg_load)
        return contribution
    
    def get_statistics(self):
        """
        获取统计信息
        
        Returns:
            dict: 统计信息
        """
        return {
            'avg_load': self.avg_load,
            'max_load': self.max_load,
            'threshold': self.avg_load * self.threshold_factor if len(self.load_history) > 0 else 0.0,
            'peak_count': self.peak_count,
            'total_penalty': self.total_peak_penalty,
            'history_length': len(self.load_history)
        }
    
    def reset(self):
        """
        重置追踪器
        """
        self.load_history.clear()
        self.avg_load = 0.0
        self.max_load = 0.0
        self.peak_count = 0
        self.total_peak_penalty = 0.0
