"""
Data interface for multi-agent system with PV coefficient adjustment
支持通过系数调整光伏数据，模拟晴天/阴天情况
"""
import numpy as np
import pandas as pd
from datetime import datetime
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from interface import DataInterface as BaseDataInterface


class MultiAgentDataInterface(BaseDataInterface):
    """
    多智能体数据接口，继承自单智能体数据接口
    支持通过系数调整光伏数据，模拟不同天气情况
    """

    def __init__(self, cons_file, prod_file, price_profile="legacy", steps_per_day=48,
                 tou_hourly_values=None, pv_coefficient=1.0):
        """
        初始化数据接口

        Args:
            cons_file: 负荷数据文件路径
            prod_file: 光伏数据文件路径
            price_profile: 电价配置文件
            steps_per_day: 每天步数
            tou_hourly_values: 时变电价列表
            pv_coefficient: 光伏系数
        """
        # 调用父类的构造函数，传递所有必要的参数
        super().__init__(cons_file, prod_file, price_profile=price_profile,
                         steps_per_day=steps_per_day, tou_hourly_values=tou_hourly_values)
        self.pv_coefficient = pv_coefficient
        
    def get_pv_generation(self, current_date, current_time_index):
        """
        获取光伏发电量（应用系数调整）
        
        Args:
            current_date: 当前日期
            current_time_index: 当前时间索引
            
        Returns:
            调整后的光伏发电量
        """
        base_pv = super().get_pv_generation(current_date, current_time_index)
        adjusted_pv = base_pv * self.pv_coefficient
        return max(0, adjusted_pv)  # 确保非负
    
    def set_pv_coefficient(self, coefficient):
        """
        设置光伏系数
        
        Args:
            coefficient: 新的光伏系数
        """
        self.pv_coefficient = coefficient

    @classmethod
    def from_storenet_ireland_2020(
        cls,
        house_id: str,
        base_dir: str = "data/storenet_ireland_2020",
        price_profile: str = "legacy",
        steps_per_day: int = 48,
        tou_hourly_values = None,
        pv_coefficient: float = 1.0
    ):
        hid = str(house_id).strip().upper()
        # 修改这里：添加 'H' 前缀
        if not hid.startswith('H'):
            hid = 'H' + hid
        cons_file = f"{base_dir}/daily_pivot_cons_2020_{hid}.csv"
        prod_file = f"{base_dir}/daily_pivot_prod_2020_{hid}.csv"
        return cls(
            cons_file,
            prod_file,
            price_profile=price_profile,
            steps_per_day=steps_per_day,
            tou_hourly_values=tou_hourly_values,
            pv_coefficient=pv_coefficient
        )


    @staticmethod
    def get_weather_coefficient(weather_type):
        """
        获取不同天气类型对应的光伏系数
        
        Args:
            weather_type: 天气类型
                - 'sunny': 晴天（高光伏）
                - 'cloudy': 阴天（低光伏）
                - 'normal': 正常
                
        Returns:
            对应的光伏系数
        """
        coefficients = {
            'sunny': 6.0,      # 晴天：
            'cloudy': 2.0,     # 阴天：
            'normal': 4.0      # 正常
        }
        return coefficients.get(weather_type, 1.0)
