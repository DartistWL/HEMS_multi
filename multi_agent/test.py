# -*- coding: utf-8 -*-
# @Time        : 2026/3/28 13:53
# @Author      : Valiantir
# @File        : test.py
# @Version     : 1.0   
# Description  :

# import os
# project_root = r"D:\Documents\Git_Program\HEMS"
# old_data_path = os.path.join(project_root, "data", "daily_pivot_cons_2011-2012.csv")
# new_data_dir = os.path.join(project_root, "data", "storenet_ireland_2020")
# print(f"旧数据文件: {old_data_path}, 存在: {os.path.exists(old_data_path)}")
# print(f"新数据目录: {new_data_dir}, 存在: {os.path.exists(new_data_dir)}")
# if os.path.exists(new_data_dir):
#     print(f"新数据文件示例: {os.listdir(new_data_dir)[:5]}")

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multi_agent.environment.multi_agent_env import MultiAgentHEMEnv

try:
    env = MultiAgentHEMEnv(
        n_agents=3,
        community_ess_capacity=36.0,
        baseline_peak=18.0,
        community_weight=0.5,
        community_credit_cost_weight=0.05,
        community_credit_benefit_weight=0.05,
        initial_credit=100.0,
        peak_penalty_exponent=3.0,
        peak_discharge_bonus=0.45,
        peak_credit_cost_reduction=0.3,
        pv_coefficients=[2.0, 2.0, 2.0],
        train_house_ids=[f"H{i}" for i in range(1, 16)],
        test_house_ids=[f"H{i}" for i in range(16, 21)]
    )
    print("环境创建成功")
    state = env.reset(mode='train', date_index=0, house_index=0)
    print("reset 成功")
    print("状态类型:", type(state))
except Exception as e:
    print("错误:", e)
    import traceback
    traceback.print_exc()