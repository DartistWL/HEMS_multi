# -*- coding: utf-8 -*-
# @Time        : 2026/3/28 13:53
# @Author      : Valiantir
# @File        : test.py
# @Version     : 1.0   
# Description  :

import os
project_root = r"D:\Documents\Git_Program\HEMS"
old_data_path = os.path.join(project_root, "data", "daily_pivot_cons_2011-2012.csv")
new_data_dir = os.path.join(project_root, "data", "storenet_ireland_2020")
print(f"旧数据文件: {old_data_path}, 存在: {os.path.exists(old_data_path)}")
print(f"新数据目录: {new_data_dir}, 存在: {os.path.exists(new_data_dir)}")
if os.path.exists(new_data_dir):
    print(f"新数据文件示例: {os.listdir(new_data_dir)[:5]}")