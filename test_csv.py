# -*- coding: utf-8 -*-
# @Time        : 2026/4/3 14:49
# @Author      : Valiantir
# @File        : test_csv.py
# @Version     : 1.0   
# Description  :

import pandas as pd
for hid in ['H16','H17','H18','H19','H20']:
    f = f"data/storenet_ireland_2020/daily_pivot_prod_2020_{hid}.csv"
    try:
        df = pd.read_csv(f, parse_dates=['date'], index_col='date')
        print(f"{hid}: OK, shape {df.shape}")
    except Exception as e:
        print(f"{hid}: ERROR {e}")