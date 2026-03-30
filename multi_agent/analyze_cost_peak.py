"""
分析成本与峰值的关系
"""
import json
import numpy as np

# 检查MAPPO的净负荷和成本关系
with open('multi_agent/visualization_data/mappo_data.json', 'r', encoding='utf-8') as f:
    mappo_data = json.load(f)

episodes = mappo_data.get('episodes', [])
if episodes:
    ep = episodes[0]
    # 检查峰值和成本
    peak_load = ep.get('peak_load', 0)
    total_cost = ep.get('total_cost', 0)
    agent_costs = ep.get('agent_costs', [0, 0, 0])
    community_loads = ep.get('community_net_loads', [])
    
    print('MAPPO Episode 0:')
    print(f'  峰值负荷: {peak_load:.2f} kW')
    print(f'  总成本: {total_cost:.2f} 元')
    print(f'  各家庭成本: {[f"{c:.2f}" for c in agent_costs]} 元')
    print(f'  平均净负荷: {np.mean(community_loads) if community_loads else 0:.2f} kW')
    print(f'  最大净负荷: {np.max(community_loads) if community_loads else 0:.2f} kW')
    
    # 检查社区储能使用情况
    comm_charge = ep.get('community_ess_charge_power', [])
    comm_discharge = ep.get('community_ess_discharge_power', [])
    print(f'  社区储能总充电: {sum(comm_charge) if comm_charge else 0:.2f} kWh')
    print(f'  社区储能总放电: {sum(comm_discharge) if comm_discharge else 0:.2f} kWh')
    
    # 检查各家庭的净负荷
    agent_loads = ep.get('agent_net_loads', [])
    print(f'  各家庭净负荷 (平均值): {[f"{np.mean(l) if isinstance(l, list) else l:.2f}" for l in agent_loads[:3]]} kW')

# 对比独立基线的峰值和成本
with open('multi_agent/visualization_data/independent_data.json', 'r', encoding='utf-8') as f:
    indep_data = json.load(f)

episodes = indep_data.get('episodes', [])
if episodes:
    ep = episodes[0]
    peak_load = ep.get('peak_load', 0)
    total_cost = ep.get('total_cost', 0)
    agent_costs = ep.get('agent_costs', [0, 0, 0])
    community_loads = ep.get('community_net_loads', [])
    
    print('\n独立基线 Episode 0:')
    print(f'  峰值负荷: {peak_load:.2f} kW')
    print(f'  总成本: {total_cost:.2f} 元')
    print(f'  各家庭成本: {[f"{c:.2f}" for c in agent_costs]} 元')
    print(f'  平均净负荷: {np.mean(community_loads) if community_loads else 0:.2f} kW')
    print(f'  最大净负荷: {np.max(community_loads) if community_loads else 0:.2f} kW')
    
    agent_loads = ep.get('agent_net_loads', [])
    print(f'  各家庭净负荷 (平均值): {[f"{np.mean(l) if isinstance(l, list) else l:.2f}" for l in agent_loads[:3]]} kW')

print('\n分析:')
print('如果MAPPO的峰值最高但成本最低，可能的原因是:')
print('1. MAPPO使用社区储能减少了从电网的直接购电（降低成本）')
print('2. 但社区储能的充放电导致了净负荷峰值的升高')
print('3. 成本计算基于与电网的直接交互，不包含社区储能交易')
