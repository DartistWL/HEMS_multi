"""
计算和对比三种方法的电力成本
Calculate and compare electricity costs for three methods
"""
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def load_data_from_json(data_file):
    """从JSON文件加载数据"""
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"数据文件不存在: {data_file}")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def calculate_costs_from_data(data_file):
    """
    从数据文件中计算成本
    
    Args:
        data_file: 数据文件路径
    
    Returns:
        dict: 包含成本信息的字典，包括每个家庭的成本和积分余额
    """
    data = load_data_from_json(data_file)
    return calculate_costs_from_data_dict(data)


def calculate_costs_from_data_dict(data):
    """
    从数据字典中计算成本（支持从文件加载的数据或comparison_data）
    
    Args:
        data: 数据字典（包含summary和episodes）
    
    Returns:
        dict: 包含成本信息的字典，包括每个家庭的成本和积分余额
    """
    # 获取summary中的total_cost
    summary = data.get('summary', {})
    total_cost = summary.get('total_cost', 0.0)
    
    # 从episodes中计算总成本和每个家庭的成本；若有 date_type，则同时统计工作日/双休日
    episodes = data.get('episodes', [])
    weekday_eps = [e for e in episodes if e.get('date_type') == '工作日']
    weekend_eps = [e for e in episodes if e.get('date_type') == '双休日']
    has_date_type = bool(weekday_eps or weekend_eps)
    
    if episodes:
        episode_costs = [ep.get('total_cost', 0.0) for ep in episodes]
        avg_cost = np.mean(episode_costs) if episode_costs else 0.0
        std_cost = np.std(episode_costs) if len(episode_costs) > 1 else 0.0
        min_cost = np.min(episode_costs) if episode_costs else 0.0
        max_cost = np.max(episode_costs) if episode_costs else 0.0
        
        # 计算每个家庭的成本（检查是否有数据）
        agent_costs = [[], [], []]  # 每个家庭在所有episode中的成本列表
        has_agent_costs = False
        
        # 收集每个家庭的积分余额（初始和最终）
        agent_initial_credits = [[], [], []]  # 每个家庭的初始积分余额（所有episode的列表）
        agent_final_credits = [[], [], []]  # 每个家庭的最终积分余额（所有episode的列表）
        has_credit_data = False
        
        for ep in episodes:
            if 'agent_costs' in ep and ep['agent_costs']:
                has_agent_costs = True
                ep_agent_costs = ep['agent_costs']
                for i in range(3):
                    if i < len(ep_agent_costs):
                        agent_costs[i].append(ep_agent_costs[i])
            
            # 收集积分余额数据
            if 'agent_credit_balances' in ep and ep['agent_credit_balances']:
                credit_balances = ep['agent_credit_balances']
                # 检查是否为空列表（[[], [], []]）或包含有效数据
                if len(credit_balances) == 3 and any(len(bal) > 0 for bal in credit_balances):
                    has_credit_data = True
                    # 每个家庭的积分余额是一个时间序列，取第一个和最后一个
                    for i in range(3):
                        if i < len(credit_balances) and credit_balances[i] and len(credit_balances[i]) > 0:
                            agent_initial_credits[i].append(credit_balances[i][0])
                            agent_final_credits[i].append(credit_balances[i][-1])
        
        # 计算每个家庭的平均成本、标准差等（仅当有数据时）
        if has_agent_costs:
            agent_avg_costs = [np.mean(costs) if costs else 0.0 for costs in agent_costs]
            agent_std_costs = [np.std(costs) if len(costs) > 1 else 0.0 for costs in agent_costs]
            agent_min_costs = [np.min(costs) if costs else 0.0 for costs in agent_costs]
            agent_max_costs = [np.max(costs) if costs else 0.0 for costs in agent_costs]
        else:
            # 如果没有分别记录各家庭成本，则不提供这些数据
            agent_avg_costs = None
            agent_std_costs = None
            agent_min_costs = None
            agent_max_costs = None
        
        # 计算每个家庭的平均积分余额（初始和最终）
        if has_credit_data and agent_initial_credits and agent_final_credits:
            # 检查是否所有家庭都有数据
            if all(len(credits) > 0 for credits in agent_initial_credits) and all(len(credits) > 0 for credits in agent_final_credits):
                agent_avg_initial_credits = [np.mean(credits) if credits else 0.0 for credits in agent_initial_credits]
                agent_avg_final_credits = [np.mean(credits) if credits else 0.0 for credits in agent_final_credits]
                agent_credit_changes = [final - initial for initial, final in zip(agent_avg_initial_credits, agent_avg_final_credits)]
            else:
                # 如果某些家庭没有数据，设置为None
                agent_avg_initial_credits = None
                agent_avg_final_credits = None
                agent_credit_changes = None
                has_credit_data = False  # 重置标志
        else:
            agent_avg_initial_credits = None
            agent_avg_final_credits = None
            agent_credit_changes = None
        
        # 若有 date_type，分别计算工作日与双休日的成本统计
        if has_date_type:
            _hours = 24.0
            def _cost_stats(eps):
                if not eps:
                    return None
                costs_list = [e.get('total_cost', 0.0) for e in eps]
                return {
                    'avg_cost': np.mean(costs_list),
                    'std_cost': np.std(costs_list) if len(costs_list) > 1 else 0.0,
                    'min_cost': np.min(costs_list),
                    'max_cost': np.max(costs_list),
                    'episode_count': len(eps),
                    'avg_cost_per_hour': np.mean(costs_list) / _hours,
                }
            result_weekday = _cost_stats(weekday_eps)
            result_weekend = _cost_stats(weekend_eps)
        else:
            result_weekday = None
            result_weekend = None
    else:
        avg_cost = total_cost
        std_cost = 0.0
        min_cost = total_cost
        max_cost = total_cost
        # 如果没有episode数据，也没有各家庭成本数据
        agent_avg_costs = None
        agent_std_costs = None
        agent_min_costs = None
        agent_max_costs = None
        has_agent_costs = False
        result_weekday = None
        result_weekend = None
    
    # 每个episode是24小时（48个时间步 * 0.5小时）
    hours_per_episode = 24.0
    
    result = {
        'avg_cost': avg_cost,
        'avg_cost_per_hour': avg_cost / hours_per_episode,  # 平均每小时成本
        'total_cost': total_cost,
        'std_cost': std_cost,
        'min_cost': min_cost,
        'max_cost': max_cost,
        'episode_costs': episode_costs if episodes else [total_cost],
        'has_agent_costs': has_agent_costs,  # 标记是否有各家庭成本数据
        'has_credit_data': has_credit_data if episodes else False  # 标记是否有积分余额数据
    }
    if result_weekday is not None:
        result['weekday'] = result_weekday
    if result_weekend is not None:
        result['weekend'] = result_weekend
    
    # 只有当有各家庭成本数据时才添加这些字段
    if has_agent_costs and agent_avg_costs is not None:
        result['agent_avg_costs'] = agent_avg_costs  # 每个家庭的平均成本
        result['agent_avg_costs_per_hour'] = [cost / hours_per_episode for cost in agent_avg_costs]  # 每个家庭平均每小时成本
        result['agent_std_costs'] = agent_std_costs
        result['agent_min_costs'] = agent_min_costs
        result['agent_max_costs'] = agent_max_costs
        result['agent_costs'] = agent_costs  # 每个家庭在所有episode中的成本列表
    
    # 只有当有积分余额数据时才添加这些字段
    if has_credit_data and agent_avg_initial_credits is not None and agent_avg_final_credits is not None:
        result['agent_avg_initial_credits'] = agent_avg_initial_credits  # 每个家庭的平均初始积分
        result['agent_avg_final_credits'] = agent_avg_final_credits  # 每个家庭的平均最终积分
        result['agent_credit_changes'] = agent_credit_changes  # 每个家庭的积分变化
    
    return result


def compare_costs(independent_data_file=None, rule_based_data_file=None, 
                  mappo_data_file=None, comparison_data_file=None,
                  save_path=None, show_plot=True):
    """
    对比三种方法的电力成本
    
    Args:
        independent_data_file: 独立训练数据文件
        rule_based_data_file: 规则基线数据文件
        mappo_data_file: MAPPO数据文件
        comparison_data_file: 对比数据文件（如果存在，优先使用）
        save_path: 保存路径
        show_plot: 是否显示图表
    """
    costs = {}
    
    # 如果提供了comparison_data文件，优先使用
    if comparison_data_file and os.path.exists(comparison_data_file):
        comparison_data = load_data_from_json(comparison_data_file)
        
        # 只有在comparison_data包含episode且带agent_costs时才使用，否则忽略，回退到单独数据文件
        use_comparison = False
        for key in ['independent', 'rule_based', 'mappo']:
            method_data = comparison_data.get(key)
            if not method_data:
                continue
            episodes = method_data.get('episodes', [])
            if episodes and isinstance(episodes, list):
                first_ep = episodes[0]
                if isinstance(first_ep, dict) and 'agent_costs' in first_ep:
                    use_comparison = True
                    break
        
        if use_comparison:
            # 从comparison_data中提取（使用calculate_costs_from_data_dict来获取完整信息，包括积分余额）
            for method_key in ['independent', 'rule_based', 'mappo']:
                method_data = comparison_data.get(method_key)
                if method_data:
                    # 构造临时数据结构，模拟单独的数据文件格式
                    temp_data = {
                        'summary': method_data.get('summary', {}),
                        'episodes': method_data.get('episodes', [])
                    }
                    # 使用calculate_costs_from_data_dict来获取完整信息（包括积分余额）
                    costs[method_key] = calculate_costs_from_data_dict(temp_data)
    
    # 从单独的数据文件加载（如果comparison_data中没有或需要更详细的信息）
    if independent_data_file and os.path.exists(independent_data_file):
        if 'independent' not in costs:
            costs['independent'] = calculate_costs_from_data(independent_data_file)
    
    if rule_based_data_file and os.path.exists(rule_based_data_file):
        if 'rule_based' not in costs:
            costs['rule_based'] = calculate_costs_from_data(rule_based_data_file)
    
    if mappo_data_file and os.path.exists(mappo_data_file):
        if 'mappo' not in costs:
            costs['mappo'] = calculate_costs_from_data(mappo_data_file)
    
    if not costs:
        raise ValueError("没有找到任何成本数据")
    
    # 打印结果
    print("=" * 80)
    print("三种方法的电力成本对比")
    print("=" * 80)
    
    method_names = {
        'independent': '独立训练基线',
        'rule_based': '固定规则基线',
        'mappo': 'MAPPO算法'
    }
    
    results = []
    for method in ['independent', 'rule_based', 'mappo']:
        if method in costs:
            cost_info = costs[method]
            print(f"\n{method_names[method]}:")
            print(f"  总平均成本: {cost_info['avg_cost']:.4f} 元/天")
            print(f"  平均每小时成本: {cost_info.get('avg_cost_per_hour', cost_info['avg_cost'] / 24.0):.4f} 元/小时")
            print(f"  标准差: {cost_info['std_cost']:.4f} 元")
            print(f"  最小成本: {cost_info['min_cost']:.4f} 元")
            print(f"  最大成本: {cost_info['max_cost']:.4f} 元")
            
            # 若有工作日/双休日分开统计，分别打印
            if 'weekday' in cost_info and cost_info['weekday']:
                wd = cost_info['weekday']
                print(f"  【工作日】平均成本: {wd['avg_cost']:.4f} 元/天 ({wd['episode_count']} 天), 每小时: {wd['avg_cost_per_hour']:.4f} 元")
            if 'weekend' in cost_info and cost_info['weekend']:
                we = cost_info['weekend']
                print(f"  【双休日】平均成本: {we['avg_cost']:.4f} 元/天 ({we['episode_count']} 天), 每小时: {we['avg_cost_per_hour']:.4f} 元")
            
            # 打印每个家庭的成本（仅当有数据时）
            if cost_info.get('has_agent_costs', False) and 'agent_avg_costs_per_hour' in cost_info:
                print(f"\n  各家庭平均每小时成本:")
                agent_hourly_costs = cost_info['agent_avg_costs_per_hour']
                max_cost = max(agent_hourly_costs) if agent_hourly_costs else 0.0
                min_cost = min(agent_hourly_costs) if agent_hourly_costs else 0.0
                
                for i in range(3):
                    agent_id = i + 1
                    avg_hourly = cost_info['agent_avg_costs_per_hour'][i]
                    avg_daily = cost_info['agent_avg_costs'][i]
                    std = cost_info['agent_std_costs'][i] if i < len(cost_info['agent_std_costs']) else 0.0
                    print(f"    家庭 {agent_id}: {avg_hourly:.4f} 元/小时 (平均 {avg_daily:.4f} 元/天, 标准差 {std:.4f})")
                
                # 检查成本合理性（仅对独立训练基线）
                if method == 'independent' and max_cost > 0:
                    cost_ratio = max_cost / min_cost if min_cost > 0 else float('inf')
                    if cost_ratio > 10:  # 如果最大成本是最小成本的10倍以上
                        print(f"\n  ⚠️  警告: 独立训练基线中，各家庭成本差异过大（最大/最小 = {cost_ratio:.2f}倍）")
                        print(f"     这可能表明训练结果不稳定或数据使用存在问题。")
                    if min_cost < 0:
                        print(f"\n  ⚠️  警告: 某些家庭的成本为负值，说明售电收益大于购电成本。")
                        print(f"     这在独立训练中可能不合理，因为所有家庭使用相同的数据。")
            elif not cost_info.get('has_agent_costs', False):
                print(f"\n  注意: 数据中未包含各家庭分别的成本信息，无法显示各家庭成本。")
            
            # 打印每个家庭的积分余额（仅当有数据时，通常只有MAPPO有）
            if cost_info.get('has_credit_data', False) and 'agent_avg_initial_credits' in cost_info:
                print(f"\n  各家庭积分余额:")
                for i in range(3):
                    agent_id = i + 1
                    initial = cost_info['agent_avg_initial_credits'][i]
                    final = cost_info['agent_avg_final_credits'][i]
                    change = cost_info['agent_credit_changes'][i]
                    change_str = f"+{change:.2f}" if change >= 0 else f"{change:.2f}"
                    print(f"    家庭 {agent_id}: 初始 {initial:.2f} → 最终 {final:.2f} (变化: {change_str})")
            
            result_dict = {
                'method': method_names[method],
                'avg_cost': cost_info['avg_cost'],
                'avg_cost_per_hour': cost_info.get('avg_cost_per_hour', cost_info['avg_cost'] / 24.0),
                'std_cost': cost_info['std_cost'],
                'has_agent_costs': cost_info.get('has_agent_costs', False),
                'has_credit_data': cost_info.get('has_credit_data', False)  # 添加积分数据标志
            }
            if 'weekday' in cost_info and cost_info['weekday']:
                result_dict['weekday'] = cost_info['weekday']
            if 'weekend' in cost_info and cost_info['weekend']:
                result_dict['weekend'] = cost_info['weekend']
            
            # 只有当有各家庭成本数据时才添加这些字段
            if result_dict['has_agent_costs']:
                result_dict['agent_avg_costs_per_hour'] = cost_info.get('agent_avg_costs_per_hour', [])
                result_dict['agent_avg_costs'] = cost_info.get('agent_avg_costs', [])
            
            # 只有当有积分余额数据时才添加这些字段
            if result_dict['has_credit_data'] and 'agent_avg_final_credits' in cost_info:
                result_dict['agent_avg_initial_credits'] = cost_info.get('agent_avg_initial_credits', [])
                result_dict['agent_avg_final_credits'] = cost_info.get('agent_avg_final_credits', [])
                result_dict['agent_credit_changes'] = cost_info.get('agent_credit_changes', [])
            
            results.append(result_dict)
    
    # 找出成本最低的方法（按每小时成本）
    if results:
        best_method = min(results, key=lambda x: x['avg_cost_per_hour'])
        print(f"\n{'='*80}")
        print(f"每小时成本最低的方法: {best_method['method']} ({best_method['avg_cost_per_hour']:.4f} 元/小时)")
        print(f"{'='*80}")
    
    # 绘制对比图（包括总成本和每个家庭的成本）
    if len(results) > 0:
        fig = plt.figure(figsize=(18, 6))
        ax1 = plt.subplot(1, 3, 1)  # 总平均每小时成本
        ax2 = plt.subplot(1, 3, 2)  # 归一化对比
        ax3 = plt.subplot(1, 3, 3)  # 每个家庭的平均每小时成本
        
        # 左图：平均每小时成本柱状图
        methods = [r['method'] for r in results]
        avg_costs_per_hour = [r['avg_cost_per_hour'] for r in results]
        std_costs_per_hour = [r['std_cost'] / 24.0 for r in results]  # 标准差也转换为每小时
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        bars = ax1.bar(methods, avg_costs_per_hour, yerr=std_costs_per_hour, capsize=5, 
                      color=colors[:len(methods)], alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax1.set_xlabel('方法', fontsize=12)
        ax1.set_ylabel('电力成本 (元/小时)', fontsize=12)
        ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # 添加数值标签
        for i, (bar, cost) in enumerate(zip(bars, avg_costs_per_hour)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + std_costs_per_hour[i] + max(avg_costs_per_hour) * 0.02,
                    f'{cost:.2f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 标注最低成本
        min_cost_idx = np.argmin(avg_costs_per_hour)
        ax1.text(bars[min_cost_idx].get_x() + bars[min_cost_idx].get_width()/2., 
                avg_costs_per_hour[min_cost_idx] - max(avg_costs_per_hour) * 0.08,
                '最低', ha='center', va='top', fontsize=10, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                fontweight='bold')
        
        # 中间图：三个家庭的积分余额（仅当有积分余额数据时）
        # 调试：打印所有results的信息
        # print(f"Debug: Number of results: {len(results)}")
        # for i, r in enumerate(results):
        #     print(f"  Result {i}: method={r.get('method')}, has_credit_data={r.get('has_credit_data', False)}, has_agent_avg_final_credits={'agent_avg_final_credits' in r}")
        
        results_with_credit_data = [r for r in results if r.get('has_credit_data', False) and 'agent_avg_final_credits' in r]
        if results_with_credit_data:
            # 优先使用MAPPO的积分数据（因为只有MAPPO有真实的积分交易）
            credit_result = None
            for r in results_with_credit_data:
                if r.get('method', '') == 'MAPPO算法':
                    credit_result = r
                    break
            # 如果没找到MAPPO，使用第一个有积分数据的方法
            if credit_result is None:
                credit_result = results_with_credit_data[0]
            credit_balances = credit_result['agent_avg_final_credits']
            
            # 验证积分余额数据：检查是否三个家庭的值都相同（这通常表示数据有问题，如固定规则基线）
            if len(credit_balances) == 3:
                if credit_balances[0] == credit_balances[1] == credit_balances[2]:
                    # 如果三个值都相同，强制使用MAPPO的数据
                    mappo_result = None
                    for r in results:
                        if r.get('method', '') == 'MAPPO算法' and r.get('has_credit_data', False) and 'agent_avg_final_credits' in r:
                            mappo_result = r
                            break
                    if mappo_result:
                        credit_balances = mappo_result['agent_avg_final_credits']
                        credit_result = mappo_result
                    else:
                        # 如果MAPPO没有积分数据，显示提示信息
                        print(f"\n[警告] 检测到积分余额数据异常（三个家庭值相同），但MAPPO没有积分数据，将使用当前数据")
            
            x = np.arange(3)  # 3个家庭
            bars2 = ax2.bar(x, credit_balances, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], 
                          alpha=0.7, edgecolor='black', linewidth=1.5)
            
            ax2.set_xlabel('家庭编号', fontsize=12)
            ax2.set_ylabel('积分余额', fontsize=12)
            ax2.set_xticks(x)
            ax2.set_xticklabels(['家庭1', '家庭2', '家庭3'])
            ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
            
            # 添加数值标签
            max_credit = max(credit_balances) if credit_balances else 100.0
            for bar, balance in zip(bars2, credit_balances):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + max_credit * 0.02,
                        f'{balance:.2f}',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            # 绘制初始积分线（100）
            initial_credit = 100.0
            ax2.axhline(y=initial_credit, color='gray', linestyle='--', linewidth=1.5, 
                       alpha=0.7, label=f'初始积分 ({initial_credit:.0f})')
            ax2.legend(loc='best', fontsize=10)
        else:
            # 如果没有积分余额数据，显示提示信息
            ax2.text(0.5, 0.5, '数据中未包含积分余额信息\n（通常只有MAPPO有积分数据）', 
                    ha='center', va='center', fontsize=12, 
                    transform=ax2.transAxes, style='italic')
            ax2.set_ylabel('积分余额', fontsize=12)
        
        # 右图：每个家庭的平均每小时成本对比（仅当有数据时）
        results_with_agent_costs = [r for r in results if r.get('has_agent_costs', False) and 'agent_avg_costs_per_hour' in r]
        if results_with_agent_costs:
            x = np.arange(3)  # 3个家庭
            width = 0.25  # 柱状图宽度
            
            for i, result in enumerate(results_with_agent_costs):
                agent_costs = result['agent_avg_costs_per_hour']
                ax3.bar(x + i * width, agent_costs, width, 
                       label=result['method'], color=colors[i], alpha=0.7, edgecolor='black', linewidth=1.5)
            
            ax3.set_xlabel('家庭编号', fontsize=12)
            ax3.set_ylabel('平均每小时成本 (元/小时)', fontsize=12)
            ax3.set_xticks(x + width * (len(results_with_agent_costs) - 1) / 2)
            ax3.set_xticklabels(['家庭1', '家庭2', '家庭3'])
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
            
            # 添加数值标签
            max_cost = max([max(r['agent_avg_costs_per_hour']) for r in results_with_agent_costs])
            for i, result in enumerate(results_with_agent_costs):
                agent_costs = result['agent_avg_costs_per_hour']
                for j, cost in enumerate(agent_costs):
                    ax3.text(x[j] + i * width, cost + max_cost * 0.02,
                            f'{cost:.2f}', ha='center', va='bottom', fontsize=9)
        else:
            # 如果没有各家庭成本数据，显示提示信息
            ax3.text(0.5, 0.5, '数据中未包含各家庭\n分别的成本信息', 
                    ha='center', va='center', fontsize=12, 
                    transform=ax3.transAxes, style='italic')
        
        # 各子图标注小写字母 a/b/c：与 y 轴标签同一条竖线（左侧）、与子图标题同一水平高度，字号加大
        for ax, label in zip([ax1, ax2, ax3], ['a', 'b', 'c']):
            ax.text(-0.05, 1.02, label, transform=ax.transAxes, fontsize=20, fontweight='bold',
                    va='bottom', ha='left')
        
        plt.tight_layout()
        
        # 保存图片（高分辨率）
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
            print(f"\n成本对比图已保存到: {save_path}")
        
        # 显示图表
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        # 若所有方法均有工作日/双休日数据，额外绘制工作日 vs 双休日成本对比图
        results_with_weekday_weekend = [r for r in results if r.get('weekday') and r.get('weekend')]
        if len(results_with_weekday_weekend) == len(results) and len(results) > 0 and save_path:
            fig2, ax_wd = plt.subplots(figsize=(10, 5))
            methods = [r['method'] for r in results]
            x = np.arange(len(methods))
            width = 0.35
            wd_costs = [r['weekday']['avg_cost'] for r in results]
            we_costs = [r['weekend']['avg_cost'] for r in results]
            ax_wd.bar(x - width/2, wd_costs, width, label='工作日', color='#4ECDC4', alpha=0.8)
            ax_wd.bar(x + width/2, we_costs, width, label='双休日', color='#FF6B6B', alpha=0.8)
            ax_wd.set_ylabel('平均成本 (元/天)', fontsize=12)
            ax_wd.set_title('各方法工作日 vs 双休日 平均电力成本对比', fontsize=14, pad=15)
            ax_wd.set_xticks(x)
            ax_wd.set_xticklabels(methods)
            ax_wd.legend()
            ax_wd.grid(True, alpha=0.3, axis='y', linestyle='--')
            plt.tight_layout()
            save_path_weekday_weekend = os.path.join(os.path.dirname(save_path), 'cost_comparison_weekday_weekend.png')
            plt.savefig(save_path_weekday_weekend, dpi=300, bbox_inches='tight')
            print(f"工作日/双休日成本对比图已保存到: {save_path_weekday_weekend}")
            if show_plot:
                plt.show()
            else:
                plt.close()
    
    return costs


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='计算和对比三种方法的电力成本')
    parser.add_argument('--comparison_data', type=str, 
                       default='multi_agent/visualization_data/comparison_data.json',
                       help='对比数据文件路径')
    parser.add_argument('--independent_data', type=str, default=None,
                       help='独立训练数据文件路径')
    parser.add_argument('--rule_based_data', type=str, default=None,
                       help='规则基线数据文件路径')
    parser.add_argument('--mappo_data', type=str, default=None,
                       help='MAPPO数据文件路径')
    parser.add_argument('--output_dir', type=str, default='multi_agent/visualization/output',
                       help='输出目录')
    parser.add_argument('--show_plot', action='store_true',
                       help='显示图表')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 如果没有指定单独的数据文件，尝试使用默认路径
    visualization_data_dir = 'multi_agent/visualization_data'
    if not args.independent_data and os.path.exists(os.path.join(visualization_data_dir, 'independent_data.json')):
        args.independent_data = os.path.join(visualization_data_dir, 'independent_data.json')
    if not args.rule_based_data and os.path.exists(os.path.join(visualization_data_dir, 'rule_based_data.json')):
        args.rule_based_data = os.path.join(visualization_data_dir, 'rule_based_data.json')
    if not args.mappo_data and os.path.exists(os.path.join(visualization_data_dir, 'mappo_data.json')):
        args.mappo_data = os.path.join(visualization_data_dir, 'mappo_data.json')
    
    # 计算和对比成本
    output_path = os.path.join(args.output_dir, 'cost_comparison.png')
    costs = compare_costs(
        independent_data_file=args.independent_data,
        rule_based_data_file=args.rule_based_data,
        mappo_data_file=args.mappo_data,
        comparison_data_file=args.comparison_data if os.path.exists(args.comparison_data) else None,
        save_path=output_path,
        show_plot=args.show_plot
    )
    
    print("\n" + "=" * 80)
    print("完成！")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断操作。")
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()
