"""
光伏场景评估结果绘图
读取 run_pv_evaluation.py 生成的 CSV/JSON，绘制不同光伏（天气）场景下的峰值、成本等对比。
默认使用 JSON（pv_eval.json），以便默认绘制光伏流向图与行为图；若仅有 CSV 可指定 --data_file xxx.csv。

用法:
  python multi_agent/plot_pv_evaluation.py
  python multi_agent/plot_pv_evaluation.py --data_file multi_agent/pv_eval_results/pv_eval.json
  python multi_agent/plot_pv_evaluation.py --data_file multi_agent/pv_eval_results/pv_eval.csv
"""
import sys
import os
import json
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def load_pv_csv(csv_path):
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            row = {}
            for k, v in r.items():
                try:
                    row[k] = float(v)
                except (ValueError, TypeError):
                    row[k] = v
            if 'method' not in row:
                row['method'] = 'mappo'
            if 'date_type' not in row and 'date_type' in r:
                row['date_type'] = r['date_type']  # 工作日/双休日
            rows.append(row)
    return rows


def _row_from_metrics(scenario, pv_coef, method, w, date_type=None):
    """从单组指标 w 构造一行扁平数据；date_type 为 None 表示旧格式（无工作日/双休日）。"""
    mc = w.get('mean_agent_costs', [0, 0, 0])
    row = {
        'scenario': scenario,
        'method': method,
        'pv_coef': pv_coef,
        'mean_peak_load': w.get('mean_peak_load', 0),
        'std_peak_load': w.get('std_peak_load', 0),
        'mean_total_cost': w.get('mean_total_cost', 0),
        'std_total_cost': w.get('std_total_cost', 0),
        'mean_agent1_cost': mc[0] if len(mc) > 0 else 0,
        'mean_agent2_cost': mc[1] if len(mc) > 1 else 0,
        'mean_agent3_cost': mc[2] if len(mc) > 2 else 0,
        'mean_cost_std': w.get('mean_cost_std', 0),
        'mean_agent_community_charge': w.get('mean_agent_community_charge', [0, 0, 0]),
        'mean_agent_community_discharge': w.get('mean_agent_community_discharge', [0, 0, 0]),
        'mean_agent_credit_change': w.get('mean_agent_credit_change', [0, 0, 0]),
    }
    if date_type is not None:
        row['date_type'] = date_type
    return row


def load_pv_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    rows = []
    # 新格式：每场景含 weekday / weekend 两组；展开为带 date_type 的扁平行
    for r in data.get('results', []):
        if 'weekday' in r and 'weekend' in r:
            for date_type, label in [('weekday', '工作日'), ('weekend', '双休日')]:
                w = r.get(date_type, {})
                rows.append(_row_from_metrics(r['scenario'], r.get('pv_coef', 0), 'mappo', w, date_type=label))
        else:
            rows.append(_row_from_metrics(r['scenario'], r.get('pv_coef', 0), 'mappo', r, date_type=None))
    for r in data.get('results_independent') or []:
        if 'weekday' in r and 'weekend' in r:
            for date_type, label in [('weekday', '工作日'), ('weekend', '双休日')]:
                w = r.get(date_type, {})
                rows.append(_row_from_metrics(r['scenario'], r.get('pv_coef', 0), 'independent', w, date_type=label))
        else:
            rows.append(_row_from_metrics(r['scenario'], r.get('pv_coef', 0), 'independent', r, date_type=None))
    return rows, data


def load_data(data_file):
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"数据文件不存在: {data_file}")
    ext = os.path.splitext(data_file)[1].lower()
    if ext == '.csv':
        return load_pv_csv(data_file), None
    if ext == '.json':
        rows, data = load_pv_json(data_file)
        return rows, data
    raise ValueError("请使用 .csv 或 .json 文件")


def _draw_pv_flow_on_ax(agent_pv_flows_48, ax, title):
    """在 ax 上绘制单户 PV 流向堆叠图（参考 plot_energy_scheduling.plot_pv_flow_direction）。"""
    # agent_pv_flows_48: list of 48 dicts, keys: direct_use, ess_charge, ev_charge, community_charge, grid_sell
    n = len(agent_pv_flows_48)
    hours = np.array([t * 0.5 for t in range(n)])
    pv_direct = np.array([agent_pv_flows_48[t].get('direct_use', 0) for t in range(n)])
    pv_ess = np.array([agent_pv_flows_48[t].get('ess_charge', 0) for t in range(n)])
    pv_ev = np.array([agent_pv_flows_48[t].get('ev_charge', 0) for t in range(n)])
    pv_community = np.array([agent_pv_flows_48[t].get('community_charge', 0) for t in range(n)])
    pv_grid_sell = np.array([agent_pv_flows_48[t].get('grid_sell', 0) for t in range(n)])
    pv_total = pv_direct + pv_ess + pv_ev + pv_community + pv_grid_sell
    
    # 检查是否有非零的充电流向（用于调试）
    has_ess = np.any(pv_ess > 1e-6)
    has_ev = np.any(pv_ev > 1e-6)
    has_community = np.any(pv_community > 1e-6)
    if not (has_ess or has_ev or has_community):
        print(f"  提示：{title} 的 PV 流向中 ess_charge/ev_charge/community_charge 均为 0，可能因为策略未选择用PV充电或PV发电量仅够直接用电/售电。")
    
    # 统一配色方案（与净负荷堆叠图协调，更美观）
    colors = {
        'direct': '#FF6B6B',      # 家庭直接用电 - 红色
        'ess': '#4ECDC4',         # ESS充电 - 青色
        'ev': '#45B7D1',          # EV充电 - 蓝色
        'community': '#32CD32',   # 向社区储能贡献 - 绿色
        'grid_sell': '#FFA500'    # 售电给电网 - 橙色
    }
    
    # 绘制堆叠图：即使值为0也绘制（保证图例完整），但只显示非零的图例项
    bottom = np.zeros(n)
    handles = []
    labels = []
    
    if np.any(pv_direct > 1e-6):
        ax.fill_between(hours, bottom, bottom + pv_direct, color=colors['direct'], alpha=0.7)
        handles.append(mpatches.Patch(facecolor=colors['direct'], alpha=0.7))
        labels.append('家庭直接用电')
    bottom += pv_direct
    
    if np.any(pv_ess > 1e-6):
        ax.fill_between(hours, bottom, bottom + pv_ess, color=colors['ess'], alpha=0.7)
        handles.append(mpatches.Patch(facecolor=colors['ess'], alpha=0.7))
        labels.append('储能充电')
    bottom += pv_ess
    
    if np.any(pv_ev > 1e-6):
        ax.fill_between(hours, bottom, bottom + pv_ev, color=colors['ev'], alpha=0.7)
        handles.append(mpatches.Patch(facecolor=colors['ev'], alpha=0.7))
        labels.append('EV充电')
    bottom += pv_ev
    
    if np.any(pv_community > 1e-6):
        ax.fill_between(hours, bottom, bottom + pv_community, color=colors['community'], alpha=0.7)
        handles.append(mpatches.Patch(facecolor=colors['community'], alpha=0.7))
        labels.append('向社区储能贡献')
    bottom += pv_community
    
    if np.any(pv_grid_sell > 1e-6):
        ax.fill_between(hours, bottom, bottom + pv_grid_sell, color=colors['grid_sell'], alpha=0.7)
        handles.append(mpatches.Patch(facecolor=colors['grid_sell'], alpha=0.7))
        labels.append('售电给电网')
    
    ax.plot(hours, pv_total, 'k-', linewidth=1.5, alpha=0.8)
    from matplotlib.lines import Line2D
    handles.append(Line2D([0],[0], color='black', linewidth=1.5))
    labels.append('PV总发电量')
    
    ax.set_xlabel('时间 (小时)', fontsize=12)
    ax.set_ylabel('功率 (kW)', fontsize=12)
    ax.set_xlim(0, 24)
    if handles:
        ax.legend(handles, labels, loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description='光伏场景评估结果绘图')
    parser.add_argument('--data_file', type=str, default='multi_agent/pv_eval_results/pv_eval.json',
                        help='评估结果文件，默认 JSON 以便默认绘制光伏流向图与行为图；也可指定 .csv')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()
    # 默认优先用 JSON（含 one_episode_agent_pv_flows 等），便于默认绘制光伏流向图；若不存在则回退到同前缀 CSV
    data_file = args.data_file
    if not os.path.exists(data_file) and data_file.endswith('.json'):
        csv_alt = data_file.replace('.json', '.csv')
        if os.path.exists(csv_alt):
            data_file = csv_alt
    elif not os.path.exists(data_file) and data_file.endswith('.csv'):
        json_alt = data_file.replace('.csv', '.json')
        if os.path.exists(json_alt):
            data_file = json_alt
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(data_file), 'figures')
    os.makedirs(args.output_dir, exist_ok=True)
    rows, raw_data = load_data(data_file)
    if not rows:
        print("没有数据")
        return
    _runs_suffix = ''
    if raw_data and raw_data.get('num_runs', 1) > 1:
        _runs_suffix = f"（共 {raw_data['num_runs']} 轮平均）"

    # 兼容：无 method 时视为全部 MAPPO
    rows_mappo = [r for r in rows if r.get('method', 'mappo') == 'mappo']
    rows_independent = [r for r in rows if r.get('method') == 'independent']
    has_compare = len(rows_independent) > 0
    # 是否按工作日/双休日分开（新格式：每场景两行）
    has_date_type = any(r.get('date_type') for r in rows_mappo)
    scenarios_order = list(dict.fromkeys([r['scenario'] for r in rows_mappo]))
    n_s = len(scenarios_order)

    # 图1：MAPPO 各场景峰值与总成本（若有 date_type 则分组：工作日 vs 双休日）
    x = np.arange(n_s)
    width = 0.35 if has_date_type else 0.6
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    if has_date_type:
        peak_wd = [next((r['mean_peak_load'] for r in rows_mappo if r['scenario'] == s and r.get('date_type') == '工作日'), 0) for s in scenarios_order]
        peak_we = [next((r['mean_peak_load'] for r in rows_mappo if r['scenario'] == s and r.get('date_type') == '双休日'), 0) for s in scenarios_order]
        peak_wd_err = [next((r.get('std_peak_load', 0) for r in rows_mappo if r['scenario'] == s and r.get('date_type') == '工作日'), 0) for s in scenarios_order]
        peak_we_err = [next((r.get('std_peak_load', 0) for r in rows_mappo if r['scenario'] == s and r.get('date_type') == '双休日'), 0) for s in scenarios_order]
        ax1.bar(x - width/2, peak_wd, width, yerr=peak_wd_err, capsize=3, label='工作日', color='steelblue', alpha=0.8)
        ax1.bar(x + width/2, peak_we, width, yerr=peak_we_err, capsize=3, label='双休日', color='skyblue', alpha=0.8)
        cost_wd = [next((r['mean_total_cost'] for r in rows_mappo if r['scenario'] == s and r.get('date_type') == '工作日'), 0) for s in scenarios_order]
        cost_we = [next((r['mean_total_cost'] for r in rows_mappo if r['scenario'] == s and r.get('date_type') == '双休日'), 0) for s in scenarios_order]
        cost_wd_err = [next((r.get('std_total_cost', 0) for r in rows_mappo if r['scenario'] == s and r.get('date_type') == '工作日'), 0) for s in scenarios_order]
        cost_we_err = [next((r.get('std_total_cost', 0) for r in rows_mappo if r['scenario'] == s and r.get('date_type') == '双休日'), 0) for s in scenarios_order]
        ax2.bar(x - width/2, cost_wd, width, yerr=cost_wd_err, capsize=3, label='工作日', color='coral', alpha=0.8)
        ax2.bar(x + width/2, cost_we, width, yerr=cost_we_err, capsize=3, label='双休日', color='lightsalmon', alpha=0.8)
    else:
        peak = [next((r['mean_peak_load'] for r in rows_mappo if r['scenario'] == s), 0) for s in scenarios_order]
        peak_err = [next((r['std_peak_load'] for r in rows_mappo if r['scenario'] == s), 0) for s in scenarios_order]
        ax1.bar(x, peak, width, yerr=peak_err, capsize=4, label='社区净负荷峰值', color='steelblue', alpha=0.8)
        cost = [next((r['mean_total_cost'] for r in rows_mappo if r['scenario'] == s), 0) for s in scenarios_order]
        cost_err = [next((r['std_total_cost'] for r in rows_mappo if r['scenario'] == s), 0) for s in scenarios_order]
        ax2.bar(x, cost, width, yerr=cost_err, capsize=4, label='总电网成本', color='coral', alpha=0.8)
    ax1.set_ylabel('峰值 (kW)')
    ax1.set_title('不同光伏场景下社区净负荷峰值')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios_order)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax2.set_ylabel('总成本 (元/天)')
    ax2.set_title('不同光伏场景下总电网成本')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios_order)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    fig.suptitle('MAPPO 在不同光伏（天气）场景下的表现' + ('（工作日 vs 双休日）' if has_date_type else '') +
                 (_runs_suffix if _runs_suffix else ''), y=1.02)
    fig.tight_layout()
    path = os.path.join(args.output_dir, 'pv_scenario_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"已保存: {path}")

    # 图2：异质光伏场景 高/中/低光伏户成本（仅 MAPPO；有 date_type 时取双休日）
    hetero_mappo = next((r for r in rows_mappo if r.get('scenario') == 'heterogeneous' and (not has_date_type or r.get('date_type') == '双休日')), None)
    if hetero_mappo is None and has_date_type:
        hetero_mappo = next((r for r in rows_mappo if r.get('scenario') == 'heterogeneous'), None)
    if hetero_mappo is not None:
        fig2, ax = plt.subplots(figsize=(6, 4))
        hh_labels = ['高光伏户\n(agent0)', '中光伏户\n(agent1)', '低光伏户\n(agent2)']
        costs = [hetero_mappo['mean_agent1_cost'], hetero_mappo['mean_agent2_cost'], hetero_mappo['mean_agent3_cost']]
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        bars = ax.bar(hh_labels, costs, color=colors, alpha=0.85, edgecolor='gray', linewidth=0.5)
        ax.set_ylabel('户均电网成本 (元/天)')
        ax.set_title('异质光伏场景下高/中/低光伏户成本对比 (MAPPO)')
        ax.grid(True, alpha=0.3, axis='y')
        for b, v in zip(bars, costs):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
        fig2.tight_layout()
        path2 = os.path.join(args.output_dir, 'pv_heterogeneous_household_comparison.png')
        fig2.savefig(path2, dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(f"已保存: {path2}")

    # 图3、4：MAPPO vs 独立基线（当存在 independent 结果时）
    n_ind = len(rows_independent)
    if has_compare and (n_ind == len(rows_mappo) or (has_date_type and n_ind == 2 * n_s)):
        x = np.arange(n_s)
        width = 0.35
        if has_date_type:
            fig3, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
            for ax, date_label, metric, ylabel in [
                (ax1, '工作日', 'mean_peak_load', '峰值 (kW)'),
                (ax2, '双休日', 'mean_peak_load', '峰值 (kW)'),
                (ax3, '工作日', 'mean_total_cost', '总成本 (元/天)'),
                (ax4, '双休日', 'mean_total_cost', '总成本 (元/天)'),
            ]:
                std_key = 'std_peak_load' if 'peak' in metric else 'std_total_cost'
                peak_m = [next((r[metric] for r in rows_mappo if r['scenario'] == s and r.get('date_type') == date_label), 0) for s in scenarios_order]
                peak_i = [next((r[metric] for r in rows_independent if r['scenario'] == s and r.get('date_type') == date_label), 0) for s in scenarios_order]
                err_m = [next((r.get(std_key, 0) for r in rows_mappo if r['scenario'] == s and r.get('date_type') == date_label), 0) for s in scenarios_order]
                err_i = [next((r.get(std_key, 0) for r in rows_independent if r['scenario'] == s and r.get('date_type') == date_label), 0) for s in scenarios_order]
                ax.bar(x - width/2, peak_m, width, yerr=err_m, capsize=3, label='MAPPO', color='steelblue', alpha=0.8)
                ax.bar(x + width/2, peak_i, width, yerr=err_i, capsize=3, label='独立基线', color='gray', alpha=0.8)
                ax.set_ylabel(ylabel)
                ax.set_title(f'各场景 {date_label}')
                ax.set_xticks(x)
                ax.set_xticklabels(scenarios_order)
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
        else:
            fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            peak_m = [next((r['mean_peak_load'] for r in rows_mappo if r['scenario'] == s), 0) for s in scenarios_order]
            peak_i = [next((r['mean_peak_load'] for r in rows_independent if r['scenario'] == s), 0) for s in scenarios_order]
            ax1.bar(x - width/2, peak_m, width, label='MAPPO', color='steelblue', alpha=0.8)
            ax1.bar(x + width/2, peak_i, width, label='独立基线', color='gray', alpha=0.8)
            ax1.set_ylabel('峰值 (kW)')
            ax1.set_title('各场景社区净负荷峰值 MAPPO vs 独立基线')
            ax1.set_xticks(x)
            ax1.set_xticklabels(scenarios_order)
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')
            cost_m = [next((r['mean_total_cost'] for r in rows_mappo if r['scenario'] == s), 0) for s in scenarios_order]
            cost_i = [next((r['mean_total_cost'] for r in rows_independent if r['scenario'] == s), 0) for s in scenarios_order]
            ax2.bar(x - width/2, cost_m, width, label='MAPPO', color='coral', alpha=0.8)
            ax2.bar(x + width/2, cost_i, width, label='独立基线', color='gray', alpha=0.8)
            ax2.set_ylabel('总成本 (元/天)')
            ax2.set_title('各场景总电网成本 MAPPO vs 独立基线')
            ax2.set_xticks(x)
            ax2.set_xticklabels(scenarios_order)
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
        fig3.suptitle('社区效益：MAPPO vs 独立基线' + ('（工作日 vs 双休日）' if has_date_type else '') + (_runs_suffix if _runs_suffix else ''), y=1.02)
        fig3.tight_layout()
        path3 = os.path.join(args.output_dir, 'pv_mappo_vs_independent_community.png')
        fig3.savefig(path3, dpi=150, bbox_inches='tight')
        plt.close(fig3)
        print(f"已保存: {path3}")

        hetero_ind = next((r for r in rows_independent if r.get('scenario') == 'heterogeneous' and (not has_date_type or r.get('date_type') == '双休日')), None)
        if hetero_ind is None:
            hetero_ind = next((r for r in rows_independent if r.get('scenario') == 'heterogeneous'), None)
        if hetero_mappo is not None and hetero_ind is not None:
            fig4, ax = plt.subplots(figsize=(7, 4))
            hh_labels = ['高光伏户\n(agent0)', '中光伏户\n(agent1)', '低光伏户\n(agent2)']
            x = np.arange(3)
            width = 0.35
            cm = [hetero_mappo['mean_agent1_cost'], hetero_mappo['mean_agent2_cost'], hetero_mappo['mean_agent3_cost']]
            ci = [hetero_ind['mean_agent1_cost'], hetero_ind['mean_agent2_cost'], hetero_ind['mean_agent3_cost']]
            # 优化配色：与成本对比图协调
            ax.bar(x - width/2, cm, width, label='MAPPO', color='#45B7D1', alpha=0.85, edgecolor='black', linewidth=0.8)
            ax.bar(x + width/2, ci, width, label='独立基线', color='#FF6B6B', alpha=0.85, edgecolor='black', linewidth=0.8)
            ax.set_xlabel('家庭', fontsize=12)
            ax.set_ylabel('户均电网成本 (元/天)', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(hh_labels)
            ax.legend(loc='upper left', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            for i, (vm, vi) in enumerate(zip(cm, ci)):
                ax.text(i - width/2, vm + 0.02, f'{vm:.3f}', ha='center', va='bottom', fontsize=9)
                ax.text(i + width/2, vi + 0.02, f'{vi:.3f}', ha='center', va='bottom', fontsize=9)
            fig4.tight_layout()
            path4 = os.path.join(args.output_dir, 'pv_heterogeneous_mappo_vs_independent.png')
            fig4.savefig(path4, dpi=600, bbox_inches='tight')
            plt.close(fig4)
            print(f"已保存: {path4}")

    # 若当前加载的是 CSV，尝试加载同目录同前缀的 JSON 以获取行为与 PV 流向数据（便于绘制光伏流向图等）
    if raw_data is None and data_file.lower().endswith('.csv'):
        json_path = os.path.join(os.path.dirname(data_file),
                                 os.path.basename(data_file).replace('.csv', '.json'))
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

    # 行为差异图：异质光伏场景下高/中/低光伏户 向社区储能充电 vs 从社区储能放电（kWh/天）；有 date_type 时已用双休日行
    hetero_for_behavior = (hetero_mappo if (hetero_mappo and (hetero_mappo.get('mean_agent_community_charge') is not None or (isinstance(hetero_mappo.get('mean_agent_community_charge'), list)))) else None)
    if hetero_for_behavior is None and raw_data is not None:
        hetero_result = next((r for r in raw_data.get('results', []) if r.get('scenario') == 'heterogeneous'), None)
        if hetero_result:
            w = hetero_result.get('weekend') or hetero_result.get('weekday') or hetero_result
            if w.get('mean_agent_community_charge') is not None or isinstance(w.get('mean_agent_community_charge'), list):
                hetero_for_behavior = w
    if hetero_for_behavior is not None:
        ch = hetero_for_behavior.get('mean_agent_community_charge', [])
        dis = hetero_for_behavior.get('mean_agent_community_discharge', [])
        cred = hetero_for_behavior.get('mean_agent_credit_change', [0, 0, 0])
        if len(ch) >= 3 and len(dis) >= 3:
            # 两子图：左=社区充放电，右=积分净变化（高光伏户成本高但积分补偿）
            fig_b, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            hh_labels = ['高光伏户\n(agent0)', '中光伏户\n(agent1)', '低光伏户\n(agent2)']
            x = np.arange(3)
            width = 0.35
            # 优化配色：与净负荷堆叠图协调
            ax1.bar(x - width/2, ch, width, label='向社区储能充电', color='#4ECDC4', alpha=0.85, edgecolor='black', linewidth=0.8)
            ax1.bar(x + width/2, dis, width, label='从社区储能放电', color='#FF6B6B', alpha=0.85, edgecolor='black', linewidth=0.8)
            ax1.set_xlabel('家庭', fontsize=12)
            ax1.set_ylabel('能量 (kWh/天)', fontsize=12)
            ax1.set_xticks(x)
            ax1.set_xticklabels(hh_labels)
            ax1.legend(loc='upper left', fontsize=10)
            ax1.grid(True, alpha=0.3, axis='y')
            # 积分净变化：正=收益，负=支出；高光伏户虽电网成本高，积分上获得补偿
            cred_3 = cred if len(cred) >= 3 else [0, 0, 0]
            colors_cred = ['#2ecc71', '#f39c12', '#e74c3c']  # 保持原配色以区分三户
            bars2 = ax2.bar(hh_labels, cred_3, color=colors_cred, alpha=0.85, edgecolor='black', linewidth=0.8)
            ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
            ax2.set_xlabel('家庭', fontsize=12)
            ax2.set_ylabel('积分净变化', fontsize=12)
            ax2.grid(True, alpha=0.3, axis='y')
            for b, v in zip(bars2, cred_3):
                ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + (0.02 if v >= 0 else -0.08),
                        f'{v:.2f}', ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)
            fig_b.tight_layout()
            path_b = os.path.join(args.output_dir, 'pv_heterogeneous_community_behavior.png')
            fig_b.savefig(path_b, dpi=600, bbox_inches='tight')
            plt.close(fig_b)
            print(f"已保存: {path_b}")

    # PV 流向图：异质光伏场景下高/中/低光伏户 24h PV 流向堆叠图（需 JSON 中含 one_episode_agent_pv_flows）
    if raw_data is not None:
        hetero_result = next((r for r in raw_data.get('results', []) if r.get('scenario') == 'heterogeneous'), None)
        pv_flows = hetero_result.get('one_episode_agent_pv_flows') if hetero_result else None
        num_steps = len(pv_flows[0]) if (pv_flows is not None and len(pv_flows) >= 3 and len(pv_flows[0]) > 0) else 0
        if pv_flows is not None and len(pv_flows) >= 3 and num_steps >= 1:
            if num_steps < 48:
                print(f"提示：PV 流向数据仅含 {num_steps} 步（非完整 24h），完整图需重新运行 run_pv_evaluation.py 生成 48 步数据。")
            fig_pv, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
            titles = ['高光伏户 (agent0)', '中光伏户 (agent1)', '低光伏户 (agent2)']
            for i, (ax, title) in enumerate(zip(axes, titles)):
                _draw_pv_flow_on_ax(pv_flows[i], ax, title)
            fig_pv.tight_layout()
            path_pv = os.path.join(args.output_dir, 'pv_heterogeneous_pv_flow.png')
            fig_pv.savefig(path_pv, dpi=600, bbox_inches='tight')
            plt.close(fig_pv)
            print(f"已保存: {path_pv}")
        elif hetero_result is not None and (pv_flows is None or len(pv_flows) < 3 or num_steps < 1):
            print("未找到异质场景 PV 流向数据（需 one_episode_agent_pv_flows 且每户至少 1 步），跳过光伏流向图。请重新运行 run_pv_evaluation.py 生成完整 JSON。")
    else:
        print("未加载 JSON，跳过光伏流向图（需使用 run_pv_evaluation.py 生成的 JSON）。")


if __name__ == '__main__':
    main()
