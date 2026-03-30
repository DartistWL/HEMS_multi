# 多智能体家庭能源管理系统 (Multi-Agent HEMS)

## 项目概述

本项目实现了基于MAPPO（Multi-Agent Proximal Policy Optimization）的多智能体强化学习系统，用于社区能源管理。系统包含3户家庭，共享一个社区储能池，通过社区积分机制和峰值负荷优化实现协作。

## 核心特性

1. **多智能体协作**：3户家庭智能体通过MAPPO算法学习协作策略
2. **社区共享储能**：36kWh共享储能池，支持家庭间能量共享
3. **动态积分系统**：基于市场价格的动态定价机制
4. **峰值负荷优化**：通过峰值惩罚实现削峰填谷
5. **多样化训练数据**：7天训练数据（4天晴天+3天阴天），3天评估数据（2011-07-10/11/12，与训练日不重叠）
6. **训练与独立基线一致**：MAPPO 与独立基线均为**一个 episode = 一天（48 步）**，7 天通过 `episode % 7` 循环（episode 0→07-03，1→07-04，…，6→07-09，7→07-03，…）

## 项目结构

```
multi_agent/
├── __init__.py
├── README.md
├── environment/              # 环境组件
│   ├── __init__.py
│   ├── multi_agent_env.py    # 多智能体环境主类
│   ├── single_agent_wrapper.py  # 单智能体环境包装
│   ├── community_ess.py      # 社区共享储能
│   ├── credit_system.py      # 社区积分系统
│   └── peak_tracker.py       # 峰值追踪器
├── baselines/                # 基线算法
│   ├── __init__.py
│   ├── independent_baseline.py  # 独立学习基线
│   └── rule_based_baseline.py   # 固定规则基线
├── algorithms/               # 强化学习算法
│   ├── __init__.py
│   └── mappo.py              # MAPPO算法实现
├── evaluation/               # 评估工具
│   ├── __init__.py
│   ├── multi_agent_evaluator.py  # 多智能体评估器
│   └── baseline_comparison.py    # 基线对比
└── utils/                    # 工具函数
    ├── __init__.py
    └── data_interface.py     # 数据接口（支持光伏系数调整）
```

## 核心设计

### 1. 奖励函数

```
总奖励 = 个人奖励 + α × 社区峰值惩罚

个人奖励：
- 电网购电成本
- 社区积分成本/收益
- 约束违反惩罚
- 用户满意度惩罚
- 套利奖励

社区峰值惩罚：
- 按贡献分配
- 归一化（除以基准峰值）
- 权重α = 0.2（固定）
```

### 2. 社区积分机制

- **动态定价**：基于电网电价、社区储能SOC、社区需求
- **购买价格**：略高于电网价，SOC越低价格越高
- **出售价格**：略低于电网价，SOC越低收购价越高

### 3. 峰值惩罚机制

- **阈值**：平均值 × 1.2
- **惩罚函数**：二次惩罚 `(excess_load)²`
- **归一化**：除以基准峰值²，与个人奖励同数量级
- **分配**：按贡献分配（每个智能体的惩罚与其对峰值的贡献成正比）
- **峰值定义**：`community_net_load = sum(agent_net_loads)`，即社区与电网的总交换功率（含「从电网购电向社区储能充电」、含「从社区储能放电减少的购电」），与电网侧真实负荷一致；每户 `net_load` 在 `single_agent_wrapper._calculate_net_load_with_pv_flow` 中已正确计入社区充放。

### 4. 多端同时充放与峰值

- 社区储能改为**多端同时充放**后，同一时间步可有多户同时从电网购电向社区充电、多户同时从社区放电。峰值计算仍为 `max(community_net_load)`，逻辑正确。
- 若观察到 **MAPPO 成本最低但峰值反而高于独立/规则基线**：多端模式允许「低价时多户同时从电网充电」，策略会倾向于低价时集中充电以降成本，易在低价时段形成新的负荷尖峰。已在 **`config.json` → `training`** 中增加/提高削峰相关参数，环境会自动读取：`community_weight`（已提至 0.8）、`queue_target_power`（15 kW）、`queue_penalty_weight`（3.5）、`peak_grid_purchase_penalty_weight`（2.5）。重新训练 MAPPO（及可选独立基线）后，峰值应有所下降。
- **社区储能充电奖励按电价分档**：**低电价**（<0.35）和**中段电价**（0.35～<high_price_threshold）给向社区储能充电的奖励，**高电价**（≥`high_price_threshold`，默认 0.6）不给充电奖励，避免高电价时买电充社区。中段电价时奖励为低电价时的 **`medium_price_charge_ratio`**（默认 0.6）。参数见 `config.json` → `training`：`high_price_threshold`、`medium_price_charge_ratio`。
- **EV 充放电仅在家时有效**：在单户环境 **`environment.py`** 的 `step()` 中，若 **EV 不在家**（`is_ev_at_home()` 为 False，由 `interface.py` 的出行时段 t1/t2 决定），会强制 `action['ev_power'] = 0`，避免策略在 EV 外出时段仍输出充放电导致净负荷图中出现不合理的“大量 EV 放电”。训练时未使用 action mask，故必须在 step 内做此强制。

## 使用方法

### 1. 训练数据配置

```python
from multi_agent.environment.multi_agent_env import MultiAgentHEMEnv

# 创建环境
env = MultiAgentHEMEnv(
    n_agents=3,
    community_ess_capacity=36.0,
    baseline_peak=50.0,
    community_weight=0.2
)

# 设置训练日期（7天：4天晴天+3天阴天）
training_dates = [
    '2011-07-03',  # 晴天
    '2011-07-04',  # 晴天
    '2011-07-05',  # 晴天
    '2011-07-06',  # 晴天
    '2011-07-07',  # 阴天
    '2011-07-08',  # 阴天
    '2011-07-09',  # 阴天
]


env.set_training_dates(training_dates, pv_coefficients_list)
```

### 2. 训练MAPPO

（见下文「训练 COMA」）COMA 训练使用 `train_coma.py`，支持 **Pop-Art** 对 Critic 的 TD target 与 Q 值做自适应归一化，利于回报快速收敛到稳定值；可通过 `config.json` → `training.use_popart_coma` 开关（默认 true）。

### 2a. 训练 COMA（含 Pop-Art）

```bash
python multi_agent/train_coma.py --num_episodes 1000 --save_dir multi_agent/algorithms/models_coma
```

- **Pop-Art**：Critic 预测归一化后的 Q，用 running mean/std 归一化 TD target，统计更新时调整 Critic 最后一层权重以保持输出语义不变；baseline 与 advantage 使用反归一化后的 Q。配置项：`config.json` → `training.use_popart_coma`（true/false）。

### 2b. 训练 MAPPO

```python
from multi_agent.algorithms.mappo import MAPPO

# 创建MAPPO算法
mappo = MAPPO(
    n_agents=3,
    state_dim=...,
    action_dim=...,
    # MAPPO超参数（经典值）
    gamma=0.99,
    lmbda=0.95,
    eps=0.2,
    epochs=10,
    lr=3e-4
)

# 训练
for episode in range(num_episodes):
    states = env.reset(mode='train')
    done = False
    
    while not done:
        actions = mappo.select_actions(states)
        next_states, rewards, dones, info = env.step(actions)
        mappo.store_transition(states, actions, rewards, next_states, dones)
        states = next_states
        done = all(dones)
    
    mappo.update()
```

### 3. 运行基线

```python
from multi_agent.baselines.independent_baseline import IndependentBaseline
from multi_agent.baselines.rule_based_baseline import RuleBasedBaseline

# 独立学习基线
independent_baseline = IndependentBaseline()
independent_baseline.train()

# 固定规则基线
rule_baseline = RuleBasedBaseline()
rule_baseline.evaluate()
```

## 评估指标

### 个人指标
- 家庭能源成本
- 用户满意度
- 约束违反率
- 社区积分余额变化

### 社区指标
- **社区峰值负荷**：max(社区净负荷)
- **峰值降低率**：(基准峰值 - 当前峰值) / 基准峰值
- **负荷平滑度**：社区净负荷的标准差
- **社区储能利用率**：社区储能的充放电效率

## 基线对比

1. **独立学习基线**：3户家庭完全独立训练，只有个人目标
2. **固定规则基线**：光伏优先自家使用，余电充入私有储能，社区储能仅在峰值时放电

## 光伏场景评估实验

训练时使用了多种天气（晴天/阴天等），评估时可在不同光伏系数下测试 MAPPO 表现，用于分析「不同光伏情况对模型的影响」。

**运行评估**（按晴天、阴天、正常等场景跑多 episode，输出 JSON/CSV）：

```bash
python multi_agent/run_pv_evaluation.py --output_dir multi_agent/pv_eval_results
python multi_agent/run_pv_evaluation.py --num_episodes 5
```

**绘图**（读取上述生成的 CSV/JSON，绘制峰值、总成本等对比图）：

```bash
python multi_agent/plot_pv_evaluation.py --data_file multi_agent/pv_eval_results/pv_eval.csv
```

- 评估日期与训练日 07-03～07-09 不重叠。光伏场景评估脚本 **按工作日与双休日分开评估**：**工作日** `EVAL_DATES_WEEKDAY`（默认 2011-07-11、07-12，周一～周二，EV 白天外出）、**双休日** `EVAL_DATES_WEEKEND`（默认 2011-07-10、07-17，周日，EV 全天在家）。每个光伏场景下分别在两类日期上跑若干 episode 并汇总峰值、总成本等，输出 JSON/CSV 中每场景含 `weekday` 与 `weekend` 两组指标；绘图会生成「工作日 vs 双休日」对比图，便于检查模型在两种时段是否都表现良好。
- 环境会在 `reset()` 时根据 `set_evaluation_dates(..., pv_coefficients_list)` 为每个家庭设置对应光伏系数。
- **异质光伏场景**：除 sunny/cloudy/normal（全社区同一天气）外，增加 **heterogeneous** 场景：户0=高光伏、户1=中光伏、户2=低光伏，用于论文中对比「高光伏户 vs 低光伏户」的户均成本与公平性。绘图脚本会额外生成 `pv_heterogeneous_household_comparison.png`（高/中/低光伏户成本对比）。
- **与独立基线对比**：加 `--compare_independent` 时，在相同光伏场景下再跑独立基线，输出 MAPPO vs 独立基线结果（JSON/CSV 含 `method` 列）。绘图会生成：`pv_mappo_vs_independent_community.png`（各场景社区峰值与总成本）、`pv_heterogeneous_mappo_vs_independent.png`（异质光伏下高/中/低光伏户成本 MAPPO vs 独立基线，供公平性/社区效益结论）。
- **行为差异**：评估时记录每户向社区储能充电/放电量（kWh/天），JSON 中含 `mean_agent_community_charge`、`mean_agent_community_discharge`。绘图会生成 `pv_heterogeneous_community_behavior.png`（异质光伏下高/中/低光伏户 向社区储能充电 vs 从社区储能放电，供论文“行为差异”结论）。社区储能按**多端口/逆变器**建模：同一时间步可**同时**向多户放电（如三户各 5kW 共 15kW）、接受多户充电，或部分户充电、部分户放电；ESS 状态按净充放更新，充电/放电功率分别按请求比例分配（受 SOC 与容量约束）。
- **PV 流向图**：异质光伏场景下保存一 episode 的 PV 流向（`one_episode_agent_pv_flows`），绘图会生成 `pv_heterogeneous_pv_flow.png`（高/中/低光伏户 24h PV 流向堆叠图，参考 `plot_energy_scheduling.plot_pv_flow_direction`）。
- 若使用贡献制积分（`credit_pricing.scheme == 'contribution_based'`），脚本会自动从 `models_contribution_based` 目录加载 MAPPO 模型。独立基线需在社区环境下训练（`independent_baseline.use_community_env=true`）且已训练好模型。

## 可视化数据收集与绘图、成本评估

- **数据收集**：`collect_visualization_data.py` 会按**工作日**与**双休日**分别收集规则基线、独立基线、MAPPO 的 episode 数据；每个 episode 带 `date_type`（`'工作日'` / `'双休日'`），并生成 `summary_weekday`、`summary_weekend` 供绘图与成本脚本使用。
- **净负荷堆叠图**：`visualization/plot_net_load_stacked.py` 若检测到数据中含 `summary_weekday`/`summary_weekend` 或 episodes 带 `date_type`，会**分别生成** `net_load_stacked_weekday.png` 与 `net_load_stacked_weekend.png`；否则只生成 `net_load_stacked_comparison.png`。函数支持参数 `date_type='weekday'` / `'weekend'` 单独绘制某一类。
- **电力成本计算**：`calculate_electricity_costs.py` 从同一套可视化数据中计算成本时，若 episodes 含 `date_type`，会**分别统计工作日与双休日**的平均成本、标准差等，并在控制台打印「工作日」「双休日」两组指标；若所有方法均有工作日/双休日数据，会额外保存 `cost_comparison_weekday_weekend.png`（各方法工作日 vs 双休日平均成本柱状图）。

## 注意事项

1. **数据路径**：确保数据文件在项目根目录的`data/`文件夹中
2. **基准峰值**：需要先运行独立基线计算基准峰值
3. **训练稳定性**：MAPPO训练可能需要较长时间，建议使用GPU加速
4. **独立基线环境参数**：独立基线在「社区储能」方案下创建的 `MultiAgentHEMEnv` 现已从 **`config.json` → `training`** 读取 `community_weight`、`peak_penalty_exponent`、`baseline_peak` 等，与 MAPPO 训练时一致，便于公平对比；排队/峰时购电惩罚由环境内部从 config 读取。
5. **评估日与过拟合**：训练在 **07-03～07-09**（7 天），评估在 **07-10、07-11、07-12**（3 天，与训练不重叠）。若独立基线在评估日上的成本只略低于规则方案，可能与**在训练集上过拟合**、在未见日期上泛化较差有关；可尝试增加训练轮数、正则或更多样化的训练日期后再评估。

## 待实现功能

- [ ] MAPPO算法实现
- [ ] 独立学习基线
- [ ] 固定规则基线
- [ ] 评估工具
- [ ] 结果可视化

## 版本历史

- v1.0.0 (2026-01-12): 初始版本，实现核心环境框架
