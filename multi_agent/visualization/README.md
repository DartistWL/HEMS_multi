# 可视化工具使用说明

## 概述

本目录包含所有用于展示项目创新点的可视化脚本。**所有脚本都只使用真实数据，不使用示例数据**。

## 已实现的可视化脚本

### 1. 社区峰值负荷对比图 (`plot_peak_comparison.py`) ✅
- **功能**：对比三种方法的社区净负荷时间序列和峰值
- **输出**：对比图 + 汇总表
- **使用方法**：
  ```bash
  python multi_agent/visualization/plot_peak_comparison.py \
      --data_file multi_agent/visualization_data/comparison_data.json
  ```

### 2. 社区储能使用可视化 (`plot_community_ess_usage.py`) ✅
- **功能**：展示社区储能SOC变化和充放电功率
- **使用方法**：
  ```bash
  python multi_agent/visualization/plot_community_ess_usage.py \
      --data_file multi_agent/visualization_data/comparison_data.json \
      --method mappo
  ```

### 3. 积分余额变化曲线 (`plot_credit_balance.py`) ✅
- **功能**：展示三个家庭的积分余额变化
- **使用方法**：
  ```bash
  python multi_agent/visualization/plot_credit_balance.py \
      --data_file multi_agent/visualization_data/comparison_data.json \
      --method mappo
  ```

### 4. 多目标权衡雷达图 (`plot_multi_objective_tradeoff.py`) ✅
- **功能**：展示三种方法在多个目标上的综合表现
- **使用方法**：
  ```bash
  python multi_agent/visualization/plot_multi_objective_tradeoff.py \
      --data_file multi_agent/visualization_data/comparison_data.json
  ```

### 5. 负荷平滑度对比 (`plot_load_smoothness.py`) ✅
- **功能**：展示负荷平滑效果（箱线图 + 标准差对比）
- **使用方法**：
  ```bash
  python multi_agent/visualization/plot_load_smoothness.py \
      --data_file multi_agent/visualization_data/comparison_data.json
  ```

### 6. 运行所有可视化 (`run_all_visualizations.py`) ✅
- **功能**：一键运行所有可视化脚本
- **使用方法**：
  ```bash
  python multi_agent/visualization/run_all_visualizations.py \
      --data_file multi_agent/visualization_data/comparison_data.json
  ```

## 使用流程

### 步骤1：收集数据（必须先完成）

使用 `collect_visualization_data.py` 脚本收集评估数据：

```bash
# 收集固定规则基线数据（可以立即运行，不需要模型）
python multi_agent/collect_visualization_data.py --method rule_based --num_episodes 3

# 收集独立学习基线数据（需要先训练模型）
python multi_agent/collect_visualization_data.py --method independent \
    --independent_model_dir multi_agent/baselines/models --num_episodes 3

# 收集MAPPO数据（需要先训练模型）
python multi_agent/collect_visualization_data.py --method mappo \
    --mappo_model_dir multi_agent/algorithms/models \
    --baseline_peak 31.01 --num_episodes 3

# 或者收集所有方法的数据
python multi_agent/collect_visualization_data.py --method all \
    --independent_model_dir multi_agent/baselines/models \
    --mappo_model_dir multi_agent/algorithms/models \
    --baseline_peak 31.01 --num_episodes 3
```

数据会保存在 `multi_agent/visualization_data/` 目录：
- `rule_based_data.json`
- `independent_data.json`
- `mappo_data.json`
- `comparison_data.json` (汇总数据，用于对比可视化)

### 步骤2：运行可视化脚本

#### 方式1：单独运行

```bash
# 峰值对比图
python multi_agent/visualization/plot_peak_comparison.py \
    --data_file multi_agent/visualization_data/comparison_data.json

# 社区储能使用图
python multi_agent/visualization/plot_community_ess_usage.py \
    --data_file multi_agent/visualization_data/comparison_data.json \
    --method mappo

# 积分余额变化图
python multi_agent/visualization/plot_credit_balance.py \
    --data_file multi_agent/visualization_data/comparison_data.json \
    --method mappo

# 多目标权衡雷达图
python multi_agent/visualization/plot_multi_objective_tradeoff.py \
    --data_file multi_agent/visualization_data/comparison_data.json

# 负荷平滑度对比图
python multi_agent/visualization/plot_load_smoothness.py \
    --data_file multi_agent/visualization_data/comparison_data.json
```

#### 方式2：一键运行所有可视化

```bash
python multi_agent/visualization/run_all_visualizations.py \
    --data_file multi_agent/visualization_data/comparison_data.json
```

## 数据文件格式

### comparison_data.json 格式

```json
{
  "baseline_peak": 31.01,
  "independent": {
    "community_net_loads": [0.0, 1.5, 2.3, ...],  // 48个值
    "peak_load": 31.01,
    "avg_load": 15.5,
    "std_load": 8.2,
    "total_cost": 120.5,
    "total_return": -50.2,
    "community_ess_soc": [0.5, 0.52, 0.51, ...],  // 48个值
    "community_ess_charge_power": [0.0, 2.5, 0.0, ...],  // 48个值
    "community_ess_discharge_power": [0.0, 0.0, 3.2, ...],  // 48个值
    "agent_credit_balances": [[1000.0, 998.5, ...], [...], [...]],  // 3个家庭，每个48个值
    "agent_net_loads": [[...], [...], [...]]  // 3个家庭，每个48个值
  },
  "rule_based": { ... },
  "mappo": { ... }
}
```

## 输出文件

所有可视化结果保存在 `multi_agent/visualization/output/` 目录中：

- `peak_comparison.png` - 峰值对比图
- `peak_comparison_summary.txt` - 峰值对比汇总表
- `community_ess_usage_mappo.png` - 社区储能使用图
- `credit_balance_mappo.png` - 积分余额变化图
- `multi_objective_tradeoff.png` - 多目标权衡雷达图
- `load_smoothness_comparison.png` - 负荷平滑度对比图

## 注意事项

1. **必须先收集数据**：所有可视化脚本都需要真实数据，不会使用示例数据
2. **数据文件路径**：确保数据文件路径正确
3. **模型训练**：独立基线和MAPPO的数据收集需要先训练模型
4. **基准峰值**：MAPPO数据收集需要提供基准峰值（通常是独立基线的峰值）

## 快速开始

1. **收集规则基线数据**（可以立即运行）：
   ```bash
   python multi_agent/collect_visualization_data.py --method rule_based --num_episodes 3
   ```

2. **运行可视化**（使用规则基线数据）：
   ```bash
   python multi_agent/visualization/plot_peak_comparison.py \
       --data_file multi_agent/visualization_data/comparison_data.json
   ```

## 可视化脚本特点

- ✅ **只使用真实数据**：所有脚本都从JSON文件读取真实数据
- ✅ **无示例数据**：不会使用任何示例或模拟数据
- ✅ **统一数据格式**：所有脚本使用相同的数据格式
- ✅ **错误处理**：如果数据不存在或格式错误，会给出明确的错误提示
