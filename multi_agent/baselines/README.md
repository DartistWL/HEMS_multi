# 基线算法说明

## 1. 固定规则基线 (Rule-Based Baseline)

### 策略规则
1. **光伏优先自家使用**：光伏发电优先满足家庭负荷
2. **余电充入私有储能**：光伏余电优先充入家庭储能
3. **社区储能峰值放电**：仅在社区净负荷超过阈值（平均值×1.2）时，从社区储能放电
4. **EV策略**：低价时段充电，高价时段放电
5. **其他设备**：洗衣机在低价时段运行，空调和热水器使用简单温度控制

### 使用方法

```python
from multi_agent.baselines.rule_based_baseline import RuleBasedBaseline
from multi_agent.environment.multi_agent_env import MultiAgentHEMEnv

# 创建环境
env = MultiAgentHEMEnv(n_agents=3, community_ess_capacity=36.0)

# 创建基线
baseline = RuleBasedBaseline(peak_threshold_factor=1.2)

# 评估
results = baseline.evaluate(env, num_episodes=10, mode='eval')
```

## 2. 独立学习基线 (Independent Learning Baseline)

### 特点
- **只训练一个模型**：所有家庭共用同一个训练好的模型，确保策略一致
- **统一随机种子**：所有环境使用相同的随机种子，确保EV调度等随机因素一致
- 每个家庭只能看到自家信息
- 奖励函数只有个人目标（无社区目标）
- 无社区储能交互
- 使用单智能体PPO算法

### 设计改进
- **问题**：之前每个家庭独立训练，导致策略不一致，成本差异很大
- **解决方案**：只训练一个模型，所有家庭共用，统一随机种子
- **优势**：确保策略一致，成本计算更合理

### 使用方法

#### 训练

```python
from multi_agent.baselines.independent_baseline import IndependentBaseline

# 创建基线（统一随机种子）
baseline = IndependentBaseline(
    n_agents=3,
    pv_coefficients=[2.0, 2.0, 2.0],  # 正常天气
    random_seed=0  # 统一随机种子
)

# 训练（只训练一个模型，所有家庭共用）
baseline.train(num_episodes=5000, save_dir='multi_agent/baselines/models')
```

#### 评估

```python
# 加载模型（所有家庭共用同一个模型）
baseline.load_models(model_dir='multi_agent/baselines/models')

# 评估（所有家庭使用相同的策略）
results = baseline.evaluate(num_episodes=10)
```

#### 计算基准峰值

```python
# 计算基准峰值（用于归一化峰值惩罚）
baseline_peak = baseline.calculate_baseline_peak(num_episodes=10)
print(f"Baseline Peak Load: {baseline_peak:.2f} kW")
```

## 评估指标

两个基线都会输出以下指标：
- **Episode Returns**：每轮总奖励
- **Episode Costs**：每轮总成本
- **Peak Loads**：社区峰值负荷
- **Peak Penalties**：峰值惩罚（仅固定规则基线）

## 注意事项

1. **独立学习基线训练时间较长**：每个智能体需要独立训练，总训练时间 = 单智能体训练时间 × 3
2. **模型保存路径**：确保`multi_agent/baselines/models/`目录存在
3. **基准峰值**：独立学习基线可以用于计算基准峰值，作为MAPPO训练时的归一化参考
