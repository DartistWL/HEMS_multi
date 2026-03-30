# 净负荷堆叠图设计文档

## 一、净负荷计算逻辑梳理

### 1.1 净负荷的定义

**净负荷 = 总消耗 - 总发电**

- **正值**：从电网购电（需求 > 供应）
- **负值**：向电网售电（供应 > 需求）

### 1.2 总消耗的组成

总消耗包括所有需要从外部获取能量的部分：

```
总消耗 = 家庭直接负荷 
      + ESS充电（从电网）
      + EV充电（从电网）
      + 从社区ESS放电（相当于获得能量）
```

**注意**：
- 家庭直接负荷 = home_load + AC + 洗衣机 + 热水器等
- 但PV可能已经直接供应了一部分，所以实际从电网购买 = 家庭直接负荷 - PV直接供应

### 1.3 总发电的组成

总发电包括所有可以抵消电网需求的部分：

```
总发电 = PV发电（直接用于家庭负荷）
      + PV充入ESS
      + PV充入EV
      + PV充入社区ESS
      + PV售给电网
      + ESS放电（用于家庭或售电）
      + EV放电（用于家庭或售电）
      + 向社区ESS充电（这实际上是消耗，减少净发电）
```

### 1.4 净负荷的完整公式

根据代码中的实现：

```python
# 家庭直接负荷
household_load = home_load + AC1 + AC2 + 洗衣机 + 热水器

# PV优先分配后，家庭还需要从电网购买的部分
grid_household_load = household_load - PV_direct_use

# 需要从电网充电的部分
grid_ess_charge = ESS_charge - PV_ess_charge
grid_ev_charge = EV_charge - PV_ev_charge
grid_community_charge = Community_charge - PV_community_charge

# 总消耗（从电网）
total_consumption = grid_household_load 
                  + grid_ess_charge
                  + grid_ev_charge
                  + grid_community_charge
                  + Community_discharge  # 从社区ESS放电

# 总发电（可以减少电网需求）
total_generation = PV_grid_sell  # PV售给电网
                 + ESS_discharge  # ESS放电
                 + EV_discharge   # EV放电

# 净负荷
net_load = total_consumption - total_generation
```

## 二、堆叠图设计方案

### 2.1 堆叠图的结构

**横轴**：时间步（0-47，共48个时间步，代表一天）

**纵轴**：功率（kW），零点在中间
- **上半部分（正值）**：需要从电网获取的能量
- **下半部分（负值）**：可以减少电网需求或售给电网的能量
- **净负荷线**：总体的净负荷曲线（可以在图中叠加显示）

### 2.2 正向堆叠（上半部分，需要从电网获取）

从下往上堆叠：

1. **家庭负荷（需要从电网购买的部分）**
   - 颜色：深蓝色
   - 数值：`household_load - pv_flow['direct_use']`

2. **ESS充电（从电网）**
   - 颜色：浅蓝色
   - 数值：`battery_charge - pv_flow['ess_charge']`

3. **EV充电（从电网）**
   - 颜色：绿色
   - 数值：`ev_charge - pv_flow['ev_charge']`

4. **向社区ESS充电（从电网）**
   - 颜色：橙色
   - 数值：`community_charge - pv_flow['community_charge']`（如果为正）

5. **从社区ESS放电**
   - 颜色：紫色
   - 数值：`max(0, community_ess_power)`（正值表示放电）

### 2.3 负向堆叠（下半部分，减少电网需求）

从上往下堆叠（绝对值）：

1. **ESS放电**
   - 颜色：青色
   - 数值：`-battery_discharge`（负值）

2. **EV放电**
   - 颜色：黄色
   - 数值：`-ev_discharge`（负值）

3. **PV售给电网**
   - 颜色：红色
   - 数值：`-pv_flow['grid_sell']`（负值）

### 2.4 净负荷线

在堆叠图上方叠加一条线，表示净负荷：
- **颜色**：黑色粗线
- **数值**：`net_load`（正值向上，负值向下）

### 2.5 峰值标注

在净负荷线上标注峰值点：
- **峰值时间**：用垂直虚线标记
- **峰值数值**：显示峰值功率值

## 三、堆叠图的优势

1. **直观展示能量流**：可以看到每个设备的功率贡献
2. **识别峰值时段**：净负荷线清楚显示峰值在哪里
3. **对比不同策略**：可以对比MAPPO和基线模型的差异
4. **优化效果可视化**：可以看到MAPPO如何通过调整各设备功率来降低峰值

## 四、实现建议

### 4.1 数据收集

需要在评估时记录每个时间步的：
- 各设备的功率（ESS、EV、社区ESS）
- PV流向信息
- 净负荷

### 4.2 绘图工具

使用`matplotlib`的`stackplot`或`bar`图：
- 正向堆叠：使用向上的柱状图或区域图
- 负向堆叠：使用向下的柱状图或区域图
- 净负荷线：使用`plot`叠加

### 4.3 多模型对比

- **选项1**：三个子图，分别显示独立训练、规则基线、MAPPO
- **选项2**：一个图，用不同透明度或线条样式叠加三个模型的净负荷线
- **选项3**：两个图，上面显示三个模型的净负荷线对比，下面显示MAPPO的详细堆叠图

## 五、示例数据结构

```python
# 每个时间步的数据
timestep_data = {
    'time': 0-47,
    'household_load': float,  # 家庭直接负荷
    'pv_direct_use': float,   # PV直接用于家庭
    'ess_charge': float,      # ESS充电功率
    'pv_ess_charge': float,   # PV充入ESS
    'ev_charge': float,       # EV充电功率
    'pv_ev_charge': float,    # PV充入EV
    'ess_discharge': float,   # ESS放电功率
    'ev_discharge': float,    # EV放电功率
    'community_ess_power': float,  # 社区ESS交互（正=放电，负=充电）
    'pv_community_charge': float,  # PV充入社区ESS
    'pv_grid_sell': float,    # PV售给电网
    'net_load': float         # 净负荷
}

# 计算堆叠图各部分
positive_stack = {
    'grid_household': household_load - pv_direct_use,
    'grid_ess_charge': max(0, ess_charge - pv_ess_charge),
    'grid_ev_charge': max(0, ev_charge - pv_ev_charge),
    'grid_community_charge': max(0, community_charge - pv_community_charge),
    'community_discharge': max(0, community_ess_power)
}

negative_stack = {
    'ess_discharge': -ess_discharge,
    'ev_discharge': -ev_discharge,
    'pv_grid_sell': -pv_grid_sell
}
```
