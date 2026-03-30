"""
分析社区储能利用率低的问题
"""
import json

# 读取配置
with open('multi_agent/config.json', 'r', encoding='utf-8') as f:
    config_data = json.load(f)
    config = config_data.get('training', {})

community_weight = config.get('community_weight', 20)
peak_discharge_bonus = config.get('peak_discharge_bonus', 0.1)
community_credit_cost_weight = config.get('community_credit_cost_weight', 0.3)
peak_credit_cost_reduction = config.get('peak_credit_cost_reduction', 0.5)
initial_credit = config.get('initial_credit', 50.0)

print("=" * 80)
print("社区储能利用率低的原因分析")
print("=" * 80)

print("\n【当前配置】")
print(f"  初始积分: {initial_credit}")
print(f"  峰值放电奖励: {peak_discharge_bonus} 元/kW")
print(f"  积分成本权重: {community_credit_cost_weight}")
print(f"  峰值时段积分成本降低: {peak_credit_cost_reduction} (降低 {(1-peak_credit_cost_reduction)*100:.0f}%)")
print(f"  社区峰值惩罚权重: {community_weight}")

print("\n【使用社区储能的经济激励分析】")
print("\n假设场景：峰值时段，从社区储能放电5kW（最大功率）")

# 典型电价
grid_price = 0.5  # 典型峰值电价

# 计算积分成本
# 从社区储能购买价格
community_price = grid_price * 1.05 * 1.2 * 1.1  # 基础价 * SOC因子 * 需求因子
energy_amount = 5.0 * 0.5  # 5kW * 0.5小时 = 2.5kWh
credit_cost = community_price * energy_amount

print(f"\n  1. 积分成本计算：")
print(f"     社区储能价格: {community_price:.3f} 积分/kWh")
print(f"     能量: {energy_amount} kWh")
print(f"     消耗积分: {credit_cost:.3f} 积分")

# 正常时段积分惩罚
normal_credit_penalty = community_credit_cost_weight * credit_cost
print(f"     正常时段积分惩罚: {normal_credit_penalty:.3f}")

# 峰值时段积分惩罚（降低后）
peak_credit_penalty = community_credit_cost_weight * peak_credit_cost_reduction * credit_cost
print(f"     峰值时段积分惩罚（降低后）: {peak_credit_penalty:.3f}")

# 峰值放电奖励
discharge_reward = peak_discharge_bonus * 5.0  # 5kW
print(f"\n  2. 峰值放电奖励：")
print(f"     奖励: {discharge_reward:.3f} (每kW {peak_discharge_bonus})")

# 净收益
net_benefit = discharge_reward - peak_credit_penalty
print(f"\n  3. 净收益：")
print(f"     奖励 - 积分惩罚 = {discharge_reward:.3f} - {peak_credit_penalty:.3f} = {net_benefit:.3f}")

# 对比避免峰值惩罚的收益
print(f"\n  4. 对比：避免峰值惩罚的收益")
print(f"     假设通过放电5kW可以避免5kW的峰值超额负荷")
print(f"     超额负荷5kW的峰值惩罚: (5^1.5) / (31.01^1.5) ≈ 0.0647")
print(f"     加权惩罚: 0.0647 * {community_weight} = {0.0647 * community_weight:.3f}")
print(f"     使用社区储能放电的净收益: {net_benefit:.3f}")
print(f"     两者比较: {net_benefit:.3f} vs {0.0647 * community_weight:.3f}")

print("\n【问题诊断】")
if net_benefit < 0:
    print("  ❌ 使用社区储能的净收益为负！智能体没有动力使用。")
elif net_benefit < 0.0647 * community_weight * 0.1:
    print("  ⚠️  使用社区储能的净收益太小，不足以激励使用。")
    print(f"     净收益 {net_benefit:.3f} 远小于避免峰值惩罚的收益 {0.0647 * community_weight:.3f}")
else:
    print("  ✓ 净收益为正，但可能仍不足以激励使用。")

print("\n【可能的解决方案】")
print("  1. 增加峰值放电奖励 (peak_discharge_bonus)")
print("     当前: 0.1 → 建议: 0.5-1.0")
print("\n  2. 降低积分成本权重 (community_credit_cost_weight)")
print("     当前: 0.3 → 建议: 0.1-0.2")
print("\n  3. 增加峰值时段积分成本降低幅度 (peak_credit_cost_reduction)")
print("     当前: 0.5 (降低50%) → 建议: 0.3-0.2 (降低70-80%)")
print("\n  4. 增加初始积分 (initial_credit)")
print("     当前: 50 → 建议: 100-200 (让智能体有更多积分可以使用)")

print("\n" + "=" * 80)
