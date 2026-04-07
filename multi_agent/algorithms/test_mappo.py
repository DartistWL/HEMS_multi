# -*- coding: utf-8 -*-
# @Time        : 2026/4/3 11:33
# @Author      : Valiantir
# @File        : test_mappo.py
# @Version     : 1.0   
# Description  :

import numpy as np
from mappo import MAPPO


# ---------- 模拟一个简单的多智能体环境 ----------
class DummyMultiAgentEnv:
    def __init__(self, n_agents=3):
        self.n_agents = n_agents
        self.agents = [DummyAgent() for _ in range(n_agents)]

    def reset(self, mode='train', date_index=0):
        # 每个智能体返回一个字典状态（模仿你的真实环境）
        states = []
        for i in range(self.n_agents):
            state = {
                'agent_own_energy': np.random.rand(),
                'agent_battery': np.random.rand(),
                'price': np.random.rand(),
                'community_ess_soc': np.random.rand(),
                # 其他字段随便加，确保长度固定
            }
            states.append(state)
        return states

    def step(self, actions):
        # actions: list of dict, 每个智能体的动作字典
        next_states = self.reset()  # 模拟下一状态
        rewards = [np.random.rand() for _ in range(self.n_agents)]
        dones = [False for _ in range(self.n_agents)]
        info = {}
        return next_states, rewards, dones, info


class DummyAgent:
    def __init__(self):
        # 模拟动作空间配置（与 mappo 中保持一致）
        self.env = DummyActionSpace()


class DummyActionSpace:
    def __init__(self):
        self.action_space = {
            'charge_power': (0, 5, 10),
            'discharge_power': (0, 5, 10),
            # 注意：mappo 中还会自动添加 'community_ess_power'
        }


# ---------- 测试 MAPPO ----------
if __name__ == "__main__":
    env = DummyMultiAgentEnv(n_agents=3)

    # 初始化 MAPPO，关闭奖励归一化（避免 reward_running_stats 问题）
    agent = MAPPO(
        env=env,
        n_agents=3,
        use_state_normalization=True,
        use_popart=True,
        use_reward_normalization=False,  # 关键：设为 False
        reward_scale=10.0,
        hidden_dim=64,
        batch_size=32
    )

    print("MAPPO 初始化成功！")

    # 收集一个 episode 的经验
    local_states = env.reset()
    for step in range(5):
        actions, log_probs = agent.select_actions(local_states)
        next_states, rewards, dones, _ = env.step(actions)

        # 构造全局状态（直接拼接局部状态向量）
        local_vectors = [agent._state_dict_to_vector(s) for s in local_states]
        global_state = np.concatenate(local_vectors)
        next_global = np.concatenate([agent._state_dict_to_vector(s) for s in next_states])

        agent.store_transition(
            local_states, global_state, actions, log_probs,
            rewards, next_states, next_global, dones
        )
        local_states = next_states
        if any(dones):
            break

    # 更新网络
    stats = agent.update()
    print("更新统计信息:", stats)
    print("测试通过！")