"""
QMIX 改进版：off-policy + replay buffer + target 网络 + 状态/奖励归一化 + TD 裁剪 + 梯度裁剪。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random

from multi_agent.algorithms.action_utils import (
    get_action_space_config,
    get_dim_sizes,
    get_ordered_action_keys,
    state_dict_to_vector,
)
from multi_agent.algorithms.common import StateRunningStats, RewardScaler


class QMIXAgentNet(nn.Module):
    """单智能体：局部状态 -> 7 个 Q head + 7 个策略分支（用于 epsilon-greedy 时取 argmax）。"""

    def __init__(self, state_dim, hidden_dim, dim_sizes):
        super(QMIXAgentNet, self).__init__()
        self.dim_sizes = dim_sizes
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.q_heads = nn.ModuleList([nn.Linear(hidden_dim, d) for d in dim_sizes])
        self.pi_branches = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, d))
            for d in dim_sizes
        ])

    def forward(self, x):
        feat = self.shared(x)
        q_list = [self.q_heads[d](feat) for d in range(len(self.dim_sizes))]
        pi_list = [self.pi_branches[d](feat) for d in range(len(self.dim_sizes))]
        return q_list, pi_list

    def q_values_for_actions(self, x, action_indices):
        q_list, _ = self.forward(x)
        batch = x.size(0)
        if isinstance(action_indices, (list, tuple)):
            arr = np.array(action_indices)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
        else:
            arr = np.array(action_indices)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
        if arr.shape[0] != batch:
            arr = np.broadcast_to(arr, (batch, arr.shape[1]))
        q = 0.0
        for d, qd in enumerate(q_list):
            aid = torch.LongTensor(arr[:, d]).to(x.device)
            aid = torch.clamp(aid, 0, self.dim_sizes[d] - 1)
            q = q + qd.gather(1, aid.unsqueeze(1)).squeeze(1)
        return q

    def greedy_actions(self, x):
        """返回 greedy 动作索引 (batch, 7)。"""
        _, pi_list = self.forward(x)
        return torch.stack([torch.argmax(pi_list[d], dim=-1) for d in range(len(self.dim_sizes))], dim=1)


class QMIXMixingNet(nn.Module):
    """单调混合: Q_tot = sum_i w_i(s_global)*Q_i, w_i = softplus(hyper(s)) >= 0。"""

    def __init__(self, n_agents, global_state_dim, hidden_dim=64):
        super(QMIXMixingNet, self).__init__()
        self.n_agents = n_agents
        self.global_state_dim = global_state_dim
        self.hyper = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_agents),
        )

    def forward(self, agent_qs, global_states):
        w = F.softplus(self.hyper(global_states))
        return (agent_qs * w).sum(dim=1)


class SumTree:
    """用于优先经验回放的求和树：O(log n) 按优先级采样、O(log n) 更新单条优先级。"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.node_count = max(1, 2 * capacity - 1)  # 叶子在 [capacity-1, 2*capacity-2]
        self.tree = np.zeros(self.node_count, dtype=np.float64)

    def _leaf_idx(self, slot):
        return self.capacity - 1 + slot

    def add(self, slot, priority):
        idx = self._leaf_idx(slot)
        self.tree[idx] = max(priority, 1e-6)
        while idx > 0:
            idx = (idx - 1) // 2
            left, right = 2 * idx + 1, 2 * idx + 2
            self.tree[idx] = self.tree[left] + (self.tree[right] if right < self.node_count else 0.0)

    def get(self, slot):
        return self.tree[self._leaf_idx(slot)]

    def total(self):
        return self.tree[0] if self.node_count > 0 else 0.0

    def sample(self, batch_size):
        total = self.total()
        if total <= 0:
            return np.random.randint(0, self.capacity, size=batch_size)
        slots = []
        for _ in range(batch_size):
            u = np.random.uniform(0, total)
            idx = 0
            while idx < self.capacity - 1:
                left = 2 * idx + 1
                right = left + 1
                if u < self.tree[left]:
                    idx = left
                else:
                    u -= self.tree[left]
                    idx = right
            slot = idx - (self.capacity - 1)
            slots.append(min(slot, self.capacity - 1))
        return np.array(slots)


class ReplayBuffer:
    """存 (local_states, global_state, joint_action_indices, reward, next_*, done)。"""

    def __init__(self, capacity, n_agents=3, local_dim=0, global_dim=0):
        self.capacity = capacity
        self.n_agents = n_agents
        self.local_dim = local_dim
        self.global_dim = global_dim
        self.local_states = np.zeros((capacity, n_agents, local_dim), dtype=np.float32)
        self.global_states = np.zeros((capacity, global_dim), dtype=np.float32)
        self.joint_actions = np.zeros((capacity, n_agents, 7), dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_local_states = np.zeros((capacity, n_agents, local_dim), dtype=np.float32)
        self.next_global_states = np.zeros((capacity, global_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def push(self, local_states, global_state, joint_action_indices, reward, next_local_states, next_global_state, done):
        for i in range(self.n_agents):
            self.local_states[self.ptr, i] = local_states[i]
            self.next_local_states[self.ptr, i] = next_local_states[i]
            for d in range(7):
                self.joint_actions[self.ptr, i, d] = joint_action_indices[i][d]
        self.global_states[self.ptr] = global_state
        self.next_global_states[self.ptr] = next_global_state
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.local_states[idx],
            self.global_states[idx],
            self.joint_actions[idx],
            self.rewards[idx],
            self.next_local_states[idx],
            self.next_global_states[idx],
            self.dones[idx],
        )


class PrioritizedReplayBuffer(ReplayBuffer):
    """优先经验回放：按 |TD error| 优先级采样，并支持重要性采样权重。"""

    def __init__(self, capacity, n_agents=3, local_dim=0, global_dim=0, alpha=0.6):
        super().__init__(capacity, n_agents, local_dim, global_dim)
        self.alpha = alpha
        self.priorities = np.zeros(capacity, dtype=np.float64)
        self.sum_tree = SumTree(capacity)
        self.max_priority = 1.0

    def push(self, local_states, global_state, joint_action_indices, reward, next_local_states, next_global_state, done):
        super().push(local_states, global_state, joint_action_indices, reward, next_local_states, next_global_state, done)
        slot = (self.ptr - 1) % self.capacity if self.ptr > 0 else self.capacity - 1
        self.sum_tree.add(slot, self.max_priority)

    def sample(self, batch_size):
        total = self.sum_tree.total()
        if total <= 0:
            idx = np.random.randint(0, max(1, self.size), size=batch_size)
        else:
            idx = self.sum_tree.sample(batch_size)
            idx = np.clip(idx, 0, self.size - 1)
        return (
            self.local_states[idx],
            self.global_states[idx],
            self.joint_actions[idx],
            self.rewards[idx],
            self.next_local_states[idx],
            self.next_global_states[idx],
            self.dones[idx],
            idx,
        )

    def get_priorities(self, indices):
        return np.array([self.sum_tree.get(i) for i in indices], dtype=np.float64)

    # def update_priorities(self, indices, priorities):
    #     priorities = np.asarray(priorities).flatten()
    #     priorities = np.where(np.isfinite(priorities), priorities, 0.0)
    #     priorities = priorities + 1e-6
    #     for i, idx in enumerate(indices):
    #         self.sum_tree.add(int(idx), float(priorities[i]))
    #     self.max_priority = max(self.max_priority, float(np.max(priorities)))
    def update_priorities(self, indices, priorities):
        priorities = np.asarray(priorities).flatten()
        priorities = np.where(np.isfinite(priorities), priorities, 0.0)
        priorities = priorities + 1e-6
        for i, idx in enumerate(indices):
            # 确保 priority 是标量
            p = priorities[i].item() if hasattr(priorities[i], 'item') else float(priorities[i])
            self.sum_tree.add(int(idx), p)
        self.max_priority = max(self.max_priority, float(np.max(priorities)))


def _copy_module(module):
    return type(module)(*module.init_args) if hasattr(module, 'init_args') else None


class QMIX:
    """
    QMIX 改进版：状态/奖励归一化、replay buffer、target 网络（软更新）、TD 裁剪、梯度裁剪。
    Off-policy：epsilon-greedy 采集，从 buffer 采样更新。
    """

    def __init__(self, env, n_agents=3, hidden_dim=128, gamma=0.96, lr=1e-4,
                 use_actor_state_normalization=False, use_critic_state_normalization=True,
                 reward_scale=10.0, use_reward_normalization=False,
                 use_vdn=False,
                 vdn_aux_coef=0.0,
                 use_prioritized_replay=False,
                 per_alpha=0.6,
                 per_beta_start=0.4,
                 per_beta_end=1.0,
                 per_beta_episodes=500,
                 replay_capacity=50000, batch_size=64, target_tau=0.005,
                 td_target_clip=500.0, max_grad_norm=10.0,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_episodes=500,
                 device=None):
        self.env = env
        self.n_agents = n_agents
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_tau = target_tau
        self.td_target_clip = td_target_clip
        self.max_grad_norm = max_grad_norm
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.use_vdn = use_vdn
        self.vdn_aux_coef = float(vdn_aux_coef) if not use_vdn else 0.0
        self.use_prioritized_replay = use_prioritized_replay
        self.per_alpha = per_alpha
        self.per_beta_start = per_beta_start
        self.per_beta_end = per_beta_end
        self.per_beta_episodes = per_beta_episodes
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.action_space_config = get_action_space_config(env)
        self.dim_sizes = get_dim_sizes(self.action_space_config)
        keys = get_ordered_action_keys(self.action_space_config)

        sample_states = env.reset(mode='train', date_index=0)
        self.local_state_dim = len(state_dict_to_vector(sample_states[0]))
        # Mixing 网络输入与 MAPPO 一致：concat(各 agent 的 local state)，不再用 env 的 global_state
        self.mixing_state_dim = self.local_state_dim * n_agents

        self.agent_nets = nn.ModuleList([
            QMIXAgentNet(self.local_state_dim, hidden_dim, self.dim_sizes).to(self.device)
            for _ in range(n_agents)
        ])
        if use_vdn:
            self.mixing_net = None
            self.target_mixing_net = None
        else:
            self.mixing_net = QMIXMixingNet(n_agents, self.mixing_state_dim, hidden_dim=64).to(self.device)
            self.target_mixing_net = QMIXMixingNet(n_agents, self.mixing_state_dim, hidden_dim=64).to(self.device)
            self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())

        self.target_agent_nets = nn.ModuleList([
            QMIXAgentNet(self.local_state_dim, hidden_dim, self.dim_sizes).to(self.device)
            for _ in range(n_agents)
        ])
        for i in range(n_agents):
            self.target_agent_nets[i].load_state_dict(self.agent_nets[i].state_dict())
        if not use_vdn:
            self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())

        params = list(self.agent_nets.parameters())
        if not use_vdn:
            params += list(self.mixing_net.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

        self.use_actor_state_normalization = use_actor_state_normalization
        self.use_critic_state_normalization = use_critic_state_normalization
        if use_actor_state_normalization or use_critic_state_normalization:
            self.local_running_stats = StateRunningStats(self.local_state_dim, self.device)
        else:
            self.local_running_stats = None

        self.reward_scale = reward_scale
        self.use_reward_normalization = use_reward_normalization
        self.reward_scaler = RewardScaler(self.device, scale=reward_scale, use_running_stats=use_reward_normalization)

        if use_prioritized_replay:
            self.replay = PrioritizedReplayBuffer(replay_capacity, n_agents, self.local_state_dim, self.mixing_state_dim, alpha=per_alpha)
        else:
            self.replay = ReplayBuffer(replay_capacity, n_agents, self.local_state_dim, self.mixing_state_dim)
        self._episode_count = 0

    def set_learning_rate(self, lr):
        """训练时可按 episode 衰减学习率，利于后期收敛到稳态。"""
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def _state_to_vector(self, state_dict):
        return state_dict_to_vector(state_dict)

    def _normalize_local(self, vec):
        if self.local_running_stats is None:
            return vec
        return self.local_running_stats.normalize_np(vec)

    def _get_epsilon(self):
        if self._episode_count >= self.epsilon_decay_episodes:
            return self.epsilon_end
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * self._episode_count / self.epsilon_decay_episodes

    def reset_buffer(self):
        pass

    def select_actions(self, local_states, deterministic=False):
        keys = get_ordered_action_keys(self.action_space_config)
        actions_dict = []
        action_indices = []
        eps = 0.0 if deterministic else self._get_epsilon()
        for i, state in enumerate(local_states):
            vec = self._state_to_vector(state)
            if self.use_actor_state_normalization:
                vec = self._normalize_local(vec)
            x = torch.FloatTensor(vec).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, pi_list = self.agent_nets[i](x)
            ad, idx = {}, []
            for d, k in enumerate(keys):
                n_a = self.dim_sizes[d]
                if not deterministic and random.random() < eps:
                    a = random.randint(0, n_a - 1)
                else:
                    a = torch.argmax(pi_list[d], dim=-1).item()
                idx.append(a)
                ad[k] = self.action_space_config[k][a]
            actions_dict.append(ad)
            action_indices.append(idx)
        return actions_dict, action_indices

    def store_transition(self, local_states, global_state, action_indices, rewards,
                        next_local_states, next_global_state, dones):
        team_r = sum(rewards)
        self.reward_scaler.update([team_r])
        scaled_r = self.reward_scaler.scale_reward(team_r)
        ls = np.array([self._state_to_vector(s) for s in local_states], dtype=np.float32)
        nls = np.array([self._state_to_vector(s) for s in next_local_states], dtype=np.float32)
        use_any_norm = self.use_actor_state_normalization or self.use_critic_state_normalization
        if use_any_norm and self.local_running_stats is not None:
            ls_stack = torch.FloatTensor(ls).to(self.device)
            self.local_running_stats.update(ls_stack)
        if self.use_actor_state_normalization:
            ls_for_agent = np.array([self._normalize_local(ls[i]) for i in range(self.n_agents)], dtype=np.float32)
            nls_for_agent = np.array([self._normalize_local(self._state_to_vector(s)) for s in next_local_states], dtype=np.float32)
        else:
            ls_for_agent = ls.copy()
            nls_for_agent = nls.copy()
        if self.use_critic_state_normalization:
            ls_for_mixing = np.array([self._normalize_local(ls[i]) for i in range(self.n_agents)], dtype=np.float32)
            nls_for_mixing = np.array([self._normalize_local(self._state_to_vector(s)) for s in next_local_states], dtype=np.float32)
        else:
            ls_for_mixing = ls.copy()
            nls_for_mixing = nls.copy()
        global_for_mixing = np.ravel(ls_for_mixing).astype(np.float32)
        next_global_for_mixing = np.ravel(nls_for_mixing).astype(np.float32)
        self.replay.push(ls_for_agent, global_for_mixing, action_indices, scaled_r, nls_for_agent, next_global_for_mixing, all(dones))

    def _q_tot(self, agent_nets, mixing_net, local_batch, global_batch, joint_actions, return_agent_qs=False):
        """local_batch (B, n_agents, local_dim), joint_actions (B, n_agents, 7).
        若 return_agent_qs=True 则返回 (q_tot, agent_qs)，否则只返回 q_tot（兼容旧调用）。"""
        B = local_batch.shape[0]
        agent_qs = []
        for i in range(self.n_agents):
            xs = torch.FloatTensor(local_batch[:, i]).to(self.device)
            idx_i = joint_actions[:, i]
            q_i = agent_nets[i].q_values_for_actions(xs, idx_i)
            if q_i.dim() == 0:
                q_i = q_i.unsqueeze(0)
            agent_qs.append(q_i)
        agent_qs = torch.stack(agent_qs, dim=1)
        if self.use_vdn:
            q_tot = agent_qs.sum(dim=1)
        else:
            gs = torch.FloatTensor(global_batch).to(self.device)
            q_tot = mixing_net(agent_qs, gs)
        if return_agent_qs:
            return q_tot, agent_qs
        return q_tot

    def update(self):
        if self.replay.size < self.batch_size:
            return None
        if self.use_prioritized_replay:
            ls, gs, ja, rewards, nls, ngs, dones, indices = self.replay.sample(self.batch_size)
        else:
            ls, gs, ja, rewards, nls, ngs, dones = self.replay.sample(self.batch_size)
            indices = None

        need_aux = not self.use_vdn and self.vdn_aux_coef > 0
        if need_aux:
            q_tot, agent_qs = self._q_tot(self.agent_nets, self.mixing_net, ls, gs, ja, return_agent_qs=True)
        else:
            q_tot = self._q_tot(self.agent_nets, self.mixing_net, ls, gs, ja)

        with torch.no_grad():
            next_actions = []
            for i in range(self.n_agents):
                xs = torch.FloatTensor(nls[:, i]).to(self.device)
                next_actions.append(self.agent_nets[i].greedy_actions(xs))
            next_ja = torch.stack(next_actions, dim=1).cpu().numpy()
            next_q_tot = self._q_tot(self.target_agent_nets, self.target_mixing_net, nls, ngs, next_ja)
            targets = rewards + self.gamma * (1.0 - dones) * next_q_tot.cpu().numpy()
            targets = np.clip(targets, -self.td_target_clip, self.td_target_clip)
            targets = torch.FloatTensor(targets).to(self.device)

        td_errors = (q_tot.detach() - targets).abs().cpu().numpy()

        if self.use_prioritized_replay:
            total = self.replay.sum_tree.total()
            priorities = self.replay.get_priorities(indices)
            probs = priorities / (total + 1e-8)
            N = self.replay.size
            beta = self.per_beta_start + (self.per_beta_end - self.per_beta_start) * min(1.0, self._episode_count / max(1, self.per_beta_episodes))
            is_weights = np.power(N * probs + 1e-8, -beta)
            is_weights = is_weights / (is_weights.max() + 1e-8)
            is_weights = torch.FloatTensor(is_weights).to(self.device)
            td_loss = (is_weights * (q_tot - targets) ** 2).mean()
        else:
            td_loss = F.mse_loss(q_tot, targets)

        if need_aux:
            q_sum = agent_qs.sum(dim=1)
            aux_loss = F.mse_loss(q_tot, q_sum)
            loss = td_loss + self.vdn_aux_coef * aux_loss
        else:
            loss = td_loss

        if not torch.isfinite(loss):
            return {'td_loss': float('nan'), 'skip_nan': True}

        self.optimizer.zero_grad()
        loss.backward()
        params = list(self.agent_nets.parameters())
        if not self.use_vdn:
            params += list(self.mixing_net.parameters())
        torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        self.optimizer.step()

        # if self.use_prioritized_replay and indices is not None:
        #     new_priorities = np.power(np.abs(td_errors).flatten() + 1e-6, self.per_alpha)
        #     self.replay.update_priorities(indices, new_priorities)
        if self.use_prioritized_replay and indices is not None:
            td_errors_flat = np.abs(td_errors).flatten()
            # 确保不含 NaN/Inf
            td_errors_flat = np.nan_to_num(td_errors_flat, nan=0.0, posinf=1e-6, neginf=0.0)
            new_priorities = np.power(td_errors_flat + 1e-6, self.per_alpha)
            new_priorities = new_priorities.astype(np.float64).ravel()
            self.replay.update_priorities(indices, new_priorities)

        for i in range(self.n_agents):
            for p, pt in zip(self.agent_nets[i].parameters(), self.target_agent_nets[i].parameters()):
                pt.data.copy_(self.target_tau * p.data + (1 - self.target_tau) * pt.data)
        if not self.use_vdn:
            for p, pt in zip(self.mixing_net.parameters(), self.target_mixing_net.parameters()):
                pt.data.copy_(self.target_tau * p.data + (1 - self.target_tau) * pt.data)

        if need_aux:
            info = {'td_loss': td_loss.item(), 'aux_loss': aux_loss.item()}
        else:
            info = {'td_loss': td_loss.item()}
        return info

    def on_episode_end(self):
        self._episode_count += 1

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        for i, net in enumerate(self.agent_nets):
            torch.save(net.state_dict(), os.path.join(save_dir, f'qmix_agent_{i}.pth'))
        if not self.use_vdn:
            torch.save(self.mixing_net.state_dict(), os.path.join(save_dir, 'qmix_mixing.pth'))
        for i, net in enumerate(self.target_agent_nets):
            torch.save(net.state_dict(), os.path.join(save_dir, f'qmix_target_agent_{i}.pth'))
        if not self.use_vdn:
            torch.save(self.target_mixing_net.state_dict(), os.path.join(save_dir, 'qmix_target_mixing.pth'))
        if self.use_actor_state_normalization or self.use_critic_state_normalization:
            torch.save({
                'local_mean': self.local_running_stats.mean.cpu(),
                'local_std': self.local_running_stats.std.cpu(),
                'local_count': self.local_running_stats.count,
            }, os.path.join(save_dir, 'qmix_stats.pth'))

    def load(self, load_dir):
        for i, net in enumerate(self.agent_nets):
            p = os.path.join(load_dir, f'qmix_agent_{i}.pth')
            if os.path.exists(p):
                net.load_state_dict(torch.load(p, map_location=self.device))
        if not self.use_vdn:
            p = os.path.join(load_dir, 'qmix_mixing.pth')
            if os.path.exists(p):
                self.mixing_net.load_state_dict(torch.load(p, map_location=self.device))
        for i, net in enumerate(self.target_agent_nets):
            p = os.path.join(load_dir, f'qmix_target_agent_{i}.pth')
            if os.path.exists(p):
                net.load_state_dict(torch.load(p, map_location=self.device))
        if not self.use_vdn:
            p = os.path.join(load_dir, 'qmix_target_mixing.pth')
            if os.path.exists(p):
                self.target_mixing_net.load_state_dict(torch.load(p, map_location=self.device))
        if self.use_actor_state_normalization or self.use_critic_state_normalization:
            sp = os.path.join(load_dir, 'qmix_stats.pth')
            if os.path.exists(sp):
                st = torch.load(sp, map_location=self.device)
                self.local_running_stats.mean = st['local_mean'].to(self.device)
                self.local_running_stats.std = st['local_std'].to(self.device)
                self.local_running_stats.count = st.get('local_count', 0)
