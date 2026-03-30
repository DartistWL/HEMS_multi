"""
COMA: Counterfactual Multi-Agent Policy Gradients（改进版）
- 状态归一化、奖励缩放
- 解析反事实基线（对每维动作求期望）、advantage 归一化、梯度裁剪
- Critic 学习率低于 Actor
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from multi_agent.algorithms.action_utils import (
    get_action_space_config,
    get_dim_sizes,
    get_ordered_action_keys,
    state_dict_to_vector,
)
from multi_agent.algorithms.common import StateRunningStats, RewardScaler
from multi_agent.algorithms.mappo import PopArtNormalizer


class COMAActor(nn.Module):
    """与 MAPPO 一致的因子化 Actor：共享骨干 + 每维度一个分支。"""

    def __init__(self, state_dim, hidden_dim, action_space_config):
        super(COMAActor, self).__init__()
        self.action_space_config = action_space_config
        keys = get_ordered_action_keys(action_space_config)
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.branches = nn.ModuleDict()
        for k in keys:
            n = len(action_space_config[k])
            self.branches[k] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, n),
            )

    def forward(self, x):
        feat = self.shared(x)
        logits = {k: self.branches[k](feat) for k in self.branches}
        return logits


class COMACritic(nn.Module):
    """集中式 Critic：输入全局状态 + 联合动作 one-hot -> Q(s,u)。支持 Pop-Art 时最后一层单独以便调整权重。"""

    def __init__(self, global_state_dim, joint_action_onehot_dim, hidden_dim=128):
        super(COMACritic, self).__init__()
        input_dim = global_state_dim + joint_action_onehot_dim
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, global_state, joint_action_onehot):
        x = torch.cat([global_state, joint_action_onehot], dim=-1)
        h = self.shared(x)
        return self.output_layer(h).squeeze(-1)


def batch_joint_action_indices_to_onehot(indices_batch, dim_sizes, device):
    """(T, 3, 7) -> (T, 135)."""
    n_agents = 3
    dim_sizes = list(dim_sizes)
    total_dim = sum(dim_sizes) * n_agents
    T = len(indices_batch)
    out = np.zeros((T, total_dim), dtype=np.float32)
    for t in range(T):
        offset = 0
        for i in range(n_agents):
            idx = indices_batch[t][i]
            for d, size in enumerate(dim_sizes):
                aid = int(idx[d]) if hasattr(idx[d], '__int__') else idx[d]
                aid = min(max(0, aid), size - 1)
                out[t, offset + aid] = 1.0
                offset += size
    return torch.FloatTensor(out).to(device)


def build_onehot_vary_agent_dim(indices_batch, agent_i, dim_d, dim_sizes, device):
    """
    构建 batch：仅 agent_i 的第 dim_d 维在 0..dim_sizes[dim_d]-1 上变化，用于解析反事实基线。
    """
    n_agents = 3
    dim_sizes = list(dim_sizes)
    n_ad = dim_sizes[dim_d]
    T = len(indices_batch)
    total_dim = sum(dim_sizes) * n_agents
    out = np.zeros((T * n_ad, total_dim), dtype=np.float32)
    for t in range(T):
        base = [list(indices_batch[t][j]) for j in range(n_agents)]
        for a in range(n_ad):
            row = [list(base[j]) for j in range(n_agents)]
            row[agent_i][dim_d] = a
            offset = 0
            for i in range(n_agents):
                for d, size in enumerate(dim_sizes):
                    aid = min(max(0, int(row[i][d])), size - 1)
                    out[t * n_ad + a, offset + aid] = 1.0
                    offset += size
    return torch.FloatTensor(out).to(device)


class COMA:
    """
    COMA 改进版：状态归一化、奖励缩放、解析反事实基线、advantage 归一化、梯度裁剪。
    """

    def __init__(self, env, n_agents=3, hidden_dim=128, gamma=0.96,
                 lr_actor=3e-4, lr_critic=5e-5,
                 use_actor_state_normalization=False, use_critic_state_normalization=True,
                 reward_scale=10.0, use_reward_normalization=False,
                 max_grad_norm=10.0, td_target_clip=200.0, ent_coef=0.01, n_actor_epochs=2,
                 use_popart=True, use_advantage_normalization=True, device=None):
        self.env = env
        self.n_agents = n_agents
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.td_target_clip = td_target_clip  # 裁剪 TD target，避免单次极差 episode 导致 critic 大更新、loss 尖峰
        self.ent_coef = ent_coef  # 熵系数，鼓励探索（与 MAPPO 一致）
        self.n_actor_epochs = max(1, n_actor_epochs)  # Actor 每批数据重复更新次数，降低方差
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.action_space_config = get_action_space_config(env)
        self.dim_sizes = get_dim_sizes(self.action_space_config)
        self.joint_action_onehot_dim = sum(self.dim_sizes) * n_agents

        sample_states = env.reset(mode='train', date_index=0)
        self.local_state_dim = len(state_dict_to_vector(sample_states[0]))
        # Critic 输入与 MAPPO 一致：concat(各 agent 的 local state)，不再用 env 的 global_state
        self.critic_global_dim = self.local_state_dim * n_agents

        self.actors = nn.ModuleList([
            COMAActor(self.local_state_dim, hidden_dim, self.action_space_config).to(self.device)
            for _ in range(n_agents)
        ])
        self.critic = COMACritic(
            self.critic_global_dim,
            self.joint_action_onehot_dim,
            hidden_dim,
        ).to(self.device)

        self.actor_optimizers = [torch.optim.Adam(a.parameters(), lr=lr_actor) for a in self.actors]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.use_actor_state_normalization = use_actor_state_normalization
        self.use_critic_state_normalization = use_critic_state_normalization
        if use_actor_state_normalization or use_critic_state_normalization:
            self.local_running_stats = StateRunningStats(self.local_state_dim, self.device)
        else:
            self.local_running_stats = None

        self.reward_scale = reward_scale
        self.use_reward_normalization = use_reward_normalization
        self.reward_scaler = RewardScaler(self.device, scale=reward_scale, use_running_stats=use_reward_normalization)

        self.use_popart = use_popart
        if self.use_popart:
            self.popart_normalizer = PopArtNormalizer(device=self.device, beta=1e-4)
        else:
            self.popart_normalizer = None

        self.use_advantage_normalization = use_advantage_normalization

        self.reset_buffer()

    def _state_to_vector(self, state_dict):
        return state_dict_to_vector(state_dict)

    def _normalize_local(self, vec):
        if self.local_running_stats is None:
            return vec
        return self.local_running_stats.normalize_np(vec)

    def _get_local_vectors(self, local_states):
        return np.array([self._state_to_vector(s) for s in local_states], dtype=np.float32)

    def reset_buffer(self):
        self.buffer = {
            'local_states': [],
            'global_states': [],
            'joint_action_indices': [],
            'rewards': [],
            'next_local_states': [],
            'next_global_states': [],
            'dones': [],
        }

    def select_actions(self, local_states, deterministic=False):
        actions_dict = []
        action_indices = []
        keys = get_ordered_action_keys(self.action_space_config)
        for i, state in enumerate(local_states):
            vec = self._state_to_vector(state)
            if self.use_actor_state_normalization:
                vec = self._normalize_local(vec)
            x = torch.FloatTensor(vec).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits_dict = self.actors[i](x)
            ad, idx = {}, []
            for k in keys:
                logits = logits_dict[k]
                # 防止 logits 过大导致 softmax 溢出产生 nan/inf
                logits = torch.clamp(logits, -20.0, 20.0)
                probs = F.softmax(logits, dim=-1)
                if not torch.isfinite(probs).all() or (probs < 0).any():
                    probs = torch.ones_like(probs) / probs.size(-1)
                a = torch.argmax(probs, dim=-1).item() if deterministic else torch.multinomial(probs, 1).item()
                idx.append(a)
                ad[k] = self.action_space_config[k][a]
            actions_dict.append(ad)
            action_indices.append(idx)
        return actions_dict, action_indices

    def store_transition(self, local_states, global_state, action_indices, rewards,
                        next_local_states, next_global_state, dones):
        ls = self._get_local_vectors(local_states)
        nls = self._get_local_vectors(next_local_states)
        team_r = sum(rewards)
        self.reward_scaler.update([team_r])
        scaled_r = self.reward_scaler.scale_reward(team_r)
        use_any_norm = self.use_actor_state_normalization or self.use_critic_state_normalization
        if use_any_norm:
            ls_stack = torch.FloatTensor(ls).to(self.device)
            self.local_running_stats.update(ls_stack)
        if self.use_actor_state_normalization:
            ls_for_actor = self._normalize_local(ls)
            nls_for_actor = np.array([self._normalize_local(self._state_to_vector(s)) for s in next_local_states], dtype=np.float32)
        else:
            ls_for_actor = ls
            nls_for_actor = nls
        if self.use_critic_state_normalization:
            ls_for_critic = self._normalize_local(ls)
            nls_for_critic = np.array([self._normalize_local(self._state_to_vector(s)) for s in next_local_states], dtype=np.float32)
        else:
            ls_for_critic = ls
            nls_for_critic = nls
        # Critic 输入 = concat(各 agent 的 local state)，与 MAPPO 一致
        global_for_critic = np.ravel(ls_for_critic).astype(np.float32)
        next_global_for_critic = np.ravel(nls_for_critic).astype(np.float32)
        self.buffer['local_states'].append(ls_for_actor)
        self.buffer['global_states'].append(global_for_critic)
        self.buffer['joint_action_indices'].append([list(a) for a in action_indices])
        self.buffer['rewards'].append(scaled_r)
        self.buffer['next_local_states'].append(nls_for_actor)
        self.buffer['next_global_states'].append(next_global_for_critic)
        self.buffer['dones'].append(all(dones))

    def update(self):
        if len(self.buffer['rewards']) == 0:
            return None
        T = len(self.buffer['rewards'])
        global_states = np.array(self.buffer['global_states'])
        next_global_states = np.array(self.buffer['next_global_states'])
        rewards = np.array(self.buffer['rewards'])
        dones = np.array(self.buffer['dones'])
        keys = get_ordered_action_keys(self.action_space_config)

        gs = torch.FloatTensor(global_states).to(self.device)
        ngs = torch.FloatTensor(next_global_states).to(self.device)

        joint_onehot = batch_joint_action_indices_to_onehot(
            self.buffer['joint_action_indices'], self.dim_sizes, self.device)

        with torch.no_grad():
            next_Q_vals = np.zeros(T, dtype=np.float32)
            if T > 1:
                next_Q_raw = self.critic(ngs[1:], joint_onehot[1:]).cpu().numpy()
                next_Q_raw = np.where(np.isfinite(next_Q_raw), next_Q_raw, 0.0)
                if self.use_popart and self.popart_normalizer is not None:
                    next_Q_raw_t = torch.FloatTensor(next_Q_raw).to(self.device)
                    next_Q_vals[:T - 1] = self.popart_normalizer.denormalize(next_Q_raw_t).cpu().numpy()
                else:
                    next_Q_vals[:T - 1] = next_Q_raw
        targets = torch.FloatTensor(
            rewards + self.gamma * (1.0 - dones.astype(np.float32)) * next_Q_vals
        ).to(self.device)
        targets = torch.where(torch.isfinite(targets), targets, torch.zeros_like(targets))
        # 裁剪 TD target，避免单次极差 episode 导致 target 极大/极小、critic_loss 尖峰与策略震荡
        if self.td_target_clip is not None and self.td_target_clip > 0:
            targets = torch.clamp(targets, -self.td_target_clip, self.td_target_clip)

        # Pop-Art：用当前 batch 的 target 更新统计，归一化 target；Critic 预测归一化后的 Q
        if self.use_popart and self.popart_normalizer is not None:
            updated = self.popart_normalizer.update(targets)
            if updated and self.popart_normalizer.old_mean is not None:
                self.popart_normalizer.adjust_network_weights(self.critic.output_layer)
            targets_for_loss = self.popart_normalizer.normalize(targets)
        else:
            targets_for_loss = targets

        Q = self.critic(gs, joint_onehot)
        critic_loss = F.mse_loss(Q, targets_for_loss)

        baselines_per_agent = []
        for i in range(self.n_agents):
            baseline_i = torch.zeros(T, device=self.device)
            for d in range(len(keys)):
                n_ad = self.dim_sizes[d]
                onehot_batch = build_onehot_vary_agent_dim(
                    self.buffer['joint_action_indices'], i, d, self.dim_sizes, self.device)
                gs_rep = gs.unsqueeze(1).expand(T, n_ad, -1).reshape(T * n_ad, -1)
                Q_id = self.critic(gs_rep, onehot_batch).reshape(T, n_ad)
                if self.use_popart and self.popart_normalizer is not None:
                    Q_id = self.popart_normalizer.denormalize(Q_id)
                local_i = np.array([self.buffer['local_states'][t][i] for t in range(T)])
                x_i = torch.FloatTensor(local_i).to(self.device)
                with torch.no_grad():
                    logits_d = self.actors[i](x_i)
                logits_d_safe = torch.clamp(logits_d[keys[d]], -20.0, 20.0)
                probs_d = F.softmax(logits_d_safe, dim=-1)
                baseline_i += (Q_id * probs_d).sum(dim=1)
            baselines_per_agent.append(baseline_i)
        baseline_avg = torch.stack(baselines_per_agent, dim=0).mean(dim=0)
        # Advantage 用真实尺度 Q（Pop-Art 时需反归一化）
        Q_real = self.popart_normalizer.denormalize(Q) if (self.use_popart and self.popart_normalizer is not None) else Q
        advantages = (Q_real - baseline_avg).detach()
        advantages = torch.where(torch.isfinite(advantages), advantages, torch.zeros_like(advantages))
        if self.use_advantage_normalization:
            adv_std = advantages.std()
            if adv_std > 1e-8:
                advantages = (advantages - advantages.mean()) / adv_std

        actor_loss = 0.0
        for i in range(self.n_agents):
            log_probs_list = []
            ent_list = []
            for t in range(T):
                local_s = self.buffer['local_states'][t][i]
                x = torch.FloatTensor(local_s).unsqueeze(0).to(self.device)
                logits_d = self.actors[i](x)
                lp = 0.0
                for d, k in enumerate(keys):
                    aid = self.buffer['joint_action_indices'][t][i][d]
                    logits_k = torch.clamp(logits_d[k], -20.0, 20.0)
                    logp = F.log_softmax(logits_k, dim=-1)
                    lp = lp + logp[0, aid]
                    probs_k = F.softmax(logits_k, dim=-1)
                    ent_list.append(-(probs_k * logp).sum(dim=-1))
                log_probs_list.append(lp)
            log_probs_t = torch.stack(log_probs_list)
            actor_loss = actor_loss - (log_probs_t * advantages).mean()
            if self.ent_coef != 0 and ent_list:
                entropy_t = torch.stack(ent_list).mean()
                actor_loss = actor_loss - self.ent_coef * entropy_t
        actor_loss = actor_loss / self.n_agents

        # 若出现 nan 则跳过本次更新，避免污染参数（但仍返回真实 loss 便于日志诊断）
        if not torch.isfinite(critic_loss) or not torch.isfinite(actor_loss):
            return {
                'actor_loss': float('nan') if not torch.isfinite(actor_loss) else actor_loss.item(),
                'critic_loss': float('nan') if not torch.isfinite(critic_loss) else critic_loss.item(),
            }

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        # Actor 多 epoch 更新（同一批 advantage，用当前策略重算 log_prob 再 backward，类似 PPO）
        for _ in range(self.n_actor_epochs):
            actor_loss = 0.0
            for i in range(self.n_agents):
                log_probs_list = []
                ent_list = []
                for t in range(T):
                    local_s = self.buffer['local_states'][t][i]
                    x = torch.FloatTensor(local_s).unsqueeze(0).to(self.device)
                    logits_d = self.actors[i](x)
                    lp = 0.0
                    for d, k in enumerate(keys):
                        aid = self.buffer['joint_action_indices'][t][i][d]
                        logits_k = torch.clamp(logits_d[k], -20.0, 20.0)
                        logp = F.log_softmax(logits_k, dim=-1)
                        lp = lp + logp[0, aid]
                        probs_k = F.softmax(logits_k, dim=-1)
                        ent_list.append(-(probs_k * logp).sum(dim=-1))
                    log_probs_list.append(lp)
                log_probs_t = torch.stack(log_probs_list)
                actor_loss = actor_loss - (log_probs_t * advantages).mean()
                if self.ent_coef != 0 and ent_list:
                    entropy_t = torch.stack(ent_list).mean()
                    actor_loss = actor_loss - self.ent_coef * entropy_t
            actor_loss = actor_loss / self.n_agents
            for j in range(self.n_agents):
                self.actor_optimizers[j].zero_grad()
            actor_loss.backward()
            for j in range(self.n_agents):
                torch.nn.utils.clip_grad_norm_(self.actors[j].parameters(), self.max_grad_norm)
                self.actor_optimizers[j].step()

        return {'actor_loss': actor_loss.item(), 'critic_loss': critic_loss.item()}

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        for i, actor in enumerate(self.actors):
            torch.save(actor.state_dict(), os.path.join(save_dir, f'coma_actor_{i}.pth'))
        torch.save(self.critic.state_dict(), os.path.join(save_dir, 'coma_critic.pth'))
        if self.use_actor_state_normalization or self.use_critic_state_normalization:
            torch.save({
                'local_mean': self.local_running_stats.mean.cpu(),
                'local_std': self.local_running_stats.std.cpu(),
                'local_count': self.local_running_stats.count,
            }, os.path.join(save_dir, 'coma_stats.pth'))
        if self.use_popart and self.popart_normalizer is not None:
            torch.save({
                'popart_mean': self.popart_normalizer.mean.cpu(),
                'popart_std': self.popart_normalizer.std.cpu(),
                'popart_running_count': self.popart_normalizer.running_count,
            }, os.path.join(save_dir, 'coma_popart.pth'))

    def load(self, load_dir):
        for i, actor in enumerate(self.actors):
            p = os.path.join(load_dir, f'coma_actor_{i}.pth')
            if os.path.exists(p):
                actor.load_state_dict(torch.load(p, map_location=self.device))
        p = os.path.join(load_dir, 'coma_critic.pth')
        if os.path.exists(p):
            self.critic.load_state_dict(torch.load(p, map_location=self.device))
        if self.use_actor_state_normalization or self.use_critic_state_normalization:
            sp = os.path.join(load_dir, 'coma_stats.pth')
            if os.path.exists(sp):
                st = torch.load(sp, map_location=self.device)
                self.local_running_stats.mean = st['local_mean'].to(self.device)
                self.local_running_stats.std = st['local_std'].to(self.device)
                self.local_running_stats.count = st['local_count']
        if self.use_popart and self.popart_normalizer is not None:
            pp = os.path.join(load_dir, 'coma_popart.pth')
            if os.path.exists(pp):
                pt = torch.load(pp, map_location=self.device)
                self.popart_normalizer.mean = pt['popart_mean'].to(self.device)
                self.popart_normalizer.std = pt['popart_std'].to(self.device)
                self.popart_normalizer.running_mean = self.popart_normalizer.mean.clone()
                self.popart_normalizer.running_std = self.popart_normalizer.std.clone()
                self.popart_normalizer.running_count = pt.get('popart_running_count', 0)
