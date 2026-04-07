"""
Multi-Agent Proximal Policy Optimization (MAPPO)
多智能体近端策略优化算法

实现CTDE（Centralized Training, Decentralized Execution）架构：
- 训练时：使用全局信息（共享Critic）
- 执行时：仅使用局部观测（独立Actor）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


class RunningStats:
    """运行统计信息，用于状态归一化（均值和标准差）"""

    def __init__(self, shape, device='cpu'):
        self.device = device
        self.mean = torch.zeros(shape, device=device)
        self.std = torch.ones(shape, device=device)
        self.count = 0

    def update(self, x):
        """更新统计信息（Welford在线算法）"""
        batch_mean = torch.mean(x, dim=0)
        batch_std = torch.std(x, dim=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        if total_count > 0:
            self.mean += delta * batch_count / total_count
            m_a = self.std ** 2 * self.count
            m_b = batch_std ** 2 * batch_count
            M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
            self.std = torch.sqrt(M2 / total_count)
            self.count = total_count


class ActorNetwork(nn.Module):
    """Actor网络：每个智能体独立，输出各动作维度的logits"""

    def __init__(self, state_dim, hidden_dim, action_space_config):
        super(ActorNetwork, self).__init__()
        # 共享特征提取层
        self.shared_backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 为每个动作维度创建一个分支（离散动作空间）
        self.action_branches = nn.ModuleDict()
        for action_name, action_values in action_space_config.items():
            if isinstance(action_values, (list, tuple)):
                action_dim = len(action_values)
                branch = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, action_dim)
                )
                self.action_branches[action_name] = branch

    def forward(self, state):
        """返回各动作分支的logits字典"""
        features = self.shared_backbone(state)
        action_logits = {}
        for action_name, branch in self.action_branches.items():
            action_logits[action_name] = branch(features)
        return action_logits


class PopArtNormalizer:
    """
    Pop-Art归一化器
    Preserving Outputs Precisely while Adaptively Rescaling Targets

    自适应地归一化targets，同时保持网络输出的一致性
    """

    def __init__(self, device='cpu', beta=1e-4):
        """
        初始化Pop-Art归一化器

        Args:
            device: 计算设备
            beta: 更新率（用于运行统计的指数移动平均）
        """
        self.device = device
        self.beta = beta  # 更新率（EMA系数）
        self.running_mean = torch.tensor(0.0, device=device)
        self.running_std = torch.tensor(1.0, device=device)
        self.running_count = 0
        self.mean = torch.tensor(0.0, device=device)
        self.std = torch.tensor(1.0, device=device)
        self.old_mean = None
        self.old_std = None

    def update(self, targets):
        """
        更新运行统计

        Args:
            targets: 目标值（returns），shape: (batch_size,)

        Returns:
            bool: 是否进行了更新（用于判断是否需要调整网络权重）
        """
        if isinstance(targets, np.ndarray):
            targets = torch.FloatTensor(targets).to(self.device)
        elif not isinstance(targets, torch.Tensor):
            targets = torch.FloatTensor([targets]).to(self.device)

        targets = targets.flatten()
        batch_mean = targets.mean().item()
        batch_std = targets.std().item()
        batch_count = len(targets)

        self.old_mean = self.mean.clone()
        self.old_std = self.std.clone()

        if batch_std < 1e-8:
            batch_std = 1.0

        updated = False
        if self.running_count == 0:
            self.running_mean = torch.tensor(batch_mean, device=self.device)
            self.running_std = torch.tensor(batch_std, device=self.device)
            updated = True
        else:
            alpha = min(1.0, self.beta * batch_count)
            delta_mean = batch_mean - self.running_mean.item()
            self.running_mean += alpha * delta_mean
            old_var = self.running_std.item() ** 2
            new_var = batch_std ** 2
            delta_var = new_var - old_var
            new_var_value = max(1e-8, old_var + alpha * delta_var)
            self.running_std = torch.tensor(new_var_value, device=self.device)
            self.running_std = torch.sqrt(self.running_std)
            updated = True

        self.running_count += batch_count
        self.mean = self.running_mean.clone()
        self.std = self.running_std.clone()
        return updated

    def normalize(self, targets):
        """
        归一化targets

        Args:
            targets: 目标值

        Returns:
            归一化后的targets
        """
        if isinstance(targets, np.ndarray):
            targets = torch.FloatTensor(targets).to(self.device)
        elif not isinstance(targets, torch.Tensor):
            targets = torch.FloatTensor([targets]).to(self.device)
        return (targets - self.mean) / (self.std + 1e-8)

    def denormalize(self, normalized_values):
        """
        反归一化值

        Args:
            normalized_values: 归一化后的值

        Returns:
            反归一化后的值
        """
        if isinstance(normalized_values, np.ndarray):
            normalized_values = torch.FloatTensor(normalized_values).to(self.device)
        elif not isinstance(normalized_values, torch.Tensor):
            normalized_values = torch.FloatTensor([normalized_values]).to(self.device)
        return normalized_values * (self.std + 1e-8) + self.mean

    def adjust_network_weights(self, last_layer):
        """
        当统计更新时，调整网络最后一层的权重和偏置以保持输出不变

        Pop-Art的核心：确保归一化参数更新后，网络输出的原始值（反归一化后）保持不变

        Args:
            last_layer: 网络的最后一层（nn.Linear）

        调整公式：
            old_output_denorm = normalized_output * old_std + old_mean
            new_output_denorm = normalized_output * new_std + new_mean

            要保持 old_output_denorm = new_output_denorm，需要：
            normalized_output * old_std + old_mean = normalized_output * new_std + new_mean
            因此：normalized_output = (new_mean - old_mean) / (old_std - new_std)

            但实际上，我们需要调整网络参数，使得：
            new_normalized_output = (old_normalized_output * old_std + old_mean - new_mean) / new_std
            new_normalized_output = old_normalized_output * (old_std / new_std) + (old_mean - new_mean) / new_std

            对于线性层 output = W * features + b：
            new_W = old_W * (old_std / new_std)
            new_b = old_b * (old_std / new_std) + (old_mean - new_mean) / new_std
            当统计更新时，调整Critic最后一层权重和偏置，保持输出不变（Pop-Art核心）
        """
        if self.old_mean is None or self.old_std is None:
            return
        old_mean_val = self.old_mean.item()
        old_std_val = self.old_std.item()
        new_mean_val = self.mean.item()
        new_std_val = self.std.item()
        if abs(old_std_val - new_std_val) < 1e-8 or new_std_val < 1e-8:
            return
        scale = old_std_val / new_std_val
        shift = (old_mean_val - new_mean_val) / new_std_val
        with torch.no_grad():
            last_layer.weight.data.mul_(scale)
            if last_layer.bias is not None:
                last_layer.bias.data.mul_(scale).add_(shift)


class CriticNetwork(nn.Module):
    """共享Critic网络：输入全局状态，输出状态价值"""

    def __init__(self, global_state_dim, hidden_dim):
        super(CriticNetwork, self).__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(hidden_dim, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier初始化，使用较小gain值避免初始输出过大"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, global_state):
        features = self.shared_net(global_state)
        return self.output_layer(features)


class MAPPO:
    """
    Multi-Agent Proximal Policy Optimization (MAPPO)

    实现集中式训练、分布式执行的MAPPO算法
    """

    def __init__(self, env, n_agents=3, local_state_dim=None, global_state_dim=None,
                 hidden_dim=128, gamma=0.99, lmbda=0.95, eps=0.2, epochs=10,
                 lr_actor=3e-4, lr_critic=3e-4, batch_size=64, ent_coef=0.01,
                 max_grad_norm=5.0, use_state_normalization=True, use_popart=True,
                 reward_scale=10.0, use_reward_normalization=False,
                 critic_loss_type='mse', huber_delta=1.0, device=None):
        """
        初始化MAPPO算法

        Args:
            env: 多智能体环境
            n_agents: 智能体数量
            local_state_dim: 局部状态维度（如果为None，将从环境自动推断）
            global_state_dim: 全局状态维度（如果为None，将从环境自动推断）
            hidden_dim: 隐藏层维度
            gamma: 折扣因子
            lmbda: GAE参数
            eps: PPO裁剪参数
            epochs: 每次更新的epoch数
            lr_actor: Actor学习率
            lr_critic: Critic学习率
            batch_size: 批次大小
            ent_coef: 熵系数
            max_grad_norm: 梯度裁剪阈值
            use_state_normalization: 是否使用状态归一化
            use_popart: 是否使用Pop-Art值归一化
            reward_scale: 奖励缩放因子
            use_reward_normalization: 是否使用奖励归一化（z-score标准化）
            critic_loss_type: Critic损失函数类型 ('mse' 或 'huber')
            huber_delta: Huber损失的delta参数（仅当critic_loss_type='huber'时使用）
            device: 计算设备
        """
        self.env = env
        self.n_agents = n_agents
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.use_state_normalization = use_state_normalization
        self.reward_scale = reward_scale
        self.use_reward_normalization = use_reward_normalization
        self.critic_loss_type = critic_loss_type.lower()
        self.huber_delta = huber_delta

        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"MAPPO using device: {self.device}")

        # 动作空间配置（从环境复制，并添加社区储能动作）
        self.action_space_config = env.agents[0].env.action_space.copy()
        self.action_space_config['community_ess_power'] = (-5.0, -2.5, 0, 2.5, 5.0)

        # 推断状态维度
        if local_state_dim is None:
            sample_local_states = env.reset(mode='train', date_index=0, house_index=0)
            sample_local_state = sample_local_states[0]
            local_state_dim = len(self._state_dict_to_vector(sample_local_state))
        self.local_state_dim = local_state_dim

        if global_state_dim is None:
            sample_local_states = env.reset(mode='train', date_index=0, house_index=0)
            sample_local_vectors = [self._state_dict_to_vector(s) for s in sample_local_states]
            global_state_dim = sum(len(vec) for vec in sample_local_vectors)
        self.global_state_dim = global_state_dim

        # 创建网络（每个智能体独立Actor，共享Critic）
        self.actors = nn.ModuleList([
            ActorNetwork(local_state_dim, hidden_dim, self.action_space_config).to(self.device)
            for _ in range(n_agents)
        ])
        self.critic = CriticNetwork(global_state_dim, hidden_dim).to(self.device)

        # 优化器
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=lr_actor) for actor in self.actors]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Pop-Art归一化
        self.use_popart = use_popart
        if self.use_popart:
            self.popart_normalizer = PopArtNormalizer(device=self.device, beta=1e-4)
        else:
            self.popart_normalizer = None

        # 状态归一化（局部和全局）
        if use_state_normalization:
            self.local_running_stats = RunningStats(local_state_dim, device=self.device)
            self.global_running_stats = RunningStats(global_state_dim, device=self.device)
        else:
            self.local_running_stats = None
            self.global_running_stats = None

        # 奖励归一化（z-score）
        if self.use_reward_normalization:
            self.reward_running_stats = RunningStats(shape=1, device=self.device)
        else:
            self.reward_running_stats = None

        self.reset_buffer()

    def _state_dict_to_vector(self, state_dict):
        """将状态字典转换为扁平的numpy向量"""
        if isinstance(state_dict, dict):
            ordered_keys = sorted(state_dict.keys())
            if isinstance(state_dict[ordered_keys[0]], (list, np.ndarray)):
                vector = []
                for k in ordered_keys:
                    val = state_dict[k]
                    if isinstance(val, (list, np.ndarray)):
                        vector.extend(val if isinstance(val, np.ndarray) else np.array(val))
                    else:
                        vector.append(val)
                return np.array(vector, dtype=np.float32)
            else:
                return np.array([state_dict[k] for k in ordered_keys], dtype=np.float32)
        else:
            return np.array(state_dict, dtype=np.float32)

    def _normalize_local_state(self, state):
        """归一化局部状态（若已启用）"""
        if self.local_running_stats is not None and self.local_running_stats.count > 0:
            state_tensor = torch.FloatTensor(state).to(self.device)
            normalized = (state_tensor - self.local_running_stats.mean) / (self.local_running_stats.std + 1e-8)
            return normalized.cpu().numpy()
        return state

    def _normalize_global_state(self, state):
        if self.global_running_stats is not None and self.global_running_stats.count > 0:
            state_tensor = torch.FloatTensor(state).to(self.device)
            normalized = (state_tensor - self.global_running_stats.mean) / (self.global_running_stats.std + 1e-8)
            return normalized.cpu().numpy()
        return state

    def reset_buffer(self):
        """清空经验池"""
        self.buffer = {
            'local_states': [[] for _ in range(self.n_agents)],
            'global_states': [],
            'actions': [[] for _ in range(self.n_agents)],
            'rewards': [[] for _ in range(self.n_agents)],
            'next_local_states': [[] for _ in range(self.n_agents)],
            'next_global_states': [],
            'dones': [[] for _ in range(self.n_agents)],
            'old_log_probs': [[] for _ in range(self.n_agents)]
        }

    def select_actions(self, local_states, deterministic=False):
        """根据局部状态选择动作（分布式执行）"""
        actions = []
        action_log_probs = []

        for agent_id, local_state in enumerate(local_states):
            state_vector = self._state_dict_to_vector(local_state)
            if self.use_state_normalization:
                state_vector = self._normalize_local_state(state_vector)

            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action_logits_dict = self.actors[agent_id](state_tensor)

            action_dict = {}
            log_prob = 0.0
            for action_name, logits in action_logits_dict.items():
                probs = F.softmax(logits, dim=-1)
                if deterministic:
                    action_idx = torch.argmax(probs, dim=-1).item()
                else:
                    action_idx = torch.multinomial(probs, 1).item()

                log_probs_action = F.log_softmax(logits, dim=-1)
                log_prob += log_probs_action[0, action_idx].item()

                action_values = self.action_space_config[action_name]
                if isinstance(action_values, (list, tuple)):
                    action_dict[action_name] = action_values[action_idx]
                else:
                    action_dict[action_name] = action_idx

            actions.append(action_dict)
            action_log_probs.append(log_prob)

        return actions, action_log_probs

    def store_transition(self, local_states, global_state, actions, action_log_probs,
                         rewards, next_local_states, next_global_state, dones):
        """
        存储经验

        Args:
            local_states: 局部状态列表
            global_state: 全局状态（用于兼容性，但不使用）
            actions: 动作列表
            action_log_probs: 动作log概率列表
            rewards: 奖励列表
            next_local_states: 下一个局部状态列表
            next_global_state: 下一个全局状态（用于兼容性，但不使用）
            dones: 终止标志列表
        """

        # 奖励缩放/归一化
        normalized_rewards = list(rewards)
        if self.use_reward_normalization and self.reward_running_stats is not None:
            rewards_array = np.array(rewards, dtype=np.float32)
            rewards_tensor = torch.FloatTensor(rewards_array.reshape(-1, 1)).to(self.device)
            self.reward_running_stats.update(rewards_tensor)
            if self.reward_running_stats.count > 0:
                mean = self.reward_running_stats.mean.item()
                std = self.reward_running_stats.std.item() + 1e-8
                for i, r in enumerate(rewards):
                    normalized_rewards[i] = (r - mean) / std

        # 将状态字典转换为向量
        local_state_vectors = []
        for s in local_states:
            if isinstance(s, dict):
                local_state_vectors.append(self._state_dict_to_vector(s))
            else:
                local_state_vectors.append(np.zeros(self.local_state_dim, dtype=np.float32))

        next_local_state_vectors = []
        for s in next_local_states:
            if isinstance(s, dict):
                next_local_state_vectors.append(self._state_dict_to_vector(s))
            else:
                next_local_state_vectors.append(np.zeros(self.local_state_dim, dtype=np.float32))

        # 状态归一化 + 构造全局状态
        if self.use_state_normalization and self.local_running_stats is not None:
            local_states_tensor = torch.FloatTensor(np.array(local_state_vectors)).to(self.device)
            self.local_running_stats.update(local_states_tensor)

            normalized_local = [self._normalize_local_state(lv) for lv in local_state_vectors]
            normalized_next_local = [self._normalize_local_state(nlv) for nlv in next_local_state_vectors]

            global_state_vector = np.concatenate(normalized_local)
            next_global_state_vector = np.concatenate(normalized_next_local)
        else:
            global_state_vector = np.concatenate(local_state_vectors)
            next_global_state_vector = np.concatenate(next_local_state_vectors)
            normalized_local = local_state_vectors
            normalized_next_local = next_local_state_vectors

        self.buffer['global_states'].append(global_state_vector)
        self.buffer['next_global_states'].append(next_global_state_vector)

        for agent_id in range(self.n_agents):
            self.buffer['local_states'][agent_id].append(normalized_local[agent_id])
            self.buffer['actions'][agent_id].append(actions[agent_id])
            self.buffer['rewards'][agent_id].append(normalized_rewards[agent_id])
            self.buffer['next_local_states'][agent_id].append(normalized_next_local[agent_id])
            self.buffer['dones'][agent_id].append(dones[agent_id])
            self.buffer['old_log_probs'][agent_id].append(action_log_probs[agent_id])

    def compute_gae(self, rewards, values_with_next, dones, next_value=0):
        """
        计算GAE (Generalized Advantage Estimation)

        Args:
            rewards: 奖励列表
            values_with_next: 价值列表（包含下一个状态的价值）
            dones: 终止标志列表
            next_value: 下一个状态的价值（已包含在values_with_next中）

        Returns:
            advantages: 优势函数
            returns: 回报
        """
        rewards = np.array(rewards)
        values_with_next = np.array(values_with_next)
        dones = np.array(dones)

        values = values_with_next[:-1]
        advantages = np.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                last_gae = delta
            else:
                delta = rewards[t] + self.gamma * values_with_next[t + 1] - values[t]
                last_gae = delta + self.gamma * self.lmbda * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def update(self):
        """更新策略和价值网络"""
        if len(self.buffer['global_states']) == 0:
            return None

        # 准备全局状态张量
        global_states = np.array(self.buffer['global_states'])
        next_global_states = np.array(self.buffer['next_global_states'])
        global_states_tensor = torch.FloatTensor(global_states).to(self.device)
        next_global_states_tensor = torch.FloatTensor(next_global_states).to(self.device)

        # 计算当前价值和下一状态价值
        with torch.no_grad():
            values = self.critic(global_states_tensor).squeeze().cpu().numpy()
            next_values = self.critic(next_global_states_tensor).squeeze().cpu().numpy()

        # 最后一个状态的价值（若未终止）
        if not self.buffer['dones'][0][-1]:
            last_next_global_state = next_global_states[-1]
            last_next_global_state_tensor = torch.FloatTensor(last_next_global_state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                last_next_value = self.critic(last_next_global_state_tensor).item()
        else:
            last_next_value = 0

        values_with_next = np.append(values, last_next_value)

        update_stats = {'actor_loss': [], 'critic_loss': [], 'entropy': []}

        # 更新每个Actor
        for agent_id in range(self.n_agents):
            local_states = np.array(self.buffer['local_states'][agent_id])
            rewards = np.array(self.buffer['rewards'][agent_id])
            dones = np.array(self.buffer['dones'][agent_id])
            old_log_probs = np.array(self.buffer['old_log_probs'][agent_id])
            actions = self.buffer['actions'][agent_id]

            if self.reward_scale != 1.0:
                rewards = rewards / self.reward_scale

            local_states_tensor = torch.FloatTensor(local_states).to(self.device)

            advantages, returns = self.compute_gae(rewards, values_with_next, dones, last_next_value)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantages_tensor = torch.FloatTensor(advantages).to(self.device)
            returns_tensor = torch.FloatTensor(returns).to(self.device)
            old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)

            # 将动作转换为索引
            action_indices_dict = {}
            for action_name in self.action_space_config.keys():
                action_values = self.action_space_config[action_name]
                action_indices = []
                for action_dict in actions:
                    action_value = action_dict[action_name]
                    if isinstance(action_values, (list, tuple)):
                        idx = action_values.index(action_value) if action_value in action_values else 0
                        action_indices.append(idx)
                    else:
                        action_indices.append(action_value)
                action_indices_dict[action_name] = torch.LongTensor(action_indices).to(self.device)

            # 多epoch更新Actor
            for _ in range(self.epochs):
                action_logits_dict = self.actors[agent_id](local_states_tensor)
                current_log_probs_list = []
                for action_name, logits in action_logits_dict.items():
                    action_indices = action_indices_dict[action_name]
                    log_probs = F.log_softmax(logits, dim=-1)
                    selected_log_probs = log_probs.gather(1, action_indices.unsqueeze(1)).squeeze(1)
                    current_log_probs_list.append(selected_log_probs.unsqueeze(1))

                current_log_probs = torch.cat(current_log_probs_list, dim=1).sum(dim=1)
                ratio = torch.exp(current_log_probs - old_log_probs_tensor)

                surr1 = ratio * advantages_tensor
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantages_tensor
                actor_loss = -torch.min(surr1, surr2).mean()

                total_entropy = 0.0
                for logits in action_logits_dict.values():
                    probs = F.softmax(logits, dim=-1)
                    log_probs = F.log_softmax(logits, dim=-1)
                    entropy = -(probs * log_probs).sum(dim=-1).mean()
                    total_entropy += entropy.item()

                self.actor_optimizers[agent_id].zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), self.max_grad_norm)
                self.actor_optimizers[agent_id].step()

                update_stats['actor_loss'].append(actor_loss.item())
                update_stats['entropy'].append(total_entropy)

        # 更新Critic（使用所有智能体的平均returns）
        all_returns = []
        for agent_id in range(self.n_agents):
            rewards = np.array(self.buffer['rewards'][agent_id])
            if self.reward_scale != 1.0:
                rewards = rewards / self.reward_scale
            dones = np.array(self.buffer['dones'][agent_id])
            _, returns = self.compute_gae(rewards, values_with_next, dones, last_next_value)
            all_returns.append(returns)

        avg_returns = np.mean(all_returns, axis=0)
        returns_tensor_critic = torch.FloatTensor(avg_returns).to(self.device)

        if self.use_popart:
            updated = self.popart_normalizer.update(returns_tensor_critic)
            if updated and self.popart_normalizer.old_mean is not None:
                self.popart_normalizer.adjust_network_weights(self.critic.output_layer)
            returns_normalized = self.popart_normalizer.normalize(returns_tensor_critic)
        else:
            returns_normalized = returns_tensor_critic

        for _ in range(self.epochs):
            values_pred = self.critic(global_states_tensor).squeeze()
            if self.critic_loss_type == 'huber':
                critic_loss = F.smooth_l1_loss(values_pred, returns_normalized, beta=self.huber_delta)
            else:
                critic_loss = F.mse_loss(values_pred, returns_normalized)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            update_stats['critic_loss'].append(critic_loss.item())

        return {
            'actor_loss': np.mean(update_stats['actor_loss']) if update_stats['actor_loss'] else 0.0,
            'critic_loss': np.mean(update_stats['critic_loss']) if update_stats['critic_loss'] else 0.0,
            'entropy': np.mean(update_stats['entropy']) if update_stats['entropy'] else 0.0
        }

    def save(self, save_dir):
        """保存模型和归一化参数"""
        os.makedirs(save_dir, exist_ok=True)
        for agent_id, actor in enumerate(self.actors):
            torch.save(actor.state_dict(), os.path.join(save_dir, f'mappo_actor_{agent_id}.pth'))
        torch.save(self.critic.state_dict(), os.path.join(save_dir, 'mappo_critic.pth'))
        if self.use_state_normalization:
            stats = {
                'local_mean': self.local_running_stats.mean.cpu(),
                'local_std': self.local_running_stats.std.cpu(),
                'local_count': self.local_running_stats.count,
                'global_mean': self.global_running_stats.mean.cpu(),
                'global_std': self.global_running_stats.std.cpu(),
                'global_count': self.global_running_stats.count
            }
            torch.save(stats, os.path.join(save_dir, 'mappo_stats.pth'))

    def load(self, load_dir):
        """加载模型和归一化参数"""
        for agent_id, actor in enumerate(self.actors):
            actor_path = os.path.join(load_dir, f'mappo_actor_{agent_id}.pth')
            if os.path.exists(actor_path):
                actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        critic_path = os.path.join(load_dir, 'mappo_critic.pth')
        if os.path.exists(critic_path):
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
        if self.use_state_normalization:
            stats_path = os.path.join(load_dir, 'mappo_stats.pth')
            if os.path.exists(stats_path):
                stats = torch.load(stats_path, map_location=self.device)
                self.local_running_stats.mean = stats['local_mean'].to(self.device)
                self.local_running_stats.std = stats['local_std'].to(self.device)
                self.local_running_stats.count = stats['local_count']
                self.global_running_stats.mean = stats['global_mean'].to(self.device)
                self.global_running_stats.std = stats['global_std'].to(self.device)
                self.global_running_stats.count = stats['global_count']