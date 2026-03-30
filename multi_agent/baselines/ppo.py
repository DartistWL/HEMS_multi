"""
经典PPO算法实现
Classic Proximal Policy Optimization (PPO) Implementation
用于独立学习基线
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


class RunningStats:
    """运行统计信息，用于状态归一化"""
    def __init__(self, shape, device='cpu'):
        self.device = device
        self.mean = torch.zeros(shape, device=device)
        self.std = torch.ones(shape, device=device)
        self.count = 0
    
    def update(self, x):
        """更新统计信息"""
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
    """Actor网络：共享主干 + 动作分支"""
    def __init__(self, state_dim, hidden_dim, action_space_config):
        super(ActorNetwork, self).__init__()
        
        # 共享特征提取网络
        self.shared_backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 为每个动作维度创建独立的分支
        self.action_branches = nn.ModuleDict()
        for action_name, action_values in action_space_config.items():
            if isinstance(action_values, (list, tuple)):
                # 离散动作空间
                action_dim = len(action_values)
                branch = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, action_dim)
                )
                self.action_branches[action_name] = branch
    
    def forward(self, state):
        """前向传播"""
        features = self.shared_backbone(state)
        action_logits = {}
        for action_name, branch in self.action_branches.items():
            action_logits[action_name] = branch(features)
        return action_logits


class CriticNetwork(nn.Module):
    """Critic网络：价值函数估计"""
    def __init__(self, state_dim, hidden_dim):
        super(CriticNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        """前向传播"""
        return self.net(state)


class HomeEnergyPPO:
    """
    经典PPO算法实现
    用于家庭能源管理的单智能体强化学习
    """
    
    def __init__(self, env, state_dim, hidden_dim=128, action_space_config=None,
                 gamma=0.99, lmbda=0.95, eps=0.2, epochs=10, lr=3e-4,
                 constraint_mode=None, use_state_normalization=True, 
                 reward_scale=10, max_grad_norm=1.0, use_reward_normalization=False,
                 use_popart=True, critic_loss_type='mse', huber_delta=1.0, device=None):
        """
        初始化PPO算法
        
        Args:
            env: 环境对象
            state_dim: 状态维度
            hidden_dim: 隐藏层维度
            action_space_config: 动作空间配置（字典）
            gamma: 折扣因子
            lmbda: GAE参数
            eps: PPO裁剪参数
            epochs: 每次更新的epoch数
            lr: 学习率
            constraint_mode: 约束模式（本实现中不使用）
            use_state_normalization: 是否使用状态归一化
            reward_scale: 奖励缩放因子，将奖励除以该值（默认1.0表示不缩放）
            max_grad_norm: 梯度裁剪的最大范数（默认0.5，常见范围0.5-1.0）
            use_reward_normalization: 是否使用奖励归一化（z-score标准化）
            use_popart: 是否使用Pop-Art值归一化
            critic_loss_type: Critic损失函数类型 ('mse' 或 'huber')
            huber_delta: Huber损失的delta参数（仅当critic_loss_type='huber'时使用）
            device: 计算设备
        """
        self.env = env
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_space_config = action_space_config or env.action_space
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps
        self.epochs = epochs
        self.use_state_normalization = use_state_normalization
        self.reward_scale = reward_scale
        self.max_grad_norm = max_grad_norm
        self.use_reward_normalization = use_reward_normalization
        self.use_popart = use_popart
        self.critic_loss_type = critic_loss_type.lower()  # 'mse' or 'huber'
        self.huber_delta = huber_delta
        
        # 设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # 创建网络
        self.actor = ActorNetwork(state_dim, hidden_dim, self.action_space_config).to(self.device)
        self.critic = CriticNetwork(state_dim, hidden_dim).to(self.device)
        
        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        # 状态归一化
        if use_state_normalization:
            self.running_stats = RunningStats(state_dim, device=self.device)
        else:
            self.running_stats = None
        
        # 奖励归一化（z-score标准化）
        if use_reward_normalization:
            self.reward_running_stats = RunningStats(shape=(1,), device=self.device)
        else:
            self.reward_running_stats = None
        
        # Pop-Art归一化（用于returns归一化）
        if use_popart:
            # 导入PopArtNormalizer（从MAPPO）
            from multi_agent.algorithms.mappo import PopArtNormalizer
            self.popart_normalizer = PopArtNormalizer(device=self.device, beta=1e-4)
        else:
            self.popart_normalizer = None
        
        # 为了兼容原有接口，添加别名
        self.shared_backbone = self.actor.shared_backbone
        self.action_branches = self.actor.action_branches
        self.value_net = self.critic
    
    def _state_dict_to_vector(self, state_dict):
        """将状态字典转换为向量"""
        if isinstance(state_dict, dict):
            # 按字母顺序排序键，确保一致性
            ordered_keys = sorted(state_dict.keys())
            return np.array([state_dict[k] for k in ordered_keys], dtype=np.float32)
        else:
            return np.array(state_dict, dtype=np.float32)
    
    def _normalize_state(self, state):
        """归一化状态"""
        if self.running_stats is not None and self.running_stats.count > 0:
            state_tensor = torch.FloatTensor(state).to(self.device)
            normalized = (state_tensor - self.running_stats.mean) / (self.running_stats.std + 1e-8)
            return normalized.cpu().numpy()
        return state
    
    def take_action(self, state):
        """
        根据状态选择动作
        
        Args:
            state: 状态字典或状态向量
        
        Returns:
            dict: 动作字典
        """
        # 转换为向量
        state_vector = self._state_dict_to_vector(state)
        
        # 归一化
        if self.use_state_normalization:
            state_vector = self._normalize_state(state_vector)
        
        # 转换为张量
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
        
        # 获取动作logits
        with torch.no_grad():
            action_logits_dict = self.actor(state_tensor)
        
        # 从每个分支采样动作
        action_dict = {}
        for action_name, logits in action_logits_dict.items():
            # 使用softmax获取概率分布
            probs = F.softmax(logits, dim=-1)
            # 采样动作
            action_idx = torch.multinomial(probs, 1).item()
            
            # 将索引转换为实际动作值
            action_values = self.action_space_config[action_name]
            if isinstance(action_values, (list, tuple)):
                action_dict[action_name] = action_values[action_idx]
            else:
                action_dict[action_name] = action_idx
        
        return action_dict
    
    def compute_gae(self, rewards, values_with_next, dones, next_value=0):
        """
        计算GAE (Generalized Advantage Estimation)
        
        Args:
            rewards: 奖励列表（长度为N）
            values_with_next: 价值列表，包含当前状态价值和下一个状态价值（长度为N+1）
            dones: 终止标志列表（长度为N）
            next_value: 下一个状态的价值（已包含在values_with_next中，此参数保留用于兼容）
        
        Returns:
            advantages: 优势函数
            returns: 回报
        """
        rewards = np.array(rewards)
        values_with_next = np.array(values_with_next)
        dones = np.array(dones)
        
        # values是当前状态的价值（前N个元素）
        values = values_with_next[:-1]  # 长度为N
        
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                last_gae = delta
            else:
                # 使用values_with_next[t+1]获取下一个状态的价值
                delta = rewards[t] + self.gamma * values_with_next[t + 1] - values[t]
                last_gae = delta + self.gamma * self.lmbda * last_gae
            advantages[t] = last_gae
        
        returns = advantages + values
        return advantages, returns
    
    def update(self, transition_dict):
        """
        更新策略网络
        
        Args:
            transition_dict: 转换字典，包含states, actions, rewards, next_states, dones
        """
        states = transition_dict['states']
        actions = transition_dict['actions']
        rewards = transition_dict['rewards']
        next_states = transition_dict.get('next_states', [None] * len(states))
        dones = transition_dict['dones']
        
        # 奖励缩放（Reward Shaping）
        if self.reward_scale != 1.0:
            rewards = [r / self.reward_scale for r in rewards]
        
        # 奖励归一化（z-score标准化，在存储和计算之前）
        normalized_rewards = list(rewards)
        if self.use_reward_normalization and self.reward_running_stats is not None:
            rewards_array = np.array(rewards, dtype=np.float32)
            rewards_tensor = torch.FloatTensor(rewards_array.reshape(-1, 1)).to(self.device)
            self.reward_running_stats.update(rewards_tensor)
            
            if self.reward_running_stats.count > 0:
                mean = self.reward_running_stats.mean.item()
                std = self.reward_running_stats.std.item() + 1e-8
                normalized_rewards = [(r - mean) / std for r in rewards]
        
        # 使用归一化后的奖励
        rewards = normalized_rewards
        
        # 转换状态为向量
        state_vectors = [self._state_dict_to_vector(s) for s in states]
        
        # 更新运行统计
        if self.running_stats is not None:
            state_array = np.array(state_vectors)
            state_tensor = torch.FloatTensor(state_array).to(self.device)
            self.running_stats.update(state_tensor)
        
        # 归一化状态
        if self.use_state_normalization:
            state_vectors = [self._normalize_state(s) for s in state_vectors]
        
        # 转换为张量
        states_tensor = torch.FloatTensor(state_vectors).to(self.device)
        
        # 计算当前状态的价值
        with torch.no_grad():
            values = self.critic(states_tensor).squeeze().cpu().numpy()
        
        # 计算下一个状态的价值（用于GAE）
        if next_states[0] is not None:
            next_state_vectors = [self._state_dict_to_vector(s) for s in next_states]
            if self.use_state_normalization:
                next_state_vectors = [self._normalize_state(s) for s in next_state_vectors]
            next_states_tensor = torch.FloatTensor(next_state_vectors).to(self.device)
            with torch.no_grad():
                next_values = self.critic(next_states_tensor).squeeze().cpu().numpy()
        else:
            next_values = np.zeros(len(rewards))
        
        # 计算最后一个状态的价值
        if not dones[-1] and next_states[-1] is not None:
            last_next_state = self._state_dict_to_vector(next_states[-1])
            if self.use_state_normalization:
                last_next_state = self._normalize_state(last_next_state)
            last_next_state_tensor = torch.FloatTensor(last_next_state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                last_next_value = self.critic(last_next_state_tensor).item()
        else:
            last_next_value = 0
        
        # 计算GAE
        # values_with_next包含当前状态价值和下一个状态价值（用于GAE计算）
        values_with_next = np.append(values, last_next_value)
        advantages, returns = self.compute_gae(rewards, values_with_next, dones, last_next_value)
        
        # Pop-Art归一化（对returns进行归一化）
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        if self.use_popart and self.popart_normalizer is not None:
            # 更新Pop-Art统计
            updated = self.popart_normalizer.update(returns_tensor)
            # 如果统计更新，调整网络权重
            if updated and self.popart_normalizer.old_mean is not None:
                self.popart_normalizer.adjust_network_weights(self.critic.net[-1])  # Critic的最后一层
            # 归一化returns
            returns_tensor = self.popart_normalizer.normalize(returns_tensor)
        
        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 转换为张量
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        
        # 将动作转换为索引
        action_indices_dict = {}
        for action_name in self.action_space_config.keys():
            action_values = self.action_space_config[action_name]
            action_indices = []
            for action_dict in actions:
                action_value = action_dict[action_name]
                if isinstance(action_values, (list, tuple)):
                    # 找到对应的索引
                    idx = action_values.index(action_value) if action_value in action_values else 0
                    action_indices.append(idx)
                else:
                    action_indices.append(action_value)
            action_indices_dict[action_name] = torch.LongTensor(action_indices).to(self.device)
        
        # 计算旧策略的log概率（在更新前保存）
        with torch.no_grad():
            old_action_logits_dict = self.actor(states_tensor)
            old_log_probs_list = []
            for action_name, logits in old_action_logits_dict.items():
                action_indices = action_indices_dict[action_name]
                log_probs = F.log_softmax(logits, dim=-1)
                selected_log_probs = log_probs.gather(1, action_indices.unsqueeze(1)).squeeze(1)
                old_log_probs_list.append(selected_log_probs.unsqueeze(1))
            old_log_probs = torch.cat(old_log_probs_list, dim=1).sum(dim=1)
        
        # 多epoch更新
        for epoch in range(self.epochs):
            # 获取当前策略的log概率
            action_logits_dict = self.actor(states_tensor)
            current_log_probs_list = []
            
            for action_name, logits in action_logits_dict.items():
                action_indices = action_indices_dict[action_name]
                log_probs = F.log_softmax(logits, dim=-1)
                selected_log_probs = log_probs.gather(1, action_indices.unsqueeze(1)).squeeze(1)
                current_log_probs_list.append(selected_log_probs.unsqueeze(1))
            
            # 合并所有动作维度的log概率（假设独立）
            current_log_probs = torch.cat(current_log_probs_list, dim=1).sum(dim=1)
            
            # 计算策略比率
            ratio = torch.exp(current_log_probs - old_log_probs)
            
            # 计算裁剪的目标
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantages_tensor
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 更新Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            
            # 更新Critic
            values_pred = self.critic(states_tensor).squeeze()
            
            # 根据损失类型选择损失函数
            if self.critic_loss_type == 'huber':
                critic_loss = F.smooth_l1_loss(values_pred, returns_tensor, reduction='mean', 
                                              delta=self.huber_delta)
            else:  # 'mse'
                critic_loss = F.mse_loss(values_pred, returns_tensor)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
        
        # 计算熵（使用最后一次更新的策略）
        with torch.no_grad():
            action_logits_dict = self.actor(states_tensor)
            total_entropy = 0.0
            for action_name, logits in action_logits_dict.items():
                probs = F.softmax(logits, dim=-1)
                log_probs = F.log_softmax(logits, dim=-1)
                entropy = -(probs * log_probs).sum(dim=-1).mean()
                total_entropy += entropy.item()
        
        # 返回最后一次更新的loss和熵
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': total_entropy
        }
