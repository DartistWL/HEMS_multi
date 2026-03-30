"""
共用工具：状态归一化、奖励缩放，供 COMA/QMIX 使用。
"""
import torch
import numpy as np


class RunningStats:
    """运行统计，用于状态归一化（与 MAPPO 逻辑一致）。"""
    def __init__(self, shape, device='cpu'):
        try:
            len(shape)
        except TypeError:
            shape = (int(shape),)
        self.device = device
        self.mean = torch.zeros(shape, device=device)
        self.std = torch.ones(shape, device=device)
        self.count = 0

    def update(self, x):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).to(self.device)
        # 若输入含 nan/inf，会污染 mean/std 导致后续归一化全部为 nan；在入口处替换为 0
        x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        batch_mean = x.mean(dim=0)
        batch_std = x.std(dim=0).clamp(min=1e-8)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        if total_count > 0:
            self.mean += delta * batch_count / total_count
            m_a = self.std ** 2 * self.count
            m_b = batch_std ** 2 * batch_count
            M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
            # 使用 min=1.0 避免早期样本少时 std 过小导致归一化后数值爆炸 -> nan
            self.std = torch.sqrt(M2 / total_count).clamp(min=1.0)
            self.count = total_count

    def normalize(self, x):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).to(self.device)
        if self.count == 0:
            return x
        return (x - self.mean) / self.std

    def normalize_np(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        x = np.asarray(x, dtype=np.float32)
        # 根本原因修复：count=0 时不能直接返回原始状态，否则大数值（如 36、100）进入网络会累积溢出产生 nan
        # 与 count>0 时一致，将输出限制在 [-10, 10]，保证网络输入始终有界
        if self.count == 0:
            return np.clip(x, -10.0, 10.0)
        m = self.mean.cpu().numpy()
        s = self.std.cpu().numpy()
        out = np.array((x - m) / (s + 1e-8), dtype=np.float32)
        out = np.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0)
        # 裁剪到合理范围，防止极端值导致后续网络 nan
        out = np.clip(out, -10.0, 10.0)
        return out


class StateRunningStats:
    """
    状态用运行统计，与 MAPPO 一致：纯 z-score，无 std 下限、无输出裁剪。
    供 COMA 状态归一化使用，避免 common.RunningStats 的 clip/std 下限导致负提升。
    """
    def __init__(self, shape, device='cpu'):
        try:
            len(shape)
        except TypeError:
            shape = (int(shape),)
        self.device = device
        self.mean = torch.zeros(shape, device=device)
        self.std = torch.ones(shape, device=device)
        self.count = 0

    def update(self, x):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).to(self.device)
        x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        batch_mean = x.mean(dim=0)
        batch_std = x.std(dim=0).clamp(min=1e-8)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        if total_count > 0:
            self.mean += delta * batch_count / total_count
            m_a = self.std ** 2 * self.count
            m_b = batch_std ** 2 * batch_count
            M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
            self.std = torch.sqrt(M2 / total_count).clamp(min=1e-8)
            self.count = total_count

    def normalize_np(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        x = np.asarray(x, dtype=np.float32)
        if self.count == 0:
            return x
        m = self.mean.cpu().numpy()
        s = self.std.cpu().numpy()
        out = np.array((x - m) / (s + 1e-8), dtype=np.float32)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out


class RewardScaler:
    """奖励缩放：running mean/std，对 return 或 step reward 做 z-score 或 scale。"""
    def __init__(self, device='cpu', scale=10.0, use_running_stats=False):
        self.device = device
        self.scale = scale
        self.use_running_stats = use_running_stats
        self.running_mean = 0.0
        self.running_std = 1.0
        self.count = 0

    def update(self, rewards):
        r = np.asarray(rewards).reshape(-1)
        if len(r) == 0:
            return
        # 防止 env 返回 nan/inf 污染 running 统计，导致后续 scale_reward 全为 nan、TD target 爆炸、梯度 nan
        r = np.where(np.isfinite(r), r, 0.0)
        batch_mean, batch_std = float(r.mean()), float(r.std())
        if batch_std < 1e-8:
            batch_std = 1.0
        n = len(r)
        total = self.count + n
        delta = batch_mean - self.running_mean
        self.running_mean += delta * n / total
        m2_old = self.running_std ** 2 * self.count
        m2_new = batch_std ** 2 * n
        M2 = m2_old + m2_new + delta ** 2 * self.count * n / total
        # 根本原因修复：running_std 下限用 1.0 而非 1e-8，否则 (r-mean)/1e-8 会产生巨大缩放奖励 -> target 爆炸 -> 梯度爆炸 -> nan
        self.running_std = max(1.0, (M2 / total) ** 0.5)
        self.count = total

    def scale_reward(self, r):
        if not self.use_running_stats or self.count == 0:
            out = float(r) / self.scale
        else:
            out = (float(r) - self.running_mean) / (self.running_std + 1e-8)
        # 若上游传入或统计异常导致 nan/inf，避免写入 replay 污染整条 TD 链
        if not np.isfinite(out):
            return 0.0
        return out

    def scale_rewards_batch(self, rewards):
        arr = np.asarray(rewards, dtype=np.float64)
        if not self.use_running_stats or self.count == 0:
            out = (arr / self.scale).astype(np.float32)
        else:
            out = ((arr - self.running_mean) / (self.running_std + 1e-8)).astype(np.float32)
        out = np.where(np.isfinite(out), out, 0.0)
        return out
