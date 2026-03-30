"""
动作空间工具：用于 COMA/QMIX 与环境动作字典的转换
不修改环境，仅根据与 MAPPO 相同的 action_space 配置做索引与 dict 的映射。
"""
import numpy as np


def get_action_space_config(env):
    """
    从多智能体环境获取与 MAPPO 一致的动作空间配置（不修改 env）。
    
    Returns:
        dict: 键为动作名，值为离散选项的 tuple/list
    """
    config = env.agents[0].env.action_space.copy()
    config['community_ess_power'] = (-5.0, -2.5, 0, 2.5, 5.0)
    return config


def get_ordered_action_keys(action_space_config):
    """与 MAPPO 一致的键顺序（插入顺序）。"""
    return list(action_space_config.keys())


def get_dim_sizes(action_space_config):
    """
    每个动作维度的选项数量（与键顺序一致）。
    
    Returns:
        list[int]: 每个维度的 size，如 [5, 5, 7, 8, 8, 7, 5]
    """
    keys = get_ordered_action_keys(action_space_config)
    return [len(action_space_config[k]) for k in keys]


def action_dict_to_indices(action_dict, action_space_config):
    """
    单个智能体的动作字典 -> 各维度的索引列表。
    
    Args:
        action_dict: 单智能体动作 dict
        action_space_config: 动作空间配置
    
    Returns:
        list[int]: 长度为 len(keys) 的索引列表
    """
    keys = get_ordered_action_keys(action_space_config)
    indices = []
    for k in keys:
        vals = action_space_config[k]
        v = action_dict.get(k, vals[0])
        if v in vals:
            indices.append(vals.index(v))
        else:
            try:
                idx = list(vals).index(v)
            except ValueError:
                idx = 0
            indices.append(idx)
    return indices


def indices_to_action_dict(indices, action_space_config):
    """
    单智能体各维度索引 -> 动作字典。
    
    Args:
        indices: list[int]，长度等于键数量
        action_space_config: 动作空间配置
    
    Returns:
        dict: 单智能体动作
    """
    keys = get_ordered_action_keys(action_space_config)
    return {k: action_space_config[k][indices[i]] for i, k in enumerate(keys)}


def action_dicts_to_indices(actions_list, action_space_config):
    """
    多智能体动作列表 [dict, dict, dict] -> [indices_0, indices_1, indices_2]。
    """
    return [action_dict_to_indices(a, action_space_config) for a in actions_list]


def indices_to_action_dicts(indices_list, action_space_config):
    """
    [indices_0, indices_1, indices_2] -> [dict, dict, dict]，可传入 env.step(actions)。
    """
    return [indices_to_action_dict(idx, action_space_config) for idx in indices_list]


def state_dict_to_vector(state_dict):
    """
    将状态字典转为向量（与 MAPPO 的 _state_dict_to_vector 逻辑一致）。
    在边界处保证返回有限值，避免环境 state 中的 nan/inf 进入算法。
    """
    if isinstance(state_dict, dict):
        ordered_keys = sorted(state_dict.keys())
        if not ordered_keys:
            return np.array([], dtype=np.float32)
        first = state_dict[ordered_keys[0]]
        if isinstance(first, (list, np.ndarray)):
            vector = []
            for k in ordered_keys:
                val = state_dict[k]
                if isinstance(val, (list, np.ndarray)):
                    vector.extend(np.array(val).ravel().tolist())
                else:
                    vector.append(float(val))
            out = np.array(vector, dtype=np.float32)
        else:
            out = np.array([float(state_dict[k]) for k in ordered_keys], dtype=np.float32)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.array(state_dict, dtype=np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def get_global_state_vector(env):
    """
    从环境的 get_community_state() 得到向量形式的全局状态（用于 Critic / Mixing）。
    在边界处保证返回有限值，避免环境偶发 nan/inf 污染 buffer 与 Critic。
    """
    d = env.get_community_state()
    vec = [
        d.get('community_ess_soc', 0.5),
        d.get('community_ess_capacity', 36.0),
        d.get('community_net_load', 0.0),
        d.get('community_avg_load', 0.0),
        d.get('peak_threshold', 25.0),
    ]
    balances = d.get('credit_balances', {})
    for i in range(3):
        vec.append(balances.get(i, 100.0))
    arr = np.array(vec, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
