"""
统一绘制 MAPPO / COMA / QMIX 回报收敛曲线，支持单算法或三算法对比在一张图。
从各算法目录下的 training_stats.json 读取 episode_returns，与 plot_mappo_returns 数据格式一致。
"""
import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 各算法默认模型目录（相对项目根）
DEFAULT_DIRS = {
    'mappo': 'multi_agent/algorithms/models',
    'coma': 'multi_agent/algorithms/models_coma',
    'qmix': 'multi_agent/algorithms/models_qmix',
}

# 绘图用颜色与标签
STYLE = {
    'mappo': {'color': '#2E86AB', 'label': 'MAPPO'},
    'coma': {'color': '#E94F37', 'label': 'COMA'},
    'qmix': {'color': '#44AF69', 'label': 'QMIX'},
}


def load_episode_returns(model_dir):
    """
    从 model_dir/training_stats.json 加载 episode_returns。
    若目录或文件不存在、或无 episode_returns，返回 None。
    """
    if not model_dir or not os.path.isdir(model_dir):
        return None
    stats_file = os.path.join(model_dir, 'training_stats.json')
    if not os.path.isfile(stats_file):
        return None
    try:
        with open(stats_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        returns = data.get('episode_returns', [])
        if not returns:
            return None
        return returns
    except Exception:
        return None


def smooth_returns(episode_returns, window=10):
    """与 plot_mappo_returns 一致：移动平均与滚动标准差。"""
    if len(episode_returns) <= window:
        return np.arange(len(episode_returns)), np.array(episode_returns), None
    smoothed = np.convolve(episode_returns, np.ones(window) / window, mode='valid')
    rolling_std = np.array([np.std(episode_returns[i:i + window]) for i in range(len(episode_returns) - window + 1)])
    x = np.arange(window - 1, len(episode_returns))
    return x, smoothed, rolling_std


def plot_returns(algorithm='all', model_dirs=None, save_path=None, show_plot=False):
    """
    绘制回报收敛图：单算法或 MAPPO/COMA/QMIX 三算法对比。
    
    Args:
        algorithm: 'mappo' | 'coma' | 'qmix' | 'all'
        model_dirs: dict 或 None。若为 None，则对 each alg 使用 DEFAULT_DIRS[alg]。
                    若为 dict，则键为 'mappo'/'coma'/'qmix'，值为对应目录。
                    也可为 str：单目录，仅当 algorithm 为单算法时使用。
        save_path: 保存路径；None 时根据 algorithm 写至对应目录或当前目录 comparison_returns.png
        show_plot: 是否弹出显示
    """
    if model_dirs is None:
        model_dirs = {}
    if isinstance(model_dirs, str):
        model_dirs = {algorithm: model_dirs}

    algs_to_plot = ['mappo', 'coma', 'qmix'] if algorithm == 'all' else [algorithm]
    data = {}
    for alg in algs_to_plot:
        dir_path = model_dirs.get(alg) or DEFAULT_DIRS.get(alg)
        if not dir_path and algorithm != 'all':
            dir_path = model_dirs.get(algorithm)
        returns = load_episode_returns(dir_path)
        if returns is not None:
            data[alg] = returns
            print(f"已加载 {STYLE[alg]['label']}: {dir_path}, {len(returns)} 个 episode")
        else:
            print(f"警告: 未找到或无效数据，跳过 {STYLE[alg]['label']} (目录: {dir_path})")

    if not data:
        print("没有可绘制的数据，请检查 model_dirs 与 training_stats.json")
        return None

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    window = 10

    for alg, returns in data.items():
        x, smoothed, std = smooth_returns(returns, window)
        style = STYLE[alg]
        if std is not None:
            ax.fill_between(x, smoothed - std, smoothed + std, alpha=0.25, color=style['color'])
        ax.plot(x, smoothed, linewidth=2, color=style['color'], label=style['label'])

    ax.set_xlabel('训练回合', fontsize=12)
    ax.set_ylabel('回报', fontsize=12)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    if save_path is None:
        if len(data) == 1:
            alg = list(data.keys())[0]
            save_path = os.path.join(model_dirs.get(alg) or DEFAULT_DIRS[alg], 'returns_curve.png')
        else:
            save_path = os.path.join(os.getcwd(), 'comparison_returns.png')
    os.makedirs(os.path.dirname(os.path.abspath(save_path)) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    print(f"回报曲线已保存到: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()
    return fig


def main():
    parser = argparse.ArgumentParser(description='统一绘制 MAPPO/COMA/QMIX 回报收敛图')
    parser.add_argument('--algorithm', type=str, default='all',
                        choices=['mappo', 'coma', 'qmix', 'all'],
                        help='要绘制的算法；all 表示三者在同一张图对比')
    parser.add_argument('--model_dir', type=str, default=None,
                        help='单算法时的模型目录（含 training_stats.json）')
    parser.add_argument('--mappo_dir', type=str, default=None, help='MAPPO 模型目录')
    parser.add_argument('--coma_dir', type=str, default=None, help='COMA 模型目录')
    parser.add_argument('--qmix_dir', type=str, default=None, help='QMIX 模型目录')
    parser.add_argument('--save_path', type=str, default=None, help='图片保存路径')
    parser.add_argument('--show', action='store_true', help='显示图表')
    args = parser.parse_args()

    model_dirs = None
    if args.model_dir:
        model_dirs = {args.algorithm: args.model_dir}
    elif args.mappo_dir or args.coma_dir or args.qmix_dir:
        model_dirs = {}
        if args.mappo_dir:
            model_dirs['mappo'] = args.mappo_dir
        if args.coma_dir:
            model_dirs['coma'] = args.coma_dir
        if args.qmix_dir:
            model_dirs['qmix'] = args.qmix_dir

    plot_returns(
        algorithm=args.algorithm,
        model_dirs=model_dirs,
        save_path=args.save_path,
        show_plot=args.show,
    )


if __name__ == '__main__':
    main()
