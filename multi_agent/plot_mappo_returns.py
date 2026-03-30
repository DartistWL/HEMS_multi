"""
单独绘制MAPPO的回报曲线
Plot MAPPO returns curve separately
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_returns_from_training_curves(model_dir='multi_agent/algorithms/models', 
                                      save_path=None, show_plot=True):
    """
    从已有的training_curves.png中，单独绘制回报曲线
    
    由于无法从图片中提取数据，这个函数会：
    1. 检查是否有保存的training_stats.json文件
    2. 如果没有，提示用户运行训练脚本时会保存数据
    
    Args:
        model_dir: 模型目录（包含training_curves.png）
        save_path: 保存路径（如果为None，则保存到model_dir）
        show_plot: 是否显示图表
    """
    # 检查是否有保存的统计数据文件
    stats_file = os.path.join(model_dir, 'training_stats.json')
    
    if os.path.exists(stats_file):
        # 从JSON文件加载数据
        with open(stats_file, 'r', encoding='utf-8') as f:
            training_stats = json.load(f)
        episode_returns = training_stats.get('episode_returns', [])
        print(f"从 {stats_file} 加载了 {len(episode_returns)} 个episode的回报数据")
    else:
        # 如果没有统计数据文件，提示用户
        print(f"警告: 未找到训练统计数据文件 {stats_file}")
        print("训练统计数据在训练过程中未被保存。")
        print("如果training_curves.png存在，您可以查看完整图片。")
        
        # 尝试从图片路径确认图片存在
        curves_img = os.path.join(model_dir, 'training_curves.png')
        if os.path.exists(curves_img):
            print(f"\n找到了训练曲线图片: {curves_img}")
            print("但由于无法从图片中提取数据，建议重新运行训练脚本并保存统计数据。")
        
        # 返回None，表示无法绘图
        return None
    
    # 绘制回报曲线：仅保留移动平均曲线与标准差阴影
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    if len(episode_returns) > 10:
        window = 10
        smoothed = np.convolve(episode_returns, np.ones(window) / window, mode='valid')
        rolling_std = np.array([np.std(episode_returns[i:i + window]) for i in range(len(episode_returns) - window + 1)])
        x_smooth = np.arange(window - 1, len(episode_returns))
        ax.fill_between(x_smooth, smoothed - rolling_std, smoothed + rolling_std, alpha=0.3, color='#4ECDC4')
        ax.plot(x_smooth, smoothed, linewidth=2.5, color='#2E86AB')
    else:
        ax.plot(episode_returns, linewidth=2, color='#2E86AB')
    
    # 添加统计信息
    if len(episode_returns) > 0:
        avg_return = np.mean(episode_returns)
        std_return = np.std(episode_returns)
        max_return = np.max(episode_returns)
        min_return = np.min(episode_returns)
        final_return = episode_returns[-1]
        
        # 计算最后100个episode的平均值
        if len(episode_returns) >= 100:
            avg_last_100 = np.mean(episode_returns[-100:])
            ax.axhline(y=avg_last_100, color='green', linestyle='--', 
                      linewidth=1.5, alpha=0.7, label=f'Avg (last 100): {avg_last_100:.2f}')
        
        # 添加文本信息（已删除左上角统计信息框）
        # info_text = f'Total Episodes: {len(episode_returns)}\n'
        # info_text += f'Average: {avg_return:.2f} ± {std_return:.2f}\n'
        # info_text += f'Max: {max_return:.2f}\n'
        # info_text += f'Min: {min_return:.2f}\n'
        # info_text += f'Final: {final_return:.2f}'
        # 
        # ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
        #        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        #        fontsize=10)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Return', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # 保存图片（高分辨率）
    if save_path is None:
        save_path = os.path.join(model_dir, 'returns_curve.png')
    
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    print(f"回报曲线已保存到: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='单独绘制MAPPO的回报曲线')
    parser.add_argument('--model_dir', type=str, default='multi_agent/algorithms/models',
                       help='模型目录路径（包含training_stats.json）')
    parser.add_argument('--save_path', type=str, default=None,
                       help='保存路径（如果为None，则保存到model_dir/returns_curve.png）')
    parser.add_argument('--show', action='store_true',
                       help='显示图表')
    
    args = parser.parse_args()
    
    plot_returns_from_training_curves(
        model_dir=args.model_dir,
        save_path=args.save_path,
        show_plot=args.show
    )
