"""
运行所有可视化脚本
Run all visualization scripts
"""
import sys
import os
import argparse
import subprocess

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)


def run_visualization(script_name, data_file, output_dir, additional_args=None):
    """
    运行可视化脚本
    
    Args:
        script_name: 脚本名称（不含.py）
        data_file: 数据文件路径
        output_dir: 输出目录
        additional_args: 额外的命令行参数（字典）
    """
    script_path = os.path.join(os.path.dirname(__file__), f'{script_name}.py')
    
    if not os.path.exists(script_path):
        print(f"警告: 脚本 {script_name}.py 不存在，跳过")
        return False
    
    cmd = [sys.executable, script_path, '--data_file', data_file, '--output_dir', output_dir]
    
    if additional_args:
        for key, value in additional_args.items():
            if value is not None:
                if isinstance(value, bool):
                    if value:
                        cmd.append(f'--{key}')
                else:
                    cmd.extend([f'--{key}', str(value)])
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"错误: 运行 {script_name}.py 失败")
        print(e.stderr)
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行所有可视化脚本')
    parser.add_argument('--data_file', type=str, required=True,
                       help='JSON数据文件路径（comparison_data.json）')
    parser.add_argument('--output_dir', type=str, default='multi_agent/visualization/output',
                       help='输出目录 (default: multi_agent/visualization/output)')
    parser.add_argument('--show_plot', action='store_true',
                       help='显示图表 (默认: False, 只保存不显示)')
    parser.add_argument('--skip_peak', action='store_true',
                       help='跳过峰值对比图')
    parser.add_argument('--skip_ess', action='store_true',
                       help='跳过社区储能使用图')
    parser.add_argument('--skip_credit', action='store_true',
                       help='跳过积分余额图')
    parser.add_argument('--skip_radar', action='store_true',
                       help='跳过雷达图')
    parser.add_argument('--skip_smoothness', action='store_true',
                       help='跳过平滑度对比图')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("运行所有可视化脚本")
    print("=" * 80)
    print(f"数据文件: {args.data_file}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 80)
    
    # 运行各个可视化脚本
    visualizations = [
        ('plot_peak_comparison', not args.skip_peak, {}),
        ('plot_community_ess_usage', not args.skip_ess, {'method': 'mappo'}),
        ('plot_credit_balance', not args.skip_credit, {'method': 'mappo'}),
        ('plot_multi_objective_tradeoff', not args.skip_radar, {}),
        ('plot_load_smoothness', not args.skip_smoothness, {}),
    ]
    
    success_count = 0
    for script_name, should_run, additional_args in visualizations:
        if not should_run:
            print(f"\n跳过: {script_name}")
            continue
        
        print(f"\n{'='*80}")
        print(f"运行: {script_name}")
        print(f"{'='*80}")
        
        additional_args['show_plot'] = args.show_plot
        
        if run_visualization(script_name, args.data_file, args.output_dir, additional_args):
            success_count += 1
            print(f"✓ {script_name} 完成")
        else:
            print(f"✗ {script_name} 失败")
    
    print("\n" + "=" * 80)
    print(f"完成！成功运行 {success_count}/{len([v for v in visualizations if v[1]])} 个可视化脚本")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断操作。")
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()
