#!/usr/bin/env python3
"""
性能对比分析工具
对比baseline和改进版本的性能指标
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt


def load_metrics(json_path: str) -> Dict:
    """加载评估指标"""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except:
        return {}


def compare_metrics(baseline_metrics: Dict, improved_metrics: Dict) -> Dict:
    """
    对比两个版本的指标
    
    Args:
        baseline_metrics: 基线版本指标
        improved_metrics: 改进版本指标
        
    Returns:
        对比结果
    """
    comparison = {
        'APE': {},
        'ATE': {},
        'ARE': {},
    }
    
    # APE对比
    if 'APE_RMSE' in baseline_metrics and 'APE_RMSE' in improved_metrics:
        baseline_ape = baseline_metrics['APE_RMSE']
        improved_ape = improved_metrics['APE_RMSE']
        improvement = (baseline_ape - improved_ape) / baseline_ape * 100
        
        comparison['APE'] = {
            'baseline': baseline_ape,
            'improved': improved_ape,
            'improvement_percent': improvement,
            'improvement_meters': baseline_ape - improved_ape,
        }
    
    # ATE对比
    if 'ATE_RMSE' in baseline_metrics and 'ATE_RMSE' in improved_metrics:
        baseline_ate = baseline_metrics['ATE_RMSE']
        improved_ate = improved_metrics['ATE_RMSE']
        improvement = (baseline_ate - improved_ate) / baseline_ate * 100
        
        comparison['ATE'] = {
            'baseline': baseline_ate,
            'improved': improved_ate,
            'improvement_percent': improvement,
            'improvement_meters': baseline_ate - improved_ate,
        }
    
    # ARE对比
    if 'ARE_RMSE' in baseline_metrics and 'ARE_RMSE' in improved_metrics:
        baseline_are = baseline_metrics['ARE_RMSE']
        improved_are = improved_metrics['ARE_RMSE']
        # 旋转误差用度数表示
        improvement = (baseline_are - improved_are) / baseline_are * 100
        
        comparison['ARE'] = {
            'baseline': baseline_are,
            'improved': improved_are,
            'improvement_percent': improvement,
            'improvement_deg': baseline_are - improved_are,
        }
    
    return comparison


def print_comparison_report(comparison: Dict, method_name: str = "Enhanced VO"):
    """
    打印对比报告
    
    Args:
        comparison: 对比结果
        method_name: 改进方法名称
    """
    print("\n" + "="*70)
    print(f"📊 性能对比报告: {method_name}")
    print("="*70)
    
    print(f"\n🎯 APE (绝对位姿误差)")
    print(f"   │")
    if comparison['APE']:
        ape = comparison['APE']
        baseline = ape['baseline']
        improved = ape['improved']
        improvement = ape['improvement_percent']
        
        print(f"   ├─ Baseline:      {baseline:.6f} m")
        print(f"   ├─ Improved:      {improved:.6f} m")
        print(f"   ├─ Improvement:   {improvement:+.2f}%")
        if improvement < 0:
            print(f"   └─ ⚠️  性能下降了 {abs(improvement):.2f}%")
        else:
            print(f"   └─ ✓ 性能提升了 {improvement:.2f}%")
    else:
        print(f"   └─ ✗ 数据不可用")
    
    print(f"\n🎯 ATE (绝对轨迹误差)")
    print(f"   │")
    if comparison['ATE']:
        ate = comparison['ATE']
        baseline = ate['baseline']
        improved = ate['improved']
        improvement = ate['improvement_percent']
        
        print(f"   ├─ Baseline:      {baseline:.6f} m")
        print(f"   ├─ Improved:      {improved:.6f} m")
        print(f"   ├─ Improvement:   {improvement:+.2f}%")
        if improvement < 0:
            print(f"   └─ ⚠️  性能下降了 {abs(improvement):.2f}%")
        else:
            print(f"   └─ ✓ 性能提升了 {improvement:.2f}%")
    else:
        print(f"   └─ ✗ 数据不可用")
    
    print(f"\n🎯 ARE (绝对旋转误差)")
    print(f"   │")
    if comparison['ARE']:
        are = comparison['ARE']
        baseline = are['baseline']
        improved = are['improved']
        improvement = are['improvement_percent']
        
        print(f"   ├─ Baseline:      {baseline:.6f}°")
        print(f"   ├─ Improved:      {improved:.6f}°")
        print(f"   ├─ Improvement:   {improvement:+.2f}%")
        if improvement < 0:
            print(f"   └─ ⚠️  性能下降了 {abs(improvement):.2f}%")
        else:
            print(f"   └─ ✓ 性能提升了 {improvement:.2f}%")
    else:
        print(f"   └─ ✗ 数据不可用")
    
    print("\n" + "="*70)


def create_comparison_chart(comparison: Dict, output_path: str = "evaluation_results/comparison.png"):
    """
    创建对比柱状图
    
    Args:
        comparison: 对比结果
        output_path: 输出图表路径
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('LVI-SAM 性能对比: Baseline vs Enhanced VO', fontsize=14, fontweight='bold')
    
    metrics = ['APE', 'ATE', 'ARE']
    colors_baseline = '#FF6B6B'
    colors_improved = '#4ECDC4'
    
    for idx, metric_name in enumerate(metrics):
        ax = axes[idx]
        metric = comparison.get(metric_name, {})
        
        if metric:
            baseline = metric['baseline']
            improved = metric['improved']
            
            x = np.arange(2)
            values = [baseline, improved]
            bars = ax.bar(x, values, color=[colors_baseline, colors_improved], alpha=0.8, edgecolor='black', linewidth=2)
            
            # 添加数值标签
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}',
                       ha='center', va='bottom', fontweight='bold', fontsize=11)
            
            # 改进百分比
            improvement = metric['improvement_percent']
            ax.text(0.5, max(values) * 0.5, f'{improvement:+.1f}%',
                   ha='center', va='center', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            ax.set_xticks(x)
            ax.set_xticklabels(['Baseline', 'Enhanced'], fontsize=11, fontweight='bold')
            ax.set_ylabel('Error (m)' if metric_name != 'ARE' else 'Error (deg)', fontsize=11, fontweight='bold')
            ax.set_title(metric_name, fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 对比图表已保存: {output_path}")
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="性能对比分析")
    parser.add_argument('--baseline-metrics', default='evaluation_results/baseline_metrics.json',
                       help='基线指标JSON文件')
    parser.add_argument('--improved-metrics', default='evaluation_results/enhanced_vo_metrics.json',
                       help='改进版本指标JSON文件')
    parser.add_argument('--method', default='Enhanced VO',
                       help='改进方法名称')
    parser.add_argument('--chart', '-c', action='store_true',
                       help='生成对比图表')
    
    args = parser.parse_args()
    
    # 加载指标
    print(f"📂 加载基线指标: {args.baseline_metrics}")
    baseline_metrics = load_metrics(args.baseline_metrics)
    
    print(f"📂 加载改进指标: {args.improved_metrics}")
    improved_metrics = load_metrics(args.improved_metrics)
    
    # 对比
    comparison = compare_metrics(baseline_metrics, improved_metrics)
    
    # 打印报告
    print_comparison_report(comparison, args.method)
    
    # 生成图表
    if args.chart:
        create_comparison_chart(comparison)
    
    # 保存对比结果
    output_json = 'evaluation_results/comparison_report.json'
    Path('evaluation_results').mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w') as f:
        # 转换为可序列化的格式
        comparison_serializable = {}
        for metric_name, metric_data in comparison.items():
            comparison_serializable[metric_name] = {
                k: float(v) if isinstance(v, (int, float, np.number)) else v
                for k, v in metric_data.items()
            }
        json.dump(comparison_serializable, f, indent=2)
    
    print(f"\n✓ 对比报告已保存: {output_json}")


if __name__ == '__main__':
    main()
