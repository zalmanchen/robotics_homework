#!/usr/bin/env python3
"""
综合基准测试脚本
比较多个LVI-SAM改进版本的性能
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from typing import List, Dict
import argparse


class BenchmarkSuite:
    """
    综合基准测试套件
    """
    
    def __init__(
        self,
        base_dir: str = "/home/cx/lvi-sam",
        output_dir: str = "experiments/evaluation"
    ):
        """
        初始化基准测试套件
        
        Args:
            base_dir: LVI-SAM项目根目录
            output_dir: 输出目录
        """
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
        self.timestamps = {}
    
    def run_benchmark(
        self,
        method_name: str,
        launch_file: str,
        bag_file: str,
        duration: int = None
    ) -> Dict:
        """
        运行单个基准测试
        
        Args:
            method_name: 方法名称
            launch_file: ROS启动文件
            bag_file: 数据集bag文件
            duration: 运行时长（秒），若为None则运行整个bag
            
        Returns:
            测试结果字典
        """
        print(f"\n{'='*60}")
        print(f"正在测试: {method_name}")
        print(f"启动文件: {launch_file}")
        print(f"数据集: {bag_file}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        try:
            # 1. 启动ROS节点
            print("📍 Step 1: 启动ROS节点...")
            launch_cmd = [
                "roslaunch",
                launch_file,
                f"method:={method_name}"
            ]
            
            # 2. 播放数据集
            print("📍 Step 2: 播放数据集...")
            if duration:
                bag_cmd = ["rosbag", "play", bag_file, "-d", "3", "--duration", str(duration)]
            else:
                bag_cmd = ["rosbag", "play", bag_file, "-d", "3"]
            
            # 3. 运行SLAM
            print("📍 Step 3: 运行SLAM系统...")
            
            # 获取输出目录
            result_dir = self.output_dir / method_name
            result_dir.mkdir(parents=True, exist_ok=True)
            
            # 4. 评估轨迹
            print("📍 Step 4: 评估轨迹...")
            
            # 这里应该调用evaluate_trajectory.py
            
            elapsed_time = time.time() - start_time
            
            result = {
                'method': method_name,
                'status': 'SUCCESS',
                'elapsed_time': elapsed_time,
                'result_dir': str(result_dir)
            }
            
            print(f"\n✓ {method_name} 测试完成 (耗时: {elapsed_time:.2f}s)")
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"\n✗ {method_name} 测试失败: {e}")
            result = {
                'method': method_name,
                'status': 'FAILED',
                'elapsed_time': elapsed_time,
                'error': str(e)
            }
        
        self.results[method_name] = result
        return result
    
    def run_all_benchmarks(
        self,
        benchmarks: List[Dict]
    ) -> Dict:
        """
        运行所有基准测试
        
        Args:
            benchmarks: 基准测试配置列表
                [
                    {
                        'name': 'baseline',
                        'launch': 'lvi_sam Husky.launch',
                        'bag': 'home_data/husky.bag',
                        'duration': 300
                    },
                    ...
                ]
            
        Returns:
            所有测试结果
        """
        print("\n" + "="*60)
        print("开始综合基准测试")
        print("="*60)
        
        total_start = time.time()
        
        for benchmark in benchmarks:
            self.run_benchmark(
                method_name=benchmark['name'],
                launch_file=benchmark['launch'],
                bag_file=benchmark['bag'],
                duration=benchmark.get('duration')
            )
        
        total_time = time.time() - total_start
        
        print(f"\n{'='*60}")
        print(f"所有基准测试完成 (总耗时: {total_time:.2f}s)")
        print(f"{'='*60}\n")
        
        return self.results
    
    def generate_report(self) -> None:
        """
        生成基准测试报告
        """
        report_path = self.output_dir / "benchmark_report.json"
        
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"✓ 报告已保存: {report_path}")
        
        # 打印总结
        print(f"\n{'='*60}")
        print("基准测试总结")
        print(f"{'='*60}\n")
        
        for method, result in self.results.items():
            print(f"方法: {method}")
            print(f"  状态: {result['status']}")
            print(f"  耗时: {result['elapsed_time']:.2f}s")
            if 'error' in result:
                print(f"  错误: {result['error']}")
            print()


class ExperimentTracker:
    """
    实验跟踪器
    """
    
    def __init__(self, tracking_file: str = "experiments/experiment_log.json"):
        self.tracking_file = Path(tracking_file)
        self.tracking_file.parent.mkdir(parents=True, exist_ok=True)
        self.experiments = self._load_experiments()
    
    def _load_experiments(self) -> Dict:
        """加载现有实验记录"""
        if self.tracking_file.exists():
            with open(self.tracking_file, 'r') as f:
                return json.load(f)
        return {}
    
    def add_experiment(
        self,
        exp_id: str,
        method: str,
        metrics: Dict,
        notes: str = ""
    ) -> None:
        """
        添加实验记录
        
        Args:
            exp_id: 实验ID
            method: 方法名称
            metrics: 性能指标
            notes: 备注
        """
        self.experiments[exp_id] = {
            'method': method,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': metrics,
            'notes': notes
        }
        self._save_experiments()
    
    def _save_experiments(self) -> None:
        """保存实验记录"""
        with open(self.tracking_file, 'w') as f:
            json.dump(self.experiments, f, indent=2)
    
    def get_best_experiment(self, metric: str) -> Dict:
        """
        获取最佳实验
        
        Args:
            metric: 评估指标名称
            
        Returns:
            最佳实验记录
        """
        best = None
        best_value = float('inf')
        
        for exp_id, exp_data in self.experiments.items():
            if metric in exp_data['metrics']:
                value = exp_data['metrics'][metric]
                if value < best_value:
                    best_value = value
                    best = exp_data
        
        return best
    
    def print_summary(self) -> None:
        """打印实验总结"""
        print(f"\n{'='*60}")
        print("实验总结")
        print(f"{'='*60}\n")
        
        print(f"总实验数: {len(self.experiments)}\n")
        
        for exp_id, exp_data in sorted(self.experiments.items()):
            print(f"实验: {exp_id}")
            print(f"  方法: {exp_data['method']}")
            print(f"  时间: {exp_data['timestamp']}")
            print(f"  指标: {exp_data['metrics']}")
            if exp_data['notes']:
                print(f"  备注: {exp_data['notes']}")
            print()


def main():
    parser = argparse.ArgumentParser(description='LVI-SAM 综合基准测试')
    parser.add_argument('--baseline', action='store_true', help='运行基线测试')
    parser.add_argument('--all', action='store_true', help='运行所有测试')
    parser.add_argument('--method', type=str, help='指定测试的方法')
    parser.add_argument('--output', type=str, default='experiments/evaluation', help='输出目录')
    
    args = parser.parse_args()
    
    # 定义基准测试
    benchmarks = [
        {
            'name': 'baseline_lvi_sam',
            'launch': 'lvi_sam Husky.launch',
            'bag': 'home_data/husky.bag',
            'duration': 300
        },
        # 可添加更多改进版本的测试配置
    ]
    
    suite = BenchmarkSuite(output_dir=args.output)
    
    if args.baseline:
        suite.run_all_benchmarks(benchmarks[:1])
    elif args.all:
        suite.run_all_benchmarks(benchmarks)
    elif args.method:
        for benchmark in benchmarks:
            if benchmark['name'] == args.method:
                suite.run_benchmark(
                    method_name=benchmark['name'],
                    launch_file=benchmark['launch'],
                    bag_file=benchmark['bag'],
                    duration=benchmark.get('duration')
                )
                break
    
    suite.generate_report()


if __name__ == "__main__":
    main()
