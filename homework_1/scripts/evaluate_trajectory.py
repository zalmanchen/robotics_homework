#!/usr/bin/env python3
"""
轨迹评估脚本
使用EVO工具计算APE、ATE、ARE等性能指标
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple

try:
    from evo.tools import file_interface
    from evo.main_ape import ape
    from evo.core import sync
    from evo.tools import plot
except ImportError:
    print("Error: EVO not installed. Install with: pip install evo")
    sys.exit(1)


class TrajectoryEvaluator:
    """
    轨迹评估工具
    """
    
    def __init__(self, output_dir: str = "evaluation_results"):
        """
        初始化评估器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def evaluate(
        self,
        estimated_traj_path: str,
        reference_traj_path: str,
        method_name: str = "Unknown"
    ) -> Dict:
        """
        评估估计轨迹相对于真值轨迹的误差
        
        Args:
            estimated_traj_path: 估计轨迹文件路径 (TUM格式)
            reference_traj_path: 真值轨迹文件路径 (TUM格式)
            method_name: 方法名称
            
        Returns:
            包含性能指标的字典
        """
        print(f"\n{'='*60}")
        print(f"评估方法: {method_name}")
        print(f"{'='*60}")
        
        # 加载轨迹
        try:
            traj_ref = file_interface.read_tum_trajectory_file(reference_traj_path)
            traj_est = file_interface.read_tum_trajectory_file(estimated_traj_path)
            print(f"✓ 成功加载参考轨迹: {len(traj_ref.positions_xyz)} 个位置")
            print(f"✓ 成功加载估计轨迹: {len(traj_est.positions_xyz)} 个位置")
        except Exception as e:
            print(f"✗ 加载轨迹失败: {e}")
            return {}
        
        # 同步轨迹（对齐时间戳）
        try:
            traj_est_sync, traj_ref_sync = sync.associate_trajectories(traj_ref, traj_est)
            print(f"✓ 同步后: {len(traj_est_sync.positions_xyz)} 个对应点")
        except Exception as e:
            print(f"✗ 轨迹同步失败: {e}")
            return {}
        
        # 计算APE (Absolute Pose Error)
        metrics = {}
        try:
            from evo.core.metrics import PoseRelation
            ape_result = ape(traj_ref_sync, traj_est_sync, PoseRelation.translation_part)
            metrics['APE_RMSE'] = float(ape_result.stats['rmse'])
            metrics['APE_Mean'] = float(ape_result.stats['mean'])
            metrics['APE_Median'] = float(ape_result.stats['median'])
            metrics['APE_Std'] = float(ape_result.stats['std'])
            metrics['APE_Min'] = float(ape_result.stats['min'])
            metrics['APE_Max'] = float(ape_result.stats['max'])
            
            print(f"\n█ APE (绝对位姿误差)")
            print(f"  RMSE:   {metrics['APE_RMSE']:.6f} m")
            print(f"  Mean:   {metrics['APE_Mean']:.6f} m")
            print(f"  Median: {metrics['APE_Median']:.6f} m")
            print(f"  Std:    {metrics['APE_Std']:.6f} m")
            print(f"  Min:    {metrics['APE_Min']:.6f} m")
            print(f"  Max:    {metrics['APE_Max']:.6f} m")
        except Exception as e:
            print(f"✗ APE计算失败: {e}")
        
        # 计算ATE (Absolute Trajectory Error)
        try:
            ate_errors = self._compute_ate(traj_ref_sync, traj_est_sync)
            metrics['ATE_RMSE'] = float(np.sqrt(np.mean(ate_errors ** 2)))
            metrics['ATE_Mean'] = float(np.mean(ate_errors))
            metrics['ATE_Std'] = float(np.std(ate_errors))
            
            print(f"\n█ ATE (绝对轨迹误差)")
            print(f"  RMSE: {metrics['ATE_RMSE']:.6f} m")
            print(f"  Mean: {metrics['ATE_Mean']:.6f} m")
            print(f"  Std:  {metrics['ATE_Std']:.6f} m")
        except Exception as e:
            print(f"✗ ATE计算失败: {e}")
        
        # 计算ARE (Absolute Rotation Error)
        try:
            are_errors = self._compute_are(traj_ref_sync, traj_est_sync)
            metrics['ARE_RMSE'] = float(np.sqrt(np.mean(are_errors ** 2)))
            metrics['ARE_Mean'] = float(np.mean(are_errors))
            metrics['ARE_Std'] = float(np.std(are_errors))
            
            print(f"\n█ ARE (绝对旋转误差)")
            print(f"  RMSE: {metrics['ARE_RMSE']:.6f} deg")
            print(f"  Mean: {metrics['ARE_Mean']:.6f} deg")
            print(f"  Std:  {metrics['ARE_Std']:.6f} deg")
        except Exception as e:
            print(f"✗ ARE计算失败: {e}")
        
        # 绘制轨迹对比图
        try:
            self._plot_trajectory_comparison(
                traj_ref_sync,
                traj_est_sync,
                method_name,
                ape_result if 'ape_result' in locals() else None
            )
        except Exception as e:
            print(f"⚠ 绘图失败: {e}")
        
        return metrics
    
    def _compute_ate(self, traj_ref, traj_est) -> np.ndarray:
        """
        计算绝对轨迹误差 (ATE)
        ATE = ||p_ref(t) - p_est(t)||
        """
        positions_ref = traj_ref.positions_xyz
        positions_est = traj_est.positions_xyz
        
        errors = np.linalg.norm(positions_ref - positions_est, axis=1)
        return errors
    
    def _compute_are(self, traj_ref, traj_est) -> np.ndarray:
        """
        计算绝对旋转误差 (ARE)
        计算两个旋转矩阵之间的角度差
        """
        from scipy.spatial.transform import Rotation
        
        orientations_ref = traj_ref.orientations_quat_wxyz
        orientations_est = traj_est.orientations_quat_wxyz
        
        errors_deg = []
        for q_ref, q_est in zip(orientations_ref, orientations_est):
            # 转换为旋转矩阵
            R_ref = Rotation.from_quat([q_ref[1], q_ref[2], q_ref[3], q_ref[0]]).as_matrix()
            R_est = Rotation.from_quat([q_est[1], q_est[2], q_est[3], q_est[0]]).as_matrix()
            
            # 计算相对旋转
            R_rel = R_est @ R_ref.T
            
            # 从旋转矩阵提取角度
            angle = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))
            errors_deg.append(np.degrees(angle))
        
        return np.array(errors_deg)
    
    def _plot_trajectory_comparison(self, traj_ref, traj_est, method_name, ape_result=None):
        """
        绘制参考轨迹和估计轨迹的对比图
        """
        pos_ref = traj_ref.positions_xyz
        pos_est = traj_est.positions_xyz
        
        fig = plt.figure(figsize=(15, 5))
        
        # XY平面
        ax1 = fig.add_subplot(131)
        ax1.plot(pos_ref[:, 0], pos_ref[:, 1], 'b-', label='Reference', linewidth=2)
        ax1.plot(pos_est[:, 0], pos_est[:, 1], 'r-', label='Estimated', linewidth=2)
        ax1.scatter(pos_ref[0, 0], pos_ref[0, 1], c='blue', s=100, marker='s', label='Start (Ref)')
        ax1.scatter(pos_est[0, 0], pos_est[0, 1], c='red', s=100, marker='s', label='Start (Est)')
        ax1.scatter(pos_ref[-1, 0], pos_ref[-1, 1], c='blue', s=100, marker='^', label='End (Ref)')
        ax1.scatter(pos_est[-1, 0], pos_est[-1, 1], c='red', s=100, marker='^', label='End (Est)')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('XY平面')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # XZ平面
        ax2 = fig.add_subplot(132)
        ax2.plot(pos_ref[:, 0], pos_ref[:, 2], 'b-', label='Reference', linewidth=2)
        ax2.plot(pos_est[:, 0], pos_est[:, 2], 'r-', label='Estimated', linewidth=2)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Z (m)')
        ax2.set_title('XZ平面')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 位置误差随时间的变化
        ax3 = fig.add_subplot(133)
        try:
            if ape_result and hasattr(ape_result, 'errors'):
                errors = ape_result.errors
            elif ape_result and hasattr(ape_result, 'error_array'):
                errors = ape_result.error_array
            else:
                # 手动计算位置误差
                errors = np.linalg.norm(pos_est - pos_ref, axis=1)
            
            ax3.plot(errors, 'g-', linewidth=2)
            ax3.fill_between(range(len(errors)), 0, errors, alpha=0.3)
            ax3.axhline(y=np.mean(errors), color='r', linestyle='--', label=f'Mean: {np.mean(errors):.4f}')
            ax3.set_xlabel('Frame Index')
            ax3.set_ylabel('APE Error (m)')
            ax3.set_title('APE随时间变化')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        except Exception as e:
            ax3.text(0.5, 0.5, f'绘图失败:\n{str(e)}', 
                    ha='center', va='center', transform=ax3.transAxes)
        
        plt.tight_layout()
        output_path = self.output_dir / f"{method_name}_trajectory.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ 轨迹对比图已保存: {output_path}")
        plt.close()
    
    def compare_methods(self, results: Dict[str, Dict]) -> None:
        """
        比较多个方法的性能
        
        Args:
            results: 多个方法的评估结果
        """
        print(f"\n{'='*80}")
        print("性能比较总结")
        print(f"{'='*80}\n")
        
        # 创建表格
        methods = list(results.keys())
        metrics_keys = ['APE_RMSE', 'ATE_RMSE', 'ARE_RMSE']
        
        # 打印表头
        print(f"{'Method':<30}", end='')
        for key in metrics_keys:
            print(f"{key:<15}", end='')
        print()
        print("-" * 80)
        
        # 打印数据
        for method in methods:
            print(f"{method:<30}", end='')
            for key in metrics_keys:
                if key in results[method]:
                    print(f"{results[method][key]:<15.6f}", end='')
                else:
                    print(f"{'N/A':<15}", end='')
            print()
        
        # 保存结果表为CSV
        csv_path = self.output_dir / "comparison_results.csv"
        with open(csv_path, 'w') as f:
            f.write("Method," + ",".join(metrics_keys) + "\n")
            for method in methods:
                f.write(f"{method},")
                values = [str(results[method].get(key, 'N/A')) for key in metrics_keys]
                f.write(",".join(values) + "\n")
        
        print(f"\n✓ 结果已保存: {csv_path}")
        
        # 绘制对比图
        self._plot_comparison_chart(results, metrics_keys)
    
    def _plot_comparison_chart(self, results: Dict[str, Dict], metrics_keys: list) -> None:
        """
        绘制方法对比柱状图
        """
        fig, axes = plt.subplots(1, len(metrics_keys), figsize=(15, 5))
        
        methods = list(results.keys())
        
        for idx, metric in enumerate(metrics_keys):
            ax = axes[idx]
            values = [results[m].get(metric, 0) for m in methods]
            
            bars = ax.bar(range(len(methods)), values, color='steelblue', alpha=0.8)
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels(methods, rotation=45, ha='right')
            ax.set_ylabel('Error (m)' if 'E' in metric else 'Error (deg)')
            ax.set_title(metric)
            ax.grid(True, alpha=0.3, axis='y')
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        output_path = self.output_dir / "method_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ 对比图已保存: {output_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='轨迹评估工具')
    parser.add_argument('--estimated', type=str, required=True, help='估计轨迹文件路径')
    parser.add_argument('--reference', type=str, required=True, help='参考轨迹文件路径')
    parser.add_argument('--method', type=str, default='Unknown', help='方法名称')
    parser.add_argument('--output', type=str, default='evaluation_results', help='输出目录')
    
    args = parser.parse_args()
    
    evaluator = TrajectoryEvaluator(output_dir=args.output)
    metrics = evaluator.evaluate(args.estimated, args.reference, args.method)
    
    # 保存指标到JSON
    json_path = Path(args.output) / f"{args.method}_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ 指标已保存: {json_path}")


if __name__ == "__main__":
    main()
