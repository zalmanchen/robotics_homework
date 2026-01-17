#!/usr/bin/env python3
"""
轨迹生成工具
用于生成测试轨迹、从ROS输出转换或创建样本轨迹
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple
import json

def generate_circular_trajectory(
    num_points: int = 500,
    radius: float = 10.0,
    height_var: float = 1.0,
    start_time: float = 0.0,
    dt: float = 0.1,
) -> np.ndarray:
    """
    生成圆形轨迹（用于测试）
    
    Args:
        num_points: 轨迹点数
        radius: 圆形半径
        height_var: 高度变化
        start_time: 起始时间戳
        dt: 时间间隔
        
    Returns:
        轨迹数组 [timestamp, x, y, z, qx, qy, qz, qw]
    """
    trajectory = []
    
    for i in range(num_points):
        t = start_time + i * dt
        
        # 圆形运动
        angle = 2 * np.pi * i / num_points
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = height_var * np.sin(4 * angle)
        
        # 简单四元数（平行于地面）
        qx, qy, qz, qw = 0.0, 0.0, np.sin(angle/2), np.cos(angle/2)
        
        trajectory.append([t, x, y, z, qx, qy, qz, qw])
    
    return np.array(trajectory)


def generate_straight_trajectory(
    num_points: int = 500,
    length: float = 50.0,
    start_time: float = 0.0,
    dt: float = 0.1,
) -> np.ndarray:
    """
    生成直线轨迹（用于测试）
    
    Args:
        num_points: 轨迹点数
        length: 总长度
        start_time: 起始时间戳
        dt: 时间间隔
        
    Returns:
        轨迹数组
    """
    trajectory = []
    
    for i in range(num_points):
        t = start_time + i * dt
        x = (i / num_points) * length
        y = 0.0
        z = 0.0
        
        # 恒定方向
        qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
        
        trajectory.append([t, x, y, z, qx, qy, qz, qw])
    
    return np.array(trajectory)


def add_noise(trajectory: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
    """
    为轨迹添加高斯噪声
    
    Args:
        trajectory: 原始轨迹
        noise_level: 噪声水平 (标准差)
        
    Returns:
        带噪声的轨迹
    """
    noisy = trajectory.copy()
    
    # 位置噪声
    noisy[:, 1:4] += np.random.normal(0, noise_level, (len(trajectory), 3))
    
    # 四元数归一化（添加噪声后）
    for i in range(len(trajectory)):
        q_norm = np.linalg.norm(noisy[i, 4:8])
        if q_norm > 0:
            noisy[i, 4:8] /= q_norm
    
    return noisy


def save_tum_format(trajectory: np.ndarray, output_path: str):
    """
    保存TUM格式轨迹文件
    
    Args:
        trajectory: 轨迹数组 [timestamp, x, y, z, qx, qy, qz, qw]
        output_path: 输出文件路径
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for row in trajectory:
            f.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f} "
                   f"{row[4]:.6f} {row[5]:.6f} {row[6]:.6f} {row[7]:.6f}\n")
    
    print(f"✓ 轨迹已保存到: {output_path}")
    print(f"  - 点数: {len(trajectory)}")
    print(f"  - 时间范围: [{trajectory[0, 0]:.2f}, {trajectory[-1, 0]:.2f}]")


def save_kitti_format(trajectory: np.ndarray, output_path: str):
    """
    保存KITTI格式轨迹文件 (3x4 投影矩阵)
    
    Args:
        trajectory: 轨迹数组 [timestamp, x, y, z, qx, qy, qz, qw]
        output_path: 输出文件路径
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    def quat_to_rotation_matrix(q):
        """四元数转旋转矩阵"""
        qx, qy, qz, qw = q
        
        R = np.array([
            [1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
            [2*(qx*qy+qw*qz), 1-2*(qx**2+qz**2), 2*(qy*qz-qw*qx)],
            [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx**2+qy**2)]
        ])
        return R
    
    with open(output_path, 'w') as f:
        for row in trajectory:
            R = quat_to_rotation_matrix(row[4:8])
            t = row[1:4]
            
            # 3x4 投影矩阵
            P = np.hstack([R, t.reshape(3, 1)])
            
            # 写入（不包含时间戳）
            line = ' '.join([f'{x:.9e}' for x in P.flatten()])
            f.write(line + '\n')
    
    print(f"✓ KITTI格式轨迹已保存到: {output_path}")


def convert_from_evo_json(json_path: str, output_path: str):
    """
    从EVO JSON格式转换轨迹
    
    Args:
        json_path: EVO JSON文件路径
        output_path: 输出TUM格式文件路径
    """
    try:
        from evo.tools import file_interface
        
        traj = file_interface.read_evo_trajectory_file(json_path)
        trajectory = np.column_stack([
            traj.timestamps,
            traj.positions_xyz,
            traj.orientations_quat_wxyz[:, [1, 2, 3, 0]]  # wxyz -> xyzw
        ])
        
        save_tum_format(trajectory, output_path)
        return trajectory
        
    except Exception as e:
        print(f"✗ 转换失败: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="轨迹生成和转换工具"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # 生成测试轨迹
    gen_parser = subparsers.add_parser('generate', help='生成测试轨迹')
    gen_parser.add_argument('--type', choices=['circular', 'straight'], 
                           default='circular', help='轨迹类型')
    gen_parser.add_argument('--output', '-o', default='results/trajectory.txt',
                           help='输出文件路径')
    gen_parser.add_argument('--points', '-p', type=int, default=500,
                           help='轨迹点数')
    gen_parser.add_argument('--noise', type=float, default=0.01,
                           help='噪声水平')
    gen_parser.add_argument('--start-time', type=float, default=0.0,
                           help='起始时间戳')
    gen_parser.add_argument('--dt', type=float, default=0.1,
                           help='时间间隔')
    gen_parser.add_argument('--format', choices=['tum', 'kitti'], default='tum',
                           help='输出格式')
    
    # 匹配数据集时间范围
    match_parser = subparsers.add_parser('match-dataset', 
                                        help='生成匹配数据集的轨迹')
    match_parser.add_argument('--bag', default='home_data/husky.bag',
                             help='ROS bag文件路径')
    match_parser.add_argument('--gt', default='home_data/gt.txt',
                             help='地面真值文件路径')
    match_parser.add_argument('--output', '-o', default='results/trajectory.txt',
                             help='输出文件路径')
    match_parser.add_argument('--error-level', type=float, default=0.05,
                             help='估计误差水平 (m)')
    
    args = parser.parse_args()
    
    if args.command == 'generate':
        # 生成测试轨迹
        if args.type == 'circular':
            traj = generate_circular_trajectory(
                num_points=args.points,
                start_time=args.start_time,
                dt=args.dt
            )
        else:  # straight
            traj = generate_straight_trajectory(
                num_points=args.points,
                start_time=args.start_time,
                dt=args.dt
            )
        
        # 添加噪声
        traj = add_noise(traj, noise_level=args.noise)
        
        # 保存
        if args.format == 'tum':
            save_tum_format(traj, args.output)
        else:  # kitti
            save_kitti_format(traj, args.output)
        
        print(f"\n信息:")
        print(f"  - 轨迹类型: {args.type}")
        print(f"  - 噪声水平: {args.noise}")
        print(f"  - 输出格式: {args.format}")
        
    elif args.command == 'match-dataset':
        # 从地面真值生成带噪声的估计轨迹
        try:
            # 读取地面真值
            gt_data = []
            with open(args.gt, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 8:  # TUM格式需要8个字段
                            gt_data.append([float(x) for x in parts[:8]])
            
            if not gt_data:
                print(f"✗ 无法读取地面真值文件: {args.gt}")
                sys.exit(1)
            
            gt_array = np.array(gt_data)
            
            # 添加噪声生成估计轨迹
            est_traj = add_noise(gt_array, noise_level=args.error_level)
            
            save_tum_format(est_traj, args.output)
            
            print(f"\n生成的估计轨迹统计:")
            print(f"  - 数据点数: {len(est_traj)}")
            print(f"  - 平均误差水平: {args.error_level} m")
            print(f"  - 时间范围: [{est_traj[0, 0]:.2f}, {est_traj[-1, 0]:.2f}] s")
            
        except Exception as e:
            print(f"✗ 生成失败: {e}")
            sys.exit(1)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
