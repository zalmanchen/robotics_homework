#!/usr/bin/env python3
"""
Complete path and trajectory planning system with result saving.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from flight_environment import FlightEnvironment
from path_planner import plan_path
from trajectory_generator import TrajectoryGenerator
import numpy as np
import os
from datetime import datetime

def main():
    # Create results directory
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Create timestamp for unique folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_folder = os.path.join(results_dir, f"planning_result_{timestamp}")
    os.makedirs(result_folder, exist_ok=True)
    
    print("\n" + "="*70)
    print("INTELLIGENT MOBILE ROBOTICS - PATH AND TRAJECTORY PLANNING")
    print("="*70)
    print(f"Results will be saved to: {result_folder}\n")
    
    # ============================================================================
    # STEP 1: PATH PLANNING
    # ============================================================================
    print("[STEP 1] Initializing flight environment...")
    env = FlightEnvironment(50)
    start = (2.0, 2.0, 1.0)
    goal = (17.0, 17.0, 2.0)
    
    print(f"Environment: {env.env_width}m × {env.env_length}m × {env.env_height}m")
    print(f"Obstacles: 50 cylinders")
    print(f"Start position: {start}")
    print(f"Goal position: {goal}\n")
    
    print("[STEP 1] Running RRT path planning algorithm...")
    path = plan_path(env, start, goal)
    
    if path is None:
        print("✗ Failed to find path. Exiting.")
        return
    
    path_length = np.sum([np.linalg.norm(path[i+1] - path[i]) for i in range(len(path)-1)])
    print(f"✓ Path found successfully!")
    print(f"  - Number of waypoints: {len(path)}")
    print(f"  - Total path length: {path_length:.2f} m\n")
    
    # Save path data
    path_file = os.path.join(result_folder, "path_waypoints.csv")
    with open(path_file, 'w') as f:
        f.write("Waypoint_ID,X(m),Y(m),Z(m)\n")
        for i, wp in enumerate(path):
            f.write(f"{i},{wp[0]:.4f},{wp[1]:.4f},{wp[2]:.4f}\n")
    print(f"✓ Path waypoints saved to: path_waypoints.csv")
    
    # ============================================================================
    # SAVE 3D PATH VISUALIZATION
    # ============================================================================
    print("\n[STEP 1.5] Generating 3D path visualization with obstacles...")
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot cylinders
    cylinders = env.cylinders
    for cx, cy, h, r in cylinders:
        z = np.linspace(0, h, 20)
        theta = np.linspace(0, 2 * np.pi, 20)
        theta, z = np.meshgrid(theta, z)
        
        x = cx + r * np.cos(theta)
        y = cy + r * np.sin(theta)
        
        ax.plot_surface(x, y, z, color='skyblue', alpha=0.6, edgecolor='none')
        
        # Top cap
        theta2 = np.linspace(0, 2*np.pi, 20)
        x_top = cx + r * np.cos(theta2)
        y_top = cy + r * np.sin(theta2)
        z_top = np.ones_like(theta2) * h
        ax.plot_trisurf(x_top, y_top, z_top, color='steelblue', alpha=0.7, edgecolor='none')
    
    # Plot path
    xs, ys, zs = path[:, 0], path[:, 1], path[:, 2]
    ax.plot(xs, ys, zs, 'r-', linewidth=2.5, label='Planned Path')
    ax.scatter(xs[0], ys[0], zs[0], s=150, c='green', marker='o', 
               edgecolors='darkgreen', linewidth=2, label='Start', zorder=10)
    ax.scatter(xs[-1], ys[-1], zs[-1], s=150, c='red', marker='s', 
               edgecolors='darkred', linewidth=2, label='Goal', zorder=10)
    
    # Plot waypoints
    ax.scatter(xs, ys, zs, s=30, c='orange', alpha=0.7, label='Waypoints', zorder=5)
    
    ax.set_xlim(0, env.env_width)
    ax.set_ylim(0, env.env_length)
    ax.set_zlim(0, env.env_height)
    ax.set_xlabel('X (m)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=11, fontweight='bold')
    ax.set_zlabel('Z (m)', fontsize=11, fontweight='bold')
    ax.set_title('3D Path Planning with Cylindrical Obstacles', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    
    env.set_axes_equal(ax)
    
    path_fig_file = os.path.join(result_folder, "3d_path_with_obstacles.png")
    plt.savefig(path_fig_file, dpi=150, bbox_inches='tight')
    print(f"✓ 3D path visualization saved to: 3d_path_with_obstacles.png")
    plt.close()
    
    # ============================================================================
    # STEP 2: TRAJECTORY PLANNING
    # ============================================================================
    print("\n[STEP 2] Running trajectory planning algorithm...")
    trajectory_generator = TrajectoryGenerator(speed=2.0)
    trajectory = trajectory_generator.generate_trajectory(path, num_points=500)
    
    if trajectory is None:
        print("✗ Failed to generate trajectory")
        return
    
    print("✓ Smooth trajectory generated successfully!")
    
    # Get statistics
    stats = trajectory_generator.get_trajectory_stats(trajectory)
    
    # Save trajectory data
    traj_data_file = os.path.join(result_folder, "trajectory_continuous.csv")
    trajectory_generator.save_trajectory_data(trajectory, traj_data_file)
    print(f"✓ Continuous trajectory data saved to: trajectory_continuous.csv")
    
    # Save waypoints data
    waypoints_file = os.path.join(result_folder, "waypoints_with_time.csv")
    trajectory_generator.save_waypoints_data(trajectory, waypoints_file)
    print(f"✓ Waypoints with timing saved to: waypoints_with_time.csv")
    
    # ============================================================================
    # SAVE TRAJECTORY PLOT
    # ============================================================================
    print("\n[STEP 2.5] Generating trajectory plot...")
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('3D Trajectory Planning Results', fontsize=14, fontweight='bold', y=0.995)
    
    t = trajectory['t']
    x = trajectory['x']
    y = trajectory['y']
    z = trajectory['z']
    waypoints = trajectory['waypoints']
    time_points = trajectory['time_points']
    
    # Plot X trajectory
    axes[0].plot(t, x, 'b-', linewidth=2.5, label='Continuous Trajectory', alpha=0.8)
    axes[0].scatter(time_points, waypoints[:, 0], color='red', s=80, zorder=5, 
                    label='Waypoints', edgecolors='darkred', linewidth=1.5)
    axes[0].set_ylabel('X Position (m)', fontsize=11, fontweight='bold')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].legend(loc='best', fontsize=10)
    axes[0].set_title('X-axis Trajectory', fontsize=12, fontweight='bold')
    
    # Plot Y trajectory
    axes[1].plot(t, y, 'g-', linewidth=2.5, label='Continuous Trajectory', alpha=0.8)
    axes[1].scatter(time_points, waypoints[:, 1], color='red', s=80, zorder=5, 
                    label='Waypoints', edgecolors='darkred', linewidth=1.5)
    axes[1].set_ylabel('Y Position (m)', fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].legend(loc='best', fontsize=10)
    axes[1].set_title('Y-axis Trajectory', fontsize=12, fontweight='bold')
    
    # Plot Z trajectory
    axes[2].plot(t, z, 'm-', linewidth=2.5, label='Continuous Trajectory', alpha=0.8)
    axes[2].scatter(time_points, waypoints[:, 2], color='red', s=80, zorder=5, 
                    label='Waypoints', edgecolors='darkred', linewidth=1.5)
    axes[2].set_ylabel('Z Position (m)', fontsize=11, fontweight='bold')
    axes[2].set_xlabel('Time (s)', fontsize=11, fontweight='bold')
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].legend(loc='best', fontsize=10)
    axes[2].set_title('Z-axis Trajectory', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    traj_fig_file = os.path.join(result_folder, "trajectory_plot.png")
    plt.savefig(traj_fig_file, dpi=150, bbox_inches='tight')
    print(f"✓ Trajectory plot saved to: trajectory_plot.png")
    plt.close()
    
    # ============================================================================
    # SAVE SUMMARY REPORT
    # ============================================================================
    print("\n[STEP 3] Generating summary report...")
    
    summary_file = os.path.join(result_folder, "planning_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("PATH AND TRAJECTORY PLANNING SUMMARY REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write("PLANNING PARAMETERS\n")
        f.write("-"*70 + "\n")
        f.write(f"Start Position: ({start[0]:.2f}, {start[1]:.2f}, {start[2]:.2f}) m\n")
        f.write(f"Goal Position: ({goal[0]:.2f}, {goal[1]:.2f}, {goal[2]:.2f}) m\n")
        f.write(f"Environment Size: {env.env_width}m × {env.env_length}m × {env.env_height}m\n")
        f.write(f"Number of Obstacles: 50 cylinders\n")
        f.write(f"Safety Margin: 0.2m\n\n")
        
        f.write("PATH PLANNING RESULTS (RRT Algorithm)\n")
        f.write("-"*70 + "\n")
        f.write(f"Number of Waypoints: {len(path)}\n")
        f.write(f"Total Path Length: {path_length:.2f} m\n")
        f.write(f"Start Point: ({path[0][0]:.2f}, {path[0][1]:.2f}, {path[0][2]:.2f})\n")
        f.write(f"End Point: ({path[-1][0]:.2f}, {path[-1][1]:.2f}, {path[-1][2]:.2f})\n\n")
        
        f.write("TRAJECTORY PLANNING RESULTS (Cubic Spline)\n")
        f.write("-"*70 + "\n")
        f.write(f"Total Execution Time: {stats['total_time']:.2f} s\n")
        f.write(f"Total Distance Traveled: {stats['total_distance']:.2f} m\n")
        f.write(f"Average Speed: {stats['average_speed']:.2f} m/s\n")
        f.write(f"Number of Trajectory Points (sampled): {len(trajectory['t'])}\n")
        f.write(f"Sampling Rate: {len(trajectory['t']) / stats['total_time']:.1f} Hz\n\n")
        
        f.write("OUTPUT FILES\n")
        f.write("-"*70 + "\n")
        f.write("1. path_waypoints.csv\n")
        f.write("   - Discrete path waypoints with coordinates\n\n")
        f.write("2. 3d_path_with_obstacles.png\n")
        f.write("   - 3D visualization of planned path and obstacles\n\n")
        f.write("3. trajectory_continuous.csv\n")
        f.write("   - Continuous trajectory data (time, x, y, z)\n\n")
        f.write("4. waypoints_with_time.csv\n")
        f.write("   - Waypoint timing information\n\n")
        f.write("5. trajectory_plot.png\n")
        f.write("   - Three subplots showing x(t), y(t), z(t) with waypoints overlaid\n\n")
        f.write("6. planning_summary.txt\n")
        f.write("   - This summary report\n\n")
        
        f.write("="*70 + "\n")
        f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n")
    
    print(f"✓ Summary report saved to: planning_summary.txt")
    
    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    print("\n" + "="*70)
    print("PLANNING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nAll results saved to directory:")
    print(f"  {result_folder}\n")
    print("Generated files:")
    print("  ✓ path_waypoints.csv")
    print("  ✓ 3d_path_with_obstacles.png")
    print("  ✓ trajectory_continuous.csv")
    print("  ✓ waypoints_with_time.csv")
    print("  ✓ trajectory_plot.png")
    print("  ✓ planning_summary.txt")
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
