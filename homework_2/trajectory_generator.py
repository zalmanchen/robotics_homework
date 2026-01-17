"""
In this file, you should implement your trajectory generation class or function.
Your method must generate a smooth 3-axis trajectory (x(t), y(t), z(t)) that 
passes through all the previously computed path points. A positional deviation 
up to 0.1 m from each path point is allowed.

You should output the generated trajectory and visualize it. The figure must
contain three subplots showing x, y, and z, respectively, with time t (in seconds)
as the horizontal axis. Additionally, you must plot the original discrete path 
points on the same figure for comparison.

You are expected to write the implementation yourself. Do NOT copy or reuse any 
existing trajectory generation code from others. Avoid using external packages 
beyond general scientific libraries such as numpy, math, or scipy. If you decide 
to use additional packages, you must clearly explain the reason in your report.
"""

import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import os
from datetime import datetime


class TrajectoryGenerator:
    """
    Trajectory generator using cubic spline interpolation.
    
    Generates a smooth 3D trajectory through waypoints using cubic splines,
    which ensures continuous position, velocity, and acceleration.
    """
    
    def __init__(self, speed=2.0):
        """
        Initialize trajectory generator.
        
        Args:
            speed: Average traversal speed along the path (m/s)
        """
        self.speed = speed
    
    def generate_trajectory(self, path, num_points=500):
        """
        Generate a smooth trajectory through path points.
        
        Args:
            path: N×3 numpy array of waypoints
            num_points: Number of interpolated points in the trajectory
            
        Returns:
            trajectory: Dictionary with 't', 'x', 'y', 'z' arrays
        """
        if path is None or len(path) < 2:
            print("Invalid path provided")
            return None
        
        path = np.array(path)
        
        # Calculate arc length between consecutive points
        distances = np.zeros(len(path))
        for i in range(1, len(path)):
            distances[i] = distances[i-1] + np.linalg.norm(path[i] - path[i-1])
        
        # Total path length
        total_distance = distances[-1]
        
        # Time for each waypoint based on speed
        time_points = distances / self.speed
        
        # Create cubic splines for each dimension
        cs_x = CubicSpline(time_points, path[:, 0])
        cs_y = CubicSpline(time_points, path[:, 1])
        cs_z = CubicSpline(time_points, path[:, 2])
        
        # Generate continuous trajectory
        t = np.linspace(0, time_points[-1], num_points)
        traj_x = cs_x(t)
        traj_y = cs_y(t)
        traj_z = cs_z(t)
        
        trajectory = {
            't': t,
            'x': traj_x,
            'y': traj_y,
            'z': traj_z,
            'waypoints': path,
            'time_points': time_points
        }
        
        return trajectory
    
    def plot_trajectory(self, trajectory, title="Trajectory Planning", save_path=None):
        """
        Plot the generated trajectory and waypoints.
        
        Args:
            trajectory: Dictionary returned by generate_trajectory()
            title: Title for the plot
            save_path: Path to save the figure (if None, not saved)
        """
        if trajectory is None:
            print("No trajectory to plot")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 9))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        t = trajectory['t']
        x = trajectory['x']
        y = trajectory['y']
        z = trajectory['z']
        waypoints = trajectory['waypoints']
        time_points = trajectory['time_points']
        
        # Plot x trajectory
        axes[0].plot(t, x, 'b-', linewidth=2, label='Trajectory')
        axes[0].scatter(time_points, waypoints[:, 0], color='r', s=100, zorder=5, label='Waypoints')
        axes[0].set_ylabel('X (m)', fontsize=12)
        axes[0].set_xlabel('Time (s)', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        axes[0].set_title('X Trajectory')
        
        # Plot y trajectory
        axes[1].plot(t, y, 'g-', linewidth=2, label='Trajectory')
        axes[1].scatter(time_points, waypoints[:, 1], color='r', s=100, zorder=5, label='Waypoints')
        axes[1].set_ylabel('Y (m)', fontsize=12)
        axes[1].set_xlabel('Time (s)', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        axes[1].set_title('Y Trajectory')
        
        # Plot z trajectory
        axes[2].plot(t, z, 'm-', linewidth=2, label='Trajectory')
        axes[2].scatter(time_points, waypoints[:, 2], color='r', s=100, zorder=5, label='Waypoints')
        axes[2].set_ylabel('Z (m)', fontsize=12)
        axes[2].set_xlabel('Time (s)', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        axes[2].set_title('Z Trajectory')
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Trajectory figure saved to: {save_path}")
        
        plt.show()
    
    def get_trajectory_stats(self, trajectory):
        """
        Print trajectory statistics.
        
        Args:
            trajectory: Dictionary returned by generate_trajectory()
        
        Returns:
            stats: Dictionary containing statistics
        """
        if trajectory is None:
            return None
        
        t = trajectory['t']
        x = trajectory['x']
        y = trajectory['y']
        z = trajectory['z']
        
        total_time = t[-1]
        total_distance = 0
        for i in range(1, len(x)):
            dist = np.sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2 + (z[i] - z[i-1])**2)
            total_distance += dist
        
        avg_speed = total_distance / total_time if total_time > 0 else 0
        num_waypoints = len(trajectory['waypoints'])
        
        stats = {
            'total_time': total_time,
            'total_distance': total_distance,
            'average_speed': avg_speed,
            'num_waypoints': num_waypoints
        }
        
        print("=" * 50)
        print("Trajectory Statistics")
        print("=" * 50)
        print(f"Total time: {total_time:.2f} s")
        print(f"Total distance: {total_distance:.2f} m")
        print(f"Average speed: {avg_speed:.2f} m/s")
        print(f"Number of waypoints: {num_waypoints}")
        print("=" * 50)
        
        return stats
    
    def save_trajectory_data(self, trajectory, file_path):
        """
        Save trajectory data to a text file.
        
        Args:
            trajectory: Dictionary returned by generate_trajectory()
            file_path: Path to save the data
        """
        if trajectory is None:
            return
        
        with open(file_path, 'w') as f:
            f.write("Time(s),X(m),Y(m),Z(m)\n")
            for i in range(len(trajectory['t'])):
                f.write(f"{trajectory['t'][i]:.4f},{trajectory['x'][i]:.4f},"
                       f"{trajectory['y'][i]:.4f},{trajectory['z'][i]:.4f}\n")
        
        print(f"✓ Trajectory data saved to: {file_path}")
    
    def save_waypoints_data(self, trajectory, file_path):
        """
        Save waypoints data to a text file.
        
        Args:
            trajectory: Dictionary returned by generate_trajectory()
            file_path: Path to save the data
        """
        if trajectory is None:
            return
        
        waypoints = trajectory['waypoints']
        time_points = trajectory['time_points']
        
        with open(file_path, 'w') as f:
            f.write("Waypoint_ID,Time(s),X(m),Y(m),Z(m)\n")
            for i, (wp, t) in enumerate(zip(waypoints, time_points)):
                f.write(f"{i},{t:.4f},{wp[0]:.4f},{wp[1]:.4f},{wp[2]:.4f}\n")
        
        print(f"✓ Waypoints data saved to: {file_path}")


def generate_smooth_trajectory(path, speed=2.0, num_points=500):
    """
    Convenience function to generate a smooth trajectory.
    
    Args:
        path: N×3 numpy array of waypoints
        speed: Average traversal speed (m/s)
        num_points: Number of interpolated points
        
    Returns:
        trajectory: Dictionary with trajectory data
    """
    generator = TrajectoryGenerator(speed=speed)
    return generator.generate_trajectory(path, num_points)

