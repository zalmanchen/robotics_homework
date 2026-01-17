"""
In this file, you should implement your own path planning class or function.
Within your implementation, you may call `env.is_collide()` and `env.is_outside()`
to verify whether candidate path points collide with obstacles or exceed the
environment boundaries.

You are required to write the path planning algorithm by yourself. Copying or calling 
any existing path planning algorithms from others is strictly
prohibited. Please avoid using external packages beyond common Python libraries
such as `numpy`, `math`, or `scipy`. If you must use additional packages, you
must clearly explain the reason in your report.
"""

import numpy as np
from math import sqrt


class PathPlanner:
    """
    Path planner using Rapidly-exploring Random Tree (RRT) algorithm.
    
    RRT is a sampling-based motion planning algorithm that incrementally builds
    a tree of collision-free paths from a start configuration towards a goal.
    """
    
    def __init__(self, env, max_iterations=8000, step_size=0.3, goal_bias=0.1):
        """
        Initialize the path planner.
        
        Args:
            env: FlightEnvironment object with is_collide() and is_outside() methods
            max_iterations: Maximum number of iterations for RRT algorithm
            step_size: Maximum distance to extend the tree in each iteration
            goal_bias: Probability (0-1) of sampling the goal position directly
        """
        self.env = env
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_bias = goal_bias
    
    def plan(self, start, goal):
        """
        Plan a collision-free path from start to goal using RRT algorithm.
        
        Args:
            start: tuple (x, y, z) - starting position
            goal: tuple (x, y, z) - goal position
            
        Returns:
            path: N×3 numpy array of waypoints, or None if no path found
        """
        start = np.array(start, dtype=float)
        goal = np.array(goal, dtype=float)
        
        # Validate start and goal
        if self.env.is_outside(start) or self.env.is_collide(start):
            print("Start position is invalid")
            return None
        
        if self.env.is_outside(goal) or self.env.is_collide(goal):
            print("Goal position is invalid")
            return None
        
        # Initialize tree
        tree_nodes = [start]
        tree_parents = [-1]
        
        for iteration in range(self.max_iterations):
            # Sample a random configuration
            if np.random.random() < self.goal_bias:
                sample = goal.copy()
            else:
                sample = self._random_sample()
            
            # Find nearest node in tree
            nearest_idx = self._find_nearest_node(tree_nodes, sample)
            nearest_node = tree_nodes[nearest_idx]
            
            # Extend tree towards sample
            new_node = self._extend_towards(nearest_node, sample)
            
            if new_node is None:
                continue
            
            # Check if new_node is valid
            if not self._is_valid(new_node):
                continue
            
            # Add new node to tree
            tree_nodes.append(new_node)
            tree_parents.append(nearest_idx)
            
            # Check if goal is reached
            if self._is_goal_reached(new_node, goal):
                # Reconstruct and smooth path
                path = self._reconstruct_path(tree_nodes, tree_parents, len(tree_nodes) - 1)
                path = self._smooth_path(path)
                return path
        
        print(f"No path found after {self.max_iterations} iterations")
        return None
    
    def _random_sample(self):
        """Sample a random point in the environment."""
        x = np.random.uniform(0, self.env.env_width)
        y = np.random.uniform(0, self.env.env_length)
        z = np.random.uniform(0, self.env.env_height)
        return np.array([x, y, z])
    
    def _find_nearest_node(self, tree_nodes, sample):
        """Find the index of the nearest node to the sample."""
        min_dist = float('inf')
        nearest_idx = 0
        
        for i, node in enumerate(tree_nodes):
            dist = np.linalg.norm(sample - node)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        return nearest_idx
    
    def _extend_towards(self, start, target):
        """Extend from start towards target by at most step_size."""
        direction = target - start
        distance = np.linalg.norm(direction)
        
        if distance < 1e-6:
            return None
        
        direction_norm = direction / distance
        extension_dist = min(self.step_size, distance)
        new_point = start + direction_norm * extension_dist
        
        if self._is_path_valid(start, new_point):
            return new_point
        
        return None
    
    def _is_valid(self, point):
        """Check if a point is valid (within bounds and collision-free)."""
        return not self.env.is_outside(point) and not self.env.is_collide(point)
    
    def _is_path_valid(self, p1, p2, num_checks=10):
        """Check if a straight line path between p1 and p2 is collision-free."""
        for i in range(num_checks + 1):
            t = i / num_checks
            point = p1 + t * (p2 - p1)
            if not self._is_valid(point):
                return False
        return True
    
    def _is_goal_reached(self, point, goal, tolerance=0.3):
        """Check if the point is close enough to the goal."""
        dist = np.linalg.norm(point - goal)
        return dist < tolerance
    
    def _reconstruct_path(self, tree_nodes, tree_parents, end_idx):
        """Reconstruct the path from start to end node."""
        path = []
        current_idx = end_idx
        
        while current_idx != -1:
            path.append(tree_nodes[current_idx])
            current_idx = tree_parents[current_idx]
        
        path.reverse()
        path.append(tree_nodes[end_idx])
        return np.array(path)
    
    def _smooth_path(self, path, max_smoothing_iterations=50):
        """Smooth the path by removing unnecessary waypoints, keeping key navigation points."""
        # Keep all points initially - they are necessary for obstacle avoidance
        # Only remove truly redundant points (collinear segments)
        smoothed = [path[0]]
        
        for i in range(1, len(path) - 1):
            # Check if point i is necessary by testing if we can skip it
            prev_point = smoothed[-1]
            next_point = path[i + 1]
            curr_point = path[i]
            
            # Only remove if direct path is valid AND the point is nearly collinear
            if self._is_path_valid(prev_point, next_point, num_checks=10):
                # Check collinearity
                v1 = next_point - prev_point
                v2 = curr_point - prev_point
                
                # Cross product magnitude (for 3D vectors)
                cross = np.linalg.norm(np.cross(v1, v2))
                distance_to_line = cross / (np.linalg.norm(v1) + 1e-6)
                
                # Only skip if very close to the line (< 0.05m deviation)
                if distance_to_line < 0.05:
                    continue
            
            # Keep this point - it's needed for obstacle avoidance
            smoothed.append(curr_point)
        
        # Always include the last point (goal)
        if len(path) > 1:
            smoothed.append(path[-1])
        
        return np.array(smoothed)


def plan_path(env, start, goal):
    """Plan a collision-free path from start to goal."""
    planner = PathPlanner(env, max_iterations=8000, step_size=0.3, goal_bias=0.1)
    return planner.plan(start, goal)
            










