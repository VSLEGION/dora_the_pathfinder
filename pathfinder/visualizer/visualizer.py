"""
Visualization module for path planning results.

This module provides functions to visualize paths, corridors, and
performance metrics.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import List, Tuple, Optional
from ..models.environment import Environment, NoFlyZone


def plot_path(path: List[Tuple[float, float]],
              waypoints: List[Tuple[float, float]],
              corridor_width: float = 5.0,
              environment: Optional[Environment] = None,
              title: str = "UAV Flight Path",
              save_path: Optional[str] = None,
              show: bool = True):
    """
    Visualize the planned flight path with corridors and constraints.
    
    Args:
        path: Planned path as list of (x, y) coordinates
        waypoints: Original waypoints
        corridor_width: Width of flight corridor
        environment: Optional environment with no-fly zones
        title: Plot title
        save_path: Optional path to save figure
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot flight corridors
    _plot_corridors(ax, waypoints, corridor_width)
    
    # Plot no-fly zones
    if environment is not None:
        _plot_no_fly_zones(ax, environment.no_fly_zones)
    
    # Plot waypoints
    wp_x = [wp[0] for wp in waypoints]
    wp_y = [wp[1] for wp in waypoints]
    ax.plot(wp_x, wp_y, 'rs', markersize=12, label='Waypoints', zorder=5)
    
    # Plot straight-line path between waypoints (reference)
    ax.plot(wp_x, wp_y, 'r--', alpha=0.3, linewidth=1, label='Direct Path')
    
    # Plot planned path
    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        ax.plot(path_x, path_y, 'b-', linewidth=2, label='Planned Path', zorder=4)
        
        # Mark start and goal
        ax.plot(path_x[0], path_y[0], 'go', markersize=15, 
                label='Start', zorder=6)
        ax.plot(path_x[-1], path_y[-1], 'r^', markersize=15, 
                label='Goal', zorder=6)
    
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def _plot_corridors(ax, waypoints: List[Tuple[float, float]], 
                    corridor_width: float):
    """Plot flight corridors as shaded regions."""
    for i in range(len(waypoints) - 1):
        wp1 = waypoints[i]
        wp2 = waypoints[i + 1]
        
        # Vector and perpendicular
        dx = wp2[0] - wp1[0]
        dy = wp2[1] - wp1[1]
        length = np.sqrt(dx**2 + dy**2)
        
        if length == 0:
            continue
        
        # Unit perpendicular vector
        px = -dy / length
        py = dx / length
        
        # Corridor corners
        corners = [
            (wp1[0] + corridor_width * px, wp1[1] + corridor_width * py),
            (wp2[0] + corridor_width * px, wp2[1] + corridor_width * py),
            (wp2[0] - corridor_width * px, wp2[1] - corridor_width * py),
            (wp1[0] - corridor_width * px, wp1[1] - corridor_width * py)
        ]
        
        corridor = patches.Polygon(corners, alpha=0.2, facecolor='green',
                                  edgecolor='green', linewidth=1,
                                  label='Flight Corridor' if i == 0 else '')
        ax.add_patch(corridor)


def _plot_no_fly_zones(ax, zones: List[NoFlyZone]):
    """Plot no-fly zones as red circles."""
    for zone in zones:
        circle = plt.Circle(zone.center, zone.radius_m, 
                          color='red', alpha=0.3, 
                          label='No-Fly Zone' if zone == zones[0] else '')
        ax.add_patch(circle)
        
        # Add zone label
        if zone.name:
            ax.text(zone.center[0], zone.center[1], zone.name,
                   ha='center', va='center', fontsize=8)


def plot_comparison(paths_dict: dict,
                   waypoints: List[Tuple[float, float]],
                   corridor_width: float = 5.0,
                   title: str = "Algorithm Comparison",
                   save_path: Optional[str] = None,
                   show: bool = True):
    """
    Compare multiple paths on the same plot.
    
    Args:
        paths_dict: Dictionary mapping algorithm names to paths
        waypoints: Original waypoints
        corridor_width: Width of flight corridor
        title: Plot title
        save_path: Optional save path
        show: Whether to display
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot corridors
    _plot_corridors(ax, waypoints, corridor_width)
    
    # Plot waypoints
    wp_x = [wp[0] for wp in waypoints]
    wp_y = [wp[1] for wp in waypoints]
    ax.plot(wp_x, wp_y, 'rs', markersize=12, label='Waypoints', zorder=5)
    
    # Plot each path with different style
    colors = ['blue', 'green', 'orange', 'purple', 'cyan']
    styles = ['-', '--', '-.', ':']
    
    for i, (name, path) in enumerate(paths_dict.items()):
        if path:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            color = colors[i % len(colors)]
            style = styles[i % len(styles)]
            ax.plot(path_x, path_y, color=color, linestyle=style,
                   linewidth=2, label=name, zorder=4)
    
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_cost_profile(path: List[Tuple[float, float]],
                     costs: List[float],
                     title: str = "Cost Profile Along Path",
                     ylabel: str = "Fuel Cost (kg)",
                     save_path: Optional[str] = None,
                     show: bool = True):
    """
    Plot cost profile along the path.
    
    Args:
        path: Path coordinates
        costs: Cumulative costs at each point
        title: Plot title
        ylabel: Y-axis label
        save_path: Optional save path
        show: Whether to display
    """
    distances = [0]
    for i in range(1, len(path)):
        dx = path[i][0] - path[i-1][0]
        dy = path[i][1] - path[i-1][1]
        dist = np.sqrt(dx**2 + dy**2)
        distances.append(distances[-1] + dist)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(distances, costs, 'b-', linewidth=2)
    ax.fill_between(distances, costs, alpha=0.3)
    
    ax.set_xlabel('Distance Along Path (m)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_performance_comparison(metrics_list: List[dict],
                               save_path: Optional[str] = None,
                               show: bool = True):
    """
    Create bar chart comparing algorithm performance.
    
    Args:
        metrics_list: List of metric dictionaries from different algorithms
        save_path: Optional save path
        show: Whether to display
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    algorithms = [m['algorithm'] for m in metrics_list]
    nodes_visited = [m['nodes_visited'] for m in metrics_list]
    exploration_ratios = [m['exploration_ratio'] * 100 for m in metrics_list]
    
    # Nodes visited comparison
    bars1 = ax1.bar(algorithms, nodes_visited, color=['#2E86AB', '#A23B72'])
    ax1.set_ylabel('Nodes Visited', fontsize=12)
    ax1.set_title('Search Efficiency', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    
    # Exploration ratio comparison
    bars2 = ax2.bar(algorithms, exploration_ratios, color=['#2E86AB', '#A23B72'])
    ax2.set_ylabel('Exploration Ratio (%)', fontsize=12)
    ax2.set_title('Graph Coverage', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    # Example visualization
    waypoints = [
        (0, 0),
        (100, 50),
        (200, 100),
        (300, 150)
    ]
    
    # Simulate a path
    path = [
        (0, 0),
        (25, 12),
        (50, 25),
        (75, 37),
        (100, 50),
        (150, 75),
        (200, 100),
        (250, 125),
        (300, 150)
    ]
    
    # Create visualization
    plot_path(path, waypoints, corridor_width=5.0, 
             title="Example UAV Path")