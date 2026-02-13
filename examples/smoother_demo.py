"""
[EXPERIMENTAL] Path smoothing comparison demo.

STATUS: Beta - The path smoother is under active development.
        Tangent-arc geometry produces valid obstacle-free paths,
        but curvature at tangent entry points is still being refined.

Compare original A* path with smoothed path side-by-side.
Shows the difference in path quality and flyability.
"""

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from pathfinder.models.aircraft import Aircraft
from pathfinder.models.environment import Environment, Wind
from pathfinder.planning.graph import FlightGraph
from pathfinder.planning.astar import AStarPlanner
from pathfinder.planning.smoother import PathSmoother


def compare_paths():
    """Generate and compare original vs smoothed paths."""
    
    print("="*80)
    print("PATH SMOOTHING COMPARISON")
    print("="*80)
    
    # Simple mission with obstacles
    waypoints = [
        (0, 0),
        (50, 25),
        (100, 50),
        (150, 75),
        (200, 100),
    ]
    
    # Create environment
    environment = Environment(wind=Wind(speed_mps=3.0, direction_deg=45))
    environment.add_no_fly_zone(center=(75, 40), radius_m=20, name="Urban Area")
    environment.add_no_fly_zone(center=(125, 65), radius_m=15, name="Airport")
    
    # Create aircraft and graph
    aircraft = Aircraft()
    
    print("\n[1/3] Building graph...")
    graph = FlightGraph(
        waypoints=waypoints,
        corridor_width=8.0,
        resolution=2.5,
        search_expansion=4.0,
        environment=environment,
        adaptive_expansion=True
    )
    
    print(f"   [OK] Graph: {len(graph.nodes)} nodes")
    
    # Find path with A*
    print("\n[2/3] Planning path with A*...")
    planner = AStarPlanner(graph, aircraft, environment)
    
    # Plan segment by segment
    complete_path = []
    for i in range(len(waypoints) - 1):
        # Simple path finding between consecutive waypoints
        temp_graph = FlightGraph(
            waypoints=[waypoints[i], waypoints[i+1]],
            corridor_width=8.0,
            resolution=2.5,
            search_expansion=4.0,
            environment=environment,
            adaptive_expansion=True
        )
        temp_planner = AStarPlanner(temp_graph, aircraft, environment)
        segment_path, _ = temp_planner.find_path()
        
        if complete_path:
            complete_path.extend(segment_path[1:])
        else:
            complete_path = segment_path
    
    print(f"   [OK] Original path: {len(complete_path)} points")
    
    # Apply smoothing
    print("\n[3/3] Applying smoothing...")
    smoother = PathSmoother(
        environment=environment,
        smoothing_factor=0.5,
        num_points=300
    )
    
    smoothed_path = smoother.smooth_path(complete_path, method='spline')
    print(f"   [OK] Smoothed path: {len(smoothed_path)} points")
    
    # Calculate metrics
    original_curvatures = smoother.calculate_path_curvature(complete_path)
    smoothed_curvatures = smoother.calculate_path_curvature(smoothed_path)
    
    orig_max_curv = max(original_curvatures) if original_curvatures else 0
    smooth_max_curv = max(smoothed_curvatures) if smoothed_curvatures else 0
    
    orig_min_radius = 1.0 / orig_max_curv if orig_max_curv > 1e-6 else float('inf')
    smooth_min_radius = 1.0 / smooth_max_curv if smooth_max_curv > 1e-6 else float('inf')
    
    print(f"\nCOMPARISON:")
    print(f"   Original min turn radius: {orig_min_radius:.1f}m")
    print(f"   Smoothed min turn radius: {smooth_min_radius:.1f}m")
    print(f"   Improvement: {(smooth_min_radius/orig_min_radius - 1)*100:.1f}%")
    
    # Visualization
    print("\n[4/4] Creating visualization...")
    fig = plt.figure(figsize=(18, 8))
    
    # Layout: 3 subplots
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[:, 0])  # Original path
    ax2 = fig.add_subplot(gs[:, 1])  # Smoothed path
    ax3 = fig.add_subplot(gs[:, 2])  # Overlay comparison
    
    # Plot 1: Original A* path
    plot_path_on_axis(ax1, complete_path, waypoints, environment, 
                     "Original A* Path (Discrete)", 'red')
    
    # Plot 2: Smoothed path
    plot_path_on_axis(ax2, smoothed_path, waypoints, environment,
                     "Smoothed Flight Path (Flyable)", 'blue')
    
    # Plot 3: Overlay both
    ax3.set_title("Overlay Comparison", fontsize=14, fontweight='bold')
    plot_environment(ax3, waypoints, environment)
    
    # Original in red
    if complete_path:
        orig_coords = np.array(complete_path)
        ax3.plot(orig_coords[:, 0], orig_coords[:, 1], 
                'r-', linewidth=2, alpha=0.6, label='Original A*')
    
    # Smoothed in blue
    if smoothed_path:
        smooth_coords = np.array(smoothed_path)
        ax3.plot(smooth_coords[:, 0], smooth_coords[:, 1],
                'b-', linewidth=3, alpha=0.8, label='Smoothed')
    
    ax3.legend(loc='best', fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    # Add metrics text
    metrics_text = f"""
    METRICS COMPARISON
    {'='*40}
    Original Path:
      Points: {len(complete_path)}
      Min turn radius: {orig_min_radius:.1f}m
      
    Smoothed Path:
      Points: {len(smoothed_path)}
      Min turn radius: {smooth_min_radius:.1f}m
      
    Improvement: {(smooth_min_radius/orig_min_radius - 1)*100:.1f}%
    """
    
    fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=10,
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.savefig('smoothing_comparison.png', dpi=300, bbox_inches='tight')
    print("   [OK] Saved: smoothing_comparison.png")
    
    plt.show()
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)


def plot_path_on_axis(ax, path, waypoints, environment, title, color):
    """Helper to plot path on axis."""
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Environment
    plot_environment(ax, waypoints, environment)
    
    # Path
    if path and len(path) > 1:
        path_coords = np.array(path)
        ax.plot(path_coords[:, 0], path_coords[:, 1],
               color=color, linewidth=3, alpha=0.8, label='Path')
        
        # Show points
        ax.scatter(path_coords[::10, 0], path_coords[::10, 1],
                  c=color, s=20, alpha=0.5, zorder=5)
    
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')


def plot_environment(ax, waypoints, environment):
    """Helper to plot environment."""
    # Waypoints
    wp_x = [wp[0] for wp in waypoints]
    wp_y = [wp[1] for wp in waypoints]
    ax.plot(wp_x, wp_y, 'rs', markersize=12, label='Waypoints', zorder=10)
    ax.plot(wp_x, wp_y, 'r--', alpha=0.3, linewidth=1)
    
    # Start and goal
    ax.plot(wp_x[0], wp_y[0], 'go', markersize=15, label='Start', zorder=11)
    ax.plot(wp_x[-1], wp_y[-1], 'r^', markersize=15, label='Goal', zorder=11)
    
    # No-fly zones
    if environment:
        for zone in environment.no_fly_zones:
            circle = plt.Circle(zone.center, zone.radius_m,
                              color='red', alpha=0.3, zorder=1)
            ax.add_patch(circle)
            if zone.name:
                ax.text(zone.center[0], zone.center[1], zone.name,
                       ha='center', va='center', fontsize=9, 
                       color='darkred', fontweight='bold')
    
    ax.set_xlabel('X Position (m)', fontsize=11)
    ax.set_ylabel('Y Position (m)', fontsize=11)


if __name__ == "__main__":
    compare_paths()