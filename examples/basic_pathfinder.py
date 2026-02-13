"""
Basic pathfinding example demonstrating the Pathfinder system.

This script shows how to:
1. Define a mission with waypoints
2. Create an aircraft model
3. Set up environmental conditions
4. Plan paths using both Dijkstra and A*
5. Compare algorithm performance
6. Visualize results
"""

import sys
sys.path.append('..')

from pathfinder.models.aircraft import Aircraft
from pathfinder.models.environment import Environment, Wind, NoFlyZone
from pathfinder.planning.dijkstra import plan_path_dijkstra
from pathfinder.planning.astar import plan_path_astar, compare_algorithms
from pathfinder.visualizer.visualizer import (
    plot_path, plot_comparison, plot_performance_comparison
)


def main():
    """Run the basic pathfinding demonstration."""
    
    print("=" * 70)
    print("PATHFINDER - UAV Path Planning Demonstration")
    print("=" * 70)
    
    # ========================================================================
    # Step 1: Define Mission Waypoints
    # ========================================================================
    print("\n[1] Defining mission waypoints...")
    
    waypoints = [
        (0, 0),        # Start: Origin
        (100, 50),     # Waypoint 1
        (200, 100),    # Waypoint 2
        (300, 150),    # Waypoint 3
        (400, 150)     # Goal: Destination
    ]
    
    print(f"    Mission has {len(waypoints)} waypoints")
    print(f"    Start: {waypoints[0]}")
    print(f"    Goal: {waypoints[-1]}")
    
    # ========================================================================
    # Step 2: Create Aircraft Model
    # ========================================================================
    print("\n[2] Creating aircraft model (ScanEagle UAV)...")
    
    aircraft = Aircraft()
    perf = aircraft.get_performance_summary()
    
    print(f"    Mass: {perf['mass_kg']:.1f} kg")
    print(f"    Cruise speed: {perf['cruise_speed_kph']:.1f} km/h")
    print(f"    Fuel burn rate: {perf['fuel_burn_rate_kg_per_hr']:.4f} kg/hr")
    print(f"    L/D ratio: {perf['lift_to_drag_ratio']:.2f}")
    
    # ========================================================================
    # Step 3: Set Up Environment
    # ========================================================================
    print("\n[3] Setting up operational environment...")
    
    # Create environment with wind
    environment = Environment(
        wind=Wind(speed_mps=5.0, direction_deg=90),  # 5 m/s wind from East
        altitude_m=150.0
    )
    
    # Add no-fly zones
    environment.add_no_fly_zone(
        center=(150, 75),
        radius_m=30,
        name="Restricted Area 1"
    )
    
    environment.add_no_fly_zone(
        center=(250, 125),
        radius_m=25,
        name="Restricted Area 2"
    )
    
    print(f"    Wind: {environment.wind.speed_mps} m/s from {environment.wind.direction_deg} deg")
    print(f"    Altitude: {environment.altitude_m} m")
    print(f"    No-fly zones: {len(environment.no_fly_zones)}")
    
    # ========================================================================
    # Step 4: Plan Path with Dijkstra's Algorithm
    # ========================================================================
    print("\n[4] Planning path with Dijkstra's algorithm...")
    
    path_dijkstra, cost_dijkstra, metrics_dijkstra = plan_path_dijkstra(
        waypoints=waypoints,
        aircraft=aircraft,
        corridor_width=5.0,
        resolution=2.0,
        environment=environment,
        cost_type='fuel'
    )
    
    print(f"    Path found: {len(path_dijkstra)} nodes")
    print(f"    Total fuel cost: {cost_dijkstra:.6f} kg")
    print(f"    Nodes explored: {metrics_dijkstra['nodes_visited']}")
    print(f"    Exploration ratio: {metrics_dijkstra['exploration_ratio']*100:.1f}%")
    
    # ========================================================================
    # Step 5: Plan Path with A* Algorithm
    # ========================================================================
    print("\n[5] Planning path with A* algorithm...")
    
    path_astar, cost_astar, metrics_astar = plan_path_astar(
        waypoints=waypoints,
        aircraft=aircraft,
        corridor_width=5.0,
        resolution=2.0,
        environment=environment,
        cost_type='fuel'
    )
    
    print(f"    Path found: {len(path_astar)} nodes")
    print(f"    Total fuel cost: {cost_astar:.6f} kg")
    print(f"    Nodes explored: {metrics_astar['nodes_visited']}")
    print(f"    Exploration ratio: {metrics_astar['exploration_ratio']*100:.1f}%")
    
    # ========================================================================
    # Step 6: Compare Algorithms
    # ========================================================================
    print("\n[6] Algorithm Comparison:")
    print("    " + "-" * 60)
    
    speedup = (metrics_dijkstra['nodes_visited'] / 
               metrics_astar['nodes_visited'] if metrics_astar['nodes_visited'] > 0 else 0)
    cost_diff = abs(cost_dijkstra - cost_astar)
    
    print(f"    A* speedup: {speedup:.2f}x faster")
    print(f"    Cost difference: {cost_diff:.8f} kg")
    print(f"    Both optimal: {'Yes' if cost_diff < 1e-6 else 'No'}")
    
    # Calculate efficiency gain
    if cost_dijkstra > 0:
        straight_line_distance = sum([
            ((waypoints[i+1][0] - waypoints[i][0])**2 + 
             (waypoints[i+1][1] - waypoints[i][1])**2)**0.5
            for i in range(len(waypoints)-1)
        ])
        straight_line_fuel = aircraft.calculate_fuel_burn(straight_line_distance)
        savings = ((straight_line_fuel - cost_astar) / straight_line_fuel) * 100
        
        print(f"\n    Fuel savings vs straight line: {savings:.2f}%")
    
    # ========================================================================
    # Step 7: Visualize Results
    # ========================================================================
    print("\n[7] Generating visualizations...")
    
    # Individual path plots
    print("    Creating A* path visualization...")
    plot_path(
        path=path_astar,
        waypoints=waypoints,
        corridor_width=5.0,
        environment=environment,
        title="A* Path Planning with Fuel Optimization",
        save_path="astar_path.png",
        show=False
    )
    
    print("    Creating Dijkstra path visualization...")
    plot_path(
        path=path_dijkstra,
        waypoints=waypoints,
        corridor_width=5.0,
        environment=environment,
        title="Dijkstra Path Planning with Fuel Optimization",
        save_path="dijkstra_path.png",
        show=False
    )
    
    # Comparison plot
    print("    Creating algorithm comparison plot...")
    plot_comparison(
        paths_dict={
            'Dijkstra': path_dijkstra,
            'A*': path_astar
        },
        waypoints=waypoints,
        corridor_width=5.0,
        title="Algorithm Comparison: Dijkstra vs A*",
        save_path="algorithm_comparison.png",
        show=False
    )
    
    # Performance comparison
    print("    Creating performance metrics plot...")
    plot_performance_comparison(
        metrics_list=[metrics_dijkstra, metrics_astar],
        save_path="performance_comparison.png",
        show=False
    )
    
    print("\n    All visualizations saved successfully!")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n[OK] Mission planned successfully")
    print(f"[OK] Total distance: {straight_line_distance:.1f} m")
    print(f"[OK] Optimal fuel cost: {cost_astar:.6f} kg")
    print(f"[OK] Flight time: {straight_line_distance / aircraft.true_airspeed_mps:.1f} seconds")
    print(f"[OK] A* explored {metrics_astar['nodes_visited']} nodes")
    print(f"[OK] {speedup:.1f}x faster than Dijkstra")
    print(f"\n[OK] Visualizations saved to current directory")
    print("=" * 70)


def advanced_example():
    """
    Advanced example with multiple cost functions.
    """
    print("\n" + "=" * 70)
    print("ADVANCED EXAMPLE - Multiple Cost Functions")
    print("=" * 70)
    
    waypoints = [(0, 0), (200, 100), (400, 200)]
    aircraft = Aircraft()
    
    # Test different cost functions
    cost_types = ['fuel', 'distance', 'time']
    
    for cost_type in cost_types:
        print(f"\n[{cost_type.upper()}] Planning with {cost_type} cost...")
        
        path, cost, metrics = plan_path_astar(
            waypoints=waypoints,
            aircraft=aircraft,
            corridor_width=5.0,
            resolution=2.0,
            cost_type=cost_type
        )
        
        print(f"    Path length: {len(path)} nodes")
        print(f"    Cost: {cost:.6f}")
        print(f"    Nodes explored: {metrics['nodes_visited']}")


if __name__ == "__main__":
    # Run basic demonstration
    main()
    
    # Uncomment to run advanced example
    # advanced_example()
    
    print("\nDemonstration complete.")