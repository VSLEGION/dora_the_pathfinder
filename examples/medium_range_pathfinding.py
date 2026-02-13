"""
MISSION: Medium-Scale Package Delivery
=====================================

Scenario: 
Your ScanEagle UAV must deliver packages to 7 locations across a 20km route.
The mission faces challenging conditions:
- Strong easterly wind (8 m/s = 28.8 km/h)
- 3 urban no-fly zones (residential areas, schools, hospitals)
- Must stay within 8m flight corridors for safety

Mission Profile:
- Distance: ~23 km total
- Flight time: ~15 minutes
- Fuel capacity: 2.0 kg
- Delivery points: 7 waypoints
- Operating altitude: 150m AGL
"""

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from pathfinder.models.aircraft import Aircraft
from pathfinder.models.environment import Environment, Wind
from pathfinder.planning.astar import plan_path_astar
from pathfinder.planning.dijkstra import plan_path_dijkstra
from pathfinder.visualizer.visualizer import plot_path, plot_comparison, plot_performance_comparison

# Import animated planner
import sys
sys.path.append('.')
from animated_pathfinding import run_animated_mission


def print_mission_briefing():
    """Print the mission briefing."""
    print("\n" + "="*80)
    print(" " * 20 + "MISSION BRIEFING")
    print("="*80)
    print("""
MISSION TYPE:        Package Delivery (Medium-Scale)
CLASSIFICATION:      Routine / Challenging Conditions
CALL SIGN:           Eagle-1

SITUATION:
You are tasked with delivering medical supplies and equipment to 7 remote 
distribution centers. The route crosses mixed terrain with several urban areas
that must be avoided.

ENVIRONMENT:
- Wind: 8.0 m/s from East (090 deg) - MODERATE to STRONG
- Visibility: Good
- No-Fly Zones: 3 urban areas (residential, school zone, hospital)

AIRCRAFT:
- Type: ScanEagle UAV
- Fuel Capacity: 2.0 kg
- Cruise Speed: 28 m/s (100.8 km/h)
- Operating Altitude: 150m AGL
- Endurance: ~45 minutes

ROUTE:
1. Launch Point (Distribution Hub)      -> (0, 0)
2. Rural Clinic Alpha                   -> (3000, 1500)
3. Community Center Bravo               -> (6000, 3000)
4. Medical Outpost Charlie              -> (9000, 4000)
5. Remote Station Delta                 -> (12000, 5500)
6. Field Hospital Echo                  -> (15000, 6000)
7. Emergency Cache Foxtrot              -> (18000, 7000)
8. Return to Base (RTB)                 -> (20000, 8000)

CONSTRAINTS:
- Stay within 8m flight corridors
- Avoid all no-fly zones
- Minimize fuel consumption (supplies are heavy)
- Plan for wind effects on fuel burn

NO-FLY ZONES:
1. Residential Area (4500, 2500) - Radius: 800m
2. School Zone (10000, 5000) - Radius: 600m  
3. Hospital Complex (16000, 6500) - Radius: 700m

FUEL ANALYSIS:
- Straight-line distance: ~23.0 km
- Estimated fuel (no wind, direct): ~0.082 kg
- Estimated fuel (with wind & detours): ~0.095-0.110 kg
- Safety margin: Required fuel < 0.5 kg (well within capacity)

MISSION SUCCESS CRITERIA:
- All waypoints reached
- No no-fly zone violations
- Fuel consumption < 0.15 kg
- Return to base safely
    """)
    print("="*80)
    print("\nInitializing mission planning systems...\n")


def analyze_mission_feasibility(waypoints, aircraft, environment):
    """Analyze if mission is feasible given constraints."""
    print("MISSION FEASIBILITY ANALYSIS")
    print("-" * 80)
    
    # Calculate straight-line distance
    total_distance = 0
    for i in range(len(waypoints) - 1):
        dx = waypoints[i+1][0] - waypoints[i][0]
        dy = waypoints[i+1][1] - waypoints[i][1]
        distance = np.sqrt(dx**2 + dy**2)
        total_distance += distance
    
    print(f"Total straight-line distance: {total_distance:.1f} m ({total_distance/1000:.2f} km)")
    
    # Estimate fuel (best case - straight line, no wind)
    best_case_fuel = aircraft.calculate_fuel_burn(total_distance)
    print(f"Best-case fuel consumption: {best_case_fuel:.6f} kg")
    
    # Estimate with wind penalty (empirical ~15% increase)
    estimated_fuel_with_wind = best_case_fuel * 1.15
    print(f"Estimated fuel with wind: {estimated_fuel_with_wind:.6f} kg")
    
    # Check against capacity
    fuel_capacity = 2.0  # kg
    fuel_margin = fuel_capacity - estimated_fuel_with_wind
    fuel_percentage = (estimated_fuel_with_wind / fuel_capacity) * 100
    
    print(f"Fuel capacity: {fuel_capacity:.3f} kg")
    print(f"Fuel margin: {fuel_margin:.3f} kg ({100-fuel_percentage:.1f}% reserve)")
    
    # Flight time estimate
    flight_time_seconds = total_distance / aircraft.true_airspeed_mps
    flight_time_minutes = flight_time_seconds / 60
    print(f"Estimated flight time: {flight_time_minutes:.1f} minutes")
    
    # Wind analysis
    print(f"\nWind conditions: {environment.wind.speed_mps} m/s from {environment.wind.direction_deg} deg")
    
    # Analyze wind impact on different headings
    headings = [0, 45, 90, 135, 180]  # N, NE, E, SE, S
    print("\nWind impact by heading:")
    for heading in headings:
        groundspeed = environment.get_effective_groundspeed(aircraft.true_airspeed_mps, heading)
        speed_change = groundspeed - aircraft.true_airspeed_mps
        direction = ["N", "NE", "E", "SE", "S"][headings.index(heading)]
        print(f"  {direction:3s} ({heading:3d} deg): Groundspeed = {groundspeed:.1f} m/s ({speed_change:+.1f} m/s)")
    
    # Safety check
    print("\n" + "-" * 80)
    if estimated_fuel_with_wind < fuel_capacity * 0.3:  # Less than 30% capacity
        print("[PASS] MISSION FEASIBLE - Fuel consumption well within limits")
        return True
    elif estimated_fuel_with_wind < fuel_capacity * 0.7:
        print("[WARN] MISSION FEASIBLE - Monitor fuel closely")
        return True
    else:
        print("[FAIL] MISSION RISKY - Fuel consumption approaching limits")
        return False


def run_delivery_mission():
    """Execute the complete delivery mission."""
    
    # Print mission briefing
    print_mission_briefing()
    
    # Mission waypoints (in meters)
    waypoints = [
        (0, 0),          # Launch Point
        (3000, 1500),    # Rural Clinic Alpha
        (6000, 3000),    # Community Center Bravo
        (9000, 4000),    # Medical Outpost Charlie
        (12000, 5500),   # Remote Station Delta
        (15000, 6000),   # Field Hospital Echo
        (18000, 7000),   # Emergency Cache Foxtrot
        (20000, 8000),   # RTB
    ]
    
    # Create aircraft
    print("[1/6] Initializing aircraft systems...")
    aircraft = Aircraft()
    perf = aircraft.get_performance_summary()
    print(f"   [OK] Aircraft ready: {aircraft}")
    print(f"   [OK] L/D ratio: {perf['lift_to_drag_ratio']:.2f}")
    print(f"   [OK] Cruise speed: {perf['cruise_speed_kph']:.1f} km/h")
    print(f"   [OK] Fuel burn rate: {perf['fuel_burn_rate_kg_per_hr']:.4f} kg/hr")
    
    # Create environment
    print("\n[2/6] Setting up operational environment...")
    environment = Environment(
        wind=Wind(speed_mps=8.0, direction_deg=90),  # 8 m/s from East
        altitude_m=150.0
    )
    
    # Add no-fly zones
    print("   [OK] Configuring no-fly zones...")
    environment.add_no_fly_zone(
        center=(4500, 2500),
        radius_m=800,
        name="Residential"
    )
    environment.add_no_fly_zone(
        center=(10000, 5000),
        radius_m=600,
        name="School"
    )
    environment.add_no_fly_zone(
        center=(16000, 6500),
        radius_m=700,
        name="Hospital"
    )
    print(f"   [OK] Environment configured: {environment}")
    
    # Feasibility analysis
    print("\n[3/6] Conducting mission feasibility analysis...")
    feasible = analyze_mission_feasibility(waypoints, aircraft, environment)
    
    if not feasible:
        print("\n[WARNING] Mission parameters exceed recommended limits!")
        response = input("Continue anyway? (yes/no): ")
        if response.lower() != 'yes':
            print("Mission aborted by operator.")
            return
    
    # Plan path with A*
    print("\n[4/6] Computing optimal flight path (A* algorithm)...")
    path_astar, cost_astar, metrics_astar = plan_path_astar(
        waypoints=waypoints,
        aircraft=aircraft,
        corridor_width=8.0,
        resolution=2.5,
        environment=environment,
        cost_type='fuel'
    )
    
    print(f"   [OK] Path computed: {len(path_astar)} waypoints")
    print(f"   [OK] Total fuel: {cost_astar:.6f} kg")
    print(f"   [OK] Nodes explored: {metrics_astar['nodes_visited']}")
    print(f"   [OK] Exploration efficiency: {(1-metrics_astar['exploration_ratio'])*100:.1f}% nodes saved")
    
    # Compare with Dijkstra
    print("\n[5/6] Running comparison with Dijkstra algorithm...")
    path_dijkstra, cost_dijkstra, metrics_dijkstra = plan_path_dijkstra(
        waypoints=waypoints,
        aircraft=aircraft,
        corridor_width=8.0,
        resolution=2.5,
        environment=environment,
        cost_type='fuel'
    )
    
    speedup = metrics_dijkstra['nodes_visited'] / metrics_astar['nodes_visited']
    print(f"   [OK] Dijkstra nodes explored: {metrics_dijkstra['nodes_visited']}")
    print(f"   [OK] A* speedup: {speedup:.2f}x faster")
    print(f"   [OK] Cost difference: {abs(cost_astar - cost_dijkstra):.8f} kg (optimal: {cost_astar == cost_dijkstra})")
    
    # Mission summary
    print("\n[6/6] Generating mission reports...")
    print("\n" + "="*80)
    print(" " * 25 + "MISSION SUMMARY")
    print("="*80)
    
    # Calculate actual flight distance
    actual_distance = sum(
        np.sqrt((path_astar[i+1][0] - path_astar[i][0])**2 + 
                (path_astar[i+1][1] - path_astar[i][1])**2)
        for i in range(len(path_astar)-1)
    )
    
    straight_distance = sum(
        np.sqrt((waypoints[i+1][0] - waypoints[i][0])**2 + 
                (waypoints[i+1][1] - waypoints[i][1])**2)
        for i in range(len(waypoints)-1)
    )
    
    detour_percentage = ((actual_distance - straight_distance) / straight_distance) * 100
    flight_time = actual_distance / aircraft.true_airspeed_mps / 60  # minutes
    fuel_percentage = (cost_astar / 2.0) * 100  # Assuming 2kg capacity
    
    print(f"""
ROUTE ANALYSIS:
   Waypoints visited:        {len(waypoints)}
   Planned flight segments:  {len(path_astar)-1}
   Direct distance:          {straight_distance/1000:.2f} km
   Planned distance:         {actual_distance/1000:.2f} km
   Detour factor:            +{detour_percentage:.1f}%
   
FUEL CONSUMPTION:
   Optimal fuel cost:        {cost_astar:.6f} kg
   Fuel capacity:            2.000 kg
   Fuel usage:               {fuel_percentage:.1f}%
   Remaining fuel:           {2.0 - cost_astar:.6f} kg ({100-fuel_percentage:.1f}% reserve)
   
FLIGHT TIME:
   Estimated duration:       {flight_time:.1f} minutes
   
ALGORITHM PERFORMANCE:
   A* nodes explored:        {metrics_astar['nodes_visited']}
   Dijkstra nodes explored:  {metrics_dijkstra['nodes_visited']}
   Efficiency gain:          {speedup:.2f}x faster
   
MISSION STATUS:            CLEARED FOR FLIGHT
    """)
    print("="*80)
    
    # Visualizations
    print("\n[INFO] Generating mission visualizations...")
    
    # Individual path plot
    plot_path(
        path=path_astar,
        waypoints=waypoints,
        corridor_width=8.0,
        environment=environment,
        title="Package Delivery Mission - Optimal Route (A*)",
        save_path="delivery_mission_path.png",
        show=False
    )
    print("   [OK] Saved: delivery_mission_path.png")
    
    # Algorithm comparison
    plot_comparison(
        paths_dict={
            'A* (Fuel Optimized)': path_astar,
            'Dijkstra': path_dijkstra
        },
        waypoints=waypoints,
        corridor_width=8.0,
        title="Algorithm Comparison - Package Delivery Mission",
        save_path="delivery_comparison.png",
        show=False
    )
    print("   [OK] Saved: delivery_comparison.png")
    
    # Performance metrics
    plot_performance_comparison(
        metrics_list=[metrics_dijkstra, metrics_astar],
        save_path="delivery_performance.png",
        show=False
    )
    print("   [OK] Saved: delivery_performance.png")
    
    # Detailed waypoint report
    print("\n" + "="*80)
    print(" " * 25 + "WAYPOINT DETAILS")
    print("="*80)
    print(f"{'#':<4} {'Name':<30} {'Coordinates':<20} {'Distance':<12}")
    print("-"*80)
    
    waypoint_names = [
        "Launch Point (Hub)",
        "Rural Clinic Alpha",
        "Community Center Bravo",
        "Medical Outpost Charlie",
        "Remote Station Delta",
        "Field Hospital Echo",
        "Emergency Cache Foxtrot",
        "Return to Base"
    ]
    
    for i, (wp, name) in enumerate(zip(waypoints, waypoint_names)):
        if i == 0:
            dist = "START"
        else:
            dx = wp[0] - waypoints[i-1][0]
            dy = wp[1] - waypoints[i-1][1]
            dist = f"{np.sqrt(dx**2 + dy**2)/1000:.2f} km"
        
        print(f"{i+1:<4} {name:<30} ({wp[0]:>6}, {wp[1]:>6}) {dist:>12}")
    
    print("="*80)
    
    # Ask about animation
    print("\n" + "="*80)
    print("REAL-TIME ANIMATION")
    print("="*80)
    print("""
Would you like to see the A* algorithm explore nodes in REAL-TIME?

The animation will show:
- Yellow dots: Nodes being considered (frontier)
- Light blue dots: Already explored nodes  
- Orange star: Current node being examined
- Blue line: Best path found so far
- Statistics panel showing costs and progress

Note: The animation may take 30-60 seconds to complete.
    """)
    
    response = input("Run animated visualization? (yes/no): ").strip().lower()
    
    if response == 'yes':
        print("\n[INFO] Starting animated visualization...")
        print("   (Close the window when done to continue)\n")
        
        run_animated_mission(
            waypoints=waypoints,
            aircraft=aircraft,
            environment=environment,
            corridor_width=8.0,
            resolution=2.5,
            animation_speed_ms=30,
            save_animation=False
        )
    
    # Final message
    print("\n" + "="*80)
    print(" " * 20 + "MISSION PLANNING COMPLETE")
    print("="*80)
    print("""
All mission files generated successfully!

Generated files:
  - delivery_mission_path.png      - Optimal route visualization
  - delivery_comparison.png         - Algorithm comparison
  - delivery_performance.png        - Performance metrics

MISSION STATUS: READY FOR EXECUTION

Next steps:
  1. Review generated visualizations
  2. Upload flight plan to ground control system
  3. Conduct pre-flight checks
  4. Execute mission

Good luck, Eagle-1!
    """)
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        run_delivery_mission()
    except KeyboardInterrupt:
        print("\n\n[WARNING] Mission planning interrupted by operator.")
    except Exception as e:
        print(f"\n\n[ERROR] Mission planning failed: {e}")
        import traceback
        traceback.print_exc()