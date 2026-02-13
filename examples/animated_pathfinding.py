"""
Animated path planning demonstration with real-time node exploration.

Shows A* search expanding through the graph in real-time, visiting all
waypoints in sequence and displaying fuel cost breakdowns.
"""

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import heapq
from typing import List, Tuple, Dict, Optional, Set
import time

from pathfinder.models.aircraft import Aircraft
from pathfinder.models.environment import Environment, Wind, NoFlyZone
from pathfinder.planning.graph import FlightGraph
from pathfinder.planning.cost import CostCalculator


class AnimatedAStarPlanner:
    """
    A* planner with real-time visualization of node exploration.
    
    IMPROVED: Visits all waypoints in sequence as required checkpoints.
    """
    
    def __init__(self, graph: FlightGraph, aircraft: Aircraft,
                 environment: Optional[Environment] = None):
        """Initialize animated planner."""
        self.graph = graph
        self.aircraft = aircraft
        self.environment = environment
        self.cost_calculator = CostCalculator(
            aircraft=aircraft,
            environment=environment,
            cost_type='fuel'
        )

        # Tracking for animation
        self.visited_nodes = []
        self.current_node = None
        self.frontier_nodes = set()
        self.path_so_far = []
        self.g_costs = {}
        self.parent = {}

        # Track waypoint visits and fuel costs
        self.waypoints_visited = []
        self.fuel_per_segment = []
        self.total_fuel = 0.0
        
    def find_path_animated(self):
        """
        Find path visiting ALL waypoints in sequence.
        
        Strategy: Plan from waypoint to waypoint, concatenate paths.
        
        Yields:
            Dictionary with current search state for animation
        """
        waypoints = self.graph.waypoints
        complete_path = []
        self.total_fuel = 0.0
        self.fuel_per_segment = []
        
        iteration_offset = 0
        
        # Plan path from each waypoint to the next
        for wp_index in range(len(waypoints) - 1):
            start_wp = waypoints[wp_index]
            goal_wp = waypoints[wp_index + 1]
            
            print(f"\n   Planning segment {wp_index + 1}: {start_wp} -> {goal_wp}")
            
            # Find path for this segment
            segment_states = list(self._find_segment_path(start_wp, goal_wp, iteration_offset))
            
            # Extract the path from final state
            if segment_states:
                final_state = segment_states[-1]
                segment_path = final_state['path']
                segment_cost = final_state['g_cost']
                
                # Add to complete path (avoid duplicating waypoints)
                if complete_path:
                    complete_path.extend(segment_path[1:])  # Skip first point (duplicate)
                else:
                    complete_path = segment_path
                
                self.total_fuel += segment_cost
                self.fuel_per_segment.append({
                    'from': start_wp,
                    'to': goal_wp,
                    'cost': segment_cost,
                    'distance': self._calculate_path_distance(segment_path)
                })
                self.waypoints_visited.append(start_wp)
                
                print(f"   [OK] Segment {wp_index + 1} complete: {len(segment_path)} nodes, {segment_cost:.6f} kg fuel")
                
                # Yield all states for this segment
                for state in segment_states:
                    state['complete_path'] = complete_path
                    state['total_fuel'] = self.total_fuel
                    state['fuel_per_segment'] = self.fuel_per_segment.copy()
                    state['current_waypoint'] = wp_index + 1
                    state['total_waypoints'] = len(waypoints)
                    yield state
                
                iteration_offset = segment_states[-1]['iteration'] + 1
        
        # Mark completion
        self.waypoints_visited.append(waypoints[-1])

        # Final state
        yield {
            'iteration': iteration_offset,
            'current': waypoints[-1],
            'visited': self.visited_nodes,
            'frontier': set(),
            'g_cost': self.total_fuel,
            'f_cost': self.total_fuel,
            'goal_reached': True,
            'path': complete_path,
            'complete_path': complete_path,
            'total_fuel': self.total_fuel,
            'fuel_per_segment': self.fuel_per_segment,
            'current_waypoint': len(waypoints),
            'total_waypoints': len(waypoints),
            'final': True
        }
    
    def _find_segment_path(self, start: Tuple[float, float], goal: Tuple[float, float],
                          iteration_offset: int):
        """Find path for a single segment between two waypoints."""
        # Priority queue
        h_start = self.cost_calculator.calculate_heuristic(start, goal)
        pq = [(h_start, 0, start)]
        
        # Reset for this segment
        g_cost = {start: 0}
        f_cost = {start: h_start}
        parent = {start: None}
        visited = set()
        
        iteration = iteration_offset
        
        while pq:
            current_f, current_g, current = heapq.heappop(pq)
            
            if current in visited:
                continue
            
            visited.add(current)
            self.visited_nodes.append(current)
            self.current_node = current
            
            # Update frontier
            self.frontier_nodes = {node for _, _, node in pq if node not in visited}
            
            # Yield current state
            yield {
                'iteration': iteration,
                'current': current,
                'visited': list(visited),
                'frontier': self.frontier_nodes,
                'g_cost': current_g,
                'f_cost': current_f,
                'goal_reached': current == goal,
                'path': self._reconstruct_path_partial(parent, start, current)
            }
            
            iteration += 1
            
            # Goal reached
            if current == goal:
                final_path = self._reconstruct_path_partial(parent, start, goal)
                yield {
                    'iteration': iteration,
                    'current': goal,
                    'visited': list(visited),
                    'frontier': set(),
                    'g_cost': current_g,
                    'f_cost': current_f,
                    'goal_reached': True,
                    'path': final_path
                }
                return
            
            # Explore neighbors
            for neighbor in self.graph.get_neighbors(current):
                if neighbor in visited:
                    continue
                
                edge_cost = self.cost_calculator.calculate_cost(current, neighbor)
                tentative_g = current_g + edge_cost
                
                if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                    g_cost[neighbor] = tentative_g
                    h = self.cost_calculator.calculate_heuristic(neighbor, goal)
                    f = tentative_g + h
                    
                    f_cost[neighbor] = f
                    parent[neighbor] = current
                    heapq.heappush(pq, (f, tentative_g, neighbor))
    
    def _reconstruct_path_partial(self, parent: dict, start: Tuple[float, float],
                                  current: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Reconstruct path from parent pointers."""
        path = []
        node = current
        while node is not None:
            path.append(node)
            node = parent.get(node)
        path.reverse()
        return path
    
    def _calculate_path_distance(self, path: List[Tuple[float, float]]) -> float:
        """Calculate total distance of a path."""
        if len(path) < 2:
            return 0.0
        
        total = 0.0
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            total += np.sqrt(dx**2 + dy**2)
        return total


class AnimatedPathfindingVisualizer:
    """
    Real-time visualizer for path planning algorithm.
    
    IMPROVED: 
    - No corridor visualization
    - Shows fuel costs per waypoint
    - Displays total fuel consumption
    """
    
    def __init__(self, graph: FlightGraph, environment: Optional[Environment] = None,
                 speed_ms: int = 50):
        """Initialize visualizer."""
        self.graph = graph
        self.environment = environment
        self.speed_ms = speed_ms
        
        # Set up the plot
        self.fig, (self.ax_main, self.ax_stats) = plt.subplots(
            1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [2, 1]}
        )
        
        self._setup_main_plot()
        self._setup_stats_plot()
        
        # Animation data
        self.states = []
        self.current_state_idx = 0
        
    def _setup_main_plot(self):
        """Set up the main path visualization with professional styling."""
        ax = self.ax_main
        
        # Set font to JetBrains Mono
        plt.rcParams['font.family'] = 'monospace'
        try:
            plt.rcParams['font.monospace'] = ['JetBrains Mono', 'DejaVu Sans Mono', 'Courier New']
        except:
            pass
        
        # Plot no-fly zones
        if self.environment:
            self._plot_no_fly_zones(ax)
        
        # Plot waypoints - smaller, cleaner design
        waypoints = self.graph.waypoints
        wp_x = [wp[0] for wp in waypoints]
        wp_y = [wp[1] for wp in waypoints]
        
        # Reference line connecting waypoints
        ax.plot(wp_x, wp_y, color='#CCCCCC', alpha=0.5, linewidth=1.0, 
                linestyle=':', label='Waypoint Sequence', zorder=2)
        
        # Waypoint markers - smaller circles
        for i, (x, y) in enumerate(waypoints):
            # Outer circle
            circle = plt.Circle((x, y), 5, color='#34495E', alpha=0.85, zorder=10)
            ax.add_patch(circle)
            
            # Inner circle
            circle_inner = plt.Circle((x, y), 3.5, color='#455A64', alpha=0.9, zorder=11)
            ax.add_patch(circle_inner)
            
            # Number
            ax.text(x, y, str(i+1), ha='center', va='center',
                   fontsize=7, fontweight='bold', color='#ECEFF1', zorder=12,
                   family='monospace')
        
        # Start marker - green
        start_x, start_y = waypoints[0]
        start_circle = plt.Circle((start_x, start_y), 6, color='#388E3C', 
                                 alpha=0.9, zorder=13)
        ax.add_patch(start_circle)
        start_inner = plt.Circle((start_x, start_y), 4.5, color='#4CAF50',
                                alpha=0.9, zorder=14)
        ax.add_patch(start_inner)
        ax.text(start_x, start_y, 'S', ha='center', va='center',
               fontsize=8, fontweight='bold', color='white', zorder=15,
               family='monospace')
        
        # Goal marker - red
        goal_x, goal_y = waypoints[-1]
        goal_circle = plt.Circle((goal_x, goal_y), 6, color='#D32F2F',
                                alpha=0.9, zorder=13)
        ax.add_patch(goal_circle)
        goal_inner = plt.Circle((goal_x, goal_y), 4.5, color='#F44336',
                                alpha=0.9, zorder=14)
        ax.add_patch(goal_inner)
        ax.text(goal_x, goal_y, 'G', ha='center', va='center',
               fontsize=8, fontweight='bold', color='white', zorder=15,
               family='monospace')
        
        # Plot elements
        self.visited_scatter = ax.scatter([], [], c='#B0BEC5', s=12, 
                                         alpha=0.4, label='Visited', zorder=3)
        self.frontier_scatter = ax.scatter([], [], c='#FFD54F', s=20, 
                                          alpha=0.6, label='Frontier', zorder=4)
        
        # UAV marker - simple delta symbol
        self.current_scatter = ax.scatter([], [], c='#FF6F00', s=120,
                                         marker='^',  # Triangle (delta)
                                         label='UAV', zorder=16,
                                         edgecolors='#E65100', linewidths=1.5)
        
        # Path line
        self.path_line, = ax.plot([], [], color='#1976D2', linewidth=2.5,
                                  alpha=0.85,
                                  label='A* Path', zorder=6)
        
        ax.set_xlabel('X Position (m)', fontsize=11, fontfamily='monospace')
        ax.set_ylabel('Y Position (m)', fontsize=11, fontfamily='monospace')
        ax.set_title('Multi-waypoint pathfinding demo', 
                    fontsize=13, fontweight='normal', fontfamily='monospace')
        ax.legend(loc='upper left', fontsize=8, framealpha=0.95)
        ax.grid(True, alpha=0.15, linestyle=':', linewidth=0.5)
        ax.set_aspect('equal')
        ax.set_facecolor('#FAFAFA')
        
    def _setup_stats_plot(self):
        """Set up the statistics panel."""
        ax = self.ax_stats
        ax.axis('off')
        
        # Set background color
        ax.set_facecolor('#F8F9FA')
        
        # Create text elements with JetBrains Mono
        self.stats_text = ax.text(0.05, 0.95, '', fontsize=10,
                                 verticalalignment='top',
                                 fontfamily='monospace',
                                 color='#2C3E50')
        
    def _plot_no_fly_zones(self, ax):
        """Plot no-fly zones with improved styling."""
        for zone in self.environment.no_fly_zones:
            # Outer circle (danger zone)
            circle_outer = plt.Circle(zone.center, zone.radius_m,
                                     color='#E74C3C', alpha=0.15, zorder=1)
            ax.add_patch(circle_outer)
            
            # Inner circle (core restricted area)
            circle_inner = plt.Circle(zone.center, zone.radius_m * 0.7,
                                     color='#C0392B', alpha=0.25, zorder=1)
            ax.add_patch(circle_inner)
            
            # Border
            circle_border = plt.Circle(zone.center, zone.radius_m,
                                      fill=False, edgecolor='#C0392B', 
                                      linewidth=2, linestyle='--',
                                      alpha=0.6, zorder=2)
            ax.add_patch(circle_border)
            
            # Label with better styling
            if zone.name:
                # Background box for text
                bbox_props = dict(boxstyle='round,pad=0.5', 
                                facecolor='white', 
                                edgecolor='#C0392B',
                                alpha=0.9, linewidth=1.5)
                ax.text(zone.center[0], zone.center[1], zone.name,
                       ha='center', va='center', fontsize=9, 
                       color='#C0392B', fontweight='bold',
                       fontfamily='monospace',
                       bbox=bbox_props, zorder=15)
    
    def animate_search(self, planner: AnimatedAStarPlanner):
        """Run animated search."""
        print("[INFO] Starting animated pathfinding...")
        print("=" * 70)
        
        # Collect all states
        for state in planner.find_path_animated():
            self.states.append(state)
            
            # Print progress
            if state['iteration'] % 20 == 0:
                current_wp = state.get('current_waypoint', 0)
                total_wp = state.get('total_waypoints', 0)
                print(f"[PROGRESS] Waypoint {current_wp}/{total_wp} | "
                      f"Iteration {state['iteration']} | "
                      f"Visited {len(state['visited'])} nodes")
        
        print(f"\n[COMPLETE] Search finished")
        print(f"  Total iterations: {len(self.states)}")
        print(f"  Waypoints visited: {planner.graph.waypoints.__len__()}")
        print(f"  Final path nodes: {len(self.states[-1]['path'])}")
        print(f"  Total fuel: {self.states[-1].get('total_fuel', 0):.6f} kg")
        
        # Print fuel breakdown
        print("\n[FUEL] Consumption by Segment:")
        print("-" * 70)
        for i, segment in enumerate(self.states[-1].get('fuel_per_segment', [])):
            print(f"  Segment {i+1}: {segment['from']} -> {segment['to']}")
            print(f"    Distance: {segment['distance']:.1f} m")
            print(f"    Fuel: {segment['cost']:.6f} kg")
        print("-" * 70)
        print(f"  TOTAL: {self.states[-1].get('total_fuel', 0):.6f} kg")
        
        print("\n[INFO] Starting animation...")
        print("[INFO] Animation will pause at end - close window when done\n")
        
        # Add extra frames at the end to pause
        final_state = self.states[-1]
        # Repeat final state 100 times (pauses for ~3-5 seconds depending on speed)
        for _ in range(100):
            self.states.append(final_state)
        
        # Create animation
        anim = FuncAnimation(
            self.fig,
            self._update_frame,
            frames=len(self.states),
            interval=self.speed_ms,
            repeat=False,  # Don't loop - stop at end
            blit=False
        )
        
        plt.tight_layout()
        return anim
    
    def _update_frame(self, frame_idx):
        """Update animation frame."""
        if frame_idx >= len(self.states):
            return
        
        state = self.states[frame_idx]
        
        # Update visited nodes
        if state['visited']:
            visited_coords = np.array(state['visited'])
            self.visited_scatter.set_offsets(visited_coords)
        
        # Update frontier nodes
        if state['frontier']:
            frontier_coords = np.array(list(state['frontier']))
            self.frontier_scatter.set_offsets(frontier_coords)
        else:
            self.frontier_scatter.set_offsets(np.empty((0, 2)))
        
        # Update UAV position
        if state['current']:
            self.current_scatter.set_offsets([state['current']])
        
        # Update path as it's being explored
        current_path = state.get('path', [])
        complete_path = state.get('complete_path', [])

        # During search: show current segment being explored
        if not state.get('final', False):
            if current_path and len(current_path) > 1:
                path_coords = np.array(current_path)
                self.path_line.set_data(path_coords[:, 0], path_coords[:, 1])

        # At end: show complete path across all segments
        else:
            if complete_path and len(complete_path) > 1:
                path_coords = np.array(complete_path)
                self.path_line.set_data(path_coords[:, 0], path_coords[:, 1])
        
        # Update statistics
        stats_str = self._format_stats(state)
        self.stats_text.set_text(stats_str)
        
        # Update title based on state
        if state.get('final', False):
            self.ax_main.set_title(
                'Mission Complete - All Waypoints Visited',
                fontsize=13, fontweight='normal', color='#388E3C',
                fontfamily='monospace'
            )
    
    def _create_uav_marker(self, heading_deg: float):
        """
        Create a UAV-shaped marker rotated to heading.
        
        Args:
            heading_deg: Heading in degrees (0 = East, 90 = North)
        
        Returns:
            Path object for the UAV shape
        """
        from matplotlib.path import Path
        
        # UAV shape (simplified top-down view)
        # Nose pointing right (0 degrees)
        uav_shape = np.array([
            [1.0, 0.0],      # Nose
            [-0.5, 0.4],     # Right wing
            [-0.3, 0.0],     # Body
            [-0.5, -0.4],    # Left wing
            [1.0, 0.0],      # Back to nose
        ])
        
        # Rotate to heading
        angle_rad = np.radians(heading_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])
        
        rotated_shape = uav_shape @ rotation_matrix.T
        
        # Create path
        codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
        return Path(rotated_shape, codes)
    
    def _format_stats(self, state: dict) -> str:
        """Format statistics text - professional, no emojis."""
        current_wp = state.get('current_waypoint', 0)
        total_wp = state.get('total_waypoints', 0)
        
        lines = [
            "MISSION PROGRESS",
            "=" * 42,
            f"Waypoint:       {current_wp:>3} / {total_wp:<3}",
            f"Iteration:      {state['iteration']:>10}",
            f"Nodes Visited:  {len(state['visited']):>10}",
            f"Frontier Size:  {len(state['frontier']):>10}",
            "",
            "CURRENT NODE",
            "=" * 42,
            f"Position X:     {state['current'][0]:>10.2f} m",
            f"Position Y:     {state['current'][1]:>10.2f} m",
            f"g(n) Cost:      {state['g_cost']:>10.6f} kg",
            f"f(n) Cost:      {state['f_cost']:>10.6f} kg",
            "",
            "PATH STATUS",
            "=" * 42,
            f"Nodes:          {len(state.get('path', [])):>10}",
        ]
        
        # Add fuel breakdown if available
        if state.get('fuel_per_segment'):
            lines.append("")
            lines.append("FUEL CONSUMPTION")
            lines.append("=" * 42)
            
            # Show up to 5 segments
            for i, seg in enumerate(state['fuel_per_segment'][:5]):
                lines.append(f"Segment {i+1}:     {seg['cost']:>10.6f} kg")
            
            if len(state['fuel_per_segment']) > 5:
                remaining = len(state['fuel_per_segment']) - 5
                lines.append(f"... {remaining} more segments")
            
            lines.append("-" * 42)
            lines.append(f"TOTAL:          {state.get('total_fuel', 0):>10.6f} kg")
        
        # Final status
        if state.get('final'):
            lines.append("")
            lines.append("=" * 42)
            lines.append("STATUS: MISSION COMPLETE")
            lines.append("=" * 42)
            lines.append(f"Waypoints:      {total_wp:>10}")
            lines.append(f"Total Fuel:     {state.get('total_fuel', 0):>10.6f} kg")
        
        return '\n'.join(lines)


def run_animated_mission(waypoints: List[Tuple[float, float]],
                        aircraft: Aircraft,
                        environment: Optional[Environment] = None,
                        corridor_width: float = 5.0,
                        resolution: float = 2.0,
                        animation_speed_ms: int = 50,
                        save_animation: bool = False):
    """Run a complete animated pathfinding mission."""
    print("ANIMATED PATHFINDING MISSION")
    print("=" * 70)
    print(f"\nMission Waypoints: {len(waypoints)}")
    for i, wp in enumerate(waypoints):
        print(f"   {i+1}. ({wp[0]:>6.1f}, {wp[1]:>6.1f})")

    print(f"\nAircraft: {aircraft}")
    if environment:
        print(f"Environment: {environment}")
    print(f"Corridor Width: {corridor_width}m")
    print(f"Resolution: {resolution}m")
    
    # Build graph with obstacle avoidance
    print("\n[1/4] Building flight graph with obstacle avoidance...")
    graph = FlightGraph(
        waypoints=waypoints,
        corridor_width=corridor_width,
        resolution=resolution,
        search_expansion=4.0,
        environment=environment
    )
    info = graph.get_graph_info()
    print(f"   [OK] Graph created: {info['num_nodes']} nodes, {info['num_edges']} edges")
    print(f"   [OK] Search width: {info['search_width']:.1f}m (allows detours)")
    print(f"   [OK] Final corridor: {info['corridor_width']:.1f}m (path constraint)")
    
    # Create planner
    print("\n[2/4] Initializing A* planner...")
    planner = AnimatedAStarPlanner(
        graph=graph,
        aircraft=aircraft,
        environment=environment
    )
    print("   [OK] Planner ready")
    
    # Create visualizer
    print("\n[3/4] Setting up visualizer...")
    visualizer = AnimatedPathfindingVisualizer(
        graph=graph,
        environment=environment,
        speed_ms=animation_speed_ms
    )
    print("   [OK] Visualizer ready")
    
    # Run animation
    print("\n[4/4] Running animated search...")
    anim = visualizer.animate_search(planner)
    
    if save_animation:
        print("\n[INFO] Saving animation as GIF...")
        anim.save('pathfinding_animation.gif', writer='pillow', fps=20)
        print("   [OK] Saved as pathfinding_animation.gif")
    
    plt.show()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("EXAMPLE MISSION: Coastal Surveillance")
    print("="*70)
    
    waypoints = [
        (0, 0),
        (50, 25),
        (100, 50),
        (150, 75),
        (200, 100),
        (250, 125),
    ]
    
    aircraft = Aircraft()
    
    environment = Environment(
        wind=Wind(speed_mps=3.0, direction_deg=45),
        altitude_m=150.0
    )
    
    environment.add_no_fly_zone(center=(75, 40), radius_m=20, name="Urban Area")
    environment.add_no_fly_zone(center=(175, 90), radius_m=15, name="Airport")
    
    run_animated_mission(
        waypoints=waypoints,
        aircraft=aircraft,
        environment=environment,
        corridor_width=8.0,
        resolution=2.5,
        animation_speed_ms=30,
        save_animation=False
    )