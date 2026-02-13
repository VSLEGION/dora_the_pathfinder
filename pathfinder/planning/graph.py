"""
Graph representation module for path planning.

This module creates and manages the flight graph with corridors around waypoints.

IMPROVEMENTS: Creates wider search space to route around obstacles.

Upcoming Features:
1. Larger search radius
2. 5m or custom flight track sizing constraints
3. Boid Simulation for obstacle avoidance

"""

import numpy as np
from typing import List, Tuple, Set, Dict, Optional
from pathfinder.models.aircraft import Aircraft
from pathfinder.models.environment import Environment


class FlightGraph:
    """
    Represents the flight space as a graph with nodes and edges.
    
    Creates a search space with flight corridors around waypoints.
    IMPROVED: Expands search space to allow routing around obstacles.
    
    The 5m corridor concept:
    - Final path must stay within 5m of the planned route
    - But during planning, we need MORE space to find routes around obstacles
    - Think of it as: "search widely, commit narrowly"
    
    Attributes:
        waypoints: User-defined waypoints
        corridor_width: Final flight corridor width (5m = path stays within 5m)
        search_expansion: How much wider to search (default 3x for obstacle avoidance)
        resolution: Grid resolution for corridor discretization in meters
        nodes: Set of all valid graph nodes (x, y)
        edges: Dictionary mapping node to list of neighbor nodes
        environment: Optional environment with constraints
    """
    
    def __init__(self, waypoints: List[Tuple[float, float]],
                 corridor_width: float = 5.0,
                 resolution: float = 2.0,
                 search_expansion: float = 3.0,
                 environment: Optional[Environment] = None,
                 adaptive_expansion: bool = True):
        """
        Initialize flight graph.
        
        Args:
            waypoints: List of (x, y) waypoints defining the flight path
            corridor_width: Final corridor width in meters (5m = stay within 5m of path)
            resolution: Grid spacing for discretization in meters
            search_expansion: Multiplier for search space (3.0 = search 3x wider)
            environment: Optional environment with no-fly zones
            adaptive_expansion: If True, auto-expand search based on obstacle size
        """
        self.waypoints = waypoints
        self.corridor_width = corridor_width
        self.resolution = resolution
        self.search_expansion = search_expansion
        self.environment = environment
        self.adaptive_expansion = adaptive_expansion
        
        # Calculate expanded search width for planning
        # ADAPTIVE: Increase search width based on largest obstacle
        if adaptive_expansion and environment and environment.no_fly_zones:
            max_obstacle_radius = max(zone.radius_m for zone in environment.no_fly_zones)
            # Need at least 3x obstacle radius to route around
            min_search_width = max_obstacle_radius * 3.0
            base_search_width = corridor_width * search_expansion
            self.search_width = max(base_search_width, min_search_width)
            print(f"Adaptive search: {self.search_width:.1f}m (obstacle-aware)")
        else:
            self.search_width = corridor_width * search_expansion
        
        self.nodes: Set[Tuple[float, float]] = set()
        self.edges: Dict[Tuple[float, float], List[Tuple[float, float]]] = {}
        
        # Build the graph
        self._build_graph()
    
    def _build_graph(self):
        """
        Build the graph structure by creating corridor nodes and edges.
        
        IMPROVED STRATEGY:
        1. Create WIDE search corridors between waypoints
        2. Add EXTRA nodes around no-fly zones for detours
        3. Connect everything into a searchable graph
        4. Path validation ensures final path stays within corridor_width
        """
        # Add waypoints as nodes
        for wp in self.waypoints:
            self.nodes.add(wp)
        
        # Create expanded corridor nodes between consecutive waypoints
        for i in range(len(self.waypoints) - 1):
            wp1 = self.waypoints[i]
            wp2 = self.waypoints[i + 1]
            
            # Create WIDE search corridor (allows routing around obstacles)
            corridor_nodes = self._create_corridor_nodes(wp1, wp2, use_search_width=True)
            self.nodes.update(corridor_nodes)
        
        # Add obstacle avoidance nodes
        if self.environment and self.environment.no_fly_zones:
            detour_nodes = self._create_detour_nodes()
            self.nodes.update(detour_nodes)
        
        # Create edges between nearby nodes
        self._create_edges()
        
        print(f"   Graph built: {len(self.nodes)} nodes, search width: {self.search_width:.1f}m")
    
    def _create_corridor_nodes(self, p1: Tuple[float, float], 
                              p2: Tuple[float, float],
                              use_search_width: bool = True) -> Set[Tuple[float, float]]:
        """
        Create nodes within the flight corridor between two waypoints.
        
        Args:
            p1: Start waypoint (x, y)
            p2: End waypoint (x, y)
            use_search_width: If True, use expanded search width; else use corridor_width
            
        Returns:
            Set of corridor nodes
        """
        corridor_nodes = set()
        
        # Vector from p1 to p2
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        segment_length = np.sqrt(dx**2 + dy**2)
        
        if segment_length == 0:
            return corridor_nodes
        
        # Unit vectors
        ux = dx / segment_length
        uy = dy / segment_length
        
        # Perpendicular unit vector (left side)
        px = -uy
        py = ux
        
        # Choose width based on whether we're searching or validating
        width = self.search_width if use_search_width else self.corridor_width
        
        # Create grid points along the segment
        num_steps = int(segment_length / self.resolution) + 1
        
        for i in range(num_steps + 1):
            # Progress along segment (0 to 1)
            t = i / num_steps if num_steps > 0 else 0
            
            # Base point on centerline
            base_x = p1[0] + t * dx
            base_y = p1[1] + t * dy
            
            # Create nodes across the corridor width
            num_lateral = int(width / self.resolution)
            
            for j in range(-num_lateral, num_lateral + 1):
                offset = j * self.resolution
                
                node_x = base_x + offset * px
                node_y = base_y + offset * py
                node = (round(node_x, 2), round(node_y, 2))
                
                # Check if node is valid (not in no-fly zones)
                if self.environment is None or self.environment.is_valid_point(node):
                    corridor_nodes.add(node)
        
        return corridor_nodes
    
    def _create_detour_nodes(self) -> Set[Tuple[float, float]]:
        """
        Create additional nodes around no-fly zones to enable detours.
        
        This is KEY for routing around obstacles!
        
        Strategy:
        - For each no-fly zone, create a ring of nodes around it
        - These nodes provide "stepping stones" to route around obstacles
        
        Returns:
            Set of detour nodes around obstacles
        """
        detour_nodes = set()
        
        if not self.environment or not self.environment.no_fly_zones:
            return detour_nodes
        
        for zone in self.environment.no_fly_zones:
            # Create nodes in a ring around the obstacle
            # Ring radius = obstacle radius + safety margin + corridor width
            safety_margin = 5.0  # 5m extra clearance
            ring_radius = zone.radius_m + safety_margin + self.search_width
            
            # Number of nodes around the ring (based on circumference)
            circumference = 2 * np.pi * ring_radius
            num_nodes = int(circumference / self.resolution)
            num_nodes = max(num_nodes, 16)  # At least 16 nodes per ring
            
            # Create ring nodes
            for i in range(num_nodes):
                angle = 2 * np.pi * i / num_nodes
                node_x = zone.center[0] + ring_radius * np.cos(angle)
                node_y = zone.center[1] + ring_radius * np.sin(angle)
                node = (round(node_x, 2), round(node_y, 2))
                
                # Only add if not in another no-fly zone
                if self.environment.is_valid_point(node):
                    # Check if node is reasonably close to any corridor
                    if self._is_near_any_corridor(node, max_distance=ring_radius * 1.5):
                        detour_nodes.add(node)
        
        return detour_nodes
    
    def _is_near_any_corridor(self, point: Tuple[float, float], 
                             max_distance: float) -> bool:
        """
        Check if a point is near any corridor segment.
        
        This prevents creating detour nodes in irrelevant areas.
        
        Args:
            point: Point to check
            max_distance: Maximum distance to corridor
            
        Returns:
            True if point is near any corridor
        """
        for i in range(len(self.waypoints) - 1):
            wp1 = self.waypoints[i]
            wp2 = self.waypoints[i + 1]
            
            distance = self._point_to_segment_distance(point, wp1, wp2)
            if distance <= max_distance:
                return True
        
        return False
    
    def _create_edges(self):
        """
        Create edges between nodes within connection range.
        
        Connects each node to nearby nodes within a specified distance.
        """
        # Connection distance (typically 2-3x resolution for 8-connectivity)
        connection_distance = self.resolution * 1.5
        
        nodes_list = list(self.nodes)
        
        for node in nodes_list:
            self.edges[node] = []
            
            for other_node in nodes_list:
                if node == other_node:
                    continue
                
                # Calculate distance
                dx = other_node[0] - node[0]
                dy = other_node[1] - node[1]
                distance = np.sqrt(dx**2 + dy**2)
                
                # Connect if within range and segment is valid
                if distance <= connection_distance:
                    if (self.environment is None or 
                        self.environment.is_valid_segment(node, other_node)):
                        self.edges[node].append(other_node)
    
    def get_neighbors(self, node: Tuple[float, float]) -> List[Tuple[float, float]]:
        """
        Get neighboring nodes for a given node.
        
        Args:
            node: Current node (x, y)
            
        Returns:
            List of neighbor nodes
        """
        return self.edges.get(node, [])
    
    def get_start_node(self) -> Tuple[float, float]:
        """Get the start node (first waypoint)."""
        return self.waypoints[0]
    
    def get_goal_node(self) -> Tuple[float, float]:
        """Get the goal node (last waypoint)."""
        return self.waypoints[-1]
    
    def is_in_corridor(self, point: Tuple[float, float]) -> bool:
        """
        Check if a point is within the FINAL flight corridor.
        
        Note: This uses corridor_width (not search_width)
        The final path must stay within corridor_width of the route.
        
        Args:
            point: Point to check (x, y)
            
        Returns:
            True if point is within corridor bounds
        """
        # Check distance to each segment
        for i in range(len(self.waypoints) - 1):
            wp1 = self.waypoints[i]
            wp2 = self.waypoints[i + 1]
            
            distance = self._point_to_segment_distance(point, wp1, wp2)
            if distance <= self.corridor_width:
                return True
        
        return False
    
    def validate_path(self, path: List[Tuple[float, float]]) -> Tuple[bool, str]:
        """
        Validate that a path stays within corridor constraints.
        
        Args:
            path: Path to validate
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not path:
            return False, "Empty path"
        
        # Check each path segment
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            
            # Check endpoints are valid
            if self.environment and not self.environment.is_valid_point(p1):
                return False, f"Point {p1} in no-fly zone"
            
            # Check segment doesn't cross no-fly zones
            if self.environment and not self.environment.is_valid_segment(p1, p2):
                return False, f"Segment {p1}-{p2} crosses no-fly zone"
        
        return True, "Path valid"
    
    def _point_to_segment_distance(self, point: Tuple[float, float],
                                   p1: Tuple[float, float],
                                   p2: Tuple[float, float]) -> float:
        """
        Calculate perpendicular distance from point to line segment.
        
        Args:
            point: Point to measure from
            p1: Segment start
            p2: Segment end
            
        Returns:
            Perpendicular distance in meters
        """
        px, py = point
        x1, y1 = p1
        x2, y2 = p2
        
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            # Segment is a point
            return np.sqrt((px - x1)**2 + (py - y1)**2)
        
        # Parameter t for closest point on line
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx**2 + dy**2)))
        
        # Closest point on segment
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        # Distance to closest point
        return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)
    
    def get_graph_info(self) -> dict:
        """
        Get information about the graph structure.
        
        Returns:
            Dictionary with graph statistics
        """
        total_edges = sum(len(neighbors) for neighbors in self.edges.values())
        
        return {
            'num_waypoints': len(self.waypoints),
            'num_nodes': len(self.nodes),
            'num_edges': total_edges,
            'corridor_width': self.corridor_width,
            'search_width': self.search_width,
            'resolution': self.resolution,
            'avg_degree': total_edges / len(self.nodes) if self.nodes else 0
        }
    
    def __repr__(self) -> str:
        """String representation of the FlightGraph."""
        info = self.get_graph_info()
        return (f"FlightGraph(waypoints={info['num_waypoints']}, "
                f"nodes={info['num_nodes']}, "
                f"search={self.search_width:.1f}m, "
                f"corridor={self.corridor_width}m)")


if __name__ == "__main__":
    # Example usage
    from pathfinder.models.environment import Environment, Wind
    
    # Define waypoints
    waypoints = [
        (0, 0),
        (100, 50),
        (200, 100),
        (300, 100)
    ]
    
    # Create environment with obstacle
    env = Environment()
    env.add_no_fly_zone(center=(150, 75), radius_m=30, name="Urban Area")
    
    # Create flight graph with obstacle avoidance
    graph = FlightGraph(
        waypoints=waypoints,
        corridor_width=5.0,
        resolution=2.0,
        search_expansion=3.0,  # Search 3x wider to find detours
        environment=env
    )
    
    print(graph)
    print("\nGraph Information:")
    info = graph.get_graph_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test point in corridor
    test_point = (50, 25)
    print(f"\nPoint {test_point} in corridor? {graph.is_in_corridor(test_point)}")
    
    # Get neighbors of start
    start = graph.get_start_node()
    neighbors = graph.get_neighbors(start)
    print(f"\nStart node {start} has {len(neighbors)} neighbors")