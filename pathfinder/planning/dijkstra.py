"""
Dijkstra's algorithm implementation

@Description: This module implements Dijkstra's shortest path algorithm using fuel burn
as the cost metric instead of simple distance.
"""

import heapq
from typing import List, Tuple, Dict, Optional
from .graph import FlightGraph
from .cost import CostCalculator
from ..models.aircraft import Aircraft
from ..models.environment import Environment


class DijkstraPlanner:
    """
    Dijkstra's algorithm path planner with fuel-based cost.
    
    Guarantees finding the optimal (minimum fuel) path by exploring
    all reachable nodes systematically.
    
    Attributes:
        graph: Flight graph with corridors
        cost_calculator: Cost calculation module
        visited_nodes: Number of nodes explored (for performance analysis)
    """
    
    def __init__(self, graph: FlightGraph, aircraft: Aircraft,
                 environment: Optional[Environment] = None,
                 cost_type: str = 'fuel'):
        """
        Initialize Dijkstra planner.
        
        Args:
            graph: Flight graph
            aircraft: Aircraft model
            environment: Optional environment model
            cost_type: Type of cost to minimize ('fuel', 'distance', 'time')
        """
        self.graph = graph
        self.aircraft = aircraft
        self.environment = environment
        self.cost_calculator = CostCalculator(
            aircraft=aircraft,
            environment=environment,
            cost_type=cost_type
        )
        self.visited_nodes = 0
    
    def find_path(self) -> Tuple[List[Tuple[float, float]], float]:
        """
        Find optimal path from start to goal using Dijkstra's algorithm.
        
        Returns:
            Tuple of (path, total_cost) where:
                - path: List of (x, y) coordinates from start to goal
                - total_cost: Total cost (fuel/distance/time) of the path
        """
        start = self.graph.get_start_node()
        goal = self.graph.get_goal_node()
        
        # Priority queue: (cost, node)
        pq = [(0, start)]
        
        # Cost from start to each node
        g_cost: Dict[Tuple[float, float], float] = {start: 0}
        
        # Parent tracking for path reconstruction
        parent: Dict[Tuple[float, float], Optional[Tuple[float, float]]] = {start: None}
        
        # Visited set
        visited = set()
        
        self.visited_nodes = 0
        
        while pq:
            current_cost, current = heapq.heappop(pq)
            
            # Skip if already visited
            if current in visited:
                continue
            
            visited.add(current)
            self.visited_nodes += 1
            
            # Goal reached
            if current == goal:
                path = self._reconstruct_path(parent, start, goal)
                return path, current_cost
            
            # Explore neighbors
            for neighbor in self.graph.get_neighbors(current):
                if neighbor in visited:
                    continue
                
                # Calculate cost to neighbor
                edge_cost = self.cost_calculator.calculate_cost(current, neighbor)
                tentative_g = current_cost + edge_cost
                
                # Update if this path is better
                if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                    g_cost[neighbor] = tentative_g
                    parent[neighbor] = current
                    heapq.heappush(pq, (tentative_g, neighbor))
        
        # No path found
        return [], float('inf')
    
    def _reconstruct_path(self, parent: Dict[Tuple[float, float], Optional[Tuple[float, float]]],
                         start: Tuple[float, float],
                         goal: Tuple[float, float]) -> List[Tuple[float, float]]:
        """
        Reconstruct path from parent pointers.
        
        Args:
            parent: Dictionary mapping node to its parent
            start: Start node
            goal: Goal node
            
        Returns:
            List of nodes from start to goal
        """
        path = []
        current = goal
        
        while current is not None:
            path.append(current)
            current = parent[current]
        
        path.reverse()
        return path
    
    def get_performance_metrics(self) -> dict:
        """
        Get performance metrics from the last search.
        
        Returns:
            Dictionary with search statistics
        """
        return {
            'algorithm': 'Dijkstra',
            'nodes_visited': self.visited_nodes,
            'total_nodes': len(self.graph.nodes),
            'exploration_ratio': self.visited_nodes / len(self.graph.nodes) if self.graph.nodes else 0
        }


def plan_path_dijkstra(waypoints: List[Tuple[float, float]],
                       aircraft: Aircraft,
                       corridor_width: float = 5.0,
                       resolution: float = 2.0,
                       environment: Optional[Environment] = None,
                       cost_type: str = 'fuel') -> Tuple[List[Tuple[float, float]], float, dict]:
    """
    Convenience function to plan a path using Dijkstra's algorithm.
    
    Args:
        waypoints: List of waypoints defining the mission
        aircraft: Aircraft model
        corridor_width: Flight corridor width in meters
        resolution: Grid resolution in meters
        environment: Optional environment with constraints
        cost_type: Type of cost to minimize
        
    Returns:
        Tuple of (path, cost, metrics)
    """
    # Build graph
    graph = FlightGraph(
        waypoints=waypoints,
        corridor_width=corridor_width,
        resolution=resolution,
        environment=environment
    )
    
    # Plan path
    planner = DijkstraPlanner(
        graph=graph,
        aircraft=aircraft,
        environment=environment,
        cost_type=cost_type
    )
    
    path, cost = planner.find_path()
    metrics = planner.get_performance_metrics()
    
    return path, cost, metrics


if __name__ == "__main__":
    # Example usage
    from ..models.aircraft import Aircraft
    from ..models.environment import Environment, Wind
    
    # Define mission waypoints
    waypoints = [
        (0, 0),
        (100, 50),
        (200, 100)
    ]
    
    # Create aircraft and environment
    aircraft = Aircraft()
    env = Environment(wind=Wind(speed_mps=3.0, direction_deg=90))
    
    # Plan path
    print("Planning path with Dijkstra's algorithm...")
    path, cost, metrics = plan_path_dijkstra(
        waypoints=waypoints,
        aircraft=aircraft,
        corridor_width=5.0,
        resolution=2.0,
        environment=env,
        cost_type='fuel'
    )
    
    print(f"\nPath found with {len(path)} nodes")
    print(f"Total fuel cost: {cost:.6f} kg")
    print("\nPerformance metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    print(f"\nFirst 5 waypoints:")
    for i, point in enumerate(path[:5]):
        print(f"  {i+1}. {point}")