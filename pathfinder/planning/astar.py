"""
A* algorithm implementation

@Description: This module implements A* search with fuel burn as both the cost metric
and heuristic, providing faster pathfinding than Dijkstra while maintaining
optimality.
"""

import heapq
from typing import List, Tuple, Dict, Optional
from .graph import FlightGraph
from .cost import CostCalculator
from ..models.aircraft import Aircraft
from ..models.environment import Environment


class AStarPlanner:
    """
    A* algorithm path planner with fuel-based heuristic.
    
    Uses fuel burn estimation as a heuristic to guide search toward the goal,
    typically exploring far fewer nodes than Dijkstra while still finding
    the optimal path.
    
    Attributes:
        graph: Flight graph with corridors
        cost_calculator: Cost calculation module
        visited_nodes: Number of nodes explored (for performance analysis)
    """
    
    def __init__(self, graph: FlightGraph, aircraft: Aircraft,
                 environment: Optional[Environment] = None,
                 cost_type: str = 'fuel'):
        """
        Initialize A* planner.
        
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
        Find optimal path from start to goal using A* algorithm.
        
        Returns:
            Tuple of (path, total_cost) where:
                - path: List of (x, y) coordinates from start to goal
                - total_cost: Total cost (fuel/distance/time) of the path
        """
        start = self.graph.get_start_node()
        goal = self.graph.get_goal_node()
        
        # Priority queue: (f_cost, g_cost, node)
        # f_cost = g_cost + h_cost (total estimated cost)
        h_start = self.cost_calculator.calculate_heuristic(start, goal)
        pq = [(h_start, 0, start)]
        
        # Cost from start to each node (g_cost)
        g_cost: Dict[Tuple[float, float], float] = {start: 0}
        
        # Estimated total cost (f_cost)
        f_cost: Dict[Tuple[float, float], float] = {start: h_start}
        
        # Parent tracking for path reconstruction
        parent: Dict[Tuple[float, float], Optional[Tuple[float, float]]] = {start: None}
        
        # Visited set
        visited = set()
        
        self.visited_nodes = 0
        
        while pq:
            current_f, current_g, current = heapq.heappop(pq)
            
            # Skip if already visited
            if current in visited:
                continue
            
            visited.add(current)
            self.visited_nodes += 1
            
            # Goal reached
            if current == goal:
                path = self._reconstruct_path(parent, start, goal)
                return path, current_g
            
            # Explore neighbors
            for neighbor in self.graph.get_neighbors(current):
                if neighbor in visited:
                    continue
                
                # Calculate cost to neighbor
                edge_cost = self.cost_calculator.calculate_cost(current, neighbor)
                tentative_g = current_g + edge_cost
                
                # Update if this path is better
                if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                    g_cost[neighbor] = tentative_g
                    
                    # Calculate heuristic to goal
                    h = self.cost_calculator.calculate_heuristic(neighbor, goal)
                    f = tentative_g + h
                    
                    f_cost[neighbor] = f
                    parent[neighbor] = current
                    heapq.heappush(pq, (f, tentative_g, neighbor))
        
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
            'algorithm': 'A*',
            'nodes_visited': self.visited_nodes,
            'total_nodes': len(self.graph.nodes),
            'exploration_ratio': self.visited_nodes / len(self.graph.nodes) if self.graph.nodes else 0
        }


def plan_path_astar(waypoints: List[Tuple[float, float]],
                    aircraft: Aircraft,
                    corridor_width: float = 5.0,
                    resolution: float = 2.0,
                    environment: Optional[Environment] = None,
                    cost_type: str = 'fuel') -> Tuple[List[Tuple[float, float]], float, dict]:
    """
    Convenience function to plan a path using A* algorithm.
    
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
    planner = AStarPlanner(
        graph=graph,
        aircraft=aircraft,
        environment=environment,
        cost_type=cost_type
    )
    
    path, cost = planner.find_path()
    metrics = planner.get_performance_metrics()
    
    return path, cost, metrics


def compare_algorithms(waypoints: List[Tuple[float, float]],
                      aircraft: Aircraft,
                      corridor_width: float = 5.0,
                      resolution: float = 2.0,
                      environment: Optional[Environment] = None) -> dict:
    """
    Compare Dijkstra and A* performance on the same problem.
    
    Args:
        waypoints: List of waypoints
        aircraft: Aircraft model
        corridor_width: Corridor width
        resolution: Grid resolution
        environment: Optional environment
        
    Returns:
        Dictionary with comparison results
    """
    from .dijkstra import plan_path_dijkstra
    
    # Run both algorithms
    path_dijkstra, cost_dijkstra, metrics_dijkstra = plan_path_dijkstra(
        waypoints, aircraft, corridor_width, resolution, environment, 'fuel'
    )
    
    path_astar, cost_astar, metrics_astar = plan_path_astar(
        waypoints, aircraft, corridor_width, resolution, environment, 'fuel'
    )
    
    # Calculate speedup
    speedup = (metrics_dijkstra['nodes_visited'] / 
               metrics_astar['nodes_visited'] if metrics_astar['nodes_visited'] > 0 else 0)
    
    return {
        'dijkstra': {
            'path_length': len(path_dijkstra),
            'cost': cost_dijkstra,
            'nodes_visited': metrics_dijkstra['nodes_visited']
        },
        'astar': {
            'path_length': len(path_astar),
            'cost': cost_astar,
            'nodes_visited': metrics_astar['nodes_visited']
        },
        'speedup_factor': speedup,
        'cost_difference': abs(cost_dijkstra - cost_astar)
    }


if __name__ == "__main__":
    # Example usage
    from ..models.aircraft import Aircraft
    from ..models.environment import Environment, Wind
    
    # Define mission waypoints
    waypoints = [
        (0, 0),
        (100, 50),
        (200, 100),
        (300, 150)
    ]
    
    # Create aircraft and environment
    aircraft = Aircraft()
    env = Environment(wind=Wind(speed_mps=3.0, direction_deg=90))
    
    # Plan path with A*
    print("Planning path with A* algorithm...")
    path, cost, metrics = plan_path_astar(
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
    
    # Compare algorithms
    print("\n" + "="*50)
    print("Algorithm Comparison")
    print("="*50)
    
    comparison = compare_algorithms(
        waypoints=waypoints,
        aircraft=aircraft,
        corridor_width=5.0,
        resolution=2.0,
        environment=env
    )
    
    print(f"\nDijkstra nodes visited: {comparison['dijkstra']['nodes_visited']}")
    print(f"A* nodes visited: {comparison['astar']['nodes_visited']}")
    print(f"Speedup factor: {comparison['speedup_factor']:.2f}x")
    print(f"Cost difference: {comparison['cost_difference']:.8f} kg (should be ~0 for optimal)")