"""
Dora the Pathfinder

A Python library for fuel-optimized path planning with dynamic flight corridors.
"""

__version__ = "0.1.0"
__author__ = "Vaishanth Srinivasan"
__license__ = "MIT"

# Import main classes for easy access
from .models.aircraft import Aircraft
from .models.environment import Environment, Wind, NoFlyZone
from .planning.graph import FlightGraph
from .planning.dijkstra import DijkstraPlanner, plan_path_dijkstra
from .planning.astar import AStarPlanner, plan_path_astar
from .planning.cost import CostCalculator

__all__ = [
    'Aircraft',
    'Environment',
    'Wind',
    'NoFlyZone',
    'FlightGraph',
    'DijkstraPlanner',
    'AStarPlanner',
    'plan_path_dijkstra',
    'plan_path_astar',
    'CostCalculator',
]