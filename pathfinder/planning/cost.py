"""
Cost calculation module for path planning.

This module provides functions to calculate path costs based on fuel burn,
distance, and other operational factors.
"""

import numpy as np
from typing import Tuple, Optional
from ..models.aircraft import Aircraft
from ..models.environment import Environment


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        p1: First point (x, y)
        p2: Second point (x, y)
        
    Returns:
        Distance in meters
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.sqrt(dx**2 + dy**2)


def calculate_heading(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Calculate heading from p1 to p2 in degrees.
    
    Args:
        p1: Start point (x, y)
        p2: End point (x, y)
        
    Returns:
        Heading in degrees (0 = North, 90 = East)
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    # atan2 gives angle from East, convert to North = 0
    heading_rad = np.arctan2(dx, dy)
    heading_deg = np.degrees(heading_rad)
    
    # Normalize to [0, 360)
    return heading_deg % 360


def fuel_cost(p1: Tuple[float, float], p2: Tuple[float, float],
              aircraft: Aircraft, environment: Optional[Environment] = None) -> float:
    """
    Calculate fuel cost to travel from p1 to p2.
    
    This is the primary cost function for fuel-optimized path planning.
    
    Args:
        p1: Start point (x, y)
        p2: End point (x, y)
        aircraft: Aircraft model
        environment: Optional environment with wind effects
        
    Returns:
        Fuel burned in kilograms
    """
    distance = euclidean_distance(p1, p2)
    
    # If no environment or no wind, use basic calculation
    if environment is None or environment.wind.speed_mps == 0:
        return aircraft.calculate_fuel_burn(distance)
    
    # Account for wind effects on groundspeed
    heading = calculate_heading(p1, p2)
    groundspeed = environment.get_effective_groundspeed(
        aircraft.true_airspeed_mps, heading
    )
    
    # Time to cover ground distance
    if groundspeed <= 0:
        return float('inf')  # Cannot make progress against wind
    
    time_s = distance / groundspeed
    
    # Fuel burn based on airspeed (power required doesn't change with wind)
    fuel_rate = aircraft.calculate_fuel_burn_rate(aircraft.true_airspeed_mps)
    total_fuel = fuel_rate * time_s
    
    return total_fuel


def fuel_heuristic(current: Tuple[float, float], goal: Tuple[float, float],
                  aircraft: Aircraft, environment: Optional[Environment] = None) -> float:
    """
    Heuristic function for A* algorithm using fuel burn estimation.
    
    This provides an optimistic (admissible) estimate of fuel cost to goal.
    
    Args:
        current: Current position (x, y)
        goal: Goal position (x, y)
        aircraft: Aircraft model
        environment: Optional environment (uses no wind for optimistic estimate)
        
    Returns:
        Estimated fuel cost to goal in kilograms
    """
    distance = euclidean_distance(current, goal)
    
    # Use basic fuel calculation without wind for admissible heuristic
    # (straight-line distance with no headwind is optimistic)
    return aircraft.calculate_fuel_burn(distance)


def distance_cost(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Simple distance-based cost function.
    
    Args:
        p1: Start point (x, y)
        p2: End point (x, y)
        
    Returns:
        Distance in meters
    """
    return euclidean_distance(p1, p2)


def time_cost(p1: Tuple[float, float], p2: Tuple[float, float],
             aircraft: Aircraft, environment: Optional[Environment] = None) -> float:
    """
    Calculate time cost to travel from p1 to p2.
    
    Args:
        p1: Start point (x, y)
        p2: End point (x, y)
        aircraft: Aircraft model
        environment: Optional environment with wind effects
        
    Returns:
        Time in seconds
    """
    distance = euclidean_distance(p1, p2)
    
    if environment is None or environment.wind.speed_mps == 0:
        return distance / aircraft.true_airspeed_mps
    
    heading = calculate_heading(p1, p2)
    groundspeed = environment.get_effective_groundspeed(
        aircraft.true_airspeed_mps, heading
    )
    
    if groundspeed <= 0:
        return float('inf')
    
    return distance / groundspeed


def composite_cost(p1: Tuple[float, float], p2: Tuple[float, float],
                  aircraft: Aircraft, environment: Optional[Environment] = None,
                  fuel_weight: float = 1.0, time_weight: float = 0.0) -> float:
    """
    Calculate composite cost with weighted fuel and time components.
    
    This allows flexibility for missions that prioritize speed vs efficiency.
    
    Args:
        p1: Start point (x, y)
        p2: End point (x, y)
        aircraft: Aircraft model
        environment: Optional environment
        fuel_weight: Weight for fuel cost (default: 1.0)
        time_weight: Weight for time cost (default: 0.0)
        
    Returns:
        Weighted composite cost
    """
    f_cost = fuel_cost(p1, p2, aircraft, environment)
    t_cost = time_cost(p1, p2, aircraft, environment)
    
    # Normalize time to comparable scale (assume 1kg fuel = 3600s acceptable trade)
    time_normalized = t_cost / 3600.0
    
    return fuel_weight * f_cost + time_weight * time_normalized


class CostCalculator:
    """
    Configurable cost calculator for path planning.
    
    Attributes:
        aircraft: Aircraft model
        environment: Environment model
        cost_type: Type of cost ('fuel', 'distance', 'time', 'composite')
        fuel_weight: Weight for fuel in composite cost
        time_weight: Weight for time in composite cost
    """
    
    def __init__(self, aircraft: Aircraft, 
                 environment: Optional[Environment] = None,
                 cost_type: str = 'fuel',
                 fuel_weight: float = 1.0,
                 time_weight: float = 0.0):
        """
        Initialize cost calculator.
        
        Args:
            aircraft: Aircraft model
            environment: Optional environment model
            cost_type: 'fuel', 'distance', 'time', or 'composite'
            fuel_weight: Weight for fuel in composite cost
            time_weight: Weight for time in composite cost
        """
        self.aircraft = aircraft
        self.environment = environment
        self.cost_type = cost_type
        self.fuel_weight = fuel_weight
        self.time_weight = time_weight
    
    def calculate_cost(self, p1: Tuple[float, float], 
                      p2: Tuple[float, float]) -> float:
        """
        Calculate cost between two points based on configured cost type.
        
        Args:
            p1: Start point (x, y)
            p2: End point (x, y)
            
        Returns:
            Cost value
        """
        if self.cost_type == 'fuel':
            return fuel_cost(p1, p2, self.aircraft, self.environment)
        elif self.cost_type == 'distance':
            return distance_cost(p1, p2)
        elif self.cost_type == 'time':
            return time_cost(p1, p2, self.aircraft, self.environment)
        elif self.cost_type == 'composite':
            return composite_cost(p1, p2, self.aircraft, self.environment,
                                self.fuel_weight, self.time_weight)
        else:
            raise ValueError(f"Unknown cost type: {self.cost_type}")
    
    def calculate_heuristic(self, current: Tuple[float, float],
                          goal: Tuple[float, float]) -> float:
        """
        Calculate heuristic estimate to goal.
        
        Args:
            current: Current position (x, y)
            goal: Goal position (x, y)
            
        Returns:
            Heuristic cost estimate
        """
        if self.cost_type == 'fuel' or self.cost_type == 'composite':
            return fuel_heuristic(current, goal, self.aircraft, self.environment)
        elif self.cost_type == 'distance':
            return euclidean_distance(current, goal)
        elif self.cost_type == 'time':
            distance = euclidean_distance(current, goal)
            return distance / self.aircraft.true_airspeed_mps
        else:
            raise ValueError(f"Unknown cost type: {self.cost_type}")


if __name__ == "__main__":
    # Example usage
    from ..models.aircraft import Aircraft
    from ..models.environment import Environment, Wind
    
    aircraft = Aircraft()
    env = Environment(wind=Wind(speed_mps=5.0, direction_deg=90))
    
    p1 = (0, 0)
    p2 = (1000, 0)  # 1km East
    
    print("Cost Comparison for 1km eastward flight:")
    print("=" * 50)
    print(f"Distance cost: {distance_cost(p1, p2):.2f} m")
    print(f"Fuel cost (no wind): {fuel_cost(p1, p2, aircraft):.6f} kg")
    print(f"Fuel cost (with wind): {fuel_cost(p1, p2, aircraft, env):.6f} kg")
    print(f"Time cost (no wind): {time_cost(p1, p2, aircraft):.2f} s")
    print(f"Time cost (with wind): {time_cost(p1, p2, aircraft, env):.2f} s")
    
    # Heuristic
    goal = (5000, 0)
    print(f"\nHeuristic to goal (5km away): {fuel_heuristic(p1, goal, aircraft):.6f} kg")