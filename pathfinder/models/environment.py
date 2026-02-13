"""
@Author: Vaishanth Srinivasan
@License: MIT

@Description: This module defines environmental conditions and constraints that affect
path planning, including wind, terrain, and restricted zones.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class Wind:
    """
    Represents wind conditions in the environment.
    
    Attributes:
        speed_mps: Wind speed in meters per second
        direction_deg: Wind direction in degrees (0 = North, 90 = East)
    """
    speed_mps: float = 0.0
    direction_deg: float = 0.0
    
    def get_component(self, heading_deg: float) -> Tuple[float, float]:
        """
        Calculate headwind and crosswind components for a given heading.
        
        Args:
            heading_deg: Aircraft heading in degrees
            
        Returns:
            Tuple of (headwind, crosswind) in m/s
            Positive headwind = tailwind, negative = headwind
        """
        wind_rad = np.radians(self.direction_deg)
        heading_rad = np.radians(heading_deg)
        
        # Relative wind angle
        relative_angle = wind_rad - heading_rad
        
        # Components
        headwind = self.speed_mps * np.cos(relative_angle)
        crosswind = self.speed_mps * np.sin(relative_angle)
        
        return headwind, crosswind


@dataclass
class NoFlyZone:
    """
    Represents a restricted airspace region.
    
    Attributes:
        center: (x, y) coordinates of zone center
        radius_m: Radius of circular no-fly zone in meters
        name: Optional name/identifier for the zone
    """
    center: Tuple[float, float]
    radius_m: float
    name: Optional[str] = None
    
    def contains_point(self, point: Tuple[float, float]) -> bool:
        """
        Check if a point is inside the no-fly zone.
        
        Args:
            point: (x, y) coordinates to check
            
        Returns:
            True if point is inside the zone
        """
        dx = point[0] - self.center[0]
        dy = point[1] - self.center[1]
        distance = np.sqrt(dx**2 + dy**2)
        return distance <= self.radius_m
    
    def intersects_segment(self, p1: Tuple[float, float], 
                          p2: Tuple[float, float]) -> bool:
        """
        Check if a line segment intersects the no-fly zone.
        
        Args:
            p1: Start point (x, y)
            p2: End point (x, y)
            
        Returns:
            True if segment intersects the zone
        """
        # Vector from p1 to p2
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        # Vector from p1 to center
        fx = self.center[0] - p1[0]
        fy = self.center[1] - p1[1]
        
        # Quadratic coefficients for intersection
        a = dx**2 + dy**2
        b = -2 * (fx * dx + fy * dy)
        c = fx**2 + fy**2 - self.radius_m**2
        
        discriminant = b**2 - 4 * a * c
        
        # No intersection if discriminant is negative
        if discriminant < 0:
            return False
        
        # Check if intersection occurs within segment (t in [0, 1])
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b + sqrt_disc) / (2 * a)
        t2 = (-b - sqrt_disc) / (2 * a)
        
        return (0 <= t1 <= 1) or (0 <= t2 <= 1)


@dataclass
class Environment:
    """
    Represents the complete operational environment for path planning.
    
    Attributes:
        wind: Wind conditions
        no_fly_zones: List of restricted airspace regions
        altitude_m: Operating altitude in meters (default: 100m)
        bounds: Optional operational area bounds ((min_x, min_y), (max_x, max_y))
    """
    wind: Wind = field(default_factory=Wind)
    no_fly_zones: List[NoFlyZone] = field(default_factory=list)
    altitude_m: float = 100.0
    bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    
    def add_no_fly_zone(self, center: Tuple[float, float], 
                       radius_m: float, name: Optional[str] = None):
        """
        Add a no-fly zone to the environment.
        
        Args:
            center: (x, y) coordinates of zone center
            radius_m: Radius in meters
            name: Optional identifier
        """
        zone = NoFlyZone(center=center, radius_m=radius_m, name=name)
        self.no_fly_zones.append(zone)
    
    def is_valid_point(self, point: Tuple[float, float]) -> bool:
        """
        Check if a point is valid (not in no-fly zones, within bounds).
        
        Args:
            point: (x, y) coordinates to check
            
        Returns:
            True if point is valid for flight
        """
        # Check bounds if specified
        if self.bounds is not None:
            min_bounds, max_bounds = self.bounds
            if not (min_bounds[0] <= point[0] <= max_bounds[0] and
                   min_bounds[1] <= point[1] <= max_bounds[1]):
                return False
        
        # Check no-fly zones
        for zone in self.no_fly_zones:
            if zone.contains_point(point):
                return False
        
        return True
    
    def is_valid_segment(self, p1: Tuple[float, float], 
                        p2: Tuple[float, float]) -> bool:
        """
        Check if a path segment is valid (doesn't intersect no-fly zones).
        
        Args:
            p1: Start point (x, y)
            p2: End point (x, y)
            
        Returns:
            True if segment is valid for flight
        """
        # Check if endpoints are valid
        if not self.is_valid_point(p1) or not self.is_valid_point(p2):
            return False
        
        # Check intersection with no-fly zones
        for zone in self.no_fly_zones:
            if zone.intersects_segment(p1, p2):
                return False
        
        return True
    
    def get_effective_groundspeed(self, airspeed_mps: float, 
                                 heading_deg: float) -> float:
        """
        Calculate effective groundspeed considering wind.
        
        Args:
            airspeed_mps: Aircraft true airspeed
            heading_deg: Aircraft heading in degrees
            
        Returns:
            Groundspeed in m/s
        """
        headwind, _ = self.wind.get_component(heading_deg)
        groundspeed = airspeed_mps + headwind
        return max(0.0, groundspeed)  # Prevent negative groundspeed
    
    def __repr__(self) -> str:
        """String representation of the Environment."""
        return (f"Environment(wind={self.wind.speed_mps:.1f}m/s, "
                f"no_fly_zones={len(self.no_fly_zones)}, "
                f"altitude={self.altitude_m}m)")


if __name__ == "__main__":
    # Example usage
    env = Environment(
        wind=Wind(speed_mps=5.0, direction_deg=90),  # 5 m/s from East
        altitude_m=150.0
    )
    
    # Add no-fly zones
    env.add_no_fly_zone(center=(100, 100), radius_m=50, name="Restricted Area 1")
    env.add_no_fly_zone(center=(200, 150), radius_m=30, name="Restricted Area 2")
    
    print(env)
    print(f"\nPoint (100, 100) valid? {env.is_valid_point((100, 100))}")
    print(f"Point (50, 50) valid? {env.is_valid_point((50, 50))}")
    print(f"Segment valid? {env.is_valid_segment((0, 0), (50, 50))}")
    
    # Wind effects
    groundspeed_north = env.get_effective_groundspeed(28.0, 0)  # Heading North
    groundspeed_east = env.get_effective_groundspeed(28.0, 90)  # Heading East
    print(f"\nGroundspeed heading North: {groundspeed_north:.2f} m/s")
    print(f"Groundspeed heading East: {groundspeed_east:.2f} m/s")