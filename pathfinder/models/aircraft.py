"""
AIRCRAFT MODELLING MODULE

@Author: Vaishanth Srinivasan
@Description: This module defines the Aircraft class with aerodynamic and propulsion
characteristics. The default parameters are based on the ScanEagle UAV platform.
"""

from dataclasses import dataclass, field
import math
from typing import Tuple


@dataclass
class Aircraft:
    """
    Represents a UAV with physical and performance characteristics.
    
    Default values are based on the Boeing ScanEagle UAV platform,
    a small reconnaissance drone widely used for surveillance missions.
    
    Attributes:
        mass_kg: Aircraft mass in kilograms (default: 28.0 kg)
        weight_N: Aircraft weight in Newtons (default: 274.6 N)
        wingspan_m: Wing span in meters (default: 3.1 m)
        true_airspeed_mps: True airspeed in meters per second (default: 28.0 m/s)
        aspect_ratio: Wing aspect ratio (wingspan^2/wing_area) (default: 12.0)
        S_m2: Wing reference area in square meters (calculated from wingspan and AR)
        CD0: Zero-lift drag coefficient (default: 0.035)
        k: Induced drag factor (default: calculated from AR and Oswald efficiency)
        eta_prop: Propeller efficiency (default: 0.75)
        sfc_kg_per_W_s: Specific fuel consumption in kg/(W*s) (default: 1.0e-7)
    """
    
    # --- PHYSICAL PROPERTIES --- 
    mass_kg: float = 28.0
    weight_N: float = field(default=None)
    wingspan_m: float = 3.1
    true_airspeed_mps: float = 28.0

    # --- WING GEOMETRY ---
    aspect_ratio: float = 12.0
    S_m2: float = field(default=None)

    # --- AERODYNAMICS ---
    CD0: float = 0.035  # Zero-lift drag coefficient
    k: float = field(default=None)  # Induced drag factor
    oswald_efficiency: float = 0.85  # Oswald efficiency factor

    # --- PROPULSION ---
    eta_prop: float = 0.75  # Propeller efficiency
    sfc_kg_per_W_s: float = 1.0e-7  # Specific fuel consumption

    def __post_init__(self):
        
        """Calculate derived parameters after initialization."""
        # Calculate weight if not provided
        
        if self.weight_N is None:
            self.weight_N = self.mass_kg * 9.80665
        
        # Calculate wing area if not provided
        if self.S_m2 is None:
            self.S_m2 = pow(self.wingspan_m, 2) / self.aspect_ratio
        
        # Calculate induced drag factor if not provided
        if self.k is None:
            self.k = 1.0 / (math.pi * self.oswald_efficiency * self.aspect_ratio)

    def calculate_lift_coefficient(self, velocity_mps: float = None) -> float:
        """
        Calculate the lift coefficient for level flight.
        
        In steady level flight, 
        Lift = Weight | Drag = Thrust
        so:
        CL = Weight / (0.5 * rho * V^2 * S)
        
        Args:
            velocity_mps: Airspeed in m/s. Uses true_airspeed_mps if not provided.
            
        Returns:
            Lift coefficient (dimensionless)
        """
        if velocity_mps is None:
            velocity_mps = self.true_airspeed_mps
        
        rho = 1.225  # Air density at sea level (kg/m^3)
        
        CL = self.weight_N / (0.5 * rho * velocity_mps**2 * self.S_m2)
        return CL

    def calculate_drag_coefficient(self, CL: float = None) -> float:
        """
        Calculate total drag coefficient using the drag polar equation.
        
        CD = CD0 + k * CL^2
        
        Args:
            CL: Lift coefficient. Calculated for cruise if not provided.
            
        Returns:
            Total drag coefficient (dimensionless)
        """
        if CL is None:
            CL = self.calculate_lift_coefficient()
        
        CD = self.CD0 + self.k * CL**2
        return CD

    def calculate_drag_force(self, velocity_mps: float = None) -> float:
        """
        Calculate total drag force in Newtons.
        
        D = 0.5 * rho * V^2 * S * CD
        
        Args:
            velocity_mps: Airspeed in m/s. Uses true_airspeed_mps if not provided.
            
        Returns:
            Drag force in Newtons
        """
        if velocity_mps is None:
            velocity_mps = self.true_airspeed_mps
        
        rho = 1.225  # Air density at sea level (kg/m^3)
        CL = self.calculate_lift_coefficient(velocity_mps)
        CD = self.calculate_drag_coefficient(CL)
        
        drag = 0.5 * rho * velocity_mps**2 * self.S_m2 * CD
        return drag

    def calculate_power_required(self, velocity_mps: float = None) -> float:
        """
        Calculate power required for level flight in Watts.
        
        P_req = D * V / eta_prop
        
        Args:
            velocity_mps: Airspeed in m/s. Uses true_airspeed_mps if not provided.
            
        Returns:
            Power required in Watts
        """
        if velocity_mps is None:
            velocity_mps = self.true_airspeed_mps
        
        drag = self.calculate_drag_force(velocity_mps)
        power = (drag * velocity_mps) / self.eta_prop
        return power

    def calculate_fuel_burn_rate(self, velocity_mps: float = None) -> float:
        """
        Calculate instantaneous fuel burn rate in kg/s.
        
        fuel_rate = SFC * P_req
        
        Args:
            velocity_mps: Airspeed in m/s. Uses true_airspeed_mps if not provided.
            
        Returns:
            Fuel burn rate in kg/s
        """
        power = self.calculate_power_required(velocity_mps)
        fuel_rate = self.sfc_kg_per_W_s * power
        return fuel_rate

    def calculate_fuel_burn(self, distance_m: float, velocity_mps: float = None) -> float:
        """
        Calculate total fuel burned for a given distance.
        
        This is the key method used as a heuristic in path planning.
        
        Args:
            distance_m: Distance to travel in meters
            velocity_mps: Airspeed in m/s. Uses true_airspeed_mps if not provided.
            
        Returns:
            Total fuel burned in kilograms
        """
        if velocity_mps is None:
            velocity_mps = self.true_airspeed_mps
        
        time_s = distance_m / velocity_mps
        fuel_rate = self.calculate_fuel_burn_rate(velocity_mps)
        total_fuel = fuel_rate * time_s
        
        return total_fuel

    def calculate_range(self, fuel_kg: float, velocity_mps: float = None) -> float:
        """
        Calculate maximum range for a given fuel quantity.
        
        Args:
            fuel_kg: Available fuel in kilograms
            velocity_mps: Airspeed in m/s. Uses true_airspeed_mps if not provided.
            
        Returns:
            Maximum range in meters
        """
        if velocity_mps is None:
            velocity_mps = self.true_airspeed_mps
        
        fuel_rate = self.calculate_fuel_burn_rate(velocity_mps)
        flight_time = fuel_kg / fuel_rate
        max_range = velocity_mps * flight_time
        
        return max_range

    def get_performance_summary(self) -> dict:
        """
        Get a summary of aircraft performance characteristics.
        
        Returns:
            Dictionary containing key performance metrics
        """
        CL = self.calculate_lift_coefficient()
        CD = self.calculate_drag_coefficient(CL)
        L_D_ratio = CL / CD if CD > 0 else 0
        
        return {
            'mass_kg': self.mass_kg,
            'cruise_speed_mps': self.true_airspeed_mps,
            'cruise_speed_kph': self.true_airspeed_mps * 3.6,
            'wing_area_m2': self.S_m2,
            'aspect_ratio': self.aspect_ratio,
            'lift_coefficient': CL,
            'drag_coefficient': CD,
            'lift_to_drag_ratio': L_D_ratio,
            'power_required_W': self.calculate_power_required(),
            'fuel_burn_rate_kg_per_s': self.calculate_fuel_burn_rate(),
            'fuel_burn_rate_kg_per_hr': self.calculate_fuel_burn_rate() * 3600
        }

    def __repr__(self) -> str:
        """String representation of the Aircraft."""
        return (f"Aircraft(mass={self.mass_kg}kg, "
                f"wingspan={self.wingspan_m}m, "
                f"cruise_speed={self.true_airspeed_mps}m/s)")


if __name__ == "__main__":
    # Example usage and validation
    uav = Aircraft()
    
    print("ScanEagle UAV Performance Summary")
    print("=" * 50)
    
    perf = uav.get_performance_summary()
    for key, value in perf.items():
        print(f"{key:.<30} {value:.4f}")
    
    print("\n" + "=" * 50)
    print(f"Fuel burn for 1000m: {uav.calculate_fuel_burn(1000):.6f} kg")
    print(f"Range with 5kg fuel: {uav.calculate_range(5.0):.0f} m")