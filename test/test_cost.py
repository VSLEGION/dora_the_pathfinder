"""
Unit tests for cost calculation functions.

Run with: pytest tests/test_cost.py
"""

import pytest
import math
from pathfinder.models.aircraft import Aircraft
from pathfinder.models.environment import Environment, Wind
from pathfinder.planning.cost import (
    euclidean_distance,
    calculate_heading,
    fuel_cost,
    fuel_heuristic,
    distance_cost,
    time_cost,
    composite_cost,
    CostCalculator
)


class TestDistanceCalculations:
    """Tests for distance and heading calculations."""
    
    def test_euclidean_distance_horizontal(self):
        """Test distance calculation for horizontal line."""
        p1 = (0, 0)
        p2 = (100, 0)
        
        distance = euclidean_distance(p1, p2)
        assert abs(distance - 100.0) < 0.001
    
    def test_euclidean_distance_vertical(self):
        """Test distance calculation for vertical line."""
        p1 = (0, 0)
        p2 = (0, 50)
        
        distance = euclidean_distance(p1, p2)
        assert abs(distance - 50.0) < 0.001
    
    def test_euclidean_distance_diagonal(self):
        """Test distance calculation for diagonal line."""
        p1 = (0, 0)
        p2 = (3, 4)
        
        distance = euclidean_distance(p1, p2)
        assert abs(distance - 5.0) < 0.001  # 3-4-5 triangle
    
    def test_euclidean_distance_same_point(self):
        """Test distance when points are the same."""
        p1 = (10, 20)
        p2 = (10, 20)
        
        distance = euclidean_distance(p1, p2)
        assert distance == 0.0
    
    def test_calculate_heading_north(self):
        """Test heading calculation for north direction."""
        p1 = (0, 0)
        p2 = (0, 100)
        
        heading = calculate_heading(p1, p2)
        assert abs(heading - 0.0) < 0.001  # North is 0 deg
    
    def test_calculate_heading_east(self):
        """Test heading calculation for east direction."""
        p1 = (0, 0)
        p2 = (100, 0)
        
        heading = calculate_heading(p1, p2)
        assert abs(heading - 90.0) < 0.001  # East is 90 deg
    
    def test_calculate_heading_south(self):
        """Test heading calculation for south direction."""
        p1 = (0, 0)
        p2 = (0, -100)
        
        heading = calculate_heading(p1, p2)
        assert abs(heading - 180.0) < 0.001  # South is 180 deg
    
    def test_calculate_heading_west(self):
        """Test heading calculation for west direction."""
        p1 = (0, 0)
        p2 = (-100, 0)
        
        heading = calculate_heading(p1, p2)
        assert abs(heading - 270.0) < 0.001  # West is 270 deg


class TestFuelCost:
    """Tests for fuel cost calculations."""
    
    def test_fuel_cost_no_wind(self):
        """Test fuel cost without wind."""
        aircraft = Aircraft()
        p1 = (0, 0)
        p2 = (1000, 0)
        
        cost = fuel_cost(p1, p2, aircraft)
        
        assert cost > 0
        # Should match direct fuel burn calculation
        expected = aircraft.calculate_fuel_burn(1000.0)
        assert abs(cost - expected) < 0.0001
    
    def test_fuel_cost_with_tailwind(self):
        """Test fuel cost with tailwind (should save time)."""
        aircraft = Aircraft()
        env = Environment(wind=Wind(speed_mps=5.0, direction_deg=90))  # East wind
        
        p1 = (0, 0)
        p2 = (1000, 0)  # Flying east (with wind)
        
        cost_with_wind = fuel_cost(p1, p2, aircraft, env)
        cost_no_wind = fuel_cost(p1, p2, aircraft)
        
        # With tailwind, should use less time (though fuel rate same)
        assert cost_with_wind > 0
    
    def test_fuel_cost_zero_distance(self):
        """Test fuel cost for zero distance."""
        aircraft = Aircraft()
        p1 = (0, 0)
        p2 = (0, 0)
        
        cost = fuel_cost(p1, p2, aircraft)
        assert cost == 0.0
    
    def test_fuel_cost_increases_with_distance(self):
        """Test fuel cost scales with distance."""
        aircraft = Aircraft()
        p1 = (0, 0)
        
        cost_1km = fuel_cost(p1, (1000, 0), aircraft)
        cost_2km = fuel_cost(p1, (2000, 0), aircraft)
        
        # Should be approximately double
        assert abs(cost_2km - 2 * cost_1km) < 0.0001


class TestFuelHeuristic:
    """Tests for fuel heuristic function."""
    
    def test_heuristic_admissible(self):
        """Test heuristic is admissible (optimistic)."""
        aircraft = Aircraft()
        current = (0, 0)
        goal = (1000, 0)
        
        h = fuel_heuristic(current, goal, aircraft)
        actual = fuel_cost(current, goal, aircraft)
        
        # Heuristic should not overestimate
        assert h <= actual + 0.0001
    
    def test_heuristic_at_goal(self):
        """Test heuristic is zero at goal."""
        aircraft = Aircraft()
        goal = (100, 100)
        
        h = fuel_heuristic(goal, goal, aircraft)
        assert h == 0.0
    
    def test_heuristic_positive(self):
        """Test heuristic is positive for non-goal states."""
        aircraft = Aircraft()
        current = (0, 0)
        goal = (1000, 1000)
        
        h = fuel_heuristic(current, goal, aircraft)
        assert h > 0


class TestOtherCostFunctions:
    """Tests for distance, time, and composite costs."""
    
    def test_distance_cost(self):
        """Test simple distance cost."""
        p1 = (0, 0)
        p2 = (100, 0)
        
        cost = distance_cost(p1, p2)
        assert abs(cost - 100.0) < 0.001
    
    def test_time_cost_no_wind(self):
        """Test time cost without wind."""
        aircraft = Aircraft(true_airspeed_mps=25.0)
        p1 = (0, 0)
        p2 = (100, 0)
        
        cost = time_cost(p1, p2, aircraft)
        expected_time = 100.0 / 25.0  # 4 seconds
        
        assert abs(cost - expected_time) < 0.001
    
    def test_time_cost_with_wind(self):
        """Test time cost with wind effects."""
        aircraft = Aircraft(true_airspeed_mps=25.0)
        env = Environment(wind=Wind(speed_mps=5.0, direction_deg=90))
        
        p1 = (0, 0)
        p2 = (1000, 0)  # Flying east with tailwind
        
        time_with_wind = time_cost(p1, p2, aircraft, env)
        time_no_wind = time_cost(p1, p2, aircraft)
        
        # Should be faster with tailwind
        assert time_with_wind < time_no_wind
    
    def test_composite_cost(self):
        """Test composite cost function."""
        aircraft = Aircraft()
        p1 = (0, 0)
        p2 = (1000, 0)
        
        # Fuel-only
        cost_fuel = composite_cost(p1, p2, aircraft, fuel_weight=1.0, time_weight=0.0)
        expected_fuel = fuel_cost(p1, p2, aircraft)
        assert abs(cost_fuel - expected_fuel) < 0.0001
        
        # Time-only
        cost_time = composite_cost(p1, p2, aircraft, fuel_weight=0.0, time_weight=1.0)
        assert cost_time > 0
        
        # Balanced
        cost_balanced = composite_cost(p1, p2, aircraft, fuel_weight=0.5, time_weight=0.5)
        assert cost_balanced > 0


class TestCostCalculator:
    """Tests for CostCalculator class."""
    
    def test_calculator_initialization(self):
        """Test cost calculator initializes correctly."""
        aircraft = Aircraft()
        calc = CostCalculator(aircraft, cost_type='fuel')
        
        assert calc.aircraft == aircraft
        assert calc.cost_type == 'fuel'
    
    def test_calculator_fuel_cost(self):
        """Test calculator with fuel cost type."""
        aircraft = Aircraft()
        calc = CostCalculator(aircraft, cost_type='fuel')
        
        p1 = (0, 0)
        p2 = (1000, 0)
        
        cost = calc.calculate_cost(p1, p2)
        expected = fuel_cost(p1, p2, aircraft)
        
        assert abs(cost - expected) < 0.0001
    
    def test_calculator_distance_cost(self):
        """Test calculator with distance cost type."""
        aircraft = Aircraft()
        calc = CostCalculator(aircraft, cost_type='distance')
        
        p1 = (0, 0)
        p2 = (100, 0)
        
        cost = calc.calculate_cost(p1, p2)
        assert abs(cost - 100.0) < 0.001
    
    def test_calculator_time_cost(self):
        """Test calculator with time cost type."""
        aircraft = Aircraft(true_airspeed_mps=25.0)
        calc = CostCalculator(aircraft, cost_type='time')
        
        p1 = (0, 0)
        p2 = (100, 0)
        
        cost = calc.calculate_cost(p1, p2)
        expected_time = 100.0 / 25.0
        
        assert abs(cost - expected_time) < 0.001
    
    def test_calculator_invalid_cost_type(self):
        """Test calculator raises error for invalid cost type."""
        aircraft = Aircraft()
        calc = CostCalculator(aircraft, cost_type='invalid')
        
        with pytest.raises(ValueError):
            calc.calculate_cost((0, 0), (100, 0))
    
    def test_calculator_heuristic(self):
        """Test calculator heuristic calculation."""
        aircraft = Aircraft()
        calc = CostCalculator(aircraft, cost_type='fuel')
        
        current = (0, 0)
        goal = (1000, 0)
        
        h = calc.calculate_heuristic(current, goal)
        expected = fuel_heuristic(current, goal, aircraft)
        
        assert abs(h - expected) < 0.0001


# Fixtures
@pytest.fixture
def default_aircraft():
    """Fixture providing default aircraft."""
    return Aircraft()


@pytest.fixture
def windy_environment():
    """Fixture providing environment with wind."""
    return Environment(wind=Wind(speed_mps=10.0, direction_deg=90))


# Integration tests
def test_cost_consistency(default_aircraft):
    """Test that different cost calculations are consistent."""
    p1 = (0, 0)
    p2 = (1000, 0)
    
    # Direct calculation
    direct_fuel = default_aircraft.calculate_fuel_burn(1000.0)
    
    # Via cost function
    cost_func_fuel = fuel_cost(p1, p2, default_aircraft)
    
    # Via calculator
    calc = CostCalculator(default_aircraft, cost_type='fuel')
    calc_fuel = calc.calculate_cost(p1, p2)
    
    # All should match
    assert abs(direct_fuel - cost_func_fuel) < 0.0001
    assert abs(direct_fuel - calc_fuel) < 0.0001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])