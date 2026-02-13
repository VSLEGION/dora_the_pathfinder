"""
Unit tests for the Aircraft model.

Run with: pytest tests/test_aircraft.py
"""

import pytest
import math
from pathfinder.models.aircraft import Aircraft


class TestAircraftInitialization:
    """Tests for aircraft initialization."""
    
    def test_default_initialization(self):
        """Test aircraft initializes with default ScanEagle parameters."""
        aircraft = Aircraft()
        
        assert aircraft.mass_kg == 28.0
        assert aircraft.wingspan_m == 3.1
        assert aircraft.true_airspeed_mps == 28.0
        assert aircraft.aspect_ratio == 12.0
        
    def test_weight_calculation(self):
        """Test weight is correctly calculated from mass."""
        aircraft = Aircraft(mass_kg=30.0)
        
        expected_weight = 30.0 * 9.80665
        assert abs(aircraft.weight_N - expected_weight) < 0.001
    
    def test_wing_area_calculation(self):
        """Test wing area is correctly calculated."""
        aircraft = Aircraft(wingspan_m=4.0, aspect_ratio=10.0)
        
        expected_area = 4.0**2 / 10.0
        assert abs(aircraft.S_m2 - expected_area) < 0.001
    
    def test_custom_parameters(self):
        """Test aircraft with custom parameters."""
        aircraft = Aircraft(
            mass_kg=25.0,
            wingspan_m=3.0,
            true_airspeed_mps=30.0
        )
        
        assert aircraft.mass_kg == 25.0
        assert aircraft.wingspan_m == 3.0
        assert aircraft.true_airspeed_mps == 30.0


class TestAerodynamicCalculations:
    """Tests for aerodynamic calculations."""
    
    def test_lift_coefficient(self):
        """Test lift coefficient calculation."""
        aircraft = Aircraft()
        CL = aircraft.calculate_lift_coefficient()
        
        # CL should be positive and reasonable (typically 0.3-1.5)
        assert 0.1 < CL < 2.0
    
    def test_drag_coefficient(self):
        """Test drag coefficient calculation."""
        aircraft = Aircraft()
        CD = aircraft.calculate_drag_coefficient()
        
        # CD should be positive and reasonable
        assert 0.01 < CD < 0.2
        
        # CD should be greater than CD0
        assert CD > aircraft.CD0
    
    def test_drag_force(self):
        """Test drag force calculation."""
        aircraft = Aircraft()
        drag = aircraft.calculate_drag_force()
        
        # Drag should be positive
        assert drag > 0
        
        # Drag should increase with velocity
        drag_slow = aircraft.calculate_drag_force(velocity_mps=20.0)
        drag_fast = aircraft.calculate_drag_force(velocity_mps=40.0)
        assert drag_fast > drag_slow
    
    def test_power_required(self):
        """Test power required calculation."""
        aircraft = Aircraft()
        power = aircraft.calculate_power_required()
        
        # Power should be positive
        assert power > 0
        
        # Power should be in reasonable range (hundreds of Watts for ScanEagle)
        assert 100 < power < 2000


class TestFuelCalculations:
    """Tests for fuel burn calculations."""
    
    def test_fuel_burn_rate(self):
        """Test fuel burn rate calculation."""
        aircraft = Aircraft()
        fuel_rate = aircraft.calculate_fuel_burn_rate()
        
        # Fuel rate should be positive
        assert fuel_rate > 0
        
        # Should be in reasonable range (grams per second)
        assert 1e-5 < fuel_rate < 1e-2
    
    def test_fuel_burn_for_distance(self):
        """Test fuel burn calculation for a specific distance."""
        aircraft = Aircraft()
        
        # 1000m should burn some fuel
        fuel_1km = aircraft.calculate_fuel_burn(1000.0)
        assert fuel_1km > 0
        
        # 2000m should burn approximately twice as much
        fuel_2km = aircraft.calculate_fuel_burn(2000.0)
        assert abs(fuel_2km - 2 * fuel_1km) < 0.0001
    
    def test_fuel_burn_zero_distance(self):
        """Test fuel burn for zero distance."""
        aircraft = Aircraft()
        fuel = aircraft.calculate_fuel_burn(0.0)
        
        assert fuel == 0.0
    
    def test_fuel_burn_different_speeds(self):
        """Test fuel burn at different speeds."""
        aircraft = Aircraft()
        
        fuel_slow = aircraft.calculate_fuel_burn(1000.0, velocity_mps=20.0)
        fuel_fast = aircraft.calculate_fuel_burn(1000.0, velocity_mps=40.0)
        
        # Both should be positive
        assert fuel_slow > 0
        assert fuel_fast > 0
        
        # Fuel burn is complex - could be higher or lower depending on drag
        # Just verify they're different and reasonable
        assert fuel_slow != fuel_fast
    
    def test_range_calculation(self):
        """Test maximum range calculation."""
        aircraft = Aircraft()
        
        # With 1kg of fuel, should get some range
        max_range = aircraft.calculate_range(fuel_kg=1.0)
        
        assert max_range > 0
        
        # More fuel should give more range (linear relationship)
        range_2kg = aircraft.calculate_range(fuel_kg=2.0)
        assert abs(range_2kg - 2 * max_range) < 1.0


class TestPerformanceSummary:
    """Tests for performance summary."""
    
    def test_performance_summary_keys(self):
        """Test performance summary contains expected keys."""
        aircraft = Aircraft()
        perf = aircraft.get_performance_summary()
        
        expected_keys = [
            'mass_kg',
            'cruise_speed_mps',
            'cruise_speed_kph',
            'wing_area_m2',
            'aspect_ratio',
            'lift_coefficient',
            'drag_coefficient',
            'lift_to_drag_ratio',
            'power_required_W',
            'fuel_burn_rate_kg_per_s',
            'fuel_burn_rate_kg_per_hr'
        ]
        
        for key in expected_keys:
            assert key in perf
    
    def test_performance_summary_values(self):
        """Test performance summary values are reasonable."""
        aircraft = Aircraft()
        perf = aircraft.get_performance_summary()
        
        # Check some key values are in reasonable ranges
        assert perf['lift_to_drag_ratio'] > 5  # Should have decent L/D
        assert perf['cruise_speed_kph'] > 50  # Should cruise faster than 50 km/h
        assert perf['fuel_burn_rate_kg_per_hr'] > 0


class TestAircraftRepresentation:
    """Tests for string representation."""
    
    def test_repr(self):
        """Test aircraft __repr__ method."""
        aircraft = Aircraft()
        repr_str = repr(aircraft)
        
        assert 'Aircraft' in repr_str
        assert '28' in repr_str  # mass
        assert '3.1' in repr_str  # wingspan


# Fixtures
@pytest.fixture
def default_aircraft():
    """Fixture providing a default aircraft instance."""
    return Aircraft()


@pytest.fixture
def custom_aircraft():
    """Fixture providing a custom aircraft instance."""
    return Aircraft(
        mass_kg=30.0,
        wingspan_m=3.5,
        true_airspeed_mps=25.0
    )


# Integration test
def test_full_mission_fuel_calculation(default_aircraft):
    """Integration test for a complete mission fuel calculation."""
    # Simulate a 10km mission
    total_distance = 10000.0  # meters
    
    total_fuel = default_aircraft.calculate_fuel_burn(total_distance)
    
    # Should burn some fuel but not an unreasonable amount
    assert total_fuel > 0
    assert total_fuel < 5.0  # Should be less than 5kg for 10km
    
    # Check consistency with range calculation
    estimated_range = default_aircraft.calculate_range(total_fuel)
    assert abs(estimated_range - total_distance) < 100.0  # Within 100m


if __name__ == "__main__":
    pytest.main([__file__, "-v"])