# Dora the Pathfinder

A Python library for fuel-optimized UAV path planning with obstacle avoidance and dynamic flight corridors.

## Overview

Dora the Pathfinder plans fuel-efficient flight paths for unmanned aerial vehicles (UAVs). Given a set of mission waypoints and environmental constraints (no-fly zones, wind), it:

1. **Builds a graph** of navigable nodes within flight corridors around each waypoint leg
2. **Finds the optimal path** using Dijkstra or A* with a physics-based fuel cost model

The fuel cost model is derived from first principles: lift coefficient, drag polar, power required, and specific fuel consumption -- all parameterized for the Boeing ScanEagle UAV by default.

## Features

- **Physics-based cost functions** -- Fuel burn calculated from aerodynamic drag, propulsive efficiency, and SFC (not just Euclidean distance)
- **Dijkstra and A* planners** -- Compare optimal vs. heuristic-guided search; A* typically explores 2-5x fewer nodes
- **Dynamic flight corridors** -- Graph nodes generated within configurable corridors around waypoint legs, with adaptive expansion near obstacles
- **Circular no-fly zones** -- Obstacles defined by center and radius; validated at graph construction
- **Wind modeling** -- Wind speed and direction affect ground speed and fuel calculations
- **Visualization** -- Matplotlib-based path plots, algorithm comparisons, and performance metrics
- **Path smoothing (beta)** -- Tangent-arc trajectory smoothing for flyable paths (experimental, see below)

## Project Structure

```
dora_the_pathfinder/
├── pathfinder/
│   ├── __init__.py
│   ├── models/
│   │   ├── aircraft.py        # ScanEagle UAV aerodynamic model
│   │   └── environment.py     # Wind, no-fly zones, terrain
│   ├── planning/
│   │   ├── graph.py           # Flight corridor graph construction
│   │   ├── cost.py            # Fuel/distance/time cost functions
│   │   ├── dijkstra.py        # Dijkstra's algorithm
│   │   ├── astar.py           # A* with fuel-burn heuristic
│   │   └── smoother.py        # Trajectory smoothing [beta]
│   └── visualizer/
│       └── visualizer.py      # Path and comparison plotting
├── examples/
│   ├── basic_pathfinder.py         # Dijkstra vs A* comparison
│   ├── medium_range_pathfinding.py # Multi-waypoint delivery mission
│   ├── animated_pathfinding.py     # Real-time A* search animation
│   └── smoother_demo.py            # Path smoothing demo [beta]
├── test/
│   ├── test_aircraft.py
│   └── test_cost.py
└── requirements.txt
```

## Installation

```bash
# Clone the repository
git clone https://github.com/VSLEGION/Dora-the-Pathfinder.git
cd dora_the_pathfinder

# Create a virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from pathfinder import Aircraft, Environment, Wind, plan_path_astar

# Define mission waypoints
waypoints = [(0, 0), (100, 50), (200, 100), (300, 150)]

# Set up environment with wind and obstacles
env = Environment(wind=Wind(speed_mps=5.0, direction_deg=90))
env.add_no_fly_zone(center=(150, 75), radius_m=30, name="Restricted Area")

# Create aircraft (defaults to ScanEagle UAV)
aircraft = Aircraft()

# Plan optimal path
path, fuel_cost, metrics = plan_path_astar(
    waypoints=waypoints,
    aircraft=aircraft,
    corridor_width=5.0,
    resolution=2.0,
    environment=env,
    cost_type='fuel'
)

print(f"Path: {len(path)} nodes, Fuel: {fuel_cost:.6f} kg")
print(f"A* explored {metrics['nodes_visited']} nodes")
```

## Running Examples

```bash
cd examples
python basic_pathfinder.py           # Dijkstra vs A* comparison
python medium_range_pathfinding.py   # Multi-waypoint delivery mission
python animated_pathfinding.py       # Real-time animated A* search
```

## How It Works

### Aircraft Model
The default aircraft is the Boeing ScanEagle (28 kg, 3.1m wingspan, 28 m/s cruise). Fuel burn is calculated from:
- Lift coefficient for level flight: `CL = W / (0.5 * rho * V^2 * S)`
- Drag polar: `CD = CD0 + k * CL^2`
- Power required: `P = D * V / eta_prop`
- Fuel rate: `fuel_rate = SFC * P`

### Graph Construction
For each waypoint-to-waypoint leg, a corridor of navigable nodes is generated. Nodes inside no-fly zones are excluded. Near obstacles, the corridor expands adaptively to ensure detour routes exist ("search widely, commit narrowly").

### Algorithm Comparison
Both Dijkstra and A* find the fuel-optimal path. A* uses the fuel-burn heuristic to guide search toward the goal, exploring significantly fewer nodes while guaranteeing the same optimal cost.

## Path Smoothing (Beta)

The `smoother.py` module is under active development. It converts discrete A* paths into smooth, flyable trajectories using tangent-arc geometry inspired by CATIA sketch constraints:

1. Simplify the discrete A* path
2. Identify segments blocked by no-fly zones
3. Compute tangent entry/exit points on obstacle clearance circles
4. Generate circular arcs between tangent points
5. Fit a gentle spline over the resulting skeleton

To try it:
```bash
cd examples
python smoother_demo.py
```

## Requirements

- Python 3.8+
- NumPy >= 1.21
- Matplotlib >= 3.5
- SciPy >= 1.7

## License

MIT License -- see [LICENSE](LICENSE) for details.

## Author

Vaishanth Srinivasan
