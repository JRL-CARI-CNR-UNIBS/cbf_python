# CBF Python: Optimal Control Barrier Function (CBF) Controller

This repository contains the Python implementation of the **Optimal Control Barrier Function (CBF)** controller for robotic manipulators with human-robot collaboration safety guarantees.

## Repository Overview

- `cbf_python/Controller/`:
  - `optimal_cbf_task_controller.py`: Main Quadratic Programming (QP) based Optimal CBF Task Controller (`BCFOptimalController`).
  - `compute_velocity_scaling_for_human_proximity.py`: Velocity scaling and deceleration limits based on human proximity.
  - `Numba_scripts/`:
    - `ssm_cbf_acc.py`: Analytical formulations and Numba-accelerated kernels for Speed and Separation Monitoring (SSM) and CBFs.
    - `numba_kernels.py`: Fast QP matrix/vector assembly and constraint evaluation.
- `cbf_python/Command_bridge/`:
  - ROS 2 and simulated hardware bridges for joint commands and perception/skeleton tracking inputs.
- `cbf_python/scripts/`:
  - `example_cbf_optimal.py`: Simulation and execution script demonstrating the optimal CBF controller on a UR10e manipulator.
  - `test/cbf_test.py`: Unit test suite verifying mathematical derivations, gradients, and QP integration.

## Requirements & Setup

### Prerequisites
- ROS 2 (Jazzy / Iron / Humble)
- Pinocchio & Eigenpy
- Python 3.10+
- Dependencies listed in `requirements.txt`

### Running with Virtual Environment

Activate the environment:
```bash
source /home/galileo/projects/python_venv/galileo_venv/bin/activate
```

Add the package path to `PYTHONPATH`:
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/cbf_python_ws/cbf_python/cbf_python
```

### Running Tests

To run the verification test suite:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest cbf_python/scripts/test/cbf_test.py
```

### Running the Controller Example

To run the standalone simulation with visual updates / daemon:
```bash
python3 -m scripts.example_cbf_optimal
```