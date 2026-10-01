# Multi-Agent DDPG for Oil-Spill Area Estimation

A simulation environment for estimating the spatial extent of an oil spill using a swarm of simulated UAVs trained with Deep Deterministic Policy Gradient (DDPG).

The project models a multi-agent reinforcement-learning problem in which five UAV agents interact with a simulated spill environment and learn coordinated behaviour for coverage and area estimation.

## Project Scope

- Multi-agent reinforcement learning with DDPG
- Five simulated UAV agents
- Gym-like custom simulation environment
- Oil-spill spatial coverage and area-estimation task
- PyTorch-based actor/critic implementation
- Numerical and environmental processing with `netCDF4`, `numba`, and `pygame`

## Repository Structure

| File | Purpose |
|---|---|
| `ddpg-env-train_v2.py` | Main simulation and training entry point |
| `train.py` | Training logic |
| `model.py` | Neural-network definitions |
| `buffer.py` | Replay-buffer implementation |
| `spill.py` | Oil-spill environment/model logic |
| `utils.py` | Supporting utilities |

## Requirements

The code is written in Python and uses PyTorch.

Core dependencies include:

```text
torch
netCDF4
numba
pygame
numpy
```

Install the required packages in an isolated Python environment before running the simulation.

## Running the Simulation

The primary entry point is:

```bash
python ddpg-env-train_v2.py
```

## Method

DDPG is an actor-critic reinforcement-learning algorithm designed for continuous action spaces. In this project, it is used to train multiple UAV agents to coordinate around the simulated spill boundary and support estimation of the affected area.

The environment is designed around an oil-spill region on the order of several thousand square metres. Agent observations, control actions and rewards are handled within the simulation code.

## Research Context

This repository is intended as research code for experimentation with:

- cooperative autonomous agents;
- continuous-control reinforcement learning;
- UAV-based environmental monitoring;
- spatial estimation under distributed sensing.

The code is provided primarily as an experimental implementation rather than a packaged production library.

## Reproducibility

For repeatable experiments, record the following when running a training session:

- Python version;
- PyTorch version;
- random seed;
- number of episodes;
- learning rates;
- replay-buffer size;
- exploration parameters;
- environment configuration.

These values should be reported alongside training curves and final estimation results when using the repository for comparative experiments.
