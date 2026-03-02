# MorphingGlider

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

A 6-DOF quaternion-based flight simulation of a morphing-wing glider for reinforcement learning research. The project provides a full pipeline from physics simulation through RL training to interpretability analysis, with rigorous statistical evaluation.

## Features

- **6-DOF Flight Dynamics** -- Quaternion-based simulation with a 41D observation space and 6D action space (right/left wing tip dx/dy/dz offsets)
- **Morphing Wing Geometry** -- Cubic Bezier spar model with curvature constraints and quasi-steady strip-theory aerodynamics
- **Curriculum Learning** -- SAC agent trained across 5 progressive phases with stability-to-tracking gating
- **Domain Randomization** -- 16+ physics parameters randomized for sim-to-real transfer
- **Baseline Controllers** -- PID, LQR, MPC, Gain-Scheduled PID, and Virtual Tendon Heuristic
- **Interpretability Pipeline** -- Machine Teaching, Latent Space MRI (PCA/t-SNE), KAN distillation, Symbolic Distillation, DAgger
- **Statistical Rigor** -- Hierarchical bootstrap confidence intervals across seeds with paired tests

## Project Structure

```
morphing_glider/
├── config.py            # Constants, run modes, hyperparameters
├── calibration.py       # Aero calibration
├── utils/               # Quaternion, numeric, visualization helpers
├── physics/             # Bezier spar, aero proxy, domain randomizer
├── environment/         # Gymnasium env, observation, reward, wrappers
├── controllers/         # PID, LQR, MPC, heuristic, SB3 wrapper
├── training/            # SAC training infrastructure, curriculum
├── evaluation/          # Metrics, evaluation, robustness tests
├── interpretability/    # Strategy analyzer, sensitivity, KAN, DAgger
├── reporting/           # Statistics, reproducibility
tests/                   # Unit tests
run_simulation.py        # Main entry point
```

## Getting Started

### Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running the Simulation

```bash
# Set RUN_MODE in config.py, then run:
python run_simulation.py
```

### Run Modes

| Mode     | Seeds | Approx. Time       | Use Case             |
|----------|-------|---------------------|----------------------|
| `dev`    | 1     | ~5 min              | Smoke test           |
| `medium` | 3     | ~2-3 hrs            | Default              |
| `paper`  | 5     | ~12+ hrs            | Publication quality  |

### Running Tests

```bash
pytest tests/
```

## Output Artifacts

| Directory / File                 | Contents                          |
|----------------------------------|-----------------------------------|
| `morphing_glider_models/`        | Trained SB3 models (.zip), VecNormalize stats, metadata |
| `morphing_glider_figures/`       | Publication plots (asymmetry, DAgger, feature importance, latent space) |
| `morphing_glider_results.json`   | Full config, calibration, validation, and paired test results |
| `tb_baseline/`, `tb_curriculum/` | TensorBoard logs                  |

## Hardware

The simulation auto-detects Apple Silicon MPS via PyTorch and falls back to CPU. Parallel training uses `SubprocVecEnv` with the `fork` start method.

## Requirements

- Python 3.9+
- numpy
- scipy
- torch
- gymnasium
- stable-baselines3
- matplotlib

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
