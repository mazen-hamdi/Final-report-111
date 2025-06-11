# Final-report-111

Author: Mohamed Mazen Hamdi / Rohan Vasudev

This repository contains a Python pipeline for analyzing a two-cell bioelectric SR-latch model. The workflow includes phase-plane diagrams, Monte-Carlo sweeps and a truth table verification. Optionally, it can perform equilibrium continuation with **PyDSTool** and its `PyCont` module.

## Requirements

- Python 3 with `numpy`, `scipy`, `matplotlib`, `pandas`, `tqdm`.
- For the continuation feature (`--cont` flag), `PyDSTool` must be installed. If it is absent, the script will skip that step and issue a warning.

## Installation

Install the required packages with pip:

```bash
pip install numpy pandas matplotlib scipy tqdm
```

Install PyDSTool if you want to run the continuation features:

```bash
pip install PyDSTool
```
## Usage

Run the script with the desired command-line options:

```bash
python3 equilibria-classification.py --phase    # phase planes and stability plots
python3 equilibria-classification.py --sweep    # Monte-Carlo parameter sweep
python3 equilibria-classification.py --truth    # deterministic truth table check
python3 equilibria-classification.py --cont     # PyCont equilibrium continuation
python3 equilibria-classification.py --bifurcation # numerical bifurcation diagram
```

Use `--all` to run the default analysis set (`phase`, `sweep`, `truth`). All outputs are saved in the `results/` directory.
