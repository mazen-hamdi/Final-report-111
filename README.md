# Equilibrium Classification and Dynamical Analysis of Coupled Neural Oscillators

**Authors:** Mohamed Mazen Hamdi, Rohan Vasudev  
**Course:** MATH 111 - Mathematical Biology  

## Abstract

This repository presents a comprehensive computational analysis framework for studying the dynamical behavior of a two-cell bioelectric SR-latch system. The analysis pipeline implements multiple numerical methods to characterize equilibrium points, stability properties, bifurcation structure, and stochastic dynamics of the coupled neural oscillator model.

## Mathematical Model

The system is governed by the following coupled differential equations:

```
dV₁/dt = V₁ - V₁³ - k·V₂ + I_S(t) + σ·ξ₁(t)
dV₂/dt = V₂ - V₂³ - k·V₁ + I_R(t) + σ·ξ₂(t)
```

where:
- `V₁, V₂`: membrane potentials of the two neural cells
- `k`: coupling strength parameter
- `I_S(t), I_R(t)`: external set/reset current inputs
- `σ`: noise intensity
- `ξ₁(t), ξ₂(t)`: independent Gaussian white noise processes

## Analysis Components

### 1. Equilibrium Classification
- **Analytical Methods**: Complete characterization of all fixed points including origin, symmetric, and anti-symmetric equilibria
- **Numerical Detection**: Hybrid grid-random search algorithm for robust equilibrium identification
- **Stability Analysis**: Linear stability classification via Jacobian eigenvalue analysis

### 2. Bifurcation Analysis
- **Parameter Continuation**: Systematic tracking of equilibrium branches as functions of coupling strength
- **Critical Point Detection**: Identification of saddle-node, transcritical, and pitchfork bifurcations
- **Stability Transitions**: Analysis of parameter regions supporting mono-, bi-, and multi-stable dynamics

### 3. Phase Portrait Analysis
- **Vector Field Visualization**: Streamline plots showing system flow structure
- **Nullcline Analysis**: Identification of curves where derivatives vanish
- **Basin of Attraction**: Computation of initial condition basins leading to different attractors

### 4. Stochastic Dynamics
- **Monte Carlo Simulations**: Statistical analysis of noise-induced transitions
- **Switching Probability**: Parameter-dependent success rates for set/reset operations
- **Dwell Time Analysis**: First-passage time measurements for memory stability assessment

### 5. Digital Logic Verification
- **Truth Table Testing**: Systematic verification of SR-latch functionality
- **Deterministic Switching**: Characterization of minimal pulse requirements
- **Parameter Robustness**: Analysis of operating ranges for reliable digital behavior

## Prerequisites

### Required Python Packages
```bash
pip install numpy scipy matplotlib pandas tqdm
```

### Optional Dependencies
```bash
pip install PyDSTool  # For advanced continuation analysis
```

## Usage Instructions

### Basic Analysis
```bash
# Complete equilibrium and bifurcation analysis
python3 equilibria-classification.py --phase --bifurcation

# Monte Carlo parameter sweep with visualization
python3 equilibria-classification.py --sweep

# Digital logic functionality verification
python3 equilibria-classification.py --truth
```

### Advanced Analysis
```bash
# Basin of attraction computation
python3 equilibria-classification.py --basins

# Memory stability under noise
python3 equilibria-classification.py --dwell

# Deterministic switching analysis
python3 equilibria-classification.py --deterministic
```

### Comprehensive Pipeline
```bash
# Execute full analysis suite
python3 equilibria-classification.py --all
```

### Animation Generation
```bash
# Single trajectory animation
python3 equilibria-classification.py --animate --k_animate 0.7 --sigma_animate 0.2

# Comprehensive animation set
python3 equilibria-classification.py --animate_all
```

## Output Structure

All results are organized in the `Results/` directory:

- **Phase Portraits**: `phaseplane_k*.png` - Vector fields with equilibria and nullclines
- **Bifurcation Diagrams**: `bifurcation_diagram*.png` - Parameter-dependent equilibrium structure
- **Stability Analysis**: `attractors_vs_k*.png` - Multistability characterization
- **Heatmaps**: `heatmap_P_vs_sigma_amp_k*.png` - Switching probability landscapes
- **Animations**: `animation_*.gif` - Dynamic trajectory visualizations
- **Data Files**: `*.csv` - Numerical results for further analysis



- **Numerical Integration**: Euler-Maruyama scheme for stochastic differential equations
- **Root Finding**: Hybrid optimization combining systematic and stochastic search
- **Stability Classification**: Linear analysis with robust eigenvalue computation
- **Parallel Processing**: Multi-core Monte Carlo simulations for computational efficiency

## Applications

This framework provides insights relevant to:
- **Computational Neuroscience**: Neural circuit dynamics and information processing
- **Bioengineering**: Design principles for bioelectric memory devices
- **Mathematical Biology**: Stochastic effects in biological switches
- **Dynamical Systems**: General theory of coupled nonlinear oscillators


