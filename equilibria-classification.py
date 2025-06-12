#!/usr/bin/env python3
"""
sr_latch_pipeline.py  ── REV‑D (2025‑05‑16)

End‑to‑end numerical study of the two‑cell bioelectric SR‑latch:
  • deterministic phase‑plane
  • equilibrium continuation with PyDSTool + PyCont
  • stochastic Euler–Maruyama simulations
  • Monte‑Carlo switching statistics + parameter sweeps
  • truth‑table verification

Author: Mohamed Mazen Hamdi / Rohan Vasudev
"""
from __future__ import annotations
import math, sys, warnings, itertools, os, multiprocessing as mp
from pathlib import Path
import argparse, logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy import optimize, linalg
from tqdm import tqdm

# ────────────────────────────────────────────────────────────── optional PyDSTool
try:
    from PyDSTool import args as pyargs, Generator
    from PyDSTool.PyCont import ContClass
    HAVE_PYDSTOOL = True
except ImportError:
    HAVE_PYDSTOOL = False
    warnings.warn("PyDSTool not found – continuation will be skipped.")

# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION AND GLOBAL PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

# Output directory for all generated plots and data
OUT_DIR = Path("Results")
OUT_DIR.mkdir(exist_ok=True)

# Parameter ranges for comprehensive analysis
K_RANGE = np.linspace(0.05, 2.0, 20)              # Coupling strength range
SIGMA_RANGE = np.logspace(-4, np.log10(2.0), 25)  # Noise intensity range (0.0001 to 2.0)
AMP_RANGE = np.linspace(-2.5, 2.5, 21)            # Pulse amplitude range (includes negative)

# Simulation parameters
DT, T_TOTAL = 1e-3, 25.0                          # Time step and total simulation time
TRIALS_PER_TASK = 30                              # Monte Carlo trials per parameter set
GRID_N = 30                                       # Phase plane grid resolution

# Computational parameters
CPU_COUNT = max(1, mp.cpu_count() - 2 if mp.cpu_count() > 1 else 1)  # Leave cores free
SEED = 42                                          # Random number generator seed
rng = np.random.default_rng(SEED)

# Plotting and logging configuration
plt.rcParams.update({'figure.dpi': 120})
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy.optimize")
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# CORE SYSTEM DYNAMICS
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

def rhs(v: np.ndarray, k: float, IS: float, IR: float) -> np.ndarray:
    """
    Right-hand side of the two-cell SR-latch differential equation.
    
    Computes the time derivatives for the coupled system:
    dV₁/dt = V₁ - V₁³ - k*V₂ + I_S
    dV₂/dt = V₂ - V₂³ - k*V₁ + I_R
    
    Parameters:
    -----------
    v : np.ndarray
        State vector [V₁, V₂]
    k : float
        Coupling strength between cells
    IS : float
        Set input current to cell 1
    IR : float  
        Reset input current to cell 2
        
    Returns:
    --------
    np.ndarray
        Time derivative vector [dV₁/dt, dV₂/dt]
    """
    v1, v2 = v
    return np.array([v1 - v1**3 - k*v2 + IS,
                     v2 - v2**3 - k*v1 + IR])

def jac(v_eq: np.ndarray, k: float) -> np.ndarray:
    """
    Jacobian matrix of the system at equilibrium point.
    
    Computes the linearization matrix for stability analysis:
    J = [[∂f₁/∂V₁, ∂f₁/∂V₂],
         [∂f₂/∂V₁, ∂f₂/∂V₂]]
    
    Parameters:
    -----------
    v_eq : np.ndarray
        Equilibrium point [V₁*, V₂*]
    k : float
        Coupling strength parameter
        
    Returns:
    --------
    np.ndarray
        2x2 Jacobian matrix
    """
    v1, v2 = v_eq
    return np.array([[1 - 3*v1**2, -k],
                     [-k,           1 - 3*v2**2]])

# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# EQUILIBRIUM ANALYSIS AND CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

def get_all_equilibria_and_classify(k: float, tol: float = 1e-9) -> list[dict]:
    """
    Find and classify all equilibria of the autonomous two-cell system.
    
    This function analytically determines the three types of equilibria:
    1. Origin (0,0) - always exists
    2. Symmetric equilibria (±v,±v) - exist when k ≤ 1
    3. Anti-symmetric equilibria (±v,∓v) - exist when k ≥ 0
    
    Each equilibrium is classified by analyzing eigenvalues of the Jacobian:
    - Attractor: all eigenvalues have negative real parts
    - Repeller: all eigenvalues have positive real parts  
    - Saddle: eigenvalues have opposite signs
    - Non-hyperbolic: one or more eigenvalues have zero real part
    
    Parameters:
    -----------
    k : float
        Coupling strength parameter
    tol : float, optional
        Numerical tolerance for eigenvalue classification
        
    Returns:
    --------
    list[dict]
        List of equilibrium dictionaries with keys:
        - 'v': equilibrium coordinates [V₁*, V₂*]
        - 'type': stability classification string
        - 'source': analytical origin (Origin/Symmetric/Anti-symmetric)
    """
    equilibria_list = []

    # 1. Origin equilibrium (0,0) - always exists
    v_origin = np.array([0.0, 0.0])
    J_origin = jac(v_origin, k)
    eig_origin = np.linalg.eigvals(J_origin)
    
    # Classify origin stability
    origin_type = "Unknown"
    if all(np.real(eig_origin) < -tol): 
        origin_type = "Attractor"
    elif all(np.real(eig_origin) > tol): 
        origin_type = "Repeller"
    elif np.real(eig_origin[0]) * np.real(eig_origin[1]) < -tol**2: 
        origin_type = "Saddle"
    else: 
        # Handle non-hyperbolic cases
        if np.any(abs(np.real(eig_origin)) < tol): 
            origin_type = "Non-Hyperbolic"
        # More specific non-hyperbolic classification
        if (np.real(eig_origin[0]) > tol and abs(np.real(eig_origin[1])) < tol) or \
           (abs(np.real(eig_origin[0])) < tol and np.real(eig_origin[1]) > tol):
            origin_type = "Unstable (non-hyperbolic)"
        elif (np.real(eig_origin[0]) < -tol and abs(np.real(eig_origin[1])) < tol) or \
             (abs(np.real(eig_origin[0])) < tol and np.real(eig_origin[1]) < -tol):
            origin_type = "Stable (non-hyperbolic)"

    equilibria_list.append({'v': v_origin, 'type': origin_type, 'source': 'Origin'})

    # 2. Symmetric equilibria: V₁ = V₂ = ±√(1-k)
    # These exist when 1-k ≥ 0, i.e., k ≤ 1
    if k <= 1.0 + tol:
        val_1_minus_k = 1.0 - k
        if val_1_minus_k >= -tol:  # Allow small numerical errors
            vs_val = np.sqrt(max(0, val_1_minus_k))
            if abs(vs_val) > tol:  # Distinct from origin
                for sign in [-1, 1]:
                    v_sym = np.array([sign * vs_val, sign * vs_val])
                    J_sym = jac(v_sym, k)
                    eig_sym = np.linalg.eigvals(J_sym)
                    
                    # Classify symmetric equilibrium stability
                    sym_type = "Unknown"
                    if all(np.real(eig_sym) < -tol): 
                        sym_type = "Attractor"
                    elif all(np.real(eig_sym) > tol): 
                        sym_type = "Repeller"
                    elif np.real(eig_sym[0]) * np.real(eig_sym[1]) < -tol**2: 
                        sym_type = "Saddle"
                    
                    equilibria_list.append({'v': v_sym, 'type': sym_type, 'source': 'Symmetric'})
    
    # 3. Anti-symmetric equilibria: V₁ = -V₂ = ±√(1+k)
    # These exist when 1+k ≥ 0, i.e., always for k ≥ 0
    val_1_plus_k = 1.0 + k
    if val_1_plus_k >= -tol:  # Should always be true for k ≥ 0
        va_val = np.sqrt(max(0, val_1_plus_k))
        if abs(va_val) > tol:  # Distinct from origin
            for sign in [-1, 1]:
                v_anti = np.array([sign * va_val, -sign * va_val])
                J_anti = jac(v_anti, k)
                eig_anti = np.linalg.eigvals(J_anti)
                
                # Classify anti-symmetric equilibrium stability
                anti_type = "Unknown"
                if all(np.real(eig_anti) < -tol): 
                    anti_type = "Attractor"
                elif all(np.real(eig_anti) > tol): 
                    anti_type = "Repeller"
                elif np.real(eig_anti[0]) * np.real(eig_anti[1]) < -tol**2: 
                    anti_type = "Saddle"
                
                equilibria_list.append({'v': v_anti, 'type': anti_type, 'source': 'Anti-symmetric'})
                
    # Remove duplicates (can occur when k=1 and symmetric coincides with origin)
    unique_equilibria = []
    seen_coords = set()
    for eq in equilibria_list:
        coord_tuple = tuple(np.round(eq['v'], 5))  # Round for numerical stability
        if coord_tuple not in seen_coords:
            unique_equilibria.append(eq)
            seen_coords.add(coord_tuple)
            
    return unique_equilibria

# ═══════════════════════════════════════════════════════════════════════════════
# NUMERICAL EQUILIBRIUM DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def find_equilibria_numerically(k: float, n_guesses: int = 500,
                                search_bounds=(-2.5, 2.5), tol: float = 1e-5) -> list[np.ndarray]:
    """
    Find equilibrium points numerically using root finding with hybrid search strategy.
    
    This function combines systematic grid-based searching with random sampling to
    robustly locate all equilibria in the phase space. Uses scipy.optimize.fsolve
    to solve the system: rhs(v, k, 0, 0) = 0.
    
    Parameters:
    -----------
    k : float
        Coupling strength parameter
    n_guesses : int, default=500
        Total number of initial guesses to try (split between grid and random)
    search_bounds : tuple, default=(-2.5, 2.5)
        (min, max) bounds for the search region in both v1 and v2 directions
    tol : float, default=1e-5
        Tolerance for considering two equilibria as duplicates
        
    Returns:
    --------
    list[np.ndarray]
        List of unique equilibrium points as 2D numpy arrays [v1, v2]
        
    Notes:
    ------
    Uses a hybrid approach:
    - First half of guesses: systematic grid covering the search space
    - Second half: random initial conditions for additional coverage
    This ensures both systematic exploration and stochastic robustness.
    """
    func = lambda v: rhs(v, k, 0.0, 0.0)
    rng_local = np.random.default_rng()
    roots = []
    
    # Add grid-based guesses for systematic coverage
    grid_size = 15
    v1_vals = np.linspace(search_bounds[0], search_bounds[1], grid_size)
    v2_vals = v1_vals
    grid_guesses = 0
    for v1 in v1_vals:
        for v2 in v2_vals:
            if grid_guesses >= n_guesses // 2:  # Use half for grid, half for random
                break
            guess = np.array([v1, v2])
            try:
                root, info, ier, _ = optimize.fsolve(func, guess, full_output=True)
                if ier == 1:
                    roots.append(root)
                grid_guesses += 1
            except Exception:
                continue
    
    # Supplement with random guesses
    remaining_guesses = n_guesses - grid_guesses
    for _ in range(remaining_guesses):
        guess = rng_local.uniform(search_bounds[0], search_bounds[1], size=2)
        try:
            root, info, ier, _ = optimize.fsolve(func, guess, full_output=True)
            if ier == 1:
                roots.append(root)
        except Exception:
            continue

    unique = []
    for r in roots:
        if not any(np.linalg.norm(r - u) < tol for u in unique):
            unique.append(r)
    return unique


def plot_bifurcation():
    """
    Generate comprehensive bifurcation diagram showing equilibrium behavior vs coupling strength.
    
    This function creates a detailed bifurcation analysis by:
    1. Scanning over a wide range of coupling values k ∈ [0.05, 2.5]
    2. Finding all equilibria numerically for each k value
    3. Classifying stability based on Jacobian eigenvalues
    4. Plotting separate diagrams for V₁ and V₂ equilibrium values
    
    The bifurcation diagram reveals:
    - Pitchfork bifurcations where symmetric equilibria emerge
    - Transcritical bifurcations affecting stability
    - Parameter regions with different numbers of attractors
    
    Outputs:
    --------
    Saves two PNG files:
    - 'bifurcation_diagram_comprehensive.png': Two-panel plot (V₁ and V₂)
    - 'bifurcation_diagram.png': Single panel focusing on V₁ dynamics
    """
    logging.info("Generating bifurcation diagram...")
    # Much wider k range with higher resolution for detailed bifurcation analysis
    k_values = np.linspace(0.05, 2.5, 200)  # Expanded range and higher resolution
    
    # Store results: k, v1_value, stability
    results = []
    
    for k in tqdm(k_values, desc="Bifurcation Analysis"):
        # Use our robust numerical finder with more guesses for better coverage
        equilibria = find_equilibria_numerically(k, n_guesses=500)  # Increased from 75
        
        for eq in equilibria:
            # Classify stability by checking eigenvalues of the Jacobian
            eigenvalues = np.linalg.eigvals(jac(eq, k))
            real_eigs = np.real(eigenvalues)
            tol = 1e-9
            
            # Robust stability classification
            if all(real_eigs < -tol):
                stability = 'stable'
            elif all(real_eigs > tol):
                stability = 'repeller'
            elif real_eigs[0] * real_eigs[1] < -tol**2:
                stability = 'saddle'
            else:
                stability = 'non-hyperbolic'
                
            results.append({'k': k, 'v1': eq[0], 'v2': eq[1], 'stability': stability})

    df = pd.DataFrame(results)
    stable_points = df[df['stability'] == 'stable']
    unstable_points = df[df['stability'] != 'stable']
    saddle_points = df[df['stability'] == 'saddle']
    repeller_points = df[df['stability'] == 'repeller']

    # Create multiple bifurcation plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot 1: V1 bifurcation diagram
    ax1.plot(stable_points['k'], stable_points['v1'], 'g.', markersize=3, label='Stable Equilibria', alpha=0.8)
    ax1.plot(saddle_points['k'], saddle_points['v1'], 'r.', markersize=2, alpha=0.6, label='Saddle Points')
    ax1.plot(repeller_points['k'], repeller_points['v1'], 'b.', markersize=2, alpha=0.5, label='Repellers')
    ax1.set_xlabel('Coupling Strength (k)')
    ax1.set_ylabel('Equilibrium value of $V_1$')
    ax1.set_title('Bifurcation Diagram: $V_1$ vs k')
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.legend()
    
    # Plot 2: V2 bifurcation diagram  
    ax2.plot(stable_points['k'], stable_points['v2'], 'g.', markersize=3, label='Stable Equilibria', alpha=0.8)
    ax2.plot(saddle_points['k'], saddle_points['v2'], 'r.', markersize=2, alpha=0.6, label='Saddle Points')
    ax2.plot(repeller_points['k'], repeller_points['v2'], 'b.', markersize=2, alpha=0.5, label='Repellers')
    ax2.set_xlabel('Coupling Strength (k)')
    ax2.set_ylabel('Equilibrium value of $V_2$')
    ax2.set_title('Bifurcation Diagram: $V_2$ vs k')
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'bifurcation_diagram_comprehensive.png', dpi=150)
    plt.close()
    
    # Also save the original single plot for V1
    plt.figure(figsize=(8, 6))
    plt.plot(stable_points['k'], stable_points['v1'], 'g.', markersize=4, label='Stable Equilibria (Attractors)')
    plt.plot(saddle_points['k'], saddle_points['v1'], 'r.', markersize=3, alpha=0.7, label='Saddle Points')
    plt.plot(repeller_points['k'], repeller_points['v1'], 'b.', markersize=2, alpha=0.5, label='Repellers')
    plt.xlabel('Coupling Strength (k)')
    plt.ylabel('Equilibrium value of $V_1$')
    plt.title('Bifurcation Diagram')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.savefig(OUT_DIR / 'bifurcation_diagram.png')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# STOCHASTIC DIFFERENTIAL EQUATION INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

def euler_maruyama(k: float,
                   IS_fun, IR_fun,
                   sigma: float,
                   v0: np.ndarray | None = None,
                   T: float = T_TOTAL,
                   dt: float = None,
                   rng: np.random.Generator | None = None,
                   return_path: bool = False) -> np.ndarray:
    """
    Integrate the stochastic neural latch equations using the Euler-Maruyama method.
    
    Solves the system of SDEs:
    dV₁/dt = V₁ - V₁³ - k·V₂ + I_S(t) + σ·ξ₁(t)
    dV₂/dt = V₂ - V₂³ - k·V₁ + I_R(t) + σ·ξ₂(t)
    
    where ξ₁(t), ξ₂(t) are independent white noise processes.
    
    Parameters:
    -----------
    k : float
        Coupling strength between the two neural units
    IS_fun, IR_fun : callable
        External current functions I_S(t) and I_R(t) for Set and Reset inputs
    sigma : float
        Noise intensity (square root of diffusion coefficient)
    v0 : np.ndarray, optional
        Initial condition [V₁(0), V₂(0)]. Defaults to [0, 0]
    T : float, default=T_TOTAL
        Total integration time
    dt : float, optional
        Time step. If None, uses adaptive step size based on noise intensity
    rng : np.random.Generator, optional
        Random number generator for reproducible results
    return_path : bool, default=False
        If True, returns full trajectory; if False, returns only final state
        
    Returns:
    --------
    np.ndarray
        If return_path=False: Final state [V₁(T), V₂(T)]
        If return_path=True: Full trajectory of shape (N+1, 2)
        
    Notes:
    ------
    Uses adaptive time stepping: smaller dt for higher noise levels to maintain
    numerical stability. The Euler-Maruyama scheme is first-order accurate for
    the deterministic part and has strong order 0.5 for the stochastic part.
    """
    if rng is None:
            rng = globals().get("rng", np.random.default_rng())
    
    # Adaptive time step for numerical stability with high noise
    if dt is None:
        dt = min(DT, 0.005 / (1 + sigma))  # Smaller steps for larger sigma
    
    v   = np.array(v0) if v0 is not None else np.zeros(2)
    N   = int(T/dt)
    sdt = math.sqrt(dt)
    t   = 0.0
    if return_path:
        path = np.zeros((N+1, 2))
        path[0] = v
        
    for i in range(N):
        v += rhs(v, k, IS_fun(t), IR_fun(t)) * dt \
             + sigma * sdt * rng.normal(size=2)
        t += dt
        if return_path:
            path[i+1] = v
           
    return path if return_path else v

# ═══════════════════════════════════════════════════════════════════════════════
# PULSE GENERATION AND STATE PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════

def rectangular_pulse(t0: float, t1: float, amp: float):
    """
    Generate a rectangular pulse function for external current injection.
    
    Parameters:
    -----------
    t0, t1 : float
        Start and end times of the pulse
    amp : float
        Pulse amplitude (can be positive or negative)
        
    Returns:
    --------
    callable
        Function I(t) that returns amp for t ∈ [t0, t1], zero otherwise
        
    Notes:
    ------
    Used to create Set pulses (positive I_S) and Reset pulses (negative I_S or positive I_R).
    """
    return lambda t: amp if (t0 <= t <= t1) else 0.0

def prepare_reset_state(k: float, amp: float, sigma: float = 0.0) -> np.ndarray:
    """
    Prepare the neural latch in a reliable Q=0 state (V₁ low, V₂ high).
    
    This function drives the system towards the Q=0 state by applying coordinated
    pulses and starting from a favorable initial condition near the anti-symmetric
    attractor configuration.
    
    Parameters:
    -----------
    k : float
        Coupling strength parameter
    amp : float
        Amplitude of the preparation pulses
    sigma : float, default=0.0
        Noise level during preparation (typically zero for reliable reset)
        
    Returns:
    --------
    np.ndarray
        Final state [V₁, V₂] representing Q=0 configuration
        
    Notes:
    ------
    Strategy:
    1. Start near theoretical Q=0 anti-symmetric equilibrium if it exists
    2. Apply coordinated pulses: positive I_R (boost V₂) and negative I_S (suppress V₁)
    3. Allow sufficient settling time (7 seconds) for convergence
    
    The Q=0 state corresponds to V₁ < 0, V₂ > 0, representing the "reset" configuration
    of the bistable neural latch.
    """
    # Use known equilibrium as starting point for better reliability
    try:
        # Start near anti-symmetric attractor if it exists
        if (1 + k) > 0.01:
            va_val = np.sqrt(1 + k)
            initial_state = np.array([-va_val * 0.9, va_val * 0.9])  # Close to target Q=0 state
        else:
            initial_state = np.array([-0.5, 0.5])  # Fallback for small k
    except:
        initial_state = np.array([-0.5, 0.5])  # Safe fallback
    
    # Use stronger, more directed pulses
    IR_pulse = rectangular_pulse(0.0, 5.0, amp)    # Push v2 high
    IS_pulse = rectangular_pulse(0.0, 5.0, -amp)   # Push v1 low
    
    return euler_maruyama(k, IS_pulse, IR_pulse, sigma, v0=initial_state, T=7.0)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE PLANE VISUALIZATION AND ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_phase_ideal(k: float):
    """
    Generate phase plane plot using analytical equilibria and nullclines.
    
    Creates a comprehensive phase portrait showing:
    1. Vector field streamlines indicating flow direction
    2. Nullclines where dV₁/dt = 0 and dV₂/dt = 0
    3. All equilibrium points with stability classification
    
    Parameters:
    -----------
    k : float
        Coupling strength parameter
        
    Mathematical Details:
    --------------------
    Nullclines are curves where one derivative vanishes:
    - V₁-nullcline: V₁ - V₁³ - k·V₂ = 0  ⟹  V₂ = (V₁ - V₁³)/k
    - V₂-nullcline: V₂ - V₂³ - k·V₁ = 0  ⟹  V₁ = (V₂ - V₂³)/k
    
    Equilibria occur at nullcline intersections and are classified by:
    - Attractors (stable): All eigenvalues have negative real parts
    - Saddles (unstable): Mixed signs in eigenvalue real parts  
    - Repellers (unstable): All eigenvalues have positive real parts
    
    Output:
    -------
    Saves 'phaseplane_k{k:.3f}.png' showing the complete phase portrait
    """
    U_coords = np.linspace(-2.5, 2.5, GRID_N)
    V_coords = np.linspace(-2.5, 2.5, GRID_N)
    Ug, Vg = np.meshgrid(U_coords, V_coords)
    
    # Calculate vector field with IS=0, IR=0
    dU, dV = rhs(np.array([Ug, Vg]), k, 0.0, 0.0)

    plt.figure(figsize=(8,7))
    plt.streamplot(Ug, Vg, dU, dV, color='0.7', density=1.2, arrowsize=0.9)

    # Nullclines: v1 - v1^3 - k*v2 = 0  => v2 = (v1 - v1^3)/k
    #             v2 - v2^3 - k*v1 = 0  => v1 = (v2 - v2^3)/k
    x_null = np.linspace(-2.5, 2.5, 400)
    if abs(k) > 1e-6:
        y_v1_nullcline = (x_null - x_null**3) / k
        x_v2_nullcline = (x_null - x_null**3) / k # Here x_null represents v2 for the second nullcline
        plt.plot(x_null, y_v1_nullcline, 'r--', lw=1.5, label='$dV_1/dt=0$')
        plt.plot(x_v2_nullcline, x_null, 'b--', lw=1.5, label='$dV_2/dt=0$')
    else: # k=0, decoupled
        plt.axvline(0, color='r', linestyle='--', lw=1.5, label='$dV_1/dt=0$ (part)')
        plt.axvline(1, color='r', linestyle='--', lw=1.5)
        plt.axvline(-1, color='r', linestyle='--', lw=1.5)
        plt.axhline(0, color='b', linestyle='--', lw=1.5, label='$dV_2/dt=0$ (part)')
        plt.axhline(1, color='b', linestyle='--', lw=1.5)
        plt.axhline(-1, color='b', linestyle='--', lw=1.5)


    equilibria_data = get_all_equilibria_and_classify(k)
    
    eq_handles = []
    eq_labels = []

    type_map = {
        "Attractor": {'marker': 'o', 'color': 'green', 'size': 80, 'label': 'Attractor'},
        "Saddle": {'marker': 'x', 'color': 'red', 'size': 80, 'label': 'Saddle'},
        "Repeller": {'marker': '^', 'color': 'blue', 'size': 80, 'label': 'Repeller'},
        "Unknown": {'marker': 's', 'color': 'gray', 'size': 60, 'label': 'Unknown'},
        "Non-Hyperbolic": {'marker': 'D', 'color': 'purple', 'size': 70, 'label': 'Non-Hyperbolic'}
    }
    # Add more specific non-hyperbolic types if needed
    type_map["Unstable (non-hyperbolic)"] = {'marker': 'D', 'color': 'orange', 'size': 70, 'label': 'Unstable (Non-Hyp.)'}
    type_map["Stable (non-hyperbolic)"] = {'marker': 'D', 'color': 'cyan', 'size': 70, 'label': 'Stable (Non-Hyp.)'}


    plotted_labels = set()
    for e in equilibria_data:
        style = type_map.get(e['type'], type_map["Unknown"])
        label = style['label'] if style['label'] not in plotted_labels else None
        if label: plotted_labels.add(style['label'])
        
        plt.scatter(e['v'][0], e['v'][1], 
                    marker=style['marker'], 
                    color=style['color'], 
                    s=style['size'], 
                    label=label,
                    edgecolors='black', zorder=5)

    plt.xlabel('$V_1$'); plt.ylabel('$V_2$')
    plt.title(f'Phase plane (k={k:.3f})')
    plt.legend(loc='upper right', frameon=True, fontsize='small')
    plt.xlim(-2.5, 2.5); plt.ylim(-2.5, 2.5); plt.grid(True, lw=0.3, alpha=0.4)
    plt.axhline(0, color='black', lw=0.5, alpha=0.5)
    plt.axvline(0, color='black', lw=0.5, alpha=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(OUT_DIR / f'phaseplane_k{k:.3f}.png')
    plt.close()


def plot_phase(k: float):
    """
    Plots the phase plane, finds equilibria numerically, classifies them,
    and plots them on the diagram.
    """
    U_coords = np.linspace(-2.5, 2.5, GRID_N)
    V_coords = np.linspace(-2.5, 2.5, GRID_N)
    Ug, Vg = np.meshgrid(U_coords, V_coords)
    
    dU, dV = rhs(np.array([Ug, Vg]), k, 0.0, 0.0)

    fig, ax = plt.subplots(figsize=(8,7))
    ax.streamplot(Ug, Vg, dU, dV, color='0.7', density=1.2, arrowsize=0.9)

    # --- Nullclines (no change here) ---
    x_null = np.linspace(-2.5, 2.5, 400)
    if abs(k) > 1e-6:
        ax.plot(x_null, (x_null - x_null**3) / k, 'r--', lw=1.5, label='$dV_1/dt=0$')
        ax.plot((x_null - x_null**3) / k, x_null, 'b--', lw=1.5, label='$dV_2/dt=0$')
    else:
        # Handle k=0 case
        ... 

    # --- Use improved numerical equilibrium finder ---
    logging.info(f"Numerically searching for equilibria for k={k:.3f}...")
    equilibria_coords = find_equilibria_numerically(k, n_guesses=500)  # Use improved version
    
    type_map = {
        "Attractor": {'marker': 'o', 'color': 'green', 'size': 80, 'label': 'Attractor'},
        "Saddle": {'marker': 'x', 'color': 'red', 'size': 80, 'label': 'Saddle'},
        "Repeller": {'marker': '^', 'color': 'blue', 'size': 80, 'label': 'Repeller'},
        "Non-Hyperbolic": {'marker': 'D', 'color': 'purple', 'size': 70, 'label': 'Non-Hyperbolic'},
        "Unknown": {'marker': 's', 'color': 'gray', 'size': 60, 'label': 'Unknown'}  
    }
    
    plotted_labels = set()
    for v_eq in equilibria_coords:
        # For each found equilibrium, classify it now
        J = jac(v_eq, k)
        eig_vals = np.linalg.eigvals(J)
        real_parts = np.real(eig_vals)
        
        # Classification logic based on eigenvalues with improved tolerance
        eq_type = "Unknown"
        tol = 1e-8  # Improved numerical tolerance
        
        if np.any(np.abs(real_parts) < tol):
            eq_type = "Non-Hyperbolic"
        elif np.all(real_parts < -tol):
            eq_type = "Attractor"
        elif np.all(real_parts > tol):
            eq_type = "Repeller"
        elif real_parts[0] * real_parts[1] < -tol**2:
            eq_type = "Saddle"

        # Plotting logic
        style = type_map.get(eq_type, type_map["Unknown"])
        label = style['label'] if style['label'] not in plotted_labels else None
        if label: plotted_labels.add(style['label'])
        
        ax.scatter(v_eq[0], v_eq[1], 
                   marker=style['marker'], color=style['color'], 
                   s=style['size'], label=label,
                   edgecolors='black', zorder=5)

    ax.set_xlabel('$V_1$'); ax.set_ylabel('$V_2$')
    ax.set_title(f'Phase plane (k={k:.3f}) with Numerically Found Equilibria')
    ax.legend(loc='upper right', frameon=True, fontsize='small')
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5); ax.grid(True, lw=0.3, alpha=0.4)
    ax.axhline(0, color='black', lw=0.5, alpha=0.5)
    ax.axvline(0, color='black', lw=0.5, alpha=0.5)
    ax.set_aspect('equal', adjustable='box')
    plt.savefig(OUT_DIR / f'phaseplane_numerical_k{k:.3f}.png')
    plt.close(fig)

def animate_trial(k: float, sigma: float, amp: float,
                  save_path: Path | None = None,
                  rng: np.random.Generator | None = None) -> None:
    """Animate a single stochastic switching trial on the phase plane."""
    if rng is None:
        rng = globals().get("rng", np.random.default_rng())

    # Prepare initial state near Q=0 then apply Set pulse
    v_start = prepare_reset_state(k, amp)
    IS_set = rectangular_pulse(0.0, 2.0, amp)
    IR_set = rectangular_pulse(0.0, 2.0, -amp if amp > 0 else 0.0)

    # Reduce simulation time for animation to generate fewer frames
    animation_T = 5.0  # Simulate for 5 seconds for the animation
    dt = DT
    path = euler_maruyama(k, IS_set, IR_set, sigma,
                          v0=v_start, T=animation_T, dt=dt, rng=rng, return_path=True)

    U_coords = np.linspace(-2.5, 2.5, GRID_N)
    V_coords = np.linspace(-2.5, 2.5, GRID_N)
    Ug, Vg = np.meshgrid(U_coords, V_coords)
    dU, dV = rhs(np.array([Ug, Vg]), k, 0.0, 0.0)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.streamplot(Ug, Vg, dU, dV, color='0.7', density=1.2, arrowsize=0.9)

    x_null = np.linspace(-2.5, 2.5, 400)
    if abs(k) > 1e-6:
        ax.plot(x_null, (x_null - x_null**3) / k, 'r--', lw=1)
        ax.plot((x_null - x_null**3) / k, x_null, 'b--', lw=1)

    point, = ax.plot([], [], 'ko', ms=5)

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_xlabel('$V_1$')
    ax.set_ylabel('$V_2$')
    ax.set_title(f'Stochastic trajectory (k={k:.2f}, sigma={sigma}, amp={amp})')
    ax.grid(True, lw=0.3, alpha=0.4)
    ax.set_aspect('equal', adjustable='box')

    def init():
        point.set_data([], [])
        return point,

    def update(frame):
        point.set_data([path[frame, 0]], [path[frame, 1]])
        return point,

    # Synchronize simulation and animation timing
    fps = 25
    frames = len(path)
    interval = 1000 / fps  # milliseconds per frame
    
    ani = animation.FuncAnimation(fig, update, frames=frames,
                                  init_func=init, blit=True, interval=interval)

    if save_path is not None:
        total_frames = len(path)
        logging.info(f"Starting animation save. Total frames: {total_frames}")
        
        def progress_callback(current_frame, total_frames_from_func):
            # current_frame is 0-indexed
            if current_frame % 50 == 0:  # Log every 50 frames to reduce output
                logging.info(f"Saving frame {current_frame + 1} of {total_frames_from_func}")

        ani.save(save_path, writer='pillow', fps=fps, progress_callback=progress_callback)
        logging.info(f"Animation saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()


# ───────────────────────────────────────────────────── PyCont continuation plot —
def antisym_eq(k: float) -> tuple[float,float]:
    """Return (+v, −v) antisymmetric equilibrium at parameter k."""
    # For v - v^3 + kv = 0 => v(1+k-v^2)=0. So v^2 = 1+k
    if 1 + k >= 0:
        v1 = math.sqrt(1 + k)
        return v1, -v1
    else: # Should not happen for k>=0
        return optimize.fsolve(lambda v: v - v**3 + k*v, 1.0)[0], -optimize.fsolve(lambda v: v - v**3 + k*v, 1.0)[0]


def continuation():
    if not HAVE_PYDSTOOL:
        logging.warning("Continuation skipped (PyDSTool missing)")
        return

    k0 = 0.20                              # start safely inside bistable zone
    v10, v20 = antisym_eq(k0)

    DS = pyargs(name='SR')
    DS.pars     = {'k': k0}
    DS.varspecs = {'v1': 'v1 - v1**3 - k*v2',
                   'v2': 'v2 - v2**3 - k*v1'}
    DS.ics      = {'v1': v10, 'v2': v20}
    ode = Generator.Vode_ODEsystem(DS)

    PC  = ContClass(ode)
    EP  = pyargs(name='EQ', type='EP-C',
                 freepars=['k'], StepSize=1e-2, MaxNumPoints=300,
                 LocBifPoints=['LP'], SaveEigen=True, MinStepSize=1e-5, MaxStepSize=1e-1) # Added step size controls
    
    PC.newCurve(EP)
    logging.info("Running forward continuation...")
    PC['EQ'].forward()
    logging.info("Running backward continuation...")
    PC['EQ'].backward()
    
    sol = PC['EQ'].sol
    ks   = np.array([s['k'] for s in sol]) # Accessing solution data correctly
    v1s  = np.array([s['v1'] for s in sol])
    # PyDSTool stability: sol[idx].labels['EP']['stab'] can be 'S' or 'U'
    # Or check eigenvalues if saved: sol[idx].labels['EP']['evals']
    
    # For plotting, we need to infer stability if not directly available or parse it
    # This part might need adjustment based on how PyCont stores stability info
    # For now, let's assume we plot all as one style or try to get stability
    
    plt.figure(figsize=(8,6))
    # Attempt to plot stable/unstable differently if possible
    # This is a simplified plotting; PyCont has its own plot methods too.
    stable_ks, stable_v1s = [], []
    unstable_ks, unstable_v1s = [], []

    # This is a guess on how stability might be stored; adjust if needed
    for point in sol:
        is_stable = True # Default, or determine from point.labels['EP']['stab'] or eigenvalues
        # Example: if 'stab' in point.labels['EP'] and point.labels['EP']['stab'] == 'U': is_stable = False
        # Example: if 'evals' in point.labels['EP']:
        #    if np.any(np.real(point.labels['EP']['evals']) > 0): is_stable = False
        
        # For now, let's just plot all points
        # if is_stable:
        #     stable_ks.append(point['k'])
        #     stable_v1s.append(point['v1'])
        # else:
        #     unstable_ks.append(point['k'])
        #     unstable_v1s.append(point['v1'])

    # Plotting all as solid for now, as stability parsing from sol needs verification
    plt.plot(ks,  v1s,  'k-', label='v1 branch')
    plt.plot(ks, -v1s,  'k-', label='-v1 branch (mirrored)')
    
    plt.xlabel('k'); plt.ylabel('$V_1^*$ equilibrium value')
    plt.title('Anti-symmetric Equilibria (PyCont)')
    plt.grid(True, lw=0.3, alpha=0.5)
    plt.legend()
    plt.savefig(OUT_DIR / 'bifurcation_pycont_antisym.png')
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# MONTE CARLO SWITCHING PROBABILITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def switch_trial(k: float, sigma: float, amp: float, rng: np.random.Generator | None = None, 
                switch_threshold: float = 0.1) -> int:
    """
    Perform a single Monte Carlo trial to test Set pulse success probability.
    
    This function simulates a complete latch switching sequence:
    1. Prepare initial Q=0 state (V₁ low, V₂ high)
    2. Apply Set pulse to attempt Q=0 → Q=1 transition
    3. Check if final state represents successful Q=1 (V₁ high, V₂ low)
    
    Parameters:
    -----------
    k : float
        Coupling strength parameter
    sigma : float  
        Noise intensity (standard deviation of Gaussian white noise)
    amp : float
        Set pulse amplitude
    rng : np.random.Generator, optional
        Random number generator for reproducible trials
    switch_threshold : float, default=0.1
        Threshold for determining successful state transitions
        
    Returns:
    --------
    int
        1 if Set pulse successfully switched Q=0 → Q=1, 0 otherwise
        
    Success Criteria:
    ----------------
    A trial is considered successful if the final state satisfies:
    - V₁ > switch_threshold (V₁ switched to high state)
    - V₂ < -switch_threshold (V₂ switched to low state)
    
    Notes:
    ------
    Each trial includes:
    - Dynamic state preparation using prepare_reset_state()
    - Set pulse: positive I_S, negative I_R (duration: 2 seconds)
    - Full simulation time T_TOTAL for complete settling
    """
    # (i) Prepare initial state, aiming for Q=0 (v1 low, v2 high).
    # For robustness, start near the anti-symmetric attractor (-va, va) if k allows, or (0,0)
    v_initial_target_q0 = np.array([-np.sqrt(max(0,1+k)), np.sqrt(max(0,1+k))]) if (1+k)>0.01 else np.array([0.0,0.0])
    
     # Prepare the state using the same routine as the reset logic so the latch
    # begins in a Q=0 configuration.
    v_start_for_set_pulse = prepare_reset_state(k, amp)

    # (ii) apply Set pulse (10–12 s) to drive V1 high, V2 low
    IS_set_pulse = rectangular_pulse(t0=0.0, t1=2.0, amp=amp)  # Shorter pulse for test
    IR_set_pulse = rectangular_pulse(t0=0.0, t1=2.0, amp=-amp if amp > 0 else 0.0) # Optional: push V2 low
    # IR_set_pulse = lambda t: 0.0 # Alternative: only pulse V1

    # Simulate for a total time T_TOTAL, which includes the pulse and settling time
    v_final = euler_maruyama(k, IS_set_pulse, IR_set_pulse, sigma,
                             v0=v_start_for_set_pulse, T=T_TOTAL, rng=rng)

    
    # Success if V1 ended high and V2 ended low
    switched_to_q1 = (v_final[0] > switch_threshold and v_final[1] < -switch_threshold)
    return int(switched_to_q1)

def mc_worker(arg_tuple):
    # k, sigma, amp = arg_tuple # Original
    # For imap_unordered, it seems the arguments are passed directly if func takes multiple args
    # If it takes one tuple, then unpack. Let's assume it's a tuple.
    if isinstance(arg_tuple, tuple) and len(arg_tuple) == 3:
        k, sigma, amp = arg_tuple
    else: # Fallback or error
        logging.error(f"mc_worker received unexpected arg format: {arg_tuple}")
        return 0.0 
    
    seed = abs(int.from_bytes(os.urandom(8), "little") ^ hash((k, sigma, amp)))
    local_rng = np.random.default_rng(seed)
    hits = sum(switch_trial(k, sigma, amp, rng=local_rng)
               for _ in range(TRIALS_PER_TASK))
    return k, sigma, amp, hits / TRIALS_PER_TASK
    
# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETER SWEEP AND HEATMAP VISUALIZATION  
# ═══════════════════════════════════════════════════════════════════════════════

def plot_2d_heatmap(df_k: pd.DataFrame, k_val: float, out_dir: Path):
    """
    Generate 2D heatmap of switching probability vs noise intensity and pulse amplitude.
    
    Parameters:
    -----------
    df_k : pd.DataFrame
        Subset of Monte Carlo results for a specific k value
    k_val : float
        The coupling strength value for this heatmap
    out_dir : Path
        Output directory for saving the heatmap
        
    Visualization Details:
    ---------------------
    - X-axis: Pulse amplitude (amp) - strength of Set pulse
    - Y-axis: Noise intensity (sigma) - stochastic fluctuation level  
    - Color scale: P(Set Success) ∈ [0,1] with viridis colormap
    - Higher amplitudes generally increase switching probability
    - Intermediate noise levels may enhance switching via stochastic resonance
    
    Output:
    -------
    Saves 'heatmap_P_vs_sigma_amp_k{k_val:.3f}.png'
    """
    if df_k.empty:
        logging.warning(f"No data to plot heatmap for k={k_val}")
        return

    try:
        heatmap_data = df_k.pivot_table(index='sigma', columns='amp', values='P_switch')
    except Exception as e:
        logging.error(f"Could not pivot data for heatmap k={k_val}: {e}")
        logging.error(f"Data for k={k_val}:\n{df_k.head()}")
        return

    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap_data, aspect='auto', origin='lower',
               extent=[AMP_RANGE.min(), AMP_RANGE.max(), SIGMA_RANGE.min(), SIGMA_RANGE.max()],
               cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(label='P(Set Success)')
    plt.xlabel('Pulse Amplitude (amp)')
    plt.ylabel('Noise Intensity (sigma)')
    plt.title(f'Set Pulse Success Probability for k = {k_val:.3f}')
    plt.savefig(out_dir / f'heatmap_P_vs_sigma_amp_k{k_val:.3f}.png')
    plt.close()

def sweep():
    params_to_run = []
    for k_val in K_RANGE:
        for s_val in SIGMA_RANGE:
            for a_val in AMP_RANGE:
                params_to_run.append((k_val, s_val, a_val))

    all_results = []
    # Using try-finally for pool to ensure it's closed
    pool = mp.Pool(processes=CPU_COUNT)
    try:
        # Using pool.map which expects a list of single arguments for the worker
        # So mc_worker should expect a single tuple.
        # Or, use starmap if mc_worker takes k, sigma, amp directly.
        # Let's stick to mc_worker taking a tuple as per its definition.
        
        # The original code used imap_unordered.
        # mc_worker now returns k, sigma, amp, P.
        for result_tuple in tqdm(pool.imap_unordered(mc_worker, params_to_run),
                                 total=len(params_to_run), desc='MC Sweep'):
            if result_tuple: # Ensure worker didn't return None or error indicator
                 all_results.append({'k':result_tuple[0], 'sigma':result_tuple[1], 
                                     'amp':result_tuple[2], 'P_switch':result_tuple[3]})
    finally:
        pool.close()
        pool.join()

    if not all_results:
        logging.error("No results collected from Monte Carlo sweep.")
        return

    df_results = pd.DataFrame(all_results)
    csv_path = OUT_DIR / 'sweep_results.csv'
    df_results.to_csv(csv_path, index=False)
    logging.info(f"Monte Carlo sweep data saved to {csv_path}")

    # Generate 2D heatmaps: P(sigma, amp) for each k
    unique_k_values = df_results['k'].unique()
    for k_val in tqdm(unique_k_values, desc="Generating Heatmaps"):
        df_k_subset = df_results[df_results['k'] == k_val]
        plot_2d_heatmap(df_k_subset, k_val, OUT_DIR)
    
    # Optional: Plot a 1D slice like before, if desired
    # Example: P(sigma) for k closest to 0.5 and amp closest to 1.0
    if not df_results.empty:
        target_k = 0.5
        target_amp = 1.0
        closest_k = df_results.iloc[(df_results['k'] - target_k).abs().argsort()[:1]]['k'].iloc[0]
        closest_amp = df_results.iloc[(df_results['amp'] - target_amp).abs().argsort()[:1]]['amp'].iloc[0]

        slice_df = df_results[(df_results['k'] == closest_k) & (df_results['amp'] == closest_amp)]
        if not slice_df.empty:
            plt.figure(figsize=(7,5))
            plt.plot(slice_df['sigma'], slice_df['P_switch'], 'o-', color='dodgerblue')
            plt.xlabel('Noise Intensity (sigma)')
            plt.ylabel('P(Set Success)')
            plt.title(f'Set Success vs. Sigma (k≈{closest_k:.2f}, amp≈{closest_amp:.2f})')
            plt.ylim(-0.05, 1.05)
            plt.grid(True, linestyle=':', alpha=0.7)
            plt.savefig(OUT_DIR / f'P_vs_sigma_k{closest_k:.2f}_amp{closest_amp:.2f}.png')
            plt.close()
        else:
            logging.warning(f"Could not find data for 1D slice plot (k≈{target_k}, amp≈{target_amp})")


# ═══════════════════════════════════════════════════════════════════════════════
# BISTABILITY AND MULTISTABILITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def bistability_vs_k():
    """
    Analyze system multistability as a function of coupling strength.
    
    This function provides a comprehensive analysis of how the number and types
    of equilibria change with the coupling parameter k. It generates detailed
    visualizations showing:
    
    1. Number of stable attractors vs k (classical bistability analysis)
    2. Complete equilibrium census including saddles and repellers
    3. Parameter regions supporting different dynamical behaviors
    
    Mathematical Context:
    --------------------
    The coupled neural latch system exhibits rich multistability:
    - k = 0: Decoupled case with 9 equilibria (3×3 grid structure)
    - k > 0: Coupling introduces new bifurcations and equilibrium merging
    - Critical k values: Points where equilibria appear/disappear or change stability
    
    Bifurcation Types:
    -----------------
    - Pitchfork bifurcations: Symmetric equilibria branch creation/destruction
    - Transcritical bifurcations: Stability exchanges between equilibria
    - Saddle-node bifurcations: Equilibrium creation/annihilation events
    
    Output:
    -------
    Generates two comprehensive plots:
    - 'attractors_vs_k_comprehensive.png': Two-panel analysis (attractors + all types)
    - 'attractors_vs_k.png': Classic bistability plot focusing on stable states
    
    The analysis uses high-resolution k sampling (100 points) to capture
    detailed bifurcation structure and transition points.
    """
    counts_attractors = []
    counts_saddles = []
    counts_repellers = []
    # Much finer k range for detailed stability analysis
    k_values_for_plot = np.linspace(min(K_RANGE), max(K_RANGE), 100)  # Higher resolution
    
    for k_val in tqdm(k_values_for_plot, desc="Bistability Analysis"):
        eq_k = get_all_equilibria_and_classify(k_val)
        counts_attractors.append(sum(1 for e in eq_k if e['type'] == 'Attractor'))
        counts_saddles.append(sum(1 for e in eq_k if 'Saddle' in e['type']))
        counts_repellers.append(sum(1 for e in eq_k if 'Repeller' in e['type']))
    
    # Create comprehensive stability plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot 1: Number of attractors (traditional bistability)
    ax1.plot(k_values_for_plot, counts_attractors, 'o-', color='darkslateblue', markersize=3, linewidth=2)
    ax1.set_xlabel('Coupling Strength (k)')
    ax1.set_ylabel('Number of Stable Attractors')
    ax1.set_title('System Stability: Attractors vs. Coupling Strength k')
    ax1.set_yticks(np.arange(0, max(counts_attractors)+2, 1))
    ax1.grid(True, linestyle=':', alpha=0.7)
    
    # Plot 2: All equilibrium types
    ax2.plot(k_values_for_plot, counts_attractors, 'o-', color='green', markersize=2, 
             linewidth=1.5, label='Attractors', alpha=0.8)
    ax2.plot(k_values_for_plot, counts_saddles, 's-', color='red', markersize=2, 
             linewidth=1.5, label='Saddles', alpha=0.8)
    ax2.plot(k_values_for_plot, counts_repellers, '^-', color='blue', markersize=2, 
             linewidth=1.5, label='Repellers', alpha=0.8)
    ax2.set_xlabel('Coupling Strength (k)')
    ax2.set_ylabel('Number of Equilibria')
    ax2.set_title('Complete Equilibrium Classification vs. k')
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'attractors_vs_k_comprehensive.png', dpi=150)
    plt.close()
    
    # Also save the original simple plot
    plt.figure(figsize=(7,5))
    plt.plot(k_values_for_plot, counts_attractors, 'o-', color='darkslateblue', markersize=4)
    plt.xlabel('Coupling Strength (k)')
    plt.ylabel('Number of Stable Attractors')
    plt.title('System Stability vs. Coupling Strength k')
    plt.yticks(np.arange(min(counts_attractors)-1 if counts_attractors else 0, 
                        max(counts_attractors)+2 if counts_attractors else 2, 1))
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.savefig(OUT_DIR / 'attractors_vs_k.png')
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# TRUTH TABLE VERIFICATION AND LATCH FUNCTIONALITY TESTING
# ═══════════════════════════════════════════════════════════════════════════════

def truth_table():
    """
    Comprehensive verification of neural latch functionality across parameter space.
    
    This function systematically tests the SR (Set-Reset) latch truth table behavior
    across multiple parameter combinations to ensure robust digital operation.
    
    Test Protocol:
    -------------
    For each (k, amplitude) parameter pair, verifies all 5 logical operations:
    
    1. HOLD Q=0: No input pulses → Q remains in reset state (V₁<0, V₂>0)
    2. SET: S=1, R=0 → Q transitions from 0 to 1 (V₁>0, V₂<0)  
    3. HOLD Q=1: No input pulses → Q remains in set state (V₁>0, V₂<0)
    4. RESET: S=0, R=1 → Q transitions from 1 to 0 (V₁<0, V₂>0)
    5. INVALID: S=1, R=1 → Undefined behavior (should not settle to Q=0 or Q=1)
    
    Parameter Coverage:
    ------------------
    - Coupling strengths: k ∈ {0.3, 0.5, 0.7, 1.0, 1.5}
    - Pulse amplitudes: amp ∈ {0.8, 1.0, 1.5, 2.0}
    - Total combinations: 5 × 4 = 20 parameter sets
    
    Deterministic Testing:
    ---------------------
    Uses σ = 0 (no noise) to isolate the deterministic switching behavior
    and verify ideal digital logic operation without stochastic effects.
    
    Success Metrics:
    ---------------
    - State classification threshold: ±0.1 voltage units
    - Q=0 state: V₁ < -0.1 AND V₂ > +0.1  
    - Q=1 state: V₁ > +0.1 AND V₂ < -0.1
    - Success rate: Fraction of operations working correctly
    
    Output:
    -------
    Generates detailed CSV report with:
    - Individual test results for each (k, amp) combination
    - Success/failure indicators for each logical operation
    - Overall functionality score per parameter set
    
    This comprehensive testing ensures the neural circuit can reliably implement
    digital latch functionality across the intended parameter operating range.
    """
    # Enhanced parameter testing - multiple k and amplitude values
    k_values_to_test = [0.3, 0.5, 0.7, 1.0, 1.5]  # Representative values from K_RANGE
    amp_values_to_test = [0.8, 1.0, 1.5, 2.0]      # Different amplitude strengths
    
    all_results = []
    
    for k_fixed in k_values_to_test:
        for amp_fixed in amp_values_to_test:
            logging.info(f"Running truth table for k={k_fixed}, amp={amp_fixed}, sigma=0.0")

            # --- Dynamically prepare initial Q=0 state ---
            logging.info("Preparing initial Q=0 state dynamically...")
            initial_state_q0 = prepare_reset_state(k_fixed, amp_fixed, sigma=0.0)
            q0_check = initial_state_q0[0] < -0.1 and initial_state_q0[1] > 0.1
            logging.info(f"  ...prepared Q=0 state: {initial_state_q0}, Q=0-like: {q0_check}")

            # --- Dynamically prepare initial Q=1 state (by "Set" pulsing the Q=0 state) ---
            logging.info("Preparing initial Q=1 state dynamically (from prepared Q=0)...")
            IS_for_q1_prep = rectangular_pulse(0, 2, amp_fixed)
            IR_for_q1_prep = rectangular_pulse(0, 2, -amp_fixed)
            initial_state_q1 = euler_maruyama(k_fixed, IS_for_q1_prep, IR_for_q1_prep, 0.0, v0=initial_state_q0, T=T_TOTAL)
            q1_check = initial_state_q1[0] > 0.1 and initial_state_q1[1] < -0.1
            logging.info(f"  ...prepared Q=1 state: {initial_state_q1}, Q=1-like: {q1_check}")

            def check_final_state(v_final, target_v1_pos=True, threshold=0.1):
                if target_v1_pos: # Q=1 like state (V1 high, V2 low)
                    return int(v_final[0] > threshold and v_final[1] < -threshold)
                else: # Q=0 like state (V1 low, V2 high)
                    return int(v_final[0] < -threshold and v_final[1] > threshold)

            # Test cases using the dynamically prepared initial_state_q0 and initial_state_q1

            # 1. Hold Q=0: Start at prepared Q=0, no pulse, should remain Q=0
            logging.info("Test 1: Hold Q=0 (start from prepared Q=0, no pulse)")
            v_final_hold_q0 = euler_maruyama(k_fixed, lambda t:0, lambda t:0, 0.0, v0=initial_state_q0, T=T_TOTAL)
            hold_q0_result = check_final_state(v_final_hold_q0, target_v1_pos=False) # Expect Q=0

            # 2. Set (from Q=0 to Q=1): Start at prepared Q=0, Set pulse, should go to Q=1
            logging.info("Test 2: Set Q0->Q1 (start from prepared Q=0, Set pulse)")
            IS_set = rectangular_pulse(0, 2, amp_fixed)
            IR_set = rectangular_pulse(0, 2, -amp_fixed) # Push V2 low for Set
            v_final_set_q1 = euler_maruyama(k_fixed, IS_set, IR_set, 0.0, v0=initial_state_q0, T=T_TOTAL)
            set_q1_result = check_final_state(v_final_set_q1, target_v1_pos=True) # Expect Q=1

            # 3. Hold Q=1: Start at prepared Q=1, no pulse, should remain Q=1
            logging.info("Test 3: Hold Q=1 (start from prepared Q=1, no pulse)")
            v_final_hold_q1 = euler_maruyama(k_fixed, lambda t:0, lambda t:0, 0.0, v0=initial_state_q1, T=T_TOTAL)
            hold_q1_result = check_final_state(v_final_hold_q1, target_v1_pos=True) # Expect Q=1
            
            # 4. Reset (from Q=1 to Q=0): Start at prepared Q=1, Reset pulse, should go to Q=0
            logging.info("Test 4: Reset Q1->Q0 (start from prepared Q=1, Reset pulse)")
            IS_reset = rectangular_pulse(0, 2, -amp_fixed) # Push V1 low for Reset
            IR_reset = rectangular_pulse(0, 2, amp_fixed)  # Push V2 high for Reset
            v_final_reset_q0 = euler_maruyama(k_fixed, IS_reset, IR_reset, 0.0, v0=initial_state_q1, T=T_TOTAL)
            reset_q0_result = check_final_state(v_final_reset_q0, target_v1_pos=False) # Expect Q=0

            # 5. Invalid (Set and Reset simultaneously from Q=0)
            logging.info("Test 5: Invalid S=R=1 (start from prepared Q=0, S=1 R=1 pulse)")
            IS_invalid = rectangular_pulse(0, 2, amp_fixed)
            IR_invalid = rectangular_pulse(0, 2, amp_fixed) # Both positive pulses (S=1, R=1)
            v_final_invalid_from_q0 = euler_maruyama(k_fixed, IS_invalid, IR_invalid, 0.0, v0=initial_state_q0, T=T_TOTAL)
            logging.info(f"  ...v_final for S=R=1 case: {v_final_invalid_from_q0}")
            is_q0_after_invalid = check_final_state(v_final_invalid_from_q0, target_v1_pos=False)
            is_q1_after_invalid = check_final_state(v_final_invalid_from_q0, target_v1_pos=True)
            # Result is 1 if it's neither Q0 nor Q1 (an "other" state)
            invalid_from_q0_result = 1 if not is_q0_after_invalid and not is_q1_after_invalid else 0

            # Store results for this k, amp combination
            result_row = {
                'k': k_fixed,
                'amp': amp_fixed,
                'Hold_Q0_stays_Q0': hold_q0_result,
                'Set_Q0_to_Q1': set_q1_result,
                'Hold_Q1_stays_Q1': hold_q1_result,
                'Reset_Q1_to_Q0': reset_q0_result,
                'Invalid_S1R1_from_Q0_not_Q0_not_Q1': invalid_from_q0_result,
                'Total_Success': hold_q0_result + set_q1_result + hold_q1_result + reset_q0_result + invalid_from_q0_result
            }
            all_results.append(result_row)
            
            logging.info(f"Truth table results for k={k_fixed}, amp={amp_fixed}: "
                        f"Total success: {result_row['Total_Success']}/5")

    # Save comprehensive results
    df_truth = pd.DataFrame(all_results)
    df_truth.to_csv(OUT_DIR / 'truth_table_comprehensive.csv', index=False)
    
    # Create summary statistics
    summary_stats = {
        'Overall_Success_Rate': df_truth['Total_Success'].mean() / 5,
        'Perfect_Operations': len(df_truth[df_truth['Total_Success'] == 5]),
        'Total_Tests': len(df_truth),
        'Best_k_amp_combo': df_truth.loc[df_truth['Total_Success'].idxmax(), ['k', 'amp']].to_dict() if not df_truth.empty else None
    }
    
    # Save summary
    summary_series = pd.Series(summary_stats)
    summary_series.to_csv(OUT_DIR / 'truth_table_summary.csv', header=False)
    
    logging.info(f"Enhanced truth table analysis complete:")
    logging.info(f"  Overall success rate: {summary_stats['Overall_Success_Rate']:.2%}")
    logging.info(f"  Perfect operations: {summary_stats['Perfect_Operations']}/{summary_stats['Total_Tests']}")
    if summary_stats['Best_k_amp_combo']:
        logging.info(f"  Best (k,amp) combo: {summary_stats['Best_k_amp_combo']}")
    
    return df_truth

# ─────────────────────────────────── deterministic (ideal world) analysis ─
def _run_deterministic_switch_test(k: float, amp: float, duration: float) -> bool:
    """
    Helper function: performs a single, noise-free switch attempt.

    Returns:
        True if the switch from Q=0 to Q=1 was successful, False otherwise.
    """
    # 1. Prepare the initial Q=0 state robustly and deterministically
    v_start_q0 = prepare_reset_state(k, amp=1.5, sigma=0.0)

    # 2. Define the "Set" pulse with the given parameters
    IS_set = rectangular_pulse(t0=0.0, t1=duration, amp=amp)
    IR_set = rectangular_pulse(t0=0.0, t1=duration, amp=-amp) # Opposing pulse

    # 3. Simulate for the pulse duration plus settling time
    settling_time = 10.0
    total_time = duration + settling_time
    v_final = euler_maruyama(k, IS_set, IR_set, sigma=0.0, v0=v_start_q0, T=total_time)

    # 4. Check if the final state is a successful Q=1 state
    # A robust check is to see if v1 is positive and v2 is negative.
    switched_successfully = (v_final[0] > 0.1 and v_final[1] < -0.1)
    return switched_successfully

def analyze_deterministic_switching():
    """
    Enhanced deterministic switching analysis with expanded parameter ranges.
    Analyzes the latch's performance in a noise-free (sigma=0) environment.
    Generates plots for:
      - Minimum switching amplitude vs. coupling (k)
      - Minimum switching duration vs. coupling (k)
    """
    logging.info("--- Starting Enhanced Deterministic Analysis (sigma=0) ---")

    # Expanded parameter ranges for comprehensive analysis
    k_to_test = np.linspace(0.1, 1.8, 15)           # Much wider k range
    amps_to_test = np.linspace(0.05, 3.0, 60)       # Wider amplitude range, higher resolution
    durations_to_test = np.linspace(0.05, 8.0, 80)  # Wider duration range, higher resolution

    results = []

    # --- Experiment 1: Find minimum amplitude for multiple durations ---
    durations_to_analyze = [0.5, 1.0, 2.0, 3.0, 5.0]  # Multiple duration points
    
    for duration in durations_to_analyze:
        logging.info(f"Finding minimum switching amplitude (pulse duration = {duration}s)...")
        for k in tqdm(k_to_test, desc=f"Testing Amplitudes (T={duration}s)"):
            min_amp = np.nan
            for amp in amps_to_test:
                if _run_deterministic_switch_test(k, amp, duration):
                    min_amp = amp
                    break # Found the first successful amplitude, so we can stop
            results.append({'k': k, 'duration': duration, 'min_amp': min_amp})

    # --- Experiment 2: Find minimum duration for multiple amplitudes ---
    amplitudes_to_analyze = [0.5, 1.0, 1.5, 2.0, 2.5]  # Multiple amplitude points
    
    for amplitude in amplitudes_to_analyze:
        logging.info(f"Finding minimum switching duration (pulse amplitude = {amplitude})...")
        for k in tqdm(k_to_test, desc=f"Testing Durations (A={amplitude})"):
            min_duration = np.nan
            for duration in durations_to_test:
                if _run_deterministic_switch_test(k, amplitude, duration):
                    min_duration = duration
                    break # Found the first successful duration
            # Add to existing results or create new entries
            existing_idx = next((i for i, r in enumerate(results) 
                               if r['k'] == k and r.get('amplitude') == amplitude), None)
            if existing_idx is not None:
                results[existing_idx]['min_duration'] = min_duration
            else:
                results.append({'k': k, 'amplitude': amplitude, 'min_duration': min_duration})

    # --- Enhanced plotting and analysis ---
    df = pd.DataFrame(results)
    
    # Separate data for amplitude and duration analysis
    amp_data = df[df['duration'].notna()].copy()
    dur_data = df[df['amplitude'].notna()].copy()
    
    # Plot 1: Minimum Amplitude vs. k for different durations
    plt.figure(figsize=(12, 8))
    for duration in durations_to_analyze:
        data_subset = amp_data[amp_data['duration'] == duration]
        if not data_subset.empty:
            plt.plot(data_subset['k'], data_subset['min_amp'], 'o-', 
                    label=f'T = {duration}s', linewidth=2, markersize=4)
    
    plt.xlabel('Coupling Strength (k)', fontsize=12)
    plt.ylabel('Minimum Switching Amplitude', fontsize=12)
    plt.title('Enhanced Deterministic Analysis: Minimum Amplitude vs. k\n' +
              '(Multiple pulse durations, expanded k range)', fontsize=13)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'deterministic_min_amplitude_enhanced.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Minimum Duration vs. k for different amplitudes
    plt.figure(figsize=(12, 8))
    for amplitude in amplitudes_to_analyze:
        data_subset = dur_data[dur_data['amplitude'] == amplitude]
        if not data_subset.empty:
            plt.plot(data_subset['k'], data_subset['min_duration'], 's-', 
                    label=f'A = {amplitude}', linewidth=2, markersize=4)
    
    plt.xlabel('Coupling Strength (k)', fontsize=12)
    plt.ylabel('Minimum Switching Duration (s)', fontsize=12)
    plt.title('Enhanced Deterministic Analysis: Minimum Duration vs. k\n' +
              '(Multiple pulse amplitudes, expanded k range)', fontsize=13)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'deterministic_min_duration_enhanced.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 3: 2D Heatmap of minimum amplitude vs (k, duration)
    if not amp_data.empty:
        pivot_amp = amp_data.pivot(index='duration', columns='k', values='min_amp')
        plt.figure(figsize=(14, 8))
        im = plt.imshow(pivot_amp.values, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(im, label='Minimum Amplitude')
        plt.xlabel('Coupling Strength (k)')
        plt.ylabel('Pulse Duration (s)')
        plt.title('2D Parameter Space: Minimum Switching Amplitude')
        
        # Set ticks to actual values
        k_ticks = np.linspace(0, len(pivot_amp.columns)-1, 8, dtype=int)
        dur_ticks = np.linspace(0, len(pivot_amp.index)-1, 5, dtype=int)
        plt.xticks(k_ticks, [f"{pivot_amp.columns[i]:.2f}" for i in k_ticks])
        plt.yticks(dur_ticks, [f"{pivot_amp.index[i]:.1f}" for i in dur_ticks])
        
        plt.tight_layout()
        plt.savefig(OUT_DIR / 'deterministic_amplitude_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # Save comprehensive data
    df.to_csv(OUT_DIR / 'deterministic_analysis_comprehensive.csv', index=False)
    
    # Generate summary statistics
    summary_stats = {
        'k_range': f"{k_to_test.min():.2f} - {k_to_test.max():.2f}",
        'amp_range_tested': f"{amps_to_test.min():.2f} - {amps_to_test.max():.2f}",
        'duration_range_tested': f"{durations_to_test.min():.2f} - {durations_to_test.max():.2f}",
        'total_tests': len(results),
        'successful_tests': len([r for r in results if not (np.isnan(r.get('min_amp', np.nan)) and np.isnan(r.get('min_duration', np.nan)))])
    }
    
    summary_series = pd.Series(summary_stats)
    summary_series.to_csv(OUT_DIR / 'deterministic_analysis_summary.csv', header=False)
    
    logging.info(f"Enhanced deterministic analysis complete:")
    logging.info(f"  K range: {summary_stats['k_range']}")
    logging.info(f"  Successful tests: {summary_stats['successful_tests']}/{summary_stats['total_tests']}")
    logging.info(f"  Results saved to {OUT_DIR}/")
    
    return df

# ═══════════════════════════════════════════════════ Basin of Attraction Analysis ═══════════════════════════════════════════════════

def find_equilibria_for_basins(k: float) -> list[np.ndarray]:
    """
    Find equilibria specifically for basin of attraction analysis.
    
    This function uses a systematic search to find stable equilibria that will serve
    as attractors for the basin analysis. It combines both analytical and numerical
    approaches for robust equilibrium detection.
    
    Parameters:
    -----------
    k : float
        Coupling strength parameter
        
    Returns:
    --------
    list[np.ndarray]
        List of equilibrium point coordinates
    """
    def equations(v):
        """System equations with zero inputs (autonomous system)"""
        return rhs(v, k, 0.0, 0.0)

    # Use a systematic grid search for robust equilibrium finding
    search_range = np.linspace(-2, 2, 5)
    guesses = [np.array([x, y]) for x in search_range for y in search_range]
    equilibria = []

    for guess in guesses:
        try:
            eq, infodict, ier, mesg = optimize.fsolve(equations, guess, full_output=True)
            # Check if the solution converged and is not already found
            if ier == 1 and not any(np.allclose(eq, existing, atol=1e-4) for existing in equilibria):
                equilibria.append(eq)
        except:
            continue

    return equilibria

def classify_equilibria_stability(equilibria: list[np.ndarray], k: float) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Classify equilibria by their stability for basin analysis.
    
    This function determines which equilibria are stable attractors (negative real 
    eigenvalues) versus unstable (positive or mixed real eigenvalues).
    
    Parameters:
    -----------
    equilibria : list[np.ndarray]
        List of equilibrium points to classify
    k : float
        Coupling strength parameter for Jacobian evaluation
        
    Returns:
    --------
    tuple[list[np.ndarray], list[np.ndarray]]
        Tuple of (stable_points, unstable_points)
    """
    stable_points = []
    unstable_points = []
    
    for eq in equilibria:
        # Calculate eigenvalues of the Jacobian at this equilibrium
        eigvals = np.linalg.eigvals(jac(eq, k))
        
        # Check if all real parts are negative (stable attractor)
        if np.all(np.real(eigvals) < 0):
            stable_points.append(eq)
        else:
            unstable_points.append(eq)
    
    return stable_points, unstable_points

def integrate_to_equilibrium_deterministic(v0: np.ndarray, k: float, 
                                         dt: float = 0.01, T: float = 50) -> np.ndarray:
    """
    Integrate system deterministically until convergence to equilibrium.
    
    This function uses simple Euler integration to evolve a trajectory from an 
    initial condition until it converges to an equilibrium (or maximum time is reached).
    Used for basin of attraction computation.
    
    Parameters:
    -----------
    v0 : np.ndarray
        Initial condition [V1, V2]
    k : float
        Coupling strength parameter
    dt : float, optional
        Integration time step (default: 0.01)
    T : float, optional
        Maximum integration time (default: 50)
        
    Returns:
    --------
    np.ndarray
        Final state after integration [V1_final, V2_final]
    """
    v = np.array(v0)
    
    # Integrate using simple Euler method until convergence
    for _ in range(int(T/dt)):
        dv = rhs(v, k, 0.0, 0.0) * dt  # No external inputs, autonomous system
        v += dv
        
        # Check for convergence (small rate of change)
        if np.linalg.norm(dv) < 1e-6:
            break
    
    return v

def plot_basins_of_attraction(k: float, resolution: int = 200, save_plot: bool = True) -> None:
    """
    Generate and plot basins of attraction for the two-cell SR-latch system.
    
    This function creates a comprehensive visualization showing:
    1. Basin boundaries between different attractors
    2. Stable equilibrium points (attractors)
    3. Phase plane structure with color-coded attraction basins
    
    The analysis reveals which initial conditions lead to which final states,
    providing insight into the system's multistability and switching behavior.
    
    Parameters:
    -----------
    k : float
        Coupling strength parameter
    resolution : int, optional
        Grid resolution for basin computation (default: 200)
        Higher resolution gives more detail but takes longer to compute
    save_plot : bool, optional
        Whether to save the plot to file (default: True)
        
    Returns:
    --------
    None
        Function saves plot and prints analysis summary
    """
    logging.info(f"Computing basins of attraction for k={k:.3f}...")
    
    # Step 1: Find and classify all equilibria
    equilibria = find_equilibria_for_basins(k)
    stable_points, unstable_points = classify_equilibria_stability(equilibria, k)

    if not stable_points:
        logging.warning(f"No stable equilibria found for k={k}. Cannot compute basins.")
        return

    logging.info(f"Found {len(stable_points)} stable attractor(s) and {len(unstable_points)} unstable point(s)")

    # Step 2: Set up computational domain
    bounds = (-2.5, 2.5)
    v1_coords = np.linspace(bounds[0], bounds[1], resolution)
    v2_coords = np.linspace(bounds[0], bounds[1], resolution)
    basin_grid = np.zeros((resolution, resolution))

    # Step 3: Compute basin membership for each grid point
    logging.info("Computing basin membership for each initial condition...")
    for i, v1 in enumerate(tqdm(v1_coords, desc="Computing Basins")):
        for j, v2 in enumerate(v2_coords):
            # Integrate from this initial condition to find final state
            v_final = integrate_to_equilibrium_deterministic([v1, v2], k)
            
            # Determine which attractor this trajectory converged to
            distances = [np.linalg.norm(v_final - attractor) for attractor in stable_points]
            closest_attractor_id = np.argmin(distances)
            
            # Store the basin ID (note [j,i] indexing for correct plot orientation)
            basin_grid[j, i] = closest_attractor_id

    # Step 4: Create comprehensive visualization
    plt.figure(figsize=(10, 8))
    
    # Create colormap with distinct colors for each basin
    cmap = plt.cm.get_cmap('tab10', len(stable_points))
    
    # Plot basin map as background
    plt.imshow(basin_grid, extent=[bounds[0], bounds[1], bounds[0], bounds[1]],
               origin='lower', cmap=cmap, interpolation='nearest', aspect='auto', alpha=0.7)

    # Plot stable attractors with emphasis
    for idx, attractor in enumerate(stable_points):
        plt.plot(attractor[0], attractor[1], 'o', markersize=12,
                 markerfacecolor=cmap(idx), markeredgecolor='black', 
                 markeredgewidth=2, label=f'Attractor {idx+1}')

    # Plot unstable equilibria if they exist
    for idx, unstable in enumerate(unstable_points):
        plt.plot(unstable[0], unstable[1], 'x', markersize=10,
                 markeredgecolor='red', markeredgewidth=3, 
                 label=f'Unstable {idx+1}' if idx == 0 else "")

    # Add nullclines for reference
    x_null = np.linspace(-2.5, 2.5, 400)
    if abs(k) > 1e-6:
        plt.plot(x_null, (x_null - x_null**3) / k, 'r--', lw=1.5, 
                alpha=0.7, label='V₁-nullcline')
        plt.plot((x_null - x_null**3) / k, x_null, 'b--', lw=1.5, 
                alpha=0.7, label='V₂-nullcline')

    # Formatting and labels
    plt.xlabel('$V_1$ (Cell 1 Voltage)', fontsize=12)
    plt.ylabel('$V_2$ (Cell 2 Voltage)', fontsize=12)
    plt.title(f'Basins of Attraction for Two-Cell SR-Latch\n' +
              f'Coupling Strength k = {k:.3f}, Resolution = {resolution}×{resolution}', fontsize=13)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot if requested
    if save_plot:
        filename = OUT_DIR / f'basins_of_attraction_k{k:.3f}_res{resolution}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        logging.info(f"Basin analysis plot saved: {filename}")
        plt.close()
    else:
        plt.show()

    # Step 5: Print analysis summary
    logging.info("═" * 60)
    logging.info(f"BASIN OF ATTRACTION ANALYSIS SUMMARY (k={k:.3f})")
    logging.info("═" * 60)
    
    for idx, attractor in enumerate(stable_points):
        basin_area = np.sum(basin_grid == idx) / (resolution * resolution)
        logging.info(f"Attractor {idx+1} at ({attractor[0]:.3f}, {attractor[1]:.3f}): "
                    f"{basin_area:.1%} of phase space")
    
    if len(stable_points) > 1:
        logging.info(f"System exhibits {len(stable_points)}-stability (multistable)")
        logging.info("Different initial conditions lead to different final states")
    else:
        logging.info("System is monostable (single attractor)")
    
    logging.info("═" * 60)

# ═══════════════════════════════════════════════════════════════════════════════
# DWELL TIME ANALYSIS (MEMORY STABILITY UNDER NOISE)
# ═══════════════════════════════════════════════════════════════════════════════

def dwell_time_trial(k: float, sigma: float, rng: np.random.Generator | None = None) -> float:
    """
    Measure first-passage time for noise-induced state transitions (dwell time analysis).
    
    This function provides a rigorous measurement of memory stability by analyzing
    how long stored information persists under stochastic perturbations.
    
    The dwell time τ is defined as the time for the system to spontaneously flip
    from a prepared Q=1 state to the Q=0 basin under pure noise dynamics.
    
    Methodology:
    -----------
    1. Deterministically prepare stable Q=1 state (V₁ > 0, V₂ < 0)
    2. Evolve system under noise only (no external inputs)
    3. Measure time until first crossing into Q=0 basin (V₁ < -0.05)
    4. Return flip time or T_MAX if no transition occurs
    
    Parameters:
    -----------
    k : float
        Coupling strength parameter
    sigma : float  
        Noise intensity (affects transition rate via Kramers theory)
    rng : np.random.Generator, optional
        Random number generator for reproducible measurements
        
    Returns:
    --------
    float
        Dwell time (first-passage time) in seconds, or T_MAX if no flip occurs
        
    Physical Interpretation:
    -----------------------
    - Larger dwell times indicate more stable memory storage
    - Dwell time typically decreases exponentially with noise intensity
    - System parameters (k) affect the energy barrier height between states
    - Related to Kramers escape time theory for noise-driven transitions
    
    Notes:
    ------
    Uses adaptive time stepping for numerical stability at high noise levels.
    The 500-second maximum simulation time balances computational cost with
    the ability to measure rare transitions in low-noise regimes.
    """
    if rng is None:
        rng = globals().get("rng", np.random.default_rng())

    T_MAX = 500.0  # Maximum simulation time
    dt = min(DT, 0.005 / (1 + sigma))  # Adaptive time step for stability
    sdt = math.sqrt(dt)

    # 1. Prepare the system in a stable Q=1 state deterministically
    # First get to Q=0, then apply Set pulse to reach Q=1
    q0_state = prepare_reset_state(k, amp=1.5, sigma=0.0)
    IS_set = rectangular_pulse(0, 2, 1.5)
    IR_set = rectangular_pulse(0, 2, -1.5)
    v = euler_maruyama(k, IS_set, IR_set, 0.0, v0=q0_state, T=10.0, dt=dt)
    
    # Verify we're in Q=1 state (v1 > 0, v2 < 0)
    if not (v[0] > 0.1 and v[1] < -0.1):
        logging.warning(f"Failed to prepare Q=1 state for k={k:.3f}, σ={sigma:.3f}")
        return T_MAX

    # 2. Evolve under pure noise until flip occurs
    t = 0.0
    N = int(T_MAX / dt)
    
    for i in range(N):
        # Check for flip to Q=0 (v1 becomes negative)
        if v[0] < -0.05:  # Crossed into Q=0 basin
            return t
        
        # Evolve one time step with noise only
        v += rhs(v, k, 0.0, 0.0) * dt + sigma * sdt * rng.normal(size=2)
        t += dt
        
    return T_MAX  # No flip occurred within T_MAX

def sweep_dwell_time(k_vals=None,
                     sigma_min=None, sigma_max=None, n_sigma=None,
                     trials=None):
    """
    Monte-Carlo estimate of the mean dwell-time before a noise-driven flip.
    
    This function rigorously measures the stability of stored memory states
    by analyzing the first-passage time for noise-induced transitions.

    Parameters
    ----------
    k_vals : list[float] | None
        Coupling strengths to test – defaults to comprehensive range from
        global K_RANGE with emphasis on critical values if None.
    sigma_min, sigma_max : float
        Inclusive σ-range. Values are swept on a log grid.
        Defaults to match global SIGMA_RANGE for consistency.
    n_sigma : int
        Number of σ points.
    trials : int
        Independent trajectories per (k, σ) sample.
    """
    # Use consistent sigma range with global configuration
    if sigma_min is None:
        sigma_min = SIGMA_RANGE.min()
    if sigma_max is None:
        sigma_max = SIGMA_RANGE.max()
    if n_sigma is None:
        n_sigma = len(SIGMA_RANGE)
        
    if k_vals is None:
        # Enhanced k values for comprehensive dwell time analysis
        # Include critical bifurcation points and representative values from K_RANGE
        k_vals = sorted(list(set([
            # Critical bifurcation points
            0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 1.8, 2.0,
            # Selected values from global K_RANGE for consistency
            K_RANGE[0], K_RANGE[len(K_RANGE)//4], K_RANGE[len(K_RANGE)//2], 
            K_RANGE[3*len(K_RANGE)//4], K_RANGE[-1]
        ])))
    
    if trials is None:
        trials = max(20, TRIALS_PER_TASK // 2)  # Reasonable default for dwell time analysis

    sigma_vals = np.logspace(np.log10(sigma_min),
                             np.log10(sigma_max), n_sigma)

    logging.info(
        f"▶︎ Dwell-time sweep → k={k_vals}, "
        f"σ∈[{sigma_min:.1e},{sigma_max:.1e}] "
        f"({n_sigma} pts), {trials} trials/pt")

    all_rows = []
    plt.figure(figsize=(12, 6))  # More horizontal aspect ratio for better data visualization

    for k in k_vals:
        mean_dt = []
        std_dt = []
        for σ in tqdm(sigma_vals, desc=f"dwell k={k:.2f}"):
            # Collect raw dwell times for statistical analysis
            dts = [dwell_time_trial(k, σ) for _ in range(trials)]
            dt_avg = float(np.mean(dts))
            dt_std = float(np.std(dts))
            mean_dt.append(dt_avg)
            std_dt.append(dt_std)
            
            # Store detailed data for CSV output
            all_rows.append({
                "k": k, 
                "sigma": σ, 
                "mean_dwell": dt_avg,
                "std_dwell": dt_std,
                "trials": trials
            })

        # Plot with error bars for scientific rigor
        plt.errorbar(sigma_vals, mean_dt, yerr=std_dt, 
                    marker='o', linestyle='-', capsize=3, linewidth=1.5,
                    markersize=4, label=f"k = {k:.2f}")

    plt.xlabel("Noise Intensity (σ)", fontsize=12)
    plt.ylabel("Average Dwell Time (s)", fontsize=12)
    plt.title("Memory Stability: Time Before Noise-Induced Flip\n" + 
              f"(Expanded range: σ∈[{sigma_min:.0e}, {sigma_max:.1f}], {n_sigma} points)", fontsize=13)
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    fig_path = OUT_DIR / "dwell_time_vs_sigma.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    logging.info(f"Dwell-time plot saved → {fig_path}")
    plt.close()

    # Save comprehensive data
    df_dwell = pd.DataFrame(all_rows)
    csv_path = OUT_DIR / "dwell_time_data.csv"
    df_dwell.to_csv(csv_path, index=False)
    logging.info(f"Raw dwell-time data saved → {csv_path}")

def generate_comprehensive_animations(subset: bool = False):
    """
    Generate animations with different starting points, amplitudes, noise levels,
    and k values. Creates unique filenames for each combination.
    
    Parameters:
    -----------
    subset : bool
        If True, generates a smaller subset for testing purposes
    """
    if subset:
        # Reduced parameter ranges for testing
        k_values = [0.50, 1.00]
        amp_values = [-1.0, 0.5, 1.0]
        sigma_values = [0.05, 0.2, 0.5]
        starting_points = [
            ("origin", np.array([0.0, 0.0])),
            ("q0_state", None),
            ("random1", np.array([0.5, -0.5])),
        ]
        logging.info("Using subset parameters for faster testing")
    else:
        # Define expanded parameter ranges
        k_values = [0.10, 0.50, 0.70, 1.00, 1.50]
        # Include negative amplitudes for exploring different dynamics
        amp_values = [-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5]
        # Expanded sigma range - all positive values for noise intensity
        sigma_values = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 1.0, 1.2]
        
        # Define different starting points with comprehensive coverage
        starting_points = [
            ("origin", np.array([0.0, 0.0])),
            ("q0_state", None),  # Will be computed using prepare_reset_state
            ("symmetric_pos", None),  # Will be computed based on k
            ("symmetric_neg", None),  # Will be computed based on k
            ("random1", np.array([0.5, -0.5])),
            ("random2", np.array([-0.8, 0.3])),
            ("random3", np.array([1.2, -1.1])),
            ("far_field1", np.array([2.0, -2.0])),
            ("far_field2", np.array([-1.8, 1.5])),
            ("near_saddle", np.array([0.1, -0.1])),
            ("high_v1", np.array([1.5, 0.2])),
            ("high_v2", np.array([0.2, 1.5])),
        ]
    
    # Use the results directory
    results_dir = OUT_DIR
    results_dir.mkdir(exist_ok=True)
    
    total_animations = len(k_values) * len(amp_values) * len(sigma_values) * len(starting_points)
    logging.info(f"Generating {total_animations} animations...")
    logging.info(f"Parameter ranges: k={len(k_values)}, amp={len(amp_values)}, sigma={len(sigma_values)}, starts={len(starting_points)}")
    
    animation_count = 0
    import time
    start_time = time.time()
    
    for k_idx, k in enumerate(tqdm(k_values, desc="K values")):
        for amp_idx, amp in enumerate(tqdm(amp_values, desc="Amplitudes", leave=False)):
            for sigma_idx, sigma in enumerate(tqdm(sigma_values, desc="Noise levels", leave=False)):
                for start_idx, (start_name, start_point) in enumerate(tqdm(starting_points, desc="Starting points", leave=False)):
                    animation_count += 1
                    
                    # Compute dynamic starting points based on k
                    if start_point is None:
                        if start_name == "q0_state":
                            start_point = prepare_reset_state(k, amp)  # Use absolute value for preparation
                        elif start_name == "symmetric_pos":
                            if k <= 1.0:
                                vs = np.sqrt(max(0, 1 - k))
                                start_point = np.array([vs, vs])
                            else:
                                start_point = np.array([0.1, 0.1])  # Fallback
                        elif start_name == "symmetric_neg":
                            if k <= 1.0:
                                vs = np.sqrt(max(0, 1 - k))
                                start_point = np.array([-vs, -vs])
                            else:
                                start_point = np.array([-0.1, -0.1])  # Fallback
                    
                    # Create unique filename with proper sign handling
                    amp_str = f"amp{amp:.2f}" if amp >= 0 else f"ampn{abs(amp):.2f}"
                    filename = f"animation_k{k:.2f}_sigma{sigma:.2f}_{amp_str}_start_{start_name}.gif"
                    save_path = results_dir / filename
                    
                    # Skip if file already exists
                    if save_path.exists():
                        logging.info(f"Skipping existing animation: {filename}")
                        continue
                    
                    try:
                        # Progress reporting with time estimation
                        if animation_count % 10 == 1 and animation_count > 1:
                            elapsed = time.time() - start_time
                            avg_time_per_animation = elapsed / (animation_count - 1)
                            remaining_animations = total_animations - animation_count + 1
                            estimated_remaining_time = avg_time_per_animation * remaining_animations
                            logging.info(f"Progress: {animation_count}/{total_animations} "
                                       f"({100*animation_count/total_animations:.1f}%) - "
                                       f"Est. remaining: {estimated_remaining_time/60:.1f} min")
                        
                        logging.info(f"Creating animation {animation_count}/{total_animations}: {filename}")
                        create_custom_animation(k, sigma, amp, start_point, start_name, save_path)
                    except Exception as e:
                        logging.error(f"Failed to create animation {filename}: {e}")
                        continue
    
    # Final summary
    total_time = time.time() - start_time
    logging.info(f"Animation generation completed! Total time: {total_time/60:.1f} minutes")
    logging.info(f"Generated {animation_count} animations in {results_dir}")

def create_custom_animation(k: float, sigma: float, amp: float, 
                          start_point: np.ndarray, start_name: str,
                          save_path: Path, rng: np.random.Generator | None = None) -> None:
    """
    Create a custom animation with specified starting point and parameters.
    """
    if rng is None:
        rng = globals().get("rng", np.random.default_rng())

    # Use the provided starting point instead of prepare_reset_state
    v_start = start_point.copy()
    
    # Define pulses - handle negative amplitudes properly
    IS_set = rectangular_pulse(0.0, 2.0, amp)
    # For negative amp, we reverse the roles or apply appropriate complementary pulse
    IR_set = rectangular_pulse(0.0, 2.0, -amp)

    # Simulation time for animation
    animation_T = 8.0  # Slightly longer for better visualization
    dt = min(DT, 0.005 / (1 + sigma))  # Adaptive time step
    
    # Run simulation
    path = euler_maruyama(k, IS_set, IR_set, sigma,
                          v0=v_start, T=animation_T, dt=dt, rng=rng, return_path=True)

    # Create phase plane background
    U_coords = np.linspace(-2.5, 2.5, GRID_N)
    V_coords = np.linspace(-2.5, 2.5, GRID_N)
    Ug, Vg = np.meshgrid(U_coords, V_coords)
    dU, dV = rhs(np.array([Ug, Vg]), k, 0.0, 0.0)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.streamplot(Ug, Vg, dU, dV, color='0.7', density=1, arrowsize=1)

    # Plot nullclines
    x_null = np.linspace(-2.5, 2.5, 400)
    if abs(k) > 1e-6:
        ax.plot(x_null, (x_null - x_null**3) / k, 'r--', lw=1.5, label='V1-nullcline')
        ax.plot((x_null - x_null**3) / k, x_null, 'b--', lw=1.5, label='V2-nullcline')

    # Plot equilibria
    equilibria = get_all_equilibria_and_classify(k)
    for eq in equilibria:
        v_eq = eq['v']
        eq_type = eq['type']
        if 'Attractor' in eq_type:
            ax.plot(v_eq[0], v_eq[1], 'go', ms=8, markerfacecolor='green', 
                   markeredgecolor='darkgreen', markeredgewidth=2)
        elif 'Saddle' in eq_type:
            ax.plot(v_eq[0], v_eq[1], 'rs', ms=8, markerfacecolor='red', 
                   markeredgecolor='darkred', markeredgewidth=2)
        elif 'Repeller' in eq_type:
            ax.plot(v_eq[0], v_eq[1], 'r^', ms=8, markerfacecolor='red', 
                   markeredgecolor='darkred', markeredgewidth=2)

    # Plot trajectory path (faded)
    ax.plot(path[:, 0], path[:, 1], 'gray', alpha=0.3, lw=1)
    
    # Moving point
    point, = ax.plot([], [], 'ko', ms=8, markerfacecolor='yellow', 
                    markeredgecolor='black', markeredgewidth=2)
    
    # Starting point marker
    start_marker, = ax.plot(v_start[0], v_start[1], 'g*', ms=12, 
                           markerfacecolor='lime', markeredgecolor='darkgreen', 
                           markeredgewidth=2, label='Start')

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_xlabel('$V_1$')
    ax.set_ylabel('$V_2$')
    ax.set_title(f'Trajectory: k={k:.2f}, σ={sigma:.2f}, amp={amp:.1f}, start={start_name}', fontsize=14)
    ax.grid(True, lw=0.3, alpha=0.4)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='upper right', fontsize=10)

    def init():
        point.set_data([], [])
        return point,

    def update(frame):
        point.set_data([path[frame, 0]], [path[frame, 1]])
        return point,

    # Create animation with slower frame rate for better visibility
    ani = animation.FuncAnimation(fig, update, frames=len(path),
                                  init_func=init, blit=True, interval=60)

    # Save animation
    total_frames = len(path)
    logging.info(f"Saving animation with {total_frames} frames to {save_path}")
    
    def progress_callback(current_frame, total_frames_from_func):
        if current_frame % 50 == 0:  # Log every 50 frames to reduce output
            logging.info(f"  Frame {current_frame + 1}/{total_frames_from_func}")

    ani.save(save_path, writer='pillow', fps=20, progress_callback=progress_callback)
    logging.info(f"Animation saved: {save_path}")
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND-LINE INTERFACE AND MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

def run_all(args):
    """
    Main execution function that coordinates all analysis components.
    
    This function serves as the central dispatcher that executes the requested
    analyses based on command-line arguments. It provides a comprehensive
    pipeline for studying the neural latch system across multiple dimensions:
    
    Analysis Components:
    -------------------
    1. Phase plane analysis: Equilibria, nullclines, and flow visualization
    2. Bifurcation analysis: Parameter-dependent stability changes
    3. Monte Carlo simulations: Switching probability under noise
    4. Truth table verification: Digital logic functionality testing
    5. Basin of attraction: Multistability and initial condition dependence
    6. Dwell time analysis: Memory stability under noise
    7. Animation generation: Dynamic trajectory visualization
    
    Parameters:
    -----------
    args : argparse.Namespace
        Parsed command-line arguments specifying which analyses to run
        
    Execution Strategy:
    ------------------
    - Analyses are executed in logical order with appropriate parameter ranges
    - Progress is logged for long-running computations
    - Results are saved in standardized formats for further analysis
    - Error handling ensures partial completion if individual analyses fail
    """
    if args.phase:
        logging.info("Generating phase plane diagrams...")
        # Expanded and varied k values for comprehensive phase plane analysis
        k_values_for_phase_plots = sorted(list(set([
            # Core critical values
            0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
            1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
            # Add values from the global range
            K_RANGE[0], K_RANGE[len(K_RANGE)//4], K_RANGE[len(K_RANGE)//2], 
            K_RANGE[3*len(K_RANGE)//4], K_RANGE[-1]
        ])))
        for k_val in tqdm(k_values_for_phase_plots, desc="Phase Planes"):
            plot_phase(k_val)
            plot_phase_ideal(k_val)  # Ideal case for comparison
        logging.info("Generating bistability vs k plot...")
        bistability_vs_k()
        
    if args.bifurcation:
        plot_bifurcation()
       
    if args.cont:
        logging.info("Attempting PyCont continuation (requires PyDSTool)...")
        if HAVE_PYDSTOOL:
            continuation()
        else:
            logging.warning("PyDSTool not installed; skipping continuation step.")

    if args.sweep:
        logging.info("Running Monte Carlo sweep for heatmaps...")
        sweep() 
    if args.dwell: # CHANGED from cli_args.dwell
        sweep_dwell_time()    
    if args.truth:
        logging.info("Running truth table verification...")
        truth_table()
    if args.deterministic: # CHANGED from cli_args.deterministic
        analyze_deterministic_switching()
        
    if args.animate:
        logging.info(f"Generating animation for k={args.k_animate}, sigma={args.sigma_animate}, amp={args.amp_animate}...")
        animation_filename = OUT_DIR / f"animation_k{args.k_animate:.2f}_sigma{args.sigma_animate:.2f}_amp{args.amp_animate:.2f}.gif"
        animate_trial(k=args.k_animate, sigma=args.sigma_animate, amp=args.amp_animate, save_path=animation_filename)
        # animate_trial function already logs the save path.
    
    if args.animate_all:
        logging.info("Generating comprehensive animations with different starting points, amplitudes, and noise levels...")
        generate_comprehensive_animations()
    
    if args.animate_subset:
        logging.info("Generating subset of animations for testing...")
        generate_comprehensive_animations(subset=True)
    
    if args.basins:
        logging.info("Generating basin of attraction analysis...")
        # Analyze basins for multiple representative k values
        k_values_for_basins = [0.1, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5]
        for k_val in tqdm(k_values_for_basins, desc="Basin Analysis"):
            plot_basins_of_attraction(k_val, resolution=150, save_plot=True)
        
    logging.info(f'All requested outputs saved in {OUT_DIR}/')

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION AND COMMAND-LINE ARGUMENT PARSING
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Coupled Neural Latch Analysis Pipeline",
        epilog="""
        This script provides a comprehensive analysis suite for studying the dynamics 
        of a two-cell bioelectric SR-latch system. The system exhibits rich multistable 
        behavior, noise-induced transitions, and complex bifurcation structure.
        
        Example usage:
          python equilibria-classification.py --all           # Run complete analysis
          python equilibria-classification.py --phase --bifurcation  # Phase planes + bifurcations
          python equilibria-classification.py --sweep --truth # Monte Carlo + logic verification
          python equilibria-classification.py --basins       # Basin of attraction analysis
          python equilibria-classification.py --animate --k_animate 0.7 --sigma_animate 0.2
        
        For questions or issues, consult the README.md file.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Core analysis options
    parser.add_argument("--all", action="store_true", 
                       help="Run all major analyses (phase, bifurcation, sweep, dwell, truth, deterministic). Excludes animation and continuation by default.")
    
    parser.add_argument("--phase", action="store_true", 
                       help="Generate phase plane diagrams and bistability vs k plot.")
    
    parser.add_argument("--bifurcation", action="store_true", 
                       help="Generate bifurcation diagram using numerical finder.")
    
    parser.add_argument("--cont", action="store_true", 
                       help="Run PyCont continuation (requires PyDSTool).")
    
    parser.add_argument("--sweep", action="store_true", 
                       help="Run Monte Carlo sweep for P(switch) heatmaps.")
    
    parser.add_argument("--dwell", action="store_true", 
                       help="Run Monte Carlo sweep for dwell time analysis.")
    
    parser.add_argument("--truth", action="store_true", 
                       help="Run truth table verification.")
    
    parser.add_argument("--deterministic", action="store_true", 
                       help="Run deterministic switching analysis (ideal world performance).")

    # Animation and visualization options
    parser.add_argument("--animate", action="store_true", 
                       help="Generate and save a trial animation.")
    
    parser.add_argument("--k_animate", type=float, default=0.5, 
                       help="Coupling strength (k) for animation. Default: 0.5")
    
    parser.add_argument("--sigma_animate", type=float, default=0.1, 
                       help="Noise intensity (sigma) for animation. Default: 0.1")
    
    parser.add_argument("--amp_animate", type=float, default=1.0, 
                       help="Pulse amplitude (amp) for animation. Default: 1.0")
    
    parser.add_argument("--animate_all", action="store_true", 
                       help="Generate comprehensive animations with different starting points, amplitudes, and noise levels for all k values.")
    
    parser.add_argument("--animate_subset", action="store_true", 
                       help="Generate a smaller subset of animations for testing (faster execution).")
    
    parser.add_argument("--basins", action="store_true", 
                       help="Generate basin of attraction analysis for multiple k values.")

    args = parser.parse_args()

    # Handle --all flag by enabling major analysis components
    if args.all:
        args.phase = True
        args.bifurcation = True
        # args.cont = False  # PyCont can be slow and has external deps, keep optional
        args.sweep = False  # MC sweep is too computationally expensive, keep optional
        args.dwell = True
        args.truth = True
        args.deterministic = True
        # Note: args.animate is not automatically enabled by --all

    # Determine if any action was requested
    action_requested = any(getattr(args, arg.dest) for arg in parser._actions 
                          if isinstance(arg, argparse._StoreTrueAction) and arg.dest not in ['help'])

    # If no action specified, show help and exit
    if not action_requested:
        # Check if any boolean flag is True
        is_any_action_explicitly_true = False
        for arg_name, arg_val in vars(args).items():
            action = next((act for act in parser._actions if act.dest == arg_name), None)
            if isinstance(action, argparse._StoreTrueAction) and arg_val:
                is_any_action_explicitly_true = True
                break
        
        if not is_any_action_explicitly_true:
            parser.print_help()
            sys.exit(0)
    
    # Execute the requested analyses
    logging.info("═" * 80)
    logging.info("COUPLED NEURAL LATCH ANALYSIS PIPELINE")
    logging.info("═" * 80)
    logging.info(f"Output directory: {OUT_DIR}")
    logging.info(f"Random seed: {SEED}")
    logging.info(f"CPU cores available: {CPU_COUNT}")
    logging.info("─" * 80)
    
    run_all(args)