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

# ──────────────────────────────────────────────────────────────── global CONFIG —
OUT_DIR          = Path("results"); OUT_DIR.mkdir(exist_ok=True)
K_RANGE          = np.linspace(0.10, 1.50, 10) # Reduced for faster testing, adjust as needed
SIGMA_RANGE      = np.linspace(0.0, 0.40, 9)
AMP_RANGE        = np.linspace(0.0, 2.0 , 9) # Reduced for faster testing
DT, T_TOTAL      = 1e-2, 20.0 # Reduced T_TOTAL for switch_trial, ensure it's enough
TRIALS_PER_TASK  = 50 # Reduced for faster testing
GRID_N           = 25
CPU_COUNT        = max(1, mp.cpu_count() - 2 if mp.cpu_count() > 1 else 1) # Leave one or two cores free
SEED             = 42                              # reproducible RNG
rng              = np.random.default_rng(SEED)

plt.rcParams.update({'figure.dpi': 120}) # Slightly reduced DPI for faster plot generation
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy.optimize")
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

# ─────────────────────────────────────────────────────────────── model dynamics —
def rhs(v: np.ndarray, k: float, IS: float, IR: float) -> np.ndarray:
    """Deterministic part  f(v,k,IS,IR)."""
    v1, v2 = v
    return np.array([v1 - v1**3 - k*v2 + IS,
                     v2 - v2**3 - k*v1 + IR])

def jac(v_eq: np.ndarray, k: float) -> np.ndarray:
    v1, v2 = v_eq
    return np.array([[1 - 3*v1**2, -k],
                     [-k,           1 - 3*v2**2]])

# ─────────────────────────────────────────────────────────── equilibrium search —
def get_all_equilibria_and_classify(k: float, tol: float = 1e-9) -> list[dict]:
    """
    Finds all equilibria (origin, symmetric, anti-symmetric) for the model
    dv/dt = v - v^3 - k*v_other and classifies them.
    Returns a list of dictionaries, each with 'v' (coordinates) and 'type' (str).
    """
    equilibria_list = []

    # 1. Origin
    v_origin = np.array([0.0, 0.0])
    J_origin = jac(v_origin, k)
    eig_origin = np.linalg.eigvals(J_origin)
    
    origin_type = "Unknown"
    if all(np.real(eig_origin) < -tol): origin_type = "Attractor"
    elif all(np.real(eig_origin) > tol): origin_type = "Repeller"
    elif np.real(eig_origin[0]) * np.real(eig_origin[1]) < -tol**2 : origin_type = "Saddle"
    else: # Handle non-hyperbolic cases or close to zero eigenvalues
        if np.any(abs(np.real(eig_origin)) < tol): origin_type = "Non-Hyperbolic"
        if (np.real(eig_origin[0]) > tol and abs(np.real(eig_origin[1])) < tol) or \
           (abs(np.real(eig_origin[0])) < tol and np.real(eig_origin[1]) > tol):
            origin_type = "Unstable (non-hyperbolic)" # e.g. one positive, one zero
        elif (np.real(eig_origin[0]) < -tol and abs(np.real(eig_origin[1])) < tol) or \
             (abs(np.real(eig_origin[0])) < tol and np.real(eig_origin[1]) < -tol):
            origin_type = "Stable (non-hyperbolic)" # e.g. one negative, one zero


    equilibria_list.append({'v': v_origin, 'type': origin_type, 'source': 'Origin'})

    # 2. Symmetric equilibria: v1 = v2 = vs
    # vs^2 = 1 - k. Exist if k <= 1.
    if k <= 1.0 + tol:
        val_1_minus_k = 1.0 - k
        if val_1_minus_k >= -tol: # Allow for small negative due to precision
            vs_val = np.sqrt(max(0, val_1_minus_k))
            if abs(vs_val) > tol: # Distinct from origin
                for sign in [-1, 1]:
                    v_sym = np.array([sign * vs_val, sign * vs_val])
                    J_sym = jac(v_sym, k)
                    eig_sym = np.linalg.eigvals(J_sym)
                    sym_type = "Unknown"
                    if all(np.real(eig_sym) < -tol): sym_type = "Attractor"
                    elif all(np.real(eig_sym) > tol): sym_type = "Repeller"
                    elif np.real(eig_sym[0]) * np.real(eig_sym[1]) < -tol**2: sym_type = "Saddle"
                    equilibria_list.append({'v': v_sym, 'type': sym_type, 'source': 'Symmetric'})
    
    # 3. Anti-symmetric equilibria: v1 = -v2 = va
    # va^2 = 1 + k. Exist for k >= 0 (always, as K_RANGE starts >= 0.1)
    val_1_plus_k = 1.0 + k
    if val_1_plus_k >= -tol: # Should always be true for k>=0
        va_val = np.sqrt(max(0, val_1_plus_k))
        if abs(va_val) > tol: # Distinct from origin
            for sign in [-1, 1]:
                v_anti = np.array([sign * va_val, -sign * va_val])
                J_anti = jac(v_anti, k)
                eig_anti = np.linalg.eigvals(J_anti)
                anti_type = "Unknown"
                if all(np.real(eig_anti) < -tol): anti_type = "Attractor"
                elif all(np.real(eig_anti) > tol): anti_type = "Repeller"
                elif np.real(eig_anti[0]) * np.real(eig_anti[1]) < -tol**2: anti_type = "Saddle"
                equilibria_list.append({'v': v_anti, 'type': anti_type, 'source': 'Anti-symmetric'})
                
    # Deduplicate (e.g. if k=1, symmetric might coincide with origin)
    unique_equilibria = []
    seen_coords = set()
    for eq in equilibria_list:
        coord_tuple = tuple(np.round(eq['v'], 5)) # Round to avoid precision issues in set
        if coord_tuple not in seen_coords:
            unique_equilibria.append(eq)
            seen_coords.add(coord_tuple)
            
    return unique_equilibria

# ─────────────────────────────────────────────── Euler–Maruyama SDE integrator —
# --------------------------- numerical equilibrium finder --------------------
def find_equilibria_numerically(k: float, n_guesses: int = 100,
                                search_bounds=(-2.5, 2.5), tol: float = 1e-5) -> list[np.ndarray]:
    """Return unique equilibrium points found via numerical root finding."""
    func = lambda v: rhs(v, k, 0.0, 0.0)
    rng_local = np.random.default_rng()
    roots = []
    for _ in range(n_guesses):
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
    """Plot equilibrium branches vs. k using the numerical finder."""
    logging.info("Generating bifurcation diagram...")
    k_vals = np.linspace(min(K_RANGE), max(K_RANGE), 100)
    records = []
    for k in tqdm(k_vals, desc="Bifurcation Analysis"):
        for eq in find_equilibria_numerically(k, n_guesses=50):
            eigs = np.linalg.eigvals(jac(eq, k))
            stable = all(np.real(eigs) < 0)
            records.append({'k': k, 'v1': eq[0], 'stable': stable})

    df = pd.DataFrame(records)
    plt.figure(figsize=(8,6))
    plt.plot(df[df['stable']]['k'], df[df['stable']]['v1'], 'bo', markersize=3, label='Stable')
    plt.plot(df[~df['stable']]['k'], df[~df['stable']]['v1'], 'ro', markersize=2, label='Unstable')
    plt.xlabel('Coupling Strength k')
    plt.ylabel('$V_1$ at equilibrium')
    plt.title('Bifurcation Diagram')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.savefig(OUT_DIR / 'bifurcation_diagram.png')
    plt.close()




def euler_maruyama(k: float,
                   IS_fun, IR_fun,
                   sigma: float,
                   v0: np.ndarray | None = None,
                   T: float = T_TOTAL,
                   dt: float = DT,
                  rng: np.random.Generator | None = None,
                   return_path: bool = False) -> np.ndarray:
    """Integrate SDE from v0 (default (0,0)).

    By default only the final state is returned.  If ``return_path`` is
    ``True`` the full trajectory array of shape (N+1, 2) is returned.
    """
    if rng is None:
            rng = globals().get("rng", np.random.default_rng())
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

# ──────────────────────────────────────────────── pulse + prep‑state utilities —
def rectangular_pulse(t0: float, t1: float, amp: float):
    """Return a function I(t) representing a rectangular current pulse."""
    return lambda t: amp if (t0 <= t <= t1) else 0.0

def prepare_reset_state(k: float, amp: float, sigma: float = 0.0) -> np.ndarray:
    """Drive latch to Q=0 (v1 low, v2 high) deterministically."""
    # For Q=0 (v1 low, v2 high), we want to push v1 down and v2 up.
    # However, the model is symmetric. Let's aim for v1 ~ -1, v2 ~ 1.
    # A strong pulse on IR (for v2) and no pulse or negative on IS (for v1)
    # For simplicity, let's use a strong positive pulse on IR to get v2 high,
    # and rely on coupling and intrinsic dynamics for v1 to go low.
    # Or, more directly, start near an anti-symmetric state.
    # The original code uses a positive IR pulse.
    IR_pulse = rectangular_pulse(0.0, 5.0, amp) # Pulse on v2
    IS_pulse = lambda t: 0.0                    # No pulse on v1
    # Start from a neutral state or slightly perturbed
    initial_state = rng.random(2) * 0.1 - 0.05 
    return euler_maruyama(k, IS_pulse, IR_pulse, sigma, v0=initial_state, T=7.0)


# ───────────────────────────────────────────────────────── phase‑plane plotting —
def plot_phase(k: float):
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
    animation_T = 5.0  # Simulate for 5 seconds for the animation (e.g., 501 frames if DT=0.01)
                       # Original T_TOTAL is 20.0, leading to 2001 frames.
    path = euler_maruyama(k, IS_set, IR_set, sigma,
                          v0=v_start, T=animation_T, dt=DT, rng=rng, return_path=True)

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

    ani = animation.FuncAnimation(fig, update, frames=len(path),
                                  init_func=init, blit=True, interval=40)

    if save_path is not None:
        total_frames = len(path)
        logging.info(f"Starting animation save. Total frames: {total_frames}")
        
        def progress_callback(current_frame, total_frames_from_func):
            # current_frame is 0-indexed
            logging.info(f"Saving frame {current_frame + 1} of {total_frames_from_func}")

        ani.save(save_path, writer='pillow', fps=25, progress_callback=progress_callback)
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

# ───────────────────────────────────────────── Monte‑Carlo switch probability —
def switch_trial(k: float, sigma: float, amp: float,rng: np.random.Generator | None = None, switch_threshold: float = 0.1 ) -> int:
    """Single trial → 1 if Set pulse (V1 high, V2 low) succeeds, 0 otherwise."""
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
    
# ──────────────────────────────────────────────── grid sweep / CSV / 2D Heatmaps —
def plot_2d_heatmap(df_k: pd.DataFrame, k_val: float, out_dir: Path):
    """Plots a 2D heatmap of P(switch) vs sigma and amp for a given k."""
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


# ───────────────────────────────────────────────────── bistability vs k (quick) —
def bistability_vs_k():
    counts = []
    k_values_for_plot = np.linspace(min(K_RANGE), max(K_RANGE), 50) # Smoother plot
    for k_val in k_values_for_plot:
        eq_k = get_all_equilibria_and_classify(k_val)
        counts.append(sum(1 for e in eq_k if e['type'] == 'Attractor'))
    
    plt.figure(figsize=(7,5))
    plt.plot(k_values_for_plot, counts, 'o-', color='darkslateblue', markersize=5)
    plt.xlabel('Coupling Strength (k)')
    plt.ylabel('Number of Stable Attractors')
    plt.title('System Stability vs. Coupling Strength k')
    plt.yticks(np.arange(min(counts)-1 if counts else 0, max(counts)+2 if counts else 2, 1)) # Integer y-ticks
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.savefig(OUT_DIR / 'attractors_vs_k.png')
    plt.close()

# ───────────────────────────────────────────── truth‑table verification (σ=0) —
def truth_table():
    # This function might need adjustments based on the refined switch_trial logic
    # and how "Set" and "Reset" pulses are defined and tested.
    # For now, keeping its structure but noting it might need review.
    k_fixed = 0.5
    amp_fixed = 1.0 # A representative amplitude for pulses
    logging.info(f"Running truth table for k={k_fixed}, amp={amp_fixed}")

    # Define states: Q=1 (v1 high, v2 low), Q=0 (v1 low, v2 high)
    # Target for Q=1: e.g., (sqrt(1+k), -sqrt(1+k))
    # Target for Q=0: e.g., (-sqrt(1+k), sqrt(1+k))
    
    # Helper to simulate and check state
    def check_final_state(v_final, target_v1_pos=True, threshold=0.1):
        if target_v1_pos: # Q=1 like state
            return int(v_final[0] > threshold and v_final[1] < -threshold)
        else: # Q=0 like state
            return int(v_final[0] < -threshold and v_final[1] > threshold)

    # 1. Hold Q=0: Start at Q=0, no pulse, should remain Q=0
    v_start_q0 = np.array([-np.sqrt(1+k_fixed), np.sqrt(1+k_fixed)]) if (1+k_fixed)>0 else np.array([0.0, 0.0])
    v_hold_q0 = euler_maruyama(k_fixed, lambda t:0, lambda t:0, 0.0, v0=v_start_q0, T=T_TOTAL)
    hold_q0_result = check_final_state(v_hold_q0, target_v1_pos=False) # Expect Q=0

    # 2. Set (from Q=0 to Q=1): Start at Q=0, Set pulse, should go to Q=1
    IS_set = rectangular_pulse(0, 2, amp_fixed)
    IR_set = rectangular_pulse(0, 2, -amp_fixed) # Push V2 low
    v_set_q1 = euler_maruyama(k_fixed, IS_set, IR_set, 0.0, v0=v_start_q0, T=T_TOTAL)
    set_q1_result = check_final_state(v_set_q1, target_v1_pos=True) # Expect Q=1

    # 3. Hold Q=1: Start at Q=1, no pulse, should remain Q=1
    v_start_q1 = np.array([np.sqrt(1+k_fixed), -np.sqrt(1+k_fixed)]) if (1+k_fixed)>0 else np.array([0.0, 0.0])
    v_hold_q1 = euler_maruyama(k_fixed, lambda t:0, lambda t:0, 0.0, v0=v_start_q1, T=T_TOTAL)
    hold_q1_result = check_final_state(v_hold_q1, target_v1_pos=True) # Expect Q=1
    
    # 4. Reset (from Q=1 to Q=0): Start at Q=1, Reset pulse, should go to Q=0
    # Reset pulse: IS low (or negative), IR high (or positive)
    IS_reset = rectangular_pulse(0, 2, -amp_fixed) # Push V1 low
    IR_reset = rectangular_pulse(0, 2, amp_fixed)  # Push V2 high
    v_reset_q0 = euler_maruyama(k_fixed, IS_reset, IR_reset, 0.0, v0=v_start_q1, T=T_TOTAL)
    reset_q0_result = check_final_state(v_reset_q0, target_v1_pos=False) # Expect Q=0

    # 5. Invalid (Set and Reset simultaneously) - behavior might be undefined or go to a specific state
    # Let's assume starting from Q=0
    IS_invalid = rectangular_pulse(0, 2, amp_fixed)
    IR_invalid = rectangular_pulse(0, 2, amp_fixed) # Both positive pulses
    v_invalid = euler_maruyama(k_fixed, IS_invalid, IR_invalid, 0.0, v0=v_start_q0, T=T_TOTAL)
    # How to classify invalid? Let's say if it's not Q=0 and not Q=1, or if it's (0,0)
    is_q0_invalid = check_final_state(v_invalid, target_v1_pos=False)
    is_q1_invalid = check_final_state(v_invalid, target_v1_pos=True)
    invalid_result = 1 if not is_q0_invalid and not is_q1_invalid else 0 # Or some other logic

    res_data = {
        'Hold_Q0_stays_Q0': hold_q0_result, # Expected 1
        'Set_Q0_to_Q1': set_q1_result,       # Expected 1
        'Hold_Q1_stays_Q1': hold_q1_result, # Expected 1
        'Reset_Q1_to_Q0': reset_q0_result,   # Expected 1
        'Invalid_input_behavior': invalid_result # Behavior specific
    }
    res_series = pd.Series(res_data)
    res_series.to_csv(OUT_DIR / 'truth_table_results.csv', header=False)
    logging.info(f"Truth table results:\n{res_series}")


# ─────────────────────────────────────────────────────────────── CLI + main —──
def run_all(args):
    if args.phase:
        logging.info("Generating phase plane diagrams...")
        k_values_for_phase_plots = sorted(list(set([0.25, 0.75, 1.0, 1.25] + [K_RANGE[0], K_RANGE[len(K_RANGE)//2], K_RANGE[-1]])))
        for k_val in tqdm(k_values_for_phase_plots, desc="Phase Planes"):
            plot_phase(k_val)
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
        sweep() # This function now also handles plotting the heatmaps
        
    if args.truth:
        logging.info("Running truth table verification...")
        truth_table()
        
    logging.info(f'All requested outputs saved in {OUT_DIR}/')

if __name__ == '__main__':
    if sys.platform.startswith('win'): # mp.freeze_support() is for Windows when freezing
        mp.freeze_support()
        
    parser = argparse.ArgumentParser(description='Two‑cell SR‑latch pipeline')
    parser.add_argument('--phase', action='store_true', help='Generate phase‑plane diagrams and basic stability plot')
    parser.add_argument('--bifurcation', action='store_true', help='Generate V1 vs k bifurcation diagram')
    parser.add_argument('--cont',  action='store_true', help='Run PyCont equilibrium continuation (if available)')
    parser.add_argument('--sweep', action='store_true', help='Run Monte‑Carlo parameter sweeps and generate heatmaps')
    parser.add_argument('--truth', action='store_true', help='Run deterministic truth‑table check')
    parser.add_argument('--all', action='store_true', help='Run all analysis parts (phase, sweep, truth)')
    
    cli_args = parser.parse_args()

    # If --all is specified, set individual flags
    if cli_args.all:
        cli_args.phase = True
        cli_args.sweep = True
        cli_args.truth = True
        cli_args.bifurcation = True
        # cli_args.cont = True # User wants to forget about PyCont

    # If no specific flag or --all is given, run a default set
    if not any([cli_args.phase, cli_args.cont, cli_args.sweep, cli_args.truth]):
        logging.info("No specific analysis selected, running default (phase, sweep, truth). Use --help for options.")
        cli_args.phase = True
        cli_args.sweep = True
        cli_args.truth = True
        cli_args.bifurcation = True
        # cli_args.cont = True # Default does not run cont

    run_all(cli_args)