import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from tqdm import tqdm

# Define the system dynamics
def rhs(v, k):
    v1, v2 = v
    return np.array([
        v1 - v1**3 - k*v2,
        v2 - v2**3 - k*v1
    ])

# Jacobian for stability analysis
def jac(v, k):
    v1, v2 = v
    return np.array([
        [1 - 3*v1**2, -k],
        [-k, 1 - 3*v2**2]
    ])

# Numerically find equilibria
def find_equilibria(k):
    def equations(v):
        return rhs(v, k)

    guesses = [np.array([x, y]) for x in np.linspace(-2, 2, 5) for y in np.linspace(-2, 2, 5)]
    equilibria = []

    for guess in guesses:
        eq, infodict, ier, mesg = fsolve(equations, guess, full_output=True)
        if ier == 1 and not any(np.allclose(eq, existing, atol=1e-4) for existing in equilibria):
            equilibria.append(eq)

    return equilibria

# Classify equilibria by stability
def classify_equilibria(equilibria, k):
    stable_points = []
    unstable_points = []
    for eq in equilibria:
        eigvals = np.linalg.eigvals(jac(eq, k))
        if np.all(np.real(eigvals) < 0):
            stable_points.append(eq)
        else:
            unstable_points.append(eq)
    return stable_points, unstable_points

# Euler method integration (fast approximation)
def integrate_to_equilibrium(v0, k, dt=0.01, T=50):
    v = np.array(v0)
    for _ in range(int(T/dt)):
        dv = rhs(v, k) * dt
        v += dv
        if np.linalg.norm(dv) < 1e-6:
            break
    return v

# Main function to plot basins of attraction
def plot_basins_of_attraction(k, resolution=200):
    # Find stable attractors
    equilibria = find_equilibria(k)
    stable_points, _ = classify_equilibria(equilibria, k)

    if not stable_points:
        print(f"No stable equilibria found for k={k}. Exiting.")
        return

    bounds = (-2.5, 2.5)
    v1_coords = np.linspace(bounds[0], bounds[1], resolution)
    v2_coords = np.linspace(bounds[0], bounds[1], resolution)
    basin_grid = np.zeros((resolution, resolution))

    # Compute basin grid
    for i, v1 in enumerate(tqdm(v1_coords, desc="Computing Basins")):
        for j, v2 in enumerate(v2_coords):
            v_final = integrate_to_equilibrium([v1, v2], k)
            distances = [np.linalg.norm(v_final - attractor) for attractor in stable_points]
            basin_id = np.argmin(distances)
            basin_grid[j, i] = basin_id  # note [j,i] indexing for correct orientation

    # Plotting
    plt.figure(figsize=(8, 7))
    cmap = plt.cm.get_cmap('tab10', len(stable_points))
    plt.imshow(basin_grid, extent=[bounds[0], bounds[1], bounds[0], bounds[1]],
               origin='lower', cmap=cmap, interpolation='nearest', aspect='auto')

    # Plot attractors
    for idx, attractor in enumerate(stable_points):
        plt.plot(attractor[0], attractor[1], 'o', markersize=10,
                 markeredgecolor='black', color=cmap(idx), label=f'Attractor {idx+1}')

    plt.xlabel('$V_1$')
    plt.ylabel('$V_2$')
    plt.title(f'Basins of Attraction (k={k:.2f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    k = 3  # Choose your desired coupling parameter
    plot_basins_of_attraction(k)
