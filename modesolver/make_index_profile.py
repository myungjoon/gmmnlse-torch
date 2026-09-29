"""
Generate 2D refractive index profiles for the mode solver (svmodes).
Saves n_{N}_{TYPE}_{wvl}.npy files compatible with test-modesolver.py.
All lengths are in microns.
"""

import numpy as np
import matplotlib.pyplot as plt


def make_grid(N, domain_size):
    """Return meshgrid (X, Y) and radial distance R for an N x N grid."""
    x = np.linspace(-domain_size / 2, domain_size / 2, N)
    X, Y = np.meshgrid(x, x, indexing='ij')
    R = np.sqrt(X**2 + Y**2)
    return X, Y, R


def grin_profile(N, domain_size, core_radius, n_core, n_clad):
    """Parabolic graded-index profile: n(r)^2 = n_core^2 (1 - 2*Delta*(r/a)^2) for r <= a."""
    _, _, R = make_grid(N, domain_size)
    delta = (n_core**2 - n_clad**2) / (2 * n_core**2)
    n_sq = n_core**2 * (1 - 2 * delta * (R / core_radius)**2)
    n = np.where(R <= core_radius, np.sqrt(n_sq), n_clad)
    return n


def step_profile(N, domain_size, core_radius, n_core, n_clad):
    """Step-index profile: n = n_core for r <= a, n_clad otherwise."""
    _, _, R = make_grid(N, domain_size)
    return np.where(R <= core_radius, n_core, n_clad)


if __name__ == '__main__':

    # Parameters
    fiber_type = 'GRIN'      # 'GRIN' or 'step'
    wavelength = 1.03       # microns (used only for the output filename)
    N = 512                  # grid points per side
    core_radius = 50.0 / 2       # microns
    domain_size = 4 * core_radius  # microns
    n_clad = 1.4497
    NA = 0.200
    n_core = np.sqrt(n_clad**2 + NA**2)

    print(f"Fiber type: {fiber_type}")
    print(f"Grid: {N} x {N}, domain {domain_size:.1f} um, pixel {domain_size / N:.4f} um")
    print(f"Core radius: {core_radius} um")
    print(f"n_core = {n_core:.6f}, n_clad = {n_clad:.6f} (NA = {NA})")

    if fiber_type == 'GRIN':
        n = grin_profile(N, domain_size, core_radius, n_core, n_clad)
    elif fiber_type == 'step':
        n = step_profile(N, domain_size, core_radius, n_core, n_clad)
    else:
        raise ValueError(f"Unknown fiber type: {fiber_type}")

    wvl_nm = int(round(wavelength * 1000))
    out_path = f'n_{N}_{fiber_type}_{wvl_nm}.npy'
    np.save(out_path, n)
    print(f"Saved {out_path} with shape {n.shape}, n in [{n.min():.6f}, {n.max():.6f}]")

    # 1D profile through the center
    n_1d = n[N // 2, :]
    x = np.linspace(-domain_size / 2, domain_size / 2, N)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, n_1d)
    ax.set_xlabel('x (μm)')
    ax.set_ylabel('n')
    ax.set_title(f'{fiber_type} index profile, center cut ({N}px)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    png_path_1d = f'n_{N}_{fiber_type}_{wvl_nm}_1D.png'
    plt.savefig(png_path_1d, dpi=300)
    plt.close(fig)
    print(f"Saved {png_path_1d}")

    # Preview plot
    extent = [-domain_size / 2, domain_size / 2, -domain_size / 2, domain_size / 2]
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(n, extent=extent, cmap='turbo', origin='lower')
    ax.set_xlabel('x (μm)')
    ax.set_ylabel('y (μm)')
    ax.set_title(f'{fiber_type} index profile ({N}x{N})')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='n')
    plt.tight_layout()
    png_path = f'n_{N}_{fiber_type}_{wvl_nm}.png'
    plt.savefig(png_path, dpi=300)
    print(f"Saved {png_path}")
