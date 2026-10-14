"""
Test script for the semivectorial mode solver (svmodes).
Tests with a GRIN fiber index profile loaded from file.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from svmodes import svmodes


def test_GRIN_fiber(wavelength, nmodes, core_radius, n_profile=None):
    """
    Test the mode solver with a GRIN fiber using loaded index profile.
    """
    print("=" * 60, flush=True)
    print("GRIN Fiber Mode Solver Test", flush=True)
    print("=" * 60, flush=True)

    if n_profile is None:
        raise ValueError("Index profile is not provided")

    # Grid parameters (from loaded profile)
    nx, ny = n_profile.shape
    wavelength_nm = int(round(wavelength * 1000))

    # Infer pixel size from the index profile by finding the core boundary.
    # The radial cut from center has n ~ n_clad outside r = core_radius.
    n_core = n_profile.max()
    n_clad = n_profile.min()
    center = nx // 2
    radial = n_profile[center, center:]
    core_boundary_px = int(np.argmax(np.abs(radial - n_clad) < 1e-4))
    if core_boundary_px == 0:
        core_boundary_px = nx // 4  # fallback
    pixel_size = core_radius / core_boundary_px  # microns / pixel

    # Computational domain
    domain_size_x = nx * pixel_size
    domain_size_y = ny * pixel_size
    x_range = (-domain_size_x / 2, domain_size_x / 2)
    y_range = (-domain_size_y / 2, domain_size_y / 2)

    # Grid spacing
    dx = pixel_size
    dy = pixel_size

    # Calculate permittivity from refractive index
    eps = n_profile**2

    # Create coordinate arrays
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)

    print(f"\nGrid size: {nx} x {ny}", flush=True)
    print(f"Domain size: {domain_size_x:.3f} x {domain_size_y:.3f} microns", flush=True)
    print(f"Grid spacing: dx = {dx:.4f} um, dy = {dy:.4f} um", flush=True)
    print(f"Inferred core boundary at pixel: {core_boundary_px}", flush=True)
    print(f"\nFiber parameters:", flush=True)
    print(f"  Wavelength: {wavelength} um", flush=True)
    print(f"  Core radius: {core_radius} um", flush=True)
    print(f"  n_max (core): {n_core:.6f}", flush=True)
    print(f"  n_min (clad): {n_clad:.6f}", flush=True)

    # Mode solver parameters
    guess = n_core + 1e-6  # Initial guess just above n_core for shift-invert
    boundary = '0000'  # Zero boundary conditions (absorbing)

    # Use scalar field for LP modes (valid for weakly guiding GRIN fibers)
    field = 'scalar'
    print(f"\n{'-'*60}", flush=True)
    print(f"Solving for {nmodes} LP modes (scalar)...", flush=True)
    print(f"{'-'*60}", flush=True)

    start_time = time.time()
    try:
        phi, neff = svmodes(wavelength, guess, nmodes, dx, dy, eps, boundary, field)
        elapsed = time.time() - start_time

        print(f"Mode solving time: {elapsed:.2f} seconds ({elapsed/nmodes:.2f} sec/mode)", flush=True)
        print(f"\nEffective indices (neff):", flush=True)
        for i, n_eff in enumerate(neff):
            if np.iscomplex(n_eff):
                print(f"  Mode {i+1}: {n_eff.real:.6f} + {n_eff.imag:.2e}j", flush=True)
            else:
                print(f"  Mode {i+1}: {n_eff:.6f}", flush=True)

        # Verify results
        print(f"\nVerification:", flush=True)
        print(f"  All neff between n_clad and n_core: ", end="", flush=True)
        neff_real = np.real(neff)
        if np.all((neff_real >= n_clad - 0.001) & (neff_real <= n_core + 0.001)):
            print("PASS", flush=True)
        else:
            print("WARNING - some neff out of expected range", flush=True)

        # Propagation constants: beta = 2*pi*neff / wavelength (rad/um)
        mode_numbers = np.arange(1, nmodes + 1)
        beta = 2 * np.pi * np.real(neff) / wavelength

        print(f"\nPropagation constants (beta):", flush=True)
        for i, b in enumerate(beta):
            print(f"  Mode {i+1}: {b:.6f} rad/um", flush=True)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(mode_numbers, beta, 'o-', color='#2563eb', markersize=6, linewidth=1)
        ax.set_xlabel('Mode number')
        ax.set_ylabel('Propagation constant β (rad/μm)')
        ax.set_title(f'Propagation Constant vs Mode Number\nGRIN fiber, {wavelength_nm} nm, {nmodes} modes')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(mode_numbers)
        plt.tight_layout()
        beta_plot_path = f'./MMF_LP_modes/beta_vs_mode_GRIN_{wavelength_nm}_{nx}_{nmodes}modes.png'
        plt.savefig(beta_plot_path, dpi=200)
        print(f"  Beta plot saved: {beta_plot_path}", flush=True)
        plt.close(fig)

        # Plot all modes in a grid
        nplot = nmodes
        ncols = int(np.ceil(np.sqrt(nplot)))
        nrows = int(np.ceil(nplot / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)
        fig.suptitle(f'LP Modes - GRIN Fiber ({nx}x{ny} grid, {wavelength_nm} nm, {nplot} modes)', fontsize=14)

        for ax in axes.flat[nplot:]:
            ax.axis('off')

        for i in range(nplot):
            ax = axes[i // ncols, i % ncols]
            mode_field = np.real(phi[:, :, i])

            # Plot mode intensity
            im = ax.imshow(np.abs(mode_field)**2,
                          extent=[x[0], x[-1], y[0], y[-1]],
                          cmap='turbo', origin='lower')

            ax.set_xlabel('x (μm)', fontsize=8)
            ax.set_ylabel('y (μm)', fontsize=8)
            ax.set_title(f'Mode {i+1}: neff={np.real(neff[i]):.5f}', fontsize=9)
            ax.set_aspect('equal')
            ax.tick_params(labelsize=7)
            plt.colorbar(im, ax=ax, label='|E|²')

        plt.tight_layout()
        plot_path = f'./MMF_LP_modes/mode_solver_test_LP_GRIN_{wavelength_nm}_{nx}.png'
        plt.savefig(plot_path, dpi=200)
        print(f"  Plot saved: {plot_path}", flush=True)

        # Reshape modes to (M, Nx, Ny) format and save
        phi_out = np.transpose(phi, (2, 0, 1))  # (Nx, Ny, M) -> (M, Nx, Ny)
        modes_path = f'./MMF_LP_modes/modes_{nx}x{ny}_GRIN_{wavelength_nm}_{nmodes}modes_py.npy'
        np.save(modes_path, phi_out)
        print(f"\nModes saved to: {modes_path}", flush=True)
        print(f"  Shape: {phi_out.shape}", flush=True)

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"ERROR after {elapsed:.2f} seconds: {e}", flush=True)
        import traceback
        traceback.print_exc()

    plt.close('all')



if __name__ == '__main__':

    wavelength = 1.030 # microns
    nmodes = 28
    Nx = 512
    Ny = 512
    core_radius = 50.0 / 2       # microns
    fiber_type = 'GRIN'
    wavelength_nm = int(round(wavelength * 1000))

    print("Mode Solver Test Suite", flush=True)
    print("=" * 60, flush=True)
    print(f"Testing svmodes with {fiber_type} fiber ({wavelength_nm} nm, {Nx}x{Ny} grid, {nmodes} modes)\n", flush=True)

    # Run tests
    index_file = f'./n_{Nx}_{fiber_type}_{wavelength_nm}.npy'
    n_profile = np.load(index_file)
    test_GRIN_fiber(wavelength, nmodes, core_radius, n_profile)

    print("\n" + "=" * 60, flush=True)
    print("All tests completed!", flush=True)
    print("=" * 60, flush=True)
